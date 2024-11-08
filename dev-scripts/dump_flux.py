import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import FluxPipeline, FluxTransformer2DModel
from diffusers.models.transformers.transformer_flux import FluxSingleTransformerBlock, FluxTransformerBlock
from diffusers.models.attention import FeedForward
import safetensors.torch
from dataclasses import dataclass
from typing import Optional

import qmodule

TensorDict = dict[str, torch.Tensor]

@dataclass
class DeepCompressorModel:
    model:  dict[str, torch.Tensor]
    smooth: dict[str, torch.Tensor]
    branch: dict[str, dict[str, torch.Tensor]]
    lora:   dict[str, torch.Tensor]

def merge_dict(old: dict, new: dict, prefix: str):
    for key, value in new.items():
        newkey = prefix + key
        assert not newkey in old
        old[newkey] = value

def group_scale(weight: torch.Tensor, num_bits: int, group_size: int) -> torch.Tensor:
    oc, ic = weight.shape
    assert ic % group_size == 0
    maxvalues = weight.reshape(oc, ic // group_size, group_size).abs().max(dim=-1).values
    qmax = 2 ** (num_bits - 1) - 1
    scale = (maxvalues.float() / qmax) 
    scale[scale == 0] = 1
    return scale.to(weight.dtype)

def ceil_div(x, y):
    return (x + y - 1) // y

def quantize_4bit(weight: torch.Tensor, wscales: torch.Tensor) -> torch.Tensor:
    oc, ic = weight.shape
    group_size = ic // wscales.shape[-1]

    # print(group_size)
    # print(weight.shape)
    # print(wscales.shape)

    # print(f"wscales={wscales}")

    qweight = weight.reshape(oc, ic // group_size, group_size).to(dtype=torch.float32) / wscales[..., None]

    # print(f"qweight={qweight}")

    qweight = qweight.reshape(oc, ic // 8, 8).round().clamp(-8, 7).to(dtype=torch.int32)
    qweight = qweight.bitwise_and_(0xf)

    shift = torch.arange(0, 32, 4, dtype=torch.int32)
    qweight = qweight.bitwise_left_shift_(shift)
    qweight = qweight.sum(dim=-1, dtype=torch.int32)

    return qweight

def dump_linear_awq(weight: torch.Tensor, bias: torch.Tensor) -> dict[str, torch.Tensor]:
    tensors = qmodule.dump_linear_awq(weight, bias, w_bit=4, group_size=64, zero_point=False)
    tensors["qweight"] = tensors["qweight"].view(dtype=torch.int32)
    return tensors

def pack_wscales(wscales: torch.Tensor) -> torch.Tensor:
    N, groups = wscales.shape

    assert wscales.dtype.itemsize == 2

    BLOCK_N = 128
    WSCALES_PACK_SIZE = 4
    WSCALES_NUM_PACKS = 1
    WSCALES_VALID_LANES = 32

    wscales = wscales.reshape(ceil_div(N, BLOCK_N), BLOCK_N, groups)
    wscales = wscales.permute(0, 2, 1)  # [..., BLOCK_N]

    wscales = wscales.reshape(*wscales.shape[0:2], WSCALES_NUM_PACKS, WSCALES_VALID_LANES // 4, WSCALES_PACK_SIZE // 2, 4, 2)
    wscales = wscales.permute(0, 1, 2, 3, 5, 4, 6)

    wscales = wscales.contiguous()
    wscales = wscales.view(groups, N)

    return wscales

# print(pack_wscales(torch.arange(0, 256, dtype=torch.int16)[..., None]))
# exit(0)

def pack_qweight(qweight: torch.Tensor) -> torch.Tensor:
    N, K = qweight.shape
    K *= 8

    assert qweight.dtype.itemsize == 4

    BLOCK_N = 128
    WARP_K = 64
    WARP_N_TILES = BLOCK_N // 16

    qweight = qweight.reshape(ceil_div(N, BLOCK_N), WARP_N_TILES, 16, ceil_div(K, WARP_K), WARP_K // 8)
    qweight = qweight.permute(0, 3, 1, 2, 4)        # [N / BLOCK_N, K / WARP_K, WARP_N_TILES, (INSN_N) => 16 , WARP_K / 8 => 8]

    # print(qweight.shape)
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#mma-16864-b-1
    assert qweight.shape[3:] == (16, 8)
    qweight = qweight.reshape(*qweight.shape[0:3], 2, 8, 2, 4)
    qweight = qweight.permute(0, 1, 2, 4, 6, 3, 5)
    assert qweight.shape[3:] == (8, 4, 2, 2)

    print(qweight.dtype)
    print(qweight.shape)

    qweight = qweight.contiguous()
    qweight = qweight.view(dtype=torch.int8)    # assume little-endian

    print(qweight.shape)

    qweight = qweight.view(N, K // 2)

    return qweight

def pack_lora(weight: torch.Tensor, is_lora_down: bool) -> torch.Tensor:
    N, R = weight.shape

    assert N % 16 == 0
    assert R % 16 == 0
    assert weight.dtype.itemsize == 2

    weight = weight.reshape(N // 16, 16, R // 16, 16)
    weight = weight.permute(0, 2, 1, 3)

    if is_lora_down:
        weight = weight.transpose(-1, -2)

    # https://docs.nvidia.com/cuda/parallel-thread-execution/#mma-16816-b-f16
    assert weight.shape[2:] == (16, 16)
    weight = weight.reshape(*weight.shape[0:2], 2, 8, 2, 4, 2)
    weight = weight.permute(0, 1, 3, 5, 2, 4, 6)

    weight = weight.contiguous()
    weight = weight.view(N, R)

    return weight

def dump_linear_w4a4(
        weight: torch.Tensor, 
        bias: torch.Tensor | None = None, 
        smooth: torch.Tensor | None = None,
        lora_down: torch.Tensor | None = None,
        lora_up: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    
    print(f"dump_linear_w4a4: weight.shape={weight.shape}")
    
    tensors = {}

    group_size = 64
    oc, ic = weight.shape
    N, K = oc, ic

    # LORA_RANK = 32

    wscales = group_scale(weight, num_bits=4, group_size=group_size)

    qweight = quantize_4bit(weight, wscales)        # [N, K / 8]
    assert qweight.shape == (N, K // 8)
    qweight = pack_qweight(qweight)
    
    wscales = pack_wscales(wscales)

    if bias is None:
        bias = torch.zeros([oc], dtype=weight.dtype)
    bias = pack_wscales(bias[..., None])
    assert bias.shape == (1, oc)
    bias = bias[0]

    if smooth is None:
        smooth = torch.ones([ic], dtype=weight.dtype)
    if smooth.dtype != weight.dtype:
        print(f"Convert smooth dtype from {smooth.dtype} to {weight.dtype}")
        smooth = smooth.to(weight.dtype)
    smooth = pack_wscales(smooth[..., None])
    assert smooth.shape == (1, ic)
    smooth = smooth[0]

    # if lora_down is None:
    #     lora_down = torch.zeros([LORA_RANK, ic], dtype=weight.dtype)
    # if lora_up is None:
    #     lora_up = torch.zeros([oc, LORA_RANK], dtype=weight.dtype)

    if not lora_down is None:
        lora_down = pack_lora(lora_down.transpose(0, 1), is_lora_down=True)
    if not lora_up is None:
        lora_up = pack_lora(lora_up, is_lora_down=False)

    tensors["qweight"] = qweight
    tensors["wscales"] = wscales
    tensors["bias"] = bias

    if not lora_down is None:
        tensors["lora_down"] = lora_down
    if not lora_up is None:
        tensors["lora_up"] = lora_up
    tensors["smooth"] = smooth

    return tensors

def dump_linear_adanorm_single(weight: torch.Tensor, bias: torch.Tensor) -> TensorDict:
    oc, ic = weight.shape
    assert oc % 3 == 0

    # shift_msa, scale_msa, gate_msa
    weight = weight.reshape(3, oc // 3, ic).transpose(0, 1).reshape(oc, ic).contiguous()
    
    bias = bias.reshape(3, oc // 3).transpose(0, 1)
    # [oc // 3, 3]
    bias = bias + torch.tensor([0, 1, 0], dtype=bias.dtype)
    bias = bias.reshape(oc).contiguous()

    return dump_linear_awq(weight, bias)

def dump_linear_adanorm_zero(weight: torch.Tensor, bias: torch.Tensor) -> dict[str, torch.Tensor]:
    oc, ic = weight.shape
    assert oc % 6 == 0

    # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
    weight = weight.reshape(6, oc // 6, ic).transpose(0, 1).reshape(oc, ic).contiguous()
    
    bias = bias.reshape(6, oc // 6).transpose(0, 1)
    # [oc // 6, 6]
    bias = bias + torch.tensor([0, 1, 0, 0, 1, 0], dtype=bias.dtype)
    bias = bias.reshape(oc).contiguous()

    return dump_linear_awq(weight, bias)

def dump_linear_layer_w4a4(layer: torch.nn.Linear):
    return dump_linear_w4a4(layer.weight.detach(), layer.bias.detach())

def dump_linear_layer_adanorm_single(block: torch.nn.Linear) -> TensorDict:
    return dump_linear_adanorm_single(block.weight.detach(), block.bias.detach())

def dump_linear_layer_adanorm_zero(block: torch.nn.Linear) -> dict[str, torch.Tensor]:
    return dump_linear_adanorm_zero(block.weight.detach(), block.bias.detach())

def dump_qkv_proj(q: torch.nn.Linear, k: torch.nn.Linear, v: torch.nn.Linear) -> TensorDict:
    qkv = [q, k, v]
    qkv_weight = torch.cat([linear.weight.detach() for linear in qkv], dim=0)
    qkv_bias = torch.cat([linear.bias.detach() for linear in qkv], dim=0)
    print(qkv_weight.shape)
    print(qkv_bias.shape)
    return dump_linear_w4a4(qkv_weight, qkv_bias)

def dump_single_transformer(block: FluxSingleTransformerBlock) -> dict[str, torch.Tensor]:
    tensors = {}

    merge_dict(tensors, dump_linear_layer_adanorm_single(block.norm.linear), "norm.linear.")
    
    merge_dict(tensors, dump_qkv_proj(block.attn.to_q, block.attn.to_k, block.attn.to_v), "qkv_proj.")

    tensors["norm_q.weight"] = block.attn.norm_q.weight.detach()
    tensors["norm_k.weight"] = block.attn.norm_k.weight.detach()

    merge_dict(tensors, dump_linear_layer_w4a4(block.proj_mlp), "mlp_fc1.")

    merge_dict(tensors, dump_linear_w4a4(
        block.proj_out.weight.detach()[:, 0:block.attn.out_dim], 
        bias=None
    ), "out_proj.")

    merge_dict(tensors, dump_linear_w4a4(
        block.proj_out.weight.detach()[:, block.attn.out_dim:], 
        bias=block.proj_out.bias.detach()
        # block.proj_out.weight.detach()
    ), "mlp_fc2.")

    # print(dict(block.named_parameters()).keys())

    return tensors
    

def dump_transformer(block: FluxTransformerBlock) -> TensorDict:
    tensors = {}

    merge_dict(tensors, dump_linear_layer_adanorm_zero(block.norm1.linear), "norm1.linear.")
    merge_dict(tensors, dump_linear_layer_adanorm_zero(block.norm1_context.linear), "norm1_context.linear.")

    merge_dict(tensors, dump_qkv_proj(block.attn.to_q, block.attn.to_k, block.attn.to_v), "qkv_proj.")
    merge_dict(tensors, dump_qkv_proj(block.attn.add_q_proj, block.attn.add_k_proj, block.attn.add_v_proj), "qkv_proj_context.")

    tensors["norm_q.weight"] = block.attn.norm_q.weight.detach()
    tensors["norm_k.weight"] = block.attn.norm_k.weight.detach()
    tensors["norm_added_q.weight"] = block.attn.norm_added_q.weight.detach()
    tensors["norm_added_k.weight"] = block.attn.norm_added_k.weight.detach()

    merge_dict(tensors, dump_linear_layer_w4a4(block.attn.to_out[0]), "out_proj.")
    merge_dict(tensors, dump_linear_layer_w4a4(block.attn.to_add_out), "out_proj_context.")

    # no params for norm2

    merge_dict(tensors, dump_linear_layer_w4a4(block.ff.net[0].proj), "mlp_fc1.")
    merge_dict(tensors, dump_linear_layer_w4a4(block.ff.net[2]), "mlp_fc2.")
    
    merge_dict(tensors, dump_linear_layer_w4a4(block.ff_context.net[0].proj), "mlp_context_fc1.")
    merge_dict(tensors, dump_linear_layer_w4a4(block.ff_context.net[2]), "mlp_context_fc2.")

    return tensors

def recip_smooth(smooth: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if smooth is None:
        return None
    return (1. / smooth.to(torch.float64)).to(smooth.dtype)

def unsmooth(weight: torch.Tensor, smooth: Optional[torch.Tensor]) -> torch.Tensor:
    if smooth is None:
        return weight
    assert smooth.ndim == 1
    assert weight.ndim == 2
    assert weight.shape[1] == smooth.shape[0]
    return (weight.to(torch.float64) / smooth[None, ...].to(torch.float64)).to(weight.dtype)

def dump_qkv_proj_svdq(
        qmodel: DeepCompressorModel,
        key_qkv: tuple[str, str, str],
        key_smooth: str,
    ) -> TensorDict:
    
    qkv_weight = torch.cat([qmodel.model[f"{k}.weight"] for k in key_qkv], dim=0)
    qkv_bias = torch.cat([qmodel.model[f"{k}.bias"] for k in key_qkv], dim=0)
    print(qkv_weight.shape)
    print(qkv_bias.shape)

    smooth = qmodel.smooth[key_smooth].to(qkv_weight.dtype).float()
    return dump_linear_w4a4(
        qkv_weight, qkv_bias, 
        smooth=smooth,  # recip_smooth(smooth),
        lora_down=unsmooth(qmodel.branch[key_smooth]["a.weight"], smooth),
        lora_up=qmodel.branch[key_smooth]["b.weight"]
    )

def dump_linear_layer_w4a4_svdq(
        qmodel: DeepCompressorModel, 
        key: str, 
        alt_smooth: bool = False, 
        smooth: Optional[torch.Tensor] = None, 
        alt_bias: bool = False,
        bias: Optional[torch.Tensor] = None,
        bias_fuse_shift: float = 0
    ) -> TensorDict:
    
    if not alt_smooth:
        smooth = qmodel.smooth[key]
    if not alt_bias:
        bias = qmodel.model[f"{key}.bias"]
    weight = qmodel.model[f"{key}.weight"]
    if smooth is not None:
        smooth = smooth.to(weight.dtype).float()
    
    if bias_fuse_shift != 0:
        oc, ic = weight.shape
        shift = torch.ones([ic], dtype=weight.dtype, device=weight.device) * bias_fuse_shift
        if smooth is not None:
            shift = (shift / smooth).to(weight.dtype)
        delta = F.linear(shift, weight)
        if bias is None:
            bias = torch.zeros([oc], dtype=weight.dtype, device=weight.device)
        bias -= delta
        
    return dump_linear_w4a4(
        weight=weight,
        bias=bias,
        smooth=smooth, # recip_smooth(smooth),
        lora_down=unsmooth(qmodel.branch[key]["a.weight"], smooth),
        lora_up=qmodel.branch[key]["b.weight"]
    )

GELU_SHIFT_VALUE = 0.171875

def dump_transformer_svdq(
        qmodel: DeepCompressorModel, 
        layer_id: int, 
        original_net: Optional[FluxTransformer2DModel],
        use_original_adanorm: bool = False,
        num_svdq_joint: int = 19,
        shift_gelu: bool = True,
        **kwargs,
    ) -> TensorDict:

    tensors = {}

    original_block: FluxTransformerBlock = original_net.transformer_blocks[layer_id]

    if layer_id >= num_svdq_joint:
        return dump_transformer(original_block)

    prefix = f"transformer_blocks.{layer_id}"
    def model(key: str):
        return qmodel.model[f"{prefix}.{key}"]
    def linear(key: str, **kwargs):
        return dump_linear_layer_w4a4_svdq(qmodel, f"{prefix}.{key}", **kwargs)
    
    if use_original_adanorm:
        merge_dict(tensors, dump_linear_layer_adanorm_zero(original_block.norm1.linear), "norm1.linear.")
        merge_dict(tensors, dump_linear_layer_adanorm_zero(original_block.norm1_context.linear), "norm1_context.linear.")
    else:
        merge_dict(tensors, dump_linear_adanorm_zero(model("norm1.linear.weight"), model("norm1.linear.bias")), "norm1.linear.")
        merge_dict(tensors, dump_linear_adanorm_zero(model("norm1_context.linear.weight"), model("norm1_context.linear.bias")), "norm1_context.linear.")

    merge_dict(tensors, dump_qkv_proj_svdq(
        qmodel, 
        (f"{prefix}.attn.to_q", f"{prefix}.attn.to_k", f"{prefix}.attn.to_v"), 
        f"{prefix}.attn.to_q"
    ), "qkv_proj.")

    merge_dict(tensors, dump_qkv_proj_svdq(
        qmodel, 
        (f"{prefix}.attn.add_q_proj", f"{prefix}.attn.add_k_proj", f"{prefix}.attn.add_v_proj"), 
        f"{prefix}.attn.add_k_proj"
    ), "qkv_proj_context.")

    tensors["norm_q.weight"] = model("attn.norm_q.weight")
    tensors["norm_k.weight"] = model("attn.norm_k.weight")
    tensors["norm_added_q.weight"] = model("attn.norm_added_q.weight")
    tensors["norm_added_k.weight"] = model("attn.norm_added_k.weight")

    # DONE GELU should be before lora up, also +shift
    # DONE smooth factor 1/smooth
    # DONE smooth fuse to lora down
    merge_dict(tensors, linear("attn.to_out.0", alt_smooth=True, smooth=None), "out_proj.")
    merge_dict(tensors, linear("attn.to_add_out", alt_smooth=True, smooth=None), "out_proj_context.")

    merge_dict(tensors, linear("ff.net.0.proj"), "mlp_fc1.")
    merge_dict(tensors, linear("ff.net.2.linear", alt_bias=True, bias=original_block.ff.net[2].bias, bias_fuse_shift=GELU_SHIFT_VALUE if shift_gelu else 0), "mlp_fc2.")
    
    merge_dict(tensors, linear("ff_context.net.0.proj"), "mlp_context_fc1.")
    merge_dict(tensors, linear("ff_context.net.2.linear", alt_bias=True, bias=original_block.ff_context.net[2].bias, bias_fuse_shift=GELU_SHIFT_VALUE if shift_gelu else 0), "mlp_context_fc2.")

    return tensors

def dump_single_transformer_svdq(
        qmodel: DeepCompressorModel, 
        layer_id: int, 
        original_net: Optional[FluxTransformer2DModel],
        use_original_adanorm: bool = False,
        num_svdq_single: int = 38,
        shift_gelu: bool = True,
        **kwargs
    ) -> TensorDict:

    tensors = {}

    original_block: FluxSingleTransformerBlock = original_net.single_transformer_blocks[layer_id]

    if layer_id >= num_svdq_single:
        return dump_single_transformer(original_block)

    prefix = f"single_transformer_blocks.{layer_id}"
    def model(key: str):
        return qmodel.model[f"{prefix}.{key}"]
    def linear(key: str, **kwargs):
        return dump_linear_layer_w4a4_svdq(qmodel, f"{prefix}.{key}", **kwargs)
    
    if use_original_adanorm:
        merge_dict(tensors, dump_linear_layer_adanorm_single(original_block.norm.linear), "norm.linear.")
    else:
        merge_dict(tensors, dump_linear_adanorm_single(model("norm.linear.weight"), model("norm.linear.bias")), "norm.linear.")
    
    merge_dict(tensors, dump_qkv_proj_svdq(
        qmodel,
        (f"{prefix}.attn.to_q", f"{prefix}.attn.to_k", f"{prefix}.attn.to_v"), 
        f"{prefix}.attn.to_q"
    ), "qkv_proj.")

    tensors["norm_q.weight"] = model("attn.norm_q.weight")
    tensors["norm_k.weight"] = model("attn.norm_k.weight")

    merge_dict(tensors, linear("proj_mlp", alt_smooth=True, smooth=qmodel.smooth[f"{prefix}.attn.to_q"]), "mlp_fc1.")

    merge_dict(tensors, linear("proj_out.linears.0", alt_smooth=True, smooth=None, alt_bias=True, bias=None), "out_proj.")
    merge_dict(tensors, linear("proj_out.linears.1.linear", alt_bias=True, bias=original_block.proj_out.bias, bias_fuse_shift=GELU_SHIFT_VALUE if shift_gelu else 0), "mlp_fc2.")

    return tensors

@torch.inference_mode()
def dump_flux(net: FluxTransformer2DModel) -> TensorDict:
    tensors = {}
    for i in range(len(net.transformer_blocks)):
        merge_dict(tensors, dump_transformer(net.transformer_blocks[i]), f"transformer_blocks.{i}.")
    for i in range(len(net.single_transformer_blocks)):
        merge_dict(tensors, dump_single_transformer(net.single_transformer_blocks[i]), f"single_transformer_blocks.{i}.")
    return tensors

@torch.inference_mode()
def dump_flux_svdq(qmodel: DeepCompressorModel, **kwargs) -> TensorDict:
    tensors = {}
    for i in range(19):
        merge_dict(tensors, dump_transformer_svdq(qmodel, i, **kwargs), f"transformer_blocks.{i}.")
    for i in range(38):
        merge_dict(tensors, dump_single_transformer_svdq(qmodel, i, **kwargs), f"single_transformer_blocks.{i}.")
    return tensors

def load_svdq(path: str) -> DeepCompressorModel:
    return DeepCompressorModel(
        model=torch.load(f"{path}/model.pt", map_location="cpu"),
        smooth=torch.load(f"{path}/smooth.pt", map_location="cpu"),
        branch=torch.load(f"{path}/branch.pt", map_location="cpu"),
        lora={}
    )
    
if __name__ == "__main__":
    use_svdq = True
    use_original_adanorm = True
    shift_gelu = True
    dev = False

    num_svdq_joint = 19
    num_svdq_single = 38

    if not use_svdq:
        pipe = FluxPipeline.from_pretrained(
            f"black-forest-labs/FLUX.1-{'dev' if dev else 'schnell'}", torch_dtype=torch.bfloat16)
        net: FluxTransformer2DModel = pipe.transformer
        print(net)
        tensors = dump_flux(net)
        dtype = pipe.dtype
    else:
        pipe = FluxPipeline.from_pretrained(
            f"black-forest-labs/FLUX.1-{'dev' if dev else 'schnell'}", torch_dtype=torch.bfloat16)
        net: FluxTransformer2DModel = pipe.transformer
        tensors = dump_flux_svdq(
            load_svdq(path="model-dev" if dev else "model-schnell"), 
            original_net=net, 
            use_original_adanorm=use_original_adanorm, 
            num_svdq_joint=num_svdq_joint, 
            num_svdq_single=num_svdq_single,
            shift_gelu=shift_gelu,
        )
        dtype = torch.bfloat16

    for k, v in tensors.items():
        assert not v.isnan().any()
        assert not v.isinf().any()
    
    safetensors.torch.save_file(
        tensors, 
        f"/tmp/flux{f'-dev' if dev else ''}{f'-svdq-{num_svdq_joint}-{num_svdq_single}' if use_svdq else ''}-divsmooth{'-shift' if shift_gelu else ''}{'-ada' if use_original_adanorm else ''}-{'bf16' if dtype == torch.bfloat16 else 'fp16'}.safetensors")

    # tensors = dump_single_transformer(net.single_transformer_blocks[0])
    # print(tensors)

    # print(dump_transformer(net.transformer_blocks[0]))
    # print(dict(net.named_parameters()))