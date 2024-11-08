import torch
import safetensors
import torch.nn.functional as F
from dump_flux import DeepCompressorModel, TensorDict, pack_wscales, pack_lora, merge_dict, unsmooth
from typing import Optional

Lora = tuple[torch.Tensor, torch.Tensor]

def load_svdq_lora(path: str, lora_path: str) -> DeepCompressorModel:
    result = DeepCompressorModel(
        model=torch.load(f"{path}/model.pt", map_location="cpu"),
        smooth=torch.load(f"{path}/smooth.pt", map_location="cpu"),
        branch=torch.load(f"{path}/branch.pt", map_location="cpu"),
        lora={}
    )

    with safetensors.safe_open(lora_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            prefix = "transformer."
            if k.startswith(prefix):
                result.lora[k.removeprefix(prefix)] = f.get_tensor(k)

    dtype = next(iter(result.branch.values()))["a.weight"].dtype
    for k, v in result.lora.items():
        if v.dtype != dtype:
            print(f"Convert lora weight {k} from {v.dtype} to {dtype}")
            result.lora[k] = v.to(dtype)

    # for k, v in result.lora.items():
    #     v.fill_(0)

    return result

# q/k/v [3072, ...] -> qkv [3072 * 3, ...]
def extend_qkv(input: torch.Tensor, id: int) -> torch.Tensor:
    oc, ic = input.shape
    tmp = torch.zeros([oc * 3, ic], dtype=input.dtype, device=input.device)
    tmp[id*oc:(id+1)*oc, ...] = input
    return tmp

def merge_lora(inputs: list[Lora]) -> Optional[Lora]:
    if len(inputs) == 0:
        return None
    
    lora_downs = [x[0] for x in inputs]
    lora_ups = [x[1] for x in inputs]

    lora_down = torch.cat(lora_downs, dim=0)
    lora_up = torch.cat(lora_ups, dim=1)

    return (lora_down, lora_up)

def merge_lora_qkv(inputs: list[Lora]) -> list[Lora]:
    if len(inputs) == 0:
        return []
    
    for x in inputs:
        if not x[0].equal(inputs[0][0]):
            return inputs
    
    lora_down = inputs[0][0]
    lora_ups = [x[1] for x in inputs]
    lora_up = torch.sum(torch.stack(lora_ups), dim=0).to(lora_down.dtype)

    return [(lora_down, lora_up)]

def dump_lora(lora_down: Optional[torch.Tensor], lora_up: Optional[torch.Tensor]) -> TensorDict:
    if lora_down is None:
        return {}
    
    rank, ic = lora_down.shape
    oc = lora_up.shape[0]
    assert lora_up.shape == (oc, rank)

    if rank % 16 != 0:
        rank_pad = (rank + 16 - 1) // 16 * 16
        tmp_down = torch.zeros([rank_pad, ic], dtype=lora_down.dtype, device=lora_down.device)
        tmp_up = torch.zeros([oc, rank_pad], dtype=lora_down.dtype, device=lora_down.device)
        tmp_down[:rank, ...] = lora_down
        tmp_up[..., :rank] = lora_up
        lora_down = tmp_down
        lora_up = tmp_up
        print(f"Pad lora rank from {rank} to {rank_pad}")
    
    lora_down = pack_lora(lora_down.transpose(0, 1), is_lora_down=True)
    lora_up = pack_lora(lora_up, is_lora_down=False)

    tensors = {}
    tensors["lora_down"] = lora_down
    tensors["lora_up"] = lora_up
    return tensors

def get_original_lora(qmodel: DeepCompressorModel, key_branch: str, key_smooth: Optional[str]) -> Lora:
    dtype = qmodel.branch[key_branch]["a.weight"].dtype
    smooth = qmodel.smooth[key_smooth].to(dtype).float() if key_smooth else None
    return (
        unsmooth(qmodel.branch[key_branch]["a.weight"], smooth), 
        qmodel.branch[key_branch]["b.weight"]
    )

def dump_linear_lora(
        qmodel: DeepCompressorModel, 
        key_lora: str,
        key_branch: str,
        key_smooth: str,
        shift_bias: bool = False,
        key_bias: Optional[str] = None,
        range_ic: slice = slice(None, None, None)) -> TensorDict:
    
    lora_original = get_original_lora(qmodel, key_branch, key_smooth)

    if f"{key_lora}.lora_A.weight" in qmodel.lora:
        

        # lora_down = qmodel.lora[f"{key}.lora_A.weight"][..., range_ic]
        # lora_up = qmodel.lora[f"{key}.lora_B.weight"]

        lora_new = (
            qmodel.lora[f"{key_lora}.lora_A.weight"][..., range_ic],
            qmodel.lora[f"{key_lora}.lora_B.weight"]
        )
        lora_down, lora_up = merge_lora([lora_original, lora_new])

        rank, ic = lora_down.shape
        oc = lora_up.shape[0]
        assert lora_up.shape == (oc, rank)

        print(f"linear at {key_lora} has rank {rank}")

        tensors = dump_lora(lora_down, lora_up)
        if shift_bias and False:    # no longer need shift bias 
            if key_bias is None:
                key_bias = f"{key_branch}.bias"
            if key_bias in qmodel.model:
                bias = qmodel.model[key_bias]
                print(f"linear at {key_lora} apply shift_bias from original bias at {key_bias}")
            else:
                bias = torch.zeros([oc], dtype=lora_up.dtype, device=lora_up.device)
                print(f"linear at {key_lora} apply shift_bias from empty original bias")
            shift = torch.empty([ic], dtype=lora_down.dtype, device=lora_down.device)
            shift = shift.fill_(0.171875)
            delta = F.linear(F.linear(shift, lora_new[0]), lora_new[1])
            print(f"shift_bias delta = {delta}")
            bias -= delta
            tensors["bias"] = pack_wscales(bias[..., None])[0]
        return tensors
    else:
        print(f"linear at {key_lora} use original lora")
        return dump_lora(*lora_original)

def dump_qkv_proj_svdq_lora(
        qmodel: DeepCompressorModel,
        key_qkv: tuple[str, str, str],
        key_smooth: str,
        key_smooth_out: str
    ) -> TensorDict:

    dtype = qmodel.branch[key_smooth]["a.weight"].dtype
    smooth_out = qmodel.smooth[key_smooth_out].to(dtype).float()
    lora_original = get_original_lora(qmodel, key_smooth, key_smooth)

    loras = []
    for i in range(3):
        key = key_qkv[i]
        if f"{key}.lora_A.weight" in qmodel.lora:
            lora_down = qmodel.lora[f"{key}.lora_A.weight"]
            lora_up = qmodel.lora[f"{key}.lora_B.weight"] 
            if i == 2:
                lora_up = (lora_up / smooth_out[..., None]).to(lora_up.dtype)
            loras.append((lora_down, extend_qkv(lora_up, i)))

    # print(loras)
    
    lora_down, lora_up = merge_lora([lora_original, *merge_lora_qkv(loras)])

    print(f"qkv_proj at {key_smooth} has rank {lora_down.shape[0]}")

    return dump_lora(lora_down, lora_up)


def dump_transformer_svdq_lora(qmodel: DeepCompressorModel, layer_id: int) -> TensorDict:
    tensors = {}

    def reorder_adanorm_linear(weight: torch.Tensor) -> torch.Tensor:
        oc, ic = weight.shape
        assert oc % 6 == 0
        return weight.reshape(6, oc // 6, ic).transpose(0, 1).reshape(oc, ic).contiguous()
    def linear(key: str, **kwargs):
        key_lora = key
        key_branch = kwargs.pop("key_branch", key_lora)
        key_smooth = kwargs.pop("key_smooth", key_branch)
        return dump_linear_lora(qmodel, key_lora, key_branch, key_smooth, **kwargs)

    prefix = f"transformer_blocks.{layer_id}"
    if f"{prefix}.norm1.linear.lora_A.weight" in qmodel.lora:
        lora_down = qmodel.lora[f"{prefix}.norm1.linear.lora_A.weight"]
        lora_up   = qmodel.lora[f"{prefix}.norm1.linear.lora_B.weight"]
        tensors[f"norm1.linear.lora_down"] = lora_down
        tensors[f"norm1.linear.lora_up"] = reorder_adanorm_linear(lora_up)

    if f"{prefix}.norm1_context.linear.lora_A.weight" in qmodel.lora:
        lora_down = qmodel.lora[f"{prefix}.norm1_context.linear.lora_A.weight"]
        lora_up   = qmodel.lora[f"{prefix}.norm1_context.linear.lora_B.weight"]
        tensors[f"norm1_context.linear.lora_down"] = lora_down
        tensors[f"norm1_context.linear.lora_up"] = reorder_adanorm_linear(lora_up)

    merge_dict(tensors, dump_qkv_proj_svdq_lora(
        qmodel, 
        (f"{prefix}.attn.to_q", f"{prefix}.attn.to_k", f"{prefix}.attn.to_v"), 
        f"{prefix}.attn.to_q",
        f"{prefix}.attn.to_out.0"
    ), "qkv_proj.")

    merge_dict(tensors, dump_qkv_proj_svdq_lora(
        qmodel, 
        (f"{prefix}.attn.add_q_proj", f"{prefix}.attn.add_k_proj", f"{prefix}.attn.add_v_proj"), 
        f"{prefix}.attn.add_k_proj",
        f"{prefix}.attn.to_out.0"
    ), "qkv_proj_context.")

    merge_dict(tensors, linear(f"{prefix}.attn.to_out.0", key_smooth=None), "out_proj.")
    merge_dict(tensors, linear(f"{prefix}.attn.to_add_out", key_smooth=None), "out_proj_context.")

    merge_dict(tensors, linear(f"{prefix}.ff.net.0.proj"), "mlp_fc1.")
    merge_dict(tensors, linear(f"{prefix}.ff.net.2", key_branch=f"{prefix}.ff.net.2.linear", shift_bias=True), "mlp_fc2.")
    
    merge_dict(tensors, linear(f"{prefix}.ff_context.net.0.proj"), "mlp_context_fc1.")
    merge_dict(tensors, linear(f"{prefix}.ff_context.net.2", key_branch=f"{prefix}.ff_context.net.2.linear", shift_bias=True), "mlp_context_fc2.")

    return tensors

def dump_single_transformer_svdq_lora(qmodel: DeepCompressorModel, layer_id: int) -> TensorDict:
    tensors = {}

    def reorder_adanorm_linear(weight: torch.Tensor) -> torch.Tensor:
        oc, ic = weight.shape
        assert oc % 3 == 0
        return weight.reshape(3, oc // 3, ic).transpose(0, 1).reshape(oc, ic).contiguous()
    def linear(key: str, **kwargs):
        key_lora = key
        key_branch = kwargs.pop("key_branch", key_lora)
        key_smooth = kwargs.pop("key_smooth", key_branch)
        return dump_linear_lora(qmodel, key_lora, key_branch, key_smooth, **kwargs)
    
    prefix = f"single_transformer_blocks.{layer_id}"
    if f"{prefix}.norm.linear.lora_A.weight" in qmodel.lora:
        lora_down = qmodel.lora[f"{prefix}.norm.linear.lora_A.weight"]
        lora_up   = qmodel.lora[f"{prefix}.norm.linear.lora_B.weight"]
        tensors[f"norm.linear.lora_down"] = lora_down
        tensors[f"norm.linear.lora_up"] = reorder_adanorm_linear(lora_up)

    merge_dict(tensors, dump_qkv_proj_svdq_lora(
        qmodel,
        (f"{prefix}.attn.to_q", f"{prefix}.attn.to_k", f"{prefix}.attn.to_v"), 
        f"{prefix}.attn.to_q",
        f"{prefix}.proj_out.linears.0"
    ), "qkv_proj.")

    merge_dict(tensors, linear(f"{prefix}.proj_mlp", key_smooth=f"{prefix}.attn.to_q"), "mlp_fc1.")

    # TODO
    out_dim = 3072

    merge_dict(tensors, linear(f"{prefix}.proj_out", 
                               key_branch=f"{prefix}.proj_out.linears.0", 
                               key_smooth=None, 
                               range_ic=slice(0, out_dim)), "out_proj.")
    
    merge_dict(tensors, linear(f"{prefix}.proj_out", 
                               key_branch=f"{prefix}.proj_out.linears.1.linear",
                               shift_bias=True, 
                               range_ic=slice(out_dim, None)), "mlp_fc2.")

    return tensors

@torch.inference_mode()
def dump_flux_svdq_lora(qmodel: DeepCompressorModel, **kwargs) -> TensorDict:
    tensors = {}
    for i in range(19):
        merge_dict(tensors, dump_transformer_svdq_lora(qmodel, i, **kwargs), f"transformer_blocks.{i}.")
    for i in range(38):
        merge_dict(tensors, dump_single_transformer_svdq_lora(qmodel, i, **kwargs), f"single_transformer_blocks.{i}.")
    return tensors

if __name__ == "__main__":
    lora_name = "realism"

    if lora_name == "sketch":
        qmodel = load_svdq_lora("model-dev", "../third_party/FLUX.1-dev-LoRA-Collections/sketch.safetensors")
    elif lora_name == "realism":
        qmodel = load_svdq_lora("model-dev", "../third_party/FLUX.1-dev-LoRA-Collections/realism.safetensors")
    elif lora_name == "anime":
        qmodel = load_svdq_lora("model-dev", "../third_party/sonny-anime-fixed/araminta_k_sonnyanime_fluxd_fixed.safetensors")
    elif lora_name == "ghibsky":
        qmodel = load_svdq_lora("model-dev", "../third_party/flux-ghibsky-illustration/lora.safetensors")
    elif lora_name == "yarn":
        qmodel = load_svdq_lora("model-dev", "../third_party/yarn_art_Flux_LoRA/pytorch_lora_weights.safetensors")
    elif lora_name == "sketch2image":
        qmodel = load_svdq_lora("model-dev", "sketch2image.safetensors")
    else:
        raise NotImplementedError

    tensors = dump_flux_svdq_lora(qmodel)

    for k, v in tensors.items():
        assert not v.isnan().any()
        assert not v.isinf().any()
    
    safetensors.torch.save_file(
        tensors, 
        f"/tmp/flux-lora-{lora_name}-bf16.safetensors")
