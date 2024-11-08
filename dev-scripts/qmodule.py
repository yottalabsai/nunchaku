import math
import torch
import torch.nn as nn

def make_divisible(c, divisor):
    return (c + divisor - 1) // divisor

def calculate_zeros_width(in_features, group_size=128, pack_num=8):
    if group_size >= 128:
        size_multiplier = 1
    elif group_size == 64:
        size_multiplier = 2
    elif group_size == 32:
        size_multiplier = 4
    else:
        raise NotImplementedError

    base_width = make_divisible(in_features // group_size, pack_num)
    base_width = make_divisible(base_width, size_multiplier) * size_multiplier
    return base_width


def pack_intweight(unpacked_qweight, interleave, kstride):
    # unpacked_qweight: [N, K]
    N = unpacked_qweight.shape[0]
    K = unpacked_qweight.shape[1]

    Packed_Kernel = unpacked_qweight.cpu().numpy().reshape(N, K // 32, 32)
    # np.arange(32).reshape(4, 4, 2).transpose(1, 0, 2) => [0, 1, 8, 9, 16, 17, 24, 25, ...]
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 4, 2).transpose(0, 1, 3, 2, 4)
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 32)

    # reorder each 8 weights for fast dequantization
    # [0, 1, 2, 3, 4, 5, 6, 7] => [0, 2, 4, 6, 1, 3, 5, 7]
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 8)
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 4, 2).transpose(0, 1, 2, 4, 3)
    Packed_Kernel = Packed_Kernel.reshape(N, K)

    # interleaving every four rows
    Packed_Kernel = Packed_Kernel.reshape(
        N // interleave, interleave, K // kstride, kstride
    )
    # N // 4, K // 64, 4, 64
    Packed_Kernel = Packed_Kernel.transpose(0, 2, 1, 3)
    Packed_Kernel = Packed_Kernel.reshape(
        N // interleave, K // kstride, kstride, interleave
    )
    # Packing -> (N // 4, K // 64, 64)
    Packed_Kernel = (
        Packed_Kernel[..., 0]
        | (Packed_Kernel[..., 1] << 4)
        | (Packed_Kernel[..., 2] << 8)
        | (Packed_Kernel[..., 3] << 12)
    )
    # reshape to (N // 4, K), FP16 format
    Packed_Kernel = Packed_Kernel.reshape(N // interleave, K)
    qweight = (
        torch.tensor(Packed_Kernel.astype("int16"))
        .to(unpacked_qweight.device)
        .contiguous()
    )
    return qweight

def pseudo_quantize_tensor(
    w, n_bit=8, zero_point=True, q_group_size=-1,
) -> tuple[torch.Tensor, torch.Tensor]:
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:  # we actually never used this
        # assert min_val is None
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = -max_int
        scales = max_val / max_int
        zeros = torch.full_like(scales, fill_value=-min_int)

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)

    return scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)


def dump_linear_awq(
    weight: torch.Tensor, 
    bias: torch.Tensor, 
    w_bit: int, 
    group_size: int, 
    zero_point: bool = True
) -> dict[str, torch.Tensor]:
    
    scales, zeros = pseudo_quantize_tensor(weight, w_bit, zero_point, group_size)

    print(scales.shape)
    print(zeros.shape)

    tensors = {}

    dtype = weight.dtype
    
    oc, ic = weight.shape

    # need scales and zeros info for real quantization
    assert scales is not None and zeros is not None
    scale_zeros = zeros * scales

    pack_num = 32 // w_bit
    qscales = torch.zeros(
        (
            scales.shape[0],
            calculate_zeros_width(ic, group_size) * pack_num,
        ),
        dtype=dtype,
        device=scales.device,
    )
    qscales[:, : scales.shape[1]] = scales
    # awq_linear.scales = scales.clone().half()
    tensors["wscales"] = qscales.transpose(1, 0).contiguous()
    if bias is not None:
        tensors["bias"] = bias.clone()

    if False:
        intweight = []
        for idx in range(ic):
            intweight.append(
                torch.round(
                    (weight.data[:, idx] + scale_zeros[:, idx // group_size])
                    / qscales[:, idx // group_size]
                ).clamp(0, 15 if zero_point else 14).to(torch.int)[:, None]
            )
        print(intweight[0].shape)
        intweight = torch.cat(intweight, dim=1)
        print(intweight.shape)

        intweight_ref = intweight
        # intweight = intweight.t().contiguous()

    assert ic % group_size == 0
    intweight = weight.reshape(oc, ic // group_size, group_size)

    # print(f"{intweight.shape} {scale_zeros[..., None].shape} {qscales[..., None].shape}")

    intweight = (intweight + scale_zeros[..., None]) / qscales[..., None]
    intweight = intweight.round_()
    intweight = intweight.clamp_(0, 15 if zero_point else 14)
    intweight = intweight.to(dtype=torch.int32)
    intweight = intweight.reshape(oc, ic)

    if False:
        print(intweight_ref - intweight)
        assert not (intweight_ref - intweight != 0).any()

    tensors["qweight"] = pack_intweight(
        intweight.contiguous(), interleave=4, kstride=64
    )

    zeros = zeros.to(dtype=torch.int32)
    scaled_zeros = torch.zeros_like(qscales)
    # scaled_zeros[:, :scales.shape[1]] = -(qscales[:, :scales.shape[1]] * (zeros.to(torch.float32) - 8.0)).to(torch.float16)
    scaled_zeros[:, : scales.shape[1]] = -(
        qscales[:, : scales.shape[1]] * (zeros.to(torch.float32))
    ).to(dtype)
    tensors["wzeros"] = scaled_zeros.transpose(1, 0).contiguous()

    return tensors
