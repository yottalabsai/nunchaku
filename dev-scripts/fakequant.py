import torch
from torch.nn import functional as F
from dump_flux import group_scale

def compare(
        ref: torch.Tensor, 
        v: torch.Tensor, 
        refname: str, 
        vname: str, 
        list_diff: bool = False):
    
    print(f"== COMPARE v={vname} vs ref={refname}")
    diff = v - ref
    print(f" - diff = {diff}")

    if list_diff:
        print(f" - diffs at {diff.nonzero()}")

    mse = diff.square().mean()
    print(f" - mse = {mse}")
    nmse = mse / ref.square().mean()
    print(f" - nmse = {nmse}")

    print(f" - mean(v/ref)={v.mean()}/{ref.mean()}")
    print(f" - var(v/ref)={v.var()}/{ref.var()}")
    print(f"== ")
    print()

def print_debug_results(debug_results: dict[str, torch.Tensor], is_ref: bool = False):
    ref = 'REF' if is_ref else ''
    for k, v in debug_results.items():
        has_nan = v.isnan().any()
        has_inf = v.isinf().any()
        
        if v.dtype.is_floating_point:
            print(f" {ref} {k}: {v.shape} ({v.dtype}) has_nan={has_nan} has_inf={has_inf} max={v.max()} min={v.min()} mean={v.mean()} var={v.var()}")
        else:
            print(f" {ref} {k}: {v.shape} ({v.dtype})")
        if has_nan:
            cnt = v.isnan().count_nonzero()
            print(f" {ref} -- {cnt} ({cnt / v.numel() * 100}%) nans at {v.isnan().nonzero()[0:10]}")
        if has_inf:
            cnt = v.isinf().count_nonzero()
            print(f" {ref} -- {cnt} ({cnt / v.numel() * 100}%) infs at {v.isinf().nonzero()[0:10]}")
        print(f" {ref} -- {v}")
        print()

def fakequant(
        act: torch.Tensor,
        weight: torch.Tensor, 
        bias: torch.Tensor,
        group_size: int = 64,
        force_cuda: bool = False,
        ):
    
    oc, ic = weight.shape
    batch_size = act.shape[0]
    assert act.shape[1] == ic

    # [oc, ic // group_size]
    wscales = group_scale(weight, num_bits=4, group_size=group_size)
    qweight = weight.reshape(oc, ic // group_size, group_size).to(dtype=torch.float32) / wscales[..., None]
    qweight = qweight.round().clamp(-8, 7)
    qweight_i = qweight.int()
    qweight = qweight * wscales[..., None]
    qweight = qweight.to(weight.dtype)
    qweight = qweight.reshape(oc, ic)
    # print(f"qweight = {qweight}")
    print_debug_results({"qweight": qweight})

    # [batch_size, ic // group_size]
    ascales = group_scale(act, num_bits=4, group_size=group_size).to(dtype=weight.dtype)
    qact = act.reshape(batch_size, ic // group_size, group_size).to(dtype=torch.float32) / ascales[..., None]
    qact = qact.round().clamp(-8, 7)
    qact_i = qact.int()
    print_debug_results({"qact_i": qact_i})
    qact = qact * ascales[..., None]
    qact = qact.to(act.dtype)
    qact = qact.reshape(batch_size, ic)
    # print(f"qact = {qact}")
    print_debug_results({"qact": qact})

    outref_q = F.linear(qact.to(qweight.dtype), qweight, bias)
    # print(f"outref_q = {outref_q}")
    print_debug_results({"outref_q": outref_q})

    ###

    if force_cuda:
        qweight_i = qweight_i.to("cuda")
        qact_i = qact_i.to("cuda")
        wscales = wscales.to("cuda")
        ascales = ascales.to("cuda")
        bias = bias.to("cuda")

    qweight = qweight_i
    qact = qact_i
    qweight = qweight.reshape(oc, ic // group_size, group_size).transpose(0, 1).transpose(1, 2)
    qact = qact.reshape(batch_size, ic // group_size, group_size).transpose(0, 1)

    # [ic // group_size, batch_size, oc]
    psum = torch.bmm(qact.float(), qweight.float())
    print(f"psum_i ({psum.shape}) = {psum} ")
    # print(psum[:, 0, 23])

    # print(f"ascales = {ascales}")
    print_debug_results({"ascales": ascales})
    print(f"ascales[0:16] = {ascales[0:16, 0]}")

    ws1 = wscales.transpose(0, 1).reshape(ic // group_size, 1, oc).repeat(1, batch_size, 1)
    as1 = ascales.transpose(0, 1).reshape(ic // group_size, batch_size, 1).repeat(1, 1, oc)
    scales = ws1 * as1

    print(f"scales = {scales}")
    # print(scales[:, 0, 23])

    psum = psum.to(dtype=act.dtype).float()
    psum = psum * scales
    print(f"psum ({psum.shape}) = {psum} ")
    # print(psum[:, 0, 23])

    # outref_q2 = psum.sum(dim=0) # .to(layer.weight.dtype)
    outref_q2 = torch.zeros_like(psum[0])
    for i in range(psum.shape[0]):
        outref_q2 = (outref_q2 + psum[i]).to(act.dtype)
    outref_q2 += bias[None, ...]
    # print(f"outref_q2 = {outref_q2}")
    print_debug_results({"outref_q2": outref_q2})

    # print(outref_q2[0, 23])

    if force_cuda:
        outref_q2 = outref_q2.to(act.device)

    return outref_q, outref_q2
