import time
import argparse
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel

import nunchaku.pipelines.flux

def get_pipe(config: str, dev: bool) -> FluxPipeline:
    version = "dev" if dev else "schnell"
    dtype = torch.bfloat16

    qencoder_path = "/NFS/raid0/user/zhangzk/models/flux-t5-tinychat-v2.pt"

    if config.startswith("svdq"):
        pipe = nunchaku.pipelines.flux.from_pretrained(
            f"black-forest-labs/FLUX.1-{version}", 
            torch_dtype=dtype,
            qmodel_path=f"/NFS/raid0/user/zhangzk/models/flux{'-dev' if dev else ''}-svdq-19-38-divsmooth-shift-ada-bf16.safetensors",
            qencoder_path=qencoder_path if config == "svdq-t5" else None
        )
    elif config.startswith("w4a4"):
        pipe = nunchaku.pipelines.flux.from_pretrained(
            f"black-forest-labs/FLUX.1-{version}", 
            torch_dtype=dtype,
            qmodel_path=f"/NFS/raid0/user/zhangzk/models/flux{'-dev' if dev else ''}-divsmooth-shift-ada-bf16.safetensors",
            qencoder_path=qencoder_path if config == "w4a4-t5" else None
        )
    elif config.startswith("bf16"):
        pipe = FluxPipeline.from_pretrained(
            f"black-forest-labs/FLUX.1-{version}", 
            torch_dtype=dtype,
        )
        if config == "bf16-t5":
            nunchaku.pipelines.flux.quantize_t5(pipe, qencoder_path)
    elif config.startswith("nf4"):
        from accelerate.utils import set_module_tensor_to_device, compute_module_sizes
        from accelerate import init_empty_weights
        from convert_nf4_flux import _replace_with_bnb_linear, create_quantized_param, check_quantized_param

        converted_state_dict = torch.load(f"/NFS/raid0/user/zhangzk/models/flux1-{version}-nf4.pt")

        with init_empty_weights():
            config = FluxTransformer2DModel.load_config(f"black-forest-labs/flux.1-{version}", subfolder="transformer")
            model = FluxTransformer2DModel.from_config(config).to(dtype)

        _replace_with_bnb_linear(model, "nf4")
        for param_name, param in converted_state_dict.items():
            param = param.to(dtype)
            print(f"{param_name}: {param.shape} check_quantized_param={check_quantized_param(model, param_name)}")
            if not check_quantized_param(model, param_name):
                set_module_tensor_to_device(model, param_name, device=0, value=param)
            else:
                create_quantized_param(model, param, param_name, target_device=0)

        pipe = FluxPipeline.from_pretrained(f"black-forest-labs/flux.1-{version}", transformer=model, torch_dtype=dtype)
        if config == "nf4-t5":
            nunchaku.pipelines.flux.quantize_t5(pipe, qencoder_path)
    else:
        raise NotImplementedError

    return pipe


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="svdq", choices=["svdq", "svdq-t5", "w4a4", "w4a4-t5", "bf16", "bf16-t5", "nf4", "nf4-t5"])
    parser.add_argument("--offload", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--torchao", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    print(f"Use config {args.config}")
    if args.offload > 0:
        print(f"Use offloading level {args.offload}")

    pipe = get_pipe(args.config, args.dev)
    print(pipe)

    
    if args.torchao:
        from torchao.quantization import quantize_, int8_dynamic_activation_int8_weight
        # pipe.transformer = autoquant(pipe.transformer, error_on_unseen=False)
        quantize_(pipe.transformer, int8_dynamic_activation_int8_weight())

    if args.offload == 2:
        pipe.enable_sequential_cpu_offload()
    elif args.offload == 1:
        pipe.enable_model_cpu_offload()
    elif args.offload == 0:
        pipe.to("cuda:0")
    else:
        raise NotImplementedError
    # assert isinstance(pipe, FluxPipeline)


    if args.compile:
        pipe.transformer.to(memory_format=torch.channels_last)
        pipe.transformer = torch.compile(
            pipe.transformer, mode="max-autotune", fullgraph=True
        )

    

    prompt = "A cat holding a sign that says hello world"
    print(f"Using prompt '{prompt}'")
    print(f"Run {args.steps} steps")

    latencies = []

    for i in range(5):
        start_time = time.time()
        out = pipe(
            prompt=prompt,
            guidance_scale=0,
            num_inference_steps=args.steps,
            generator=torch.Generator(device="cpu").manual_seed(233),
        ).images[0]
        end_time = time.time()
        latencies.append(end_time - start_time)
        torch.cuda.empty_cache()
    latencies = sorted(latencies)
    latencies = latencies[1:-1]

    out.save("output.png")
    print(f"Elapsed: {sum(latencies) / len(latencies)} seconds")

    print(f"Torch max_memory_allocated={torch.cuda.max_memory_allocated()}")