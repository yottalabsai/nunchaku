import time

import torch
import diffusers
from diffusers import FluxPipeline

import nunchaku.pipelines.flux

if __name__ == "__main__":
    QUANT = False
    SEED = 1
    DEV = True
    LORA_NAME = "anime"

    pipe = nunchaku.pipelines.flux.from_pretrained(
        f"black-forest-labs/FLUX.1-{'dev' if DEV else 'schnell'}", 
        torch_dtype=torch.bfloat16,
        qmodel_path=f"/NFS/raid0/user/zhangzk/models/flux{'-dev' if DEV else ''}-svdq-19-38-divsmooth-shift-ada-bf16.safetensors",
        qencoder_path="/NFS/raid0/user/zhangzk/models/flux-t5-tinychat-v2.pt" if QUANT else None,
    )
    if LORA_NAME:
        pipe.transformer.nunchaku_update_params(f"/tmp/flux-lora-{LORA_NAME}-bf16.safetensors")
        pipe.transformer.nunchaku_set_lora_scale(0.4)
    print("Moving model to CUDA")
    pipe.to("cuda:0")
    print("Done")

    # prompt = "A cat holding a sign that says hello world"
    # prompt = "A cyberpunk cat holding a huge neon sign that says \"SVDQuant is lite and fast\""
    prompt = "girl, neck tuft, white hair ,sheep horns, blue eyes, nm22 style"
    # prompt = "GHIBSKY style, the most beautiful place in the universe"
    # prompt = "the joker, yarn art style"
    print(f"Using prompt '{prompt}'")

    latencies = []

    diffusers.training_utils.set_seed(SEED)

    start_time = time.time()
    out = pipe(
        prompt=prompt,
        guidance_scale=3.5 if DEV else 0,
        num_inference_steps=50 if DEV else 4,
        generator=torch.Generator(device="cpu").manual_seed(SEED),
    ).images[0]
    end_time = time.time()
    latencies.append(end_time - start_time)

    out.save(f"output{'-dev' if DEV else ''}-{SEED}-{'quant' if QUANT else 'noquant'}.png")
    print(f"Elapsed: {sum(latencies) / len(latencies)} seconds")
