from safetensors.torch import safe_open, save_file


def main():
    input_path1 = "app/i2i/pretrained/converted/sketch.safetensors"
    input_path2 = "app/i2i/pretrained/original/flux-lora-sketch2image-bf16.safetensors"

    sd1 = {}
    with safe_open(input_path1, framework="pt") as f:
        for k in f.keys():
            sd1[k] = f.get_tensor(k)

    sd2 = {}
    with safe_open(input_path2, framework="pt") as f:
        for k in f.keys():
            sd2[k] = f.get_tensor(k)

    for k in sd1.keys():
        if "lora" not in k:
            print(k)
            sd2[k.replace("transformer.", "")] = sd1[k]

    save_file(sd2, "svdq-flux.1-pix2pix-turbo-sketch2image.safetensors")


if __name__ == "__main__":
    main()
