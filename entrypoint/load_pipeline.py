from diffusers import FluxPipeline
import torch
from peft.tuners import lora

from entrypoint.openai.log import setup_logging
from entrypoint.vars import LORA_PATHS, SVDQ_LORA_PATHS
from nunchaku.models.transformer_flux import NunchakuFluxTransformer2dModel

logger = setup_logging()

def get_pipeline(
    model_name: str,
    precision: str,
    use_qencoder: bool = False,
    lora_name: str = "None",
    lora_weight: float = 1,
    device: str | torch.device = "cuda",
    pipeline_init_kwargs: dict = {},
) -> FluxPipeline:
    logger.info(f"Loading model {model_name} with precision {precision}, use_qencoder={use_qencoder}, lora_name={lora_name}, lora_weight={lora_weight}, device={device}")
    if model_name == "schnell":
        if precision == "int4":
            assert torch.device(device).type == "cuda", "int4 only supported on CUDA devices"
            transformer = NunchakuFluxTransformer2dModel.from_pretrained("mit-han-lab/svdq-int4-flux.1-schnell")
            pipeline_init_kwargs["transformer"] = transformer
            if use_qencoder:
                from nunchaku.models.text_encoder import NunchakuT5EncoderModel

                text_encoder_2 = NunchakuT5EncoderModel.from_pretrained("mit-han-lab/svdq-flux.1-t5")
                pipeline_init_kwargs["text_encoder_2"] = text_encoder_2
        else:
            assert precision == "bf16"
        pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16, **pipeline_init_kwargs
        )
    elif model_name == "dev":
        if precision == "int4":
            transformer = NunchakuFluxTransformer2dModel.from_pretrained("mit-han-lab/svdq-int4-flux.1-dev")
            if lora_name not in ["All", "None"]:
                transformer.update_lora_params(SVDQ_LORA_PATHS[lora_name])
                transformer.set_lora_strength(lora_weight)
            pipeline_init_kwargs["transformer"] = transformer
            if use_qencoder:
                from nunchaku.models.text_encoder import NunchakuT5EncoderModel

                text_encoder_2 = NunchakuT5EncoderModel.from_pretrained("mit-han-lab/svdq-flux.1-t5")
                pipeline_init_kwargs["text_encoder_2"] = text_encoder_2
            pipeline = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, **pipeline_init_kwargs
            )
        else:
            assert precision == "bf16"
            pipeline = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, **pipeline_init_kwargs
            )
            if lora_name == "All":
                # Pre-load all the LoRA weights for demo use
                for name, path in LORA_PATHS.items():
                    pipeline.load_lora_weights(path["name_or_path"], weight_name=path["weight_name"], adapter_name=name)
                for m in pipeline.transformer.modules():
                    if isinstance(m, lora.LoraLayer):
                        m.set_adapter(m.scaling.keys())
                        for name in m.scaling.keys():
                            m.scaling[name] = 0
            elif lora_name != "None":
                path = LORA_PATHS[lora_name]
                pipeline.load_lora_weights(
                    path["name_or_path"], weight_name=path["weight_name"], adapter_name=lora_name
                )
                for m in pipeline.transformer.modules():
                    if isinstance(m, lora.LoraLayer):
                        for name in m.scaling.keys():
                            m.scaling[name] = lora_weight
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")
    pipeline = pipeline.to(device)

    return pipeline