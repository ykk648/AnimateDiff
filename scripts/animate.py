import argparse
import datetime
import inspect
import os
from omegaconf import OmegaConf

import torch
import PIL
import diffusers
from diffusers import AutoencoderKL, DDIMScheduler, ControlNetModel

from animatediff.models.unet_2d_condition import AnimateDiffUNet2DConditionModel
from animatediff.models.free_lunch_utils import register_free_upblock2d,register_free_crossattn_upblock2d

from animatediff.pipelines.stablediffusion_animatediff_pipeline import StableDiffusionAnimationPipeline
from animatediff.pipelines.stablediffusion_animatediff_inpainting_pipeline import StableDiffusionAnimationInpaintingPipeline
from animatediff.pipelines.stablediffusion_controlnet_animatediff_pipeline import \
    StableDiffusionControlNetAnimateDiffPipeline

from animatediff.utils.util import save_videos_grid
from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, \
    convert_ldm_vae_checkpoint
from animatediff.utils.convert_lora_safetensor_to_diffusers import convert_lora, convert_motion_lora_ckpt_to_diffusers
from ip_adapter.ip_adapter import IPAdapterPlus, IPAdapter
from sd_lib.controlnet import ControlNet

from safetensors import safe_open
from pathlib import Path
import shutil


def pipeline_loading(motion_module, model_config, inference_config):
    # Init unet
    unet = AnimateDiffUNet2DConditionModel.from_pretrained(args.pretrained_model_path, subfolder="unet",
                                                           unet_additional_kwargs=OmegaConf.to_container(
                                                               inference_config.unet_additional_kwargs))

    # controlnet but not reference
    controlnet_list, controlnet_condition_image_list, controlnet_frame_list = [], [], []
    if model_config.enable_controlnet:
        print('init controlnet model.')
        for controlnet_n in model_config.controlnets:
            controlnet_list.append(ControlNet(model_config.controlnets[controlnet_n].controlnet_name, height=args.H, width=args.W, dtype=args.dtype))
            controlnet_condition_image = model_config.controlnets[controlnet_n].controlnet_image
            if hasattr(model_config.controlnets[controlnet_n], 'controlnet_mask_image'):
                controlnet_condition_image = [controlnet_condition_image,model_config.controlnets[controlnet_n].controlnet_mask_image]
            controlnet_condition_image_list.append(controlnet_condition_image)
            controlnet_frame_list.append(model_config.controlnets[controlnet_n].controlnet_frame)

        pipeline = StableDiffusionControlNetAnimateDiffPipeline.from_pretrained(
            args.pretrained_model_path,
            unet=unet,
            controlnet=controlnet_list,
            feature_extractor=None,
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
            safety_checker=None,
            torch_dtype=args.dtype
        )
    # without controlnet, inpainting
    elif hasattr(model_config, 'inpainting') and model_config.inpainting:
        # vae = AutoencoderKL.from_pretrained('models/StableDiffusion/stable-diffusion-v1-5', subfolder="vae")
        pipeline = StableDiffusionAnimationInpaintingPipeline.from_pretrained(
            args.pretrained_model_path,
            unet=unet,
            # vae=vae,
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
            torch_dtype=args.dtype
        )
    # without controlnet
    else:
        pipeline = StableDiffusionAnimationPipeline.from_pretrained(
            args.pretrained_model_path,
            unet=unet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
            torch_dtype=args.dtype
        )

    # AnimateDiff Motion Module
    motion_module_state_dict = torch.load(motion_module, map_location="cpu")
    missing, unexpected = pipeline.unet.load_state_dict(motion_module_state_dict, strict=False)
    assert len(unexpected) == 0

    # Must enable before IpAdapter init, don't know why
    pipeline.enable_xformers_memory_efficient_attention()

    # IPAdapter
    if model_config.enable_ipadapter:
        print('init IPAdapter model.')
        ipap = IPAdapterPlus(pipeline, f'{args.ip_adapter_model_dir}/clip_image_encoder',
                             f'{args.ip_adapter_model_dir}/ip-adapter-plus_sd15.bin', "cuda", num_tokens=16)
        # ipap = IPAdapter(pipeline, f'{args.ip_adapter_model_dir}/clip_image_encoder',
        #                      f'{args.ip_adapter_model_dir}/ip-adapter_sd15.bin', "cuda")
        pipeline = ipap.return_pipe()
    else:
        ipap = None

    # FreeU
    if hasattr(model_config, 'enable_freeu') and model_config.enable_freeu:
        print('Inject freeU.')
        register_free_upblock2d(pipeline, b1=1.5, b2=1.6, s1=0.9, s2=0.2)
        register_free_crossattn_upblock2d(pipeline, b1=1.5, b2=1.6, s1=0.9, s2=0.2)

    pipeline.enable_model_cpu_offload()
    # pipeline.to("cuda")

    # Third party SD base model
    if model_config.base != "":
        print(f'Load model again from {model_config.base}')
        base_state_dict = {}
        with safe_open(model_config.base, framework="pt", device="cpu") as f:
            for key in f.keys():
                base_state_dict[key] = f.get_tensor(key)
        converted_vae_checkpoint = convert_ldm_vae_checkpoint(base_state_dict, pipeline.vae.config)
        pipeline.vae.load_state_dict(converted_vae_checkpoint)
        converted_unet_checkpoint = convert_ldm_unet_checkpoint(base_state_dict, pipeline.unet.config)
        pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)
        # pipeline.text_encoder = convert_ldm_clip_checkpoint(base_state_dict)

    # LORA
    if model_config.lora != "":
        print(f'Load lora from {model_config.lora}')
        if model_config.lora.endswith(".ckpt"):
            state_dict = torch.load(model_config.lora)
            pipeline.unet.load_state_dict(state_dict)

        elif model_config.lora.endswith(".safetensors"):
            state_dict = {}
            with safe_open(model_config.lora, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)

        pipeline = convert_lora(pipeline, state_dict, alpha=model_config.lora_alpha)

        # additional networks
        if hasattr(model_config, 'additional_networks') and len(model_config.additional_networks) > 0:
            for lora_weights in model_config.additional_networks:
                add_state_dict = {}
                (lora_path, lora_alpha) = lora_weights.split(':')
                print(f"loading lora {lora_path} with weight {lora_alpha}")
                lora_alpha = float(lora_alpha.strip())
                with safe_open(lora_path.strip(), framework="pt", device="cpu") as f:
                    for key in f.keys():
                        add_state_dict[key] = f.get_tensor(key)
                pipeline = convert_lora(pipeline, add_state_dict, alpha=lora_alpha)

    # motion LORA
    if hasattr(model_config, 'motion_module_lora_configs'):
        for motion_module_lora_config in model_config.motion_module_lora_configs:
            path, alpha = motion_module_lora_config["path"], motion_module_lora_config["alpha"]
            print(f"load motion LoRA from {path}")
    
            motion_lora_state_dict = torch.load(path, map_location="cpu")
            motion_lora_state_dict = motion_lora_state_dict[
                "state_dict"] if "state_dict" in motion_lora_state_dict else motion_lora_state_dict

            pipeline = convert_motion_lora_ckpt_to_diffusers(pipeline, motion_lora_state_dict, alpha)
    return pipeline, ipap, controlnet_condition_image_list, controlnet_frame_list


def main(args):
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    savedir = f"samples/{Path(args.config).stem}-{time_str}"
    os.makedirs(savedir)
    inference_config = OmegaConf.load(args.inference_config)

    config = OmegaConf.load(args.config)
    samples = []

    sample_idx = 0

    for model_idx, (config_key, model_config) in enumerate(list(config.items())):
        motion_modules = model_config.motion_module
        motion_modules = [motion_modules] if isinstance(motion_modules, str) else list(motion_modules)
        for motion_module in motion_modules:

            pipeline, ipap, controlnet_condition_image_list, controlnet_frame_list = pipeline_loading(motion_module, model_config, inference_config)

            ### <<< create validation pipeline <<< ###

            prompts = model_config.prompt
            n_prompts = list(model_config.n_prompt) * len(prompts) if len(
                model_config.n_prompt) == 1 else model_config.n_prompt
            init_image = model_config.init_image if hasattr(model_config, 'init_image') else None

            random_seeds = model_config.get("seed", [-1])
            random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
            random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds
            config[config_key].random_seed = []

            for prompt_idx, (prompt, n_prompt, random_seed) in enumerate(zip(prompts, n_prompts, random_seeds)):

                # manually set random seed for reproduction
                if random_seed != -1:
                    torch.manual_seed(random_seed)
                else:
                    torch.seed()
                config[config_key].random_seed.append(torch.initial_seed())

                print(f"current seed: {torch.initial_seed()}")
                print(f"sampling {prompt} ...")

                if model_config.enable_ipadapter:
                    prompt_embeds = ipap.get_prompts(prompt, n_prompt, PIL.Image.open(init_image),
                                                     strength=model_config.ip_strength)
                else:
                    prompt_embeds = None

                sample = pipeline(
                    prompt,
                    init_image=init_image,
                    control_image=controlnet_condition_image_list,
                    denoise_strength=model_config.denoise_strength,
                    negative_prompt=n_prompt,
                    prompt_embeds=prompt_embeds,
                    num_inference_steps=model_config.steps,
                    guidance_scale=model_config.guidance_scale,
                    width=args.W,
                    height=args.H,
                    video_length=args.L,
                    # controlnet
                    controlnet_frame=controlnet_frame_list,
                    # reference
                    reference_attn=model_config.enable_reference,
                    reference_adain=False,
                    style_fidelity=model_config.style_fidelity,
                ).videos
                samples.append(sample)

                prompt = "-".join((prompt.replace("/", "").split(" ")[:10]))
                save_videos_grid(sample, f"{savedir}/sample/{sample_idx}-{prompt}.gif")
                print(f"save to {savedir}/sample/{prompt}.gif")

                sample_idx += 1

    samples = torch.concat(samples)
    save_videos_grid(samples, f"{savedir}/sample.gif", n_rows=4)

    OmegaConf.save(config, f"{savedir}/config.yaml")
    if init_image is not None:
        shutil.copy(init_image, f"{savedir}/init_image.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, default="models/StableDiffusion/stable-diffusion-v1-5", )
    parser.add_argument("--ip_adapter_model_dir", type=str, default="models/IP_Adapter", )
    parser.add_argument("--controlnet_model_dir", type=str, default="models/ControlNet", )
    parser.add_argument("--inference_config", type=str, default="configs/inference/inference-v2.yaml")
    parser.add_argument("--dtype", type=torch.dtype, default=torch.float32)
    parser.add_argument("--config", type=str, required=True)

    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--H", type=int, default=512)

    args = parser.parse_args()
    main(args)
