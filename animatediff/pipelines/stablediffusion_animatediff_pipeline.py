# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py

from typing import Callable, List, Optional, Union, Tuple, Any
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm
from einops import rearrange, repeat

from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.models import AutoencoderKL
from diffusers.pipelines import StableDiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import (
    BaseOutput,
)

from einops import rearrange

from ..models.unet_2d_condition import AnimateDiffUNet2DConditionModel
from ..utils.latents_maker import prepare_latents, decode_latents, get_timesteps


@dataclass
class AnimationPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class StableDiffusionAnimationPipeline(StableDiffusionPipeline):
    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: AnimateDiffUNet2DConditionModel,
            scheduler: Union[
                DDIMScheduler,
                PNDMScheduler,
                LMSDiscreteScheduler,
                EulerDiscreteScheduler,
                EulerAncestralDiscreteScheduler,
                DPMSolverMultistepScheduler,
            ],
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            requires_safety_checker=requires_safety_checker,
        )

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]],
            video_length: Optional[int],
            init_image: str = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            prompt_embeds=None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_videos_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "tensor",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            **kwargs,
    ):
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # Define call parameters
        # batch_size = 1 if isinstance(prompt, str) else len(prompt)
        batch_size = 1
        if latents is not None:
            batch_size = latents.shape[0]
        if isinstance(prompt, list):
            batch_size = len(prompt)

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode input prompt
        if prompt_embeds is None:
            prompt = prompt if isinstance(prompt, list) else [prompt] * batch_size
            if negative_prompt is not None:
                negative_prompt = negative_prompt if isinstance(negative_prompt, list) else [
                                                                                                negative_prompt] * batch_size
            prompt_embeds = self._encode_prompt(
                prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
            )

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # ref diffusers img2img
        denoise_strength = kwargs['denoise_strength']  # hyper-parameters
        timesteps, num_inference_steps = get_timesteps(self.scheduler, num_inference_steps, denoise_strength)
        noise_timestep = timesteps[0:1]
        noise_timestep = noise_timestep.repeat(batch_size * num_videos_per_prompt)

        # Prepare latent variables
        num_channels_latents = self.unet.config.in_channels # 4
        latents = prepare_latents(self,
                                  init_image,
                                  batch_size * num_videos_per_prompt,
                                  num_channels_latents,
                                  video_length,
                                  height,
                                  width,
                                  prompt_embeds.dtype,
                                  device,
                                  generator,
                                  timestep=noise_timestep,
                                  latents=latents,
                                  )
        latents_dtype = latents.dtype

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        # torch.Size([32, 77, 768])
        prompt_embeds = repeat(prompt_embeds, 'b n c -> (b f) n c', f=video_length)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                # torch.Size([2, 4, 16, 64, 64])
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                # torch.Size([32, 4, 64, 64])
                latent_model_input = rearrange(latent_model_input, "b c f h w -> (b f) c h w")
                noise_pred = self.unet(
                    latent_model_input.to(device, dtype=self.unet.dtype),
                    t,
                    encoder_hidden_states=prompt_embeds.to(device, dtype=self.unet.dtype),
                ).sample.to(dtype=latents_dtype)
                # torch.Size([2, 4, 16, 64, 64])
                noise_pred = rearrange(noise_pred, "(b f) c h w -> b c f h w", f=video_length)

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # Post-processing
        video = decode_latents(self, latents)

        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)

        if not return_dict:
            return video

        return AnimationPipelineOutput(videos=video)
