# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
# 230830 by ykk648: mod from https://github.com/huggingface/diffusers/blob/8ccb619416a36f5951dac654a92e869d76db4bbc/src/diffusers/pipelines/controlnet/pipeline_controlnet.py

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import torch
import PIL
from einops import rearrange, repeat

from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor

from diffusers.models import AutoencoderKL, ControlNetModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.pipelines import StableDiffusionControlNetPipeline
from diffusers.utils import (
    BaseOutput,
)
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from diffusers.models.attention import BasicTransformerBlock
from einops import rearrange

from ..models.unet_2d_condition import AnimateDiffUNet2DConditionOutput
from ..utils.latents_maker import prepare_latents, decode_latents, get_timesteps
from ..models.unet_2d_blocks import CrossAttnDownBlock2D, CrossAttnUpBlock2D, DownBlock2D, UpBlock2D


def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result

@dataclass
class StableDiffusionControlNetAnimateDiffPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class StableDiffusionControlNetAnimateDiffPipeline(StableDiffusionControlNetPipeline):
    _optional_components = []

    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: AnimateDiffUNet2DConditionOutput,
            controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel],
            scheduler: Union[
                DDIMScheduler,
                PNDMScheduler,
                LMSDiscreteScheduler,
                EulerDiscreteScheduler,
                EulerAncestralDiscreteScheduler,
                DPMSolverMultistepScheduler,
            ],
            safety_checker: None,
            feature_extractor: None,
            requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=None,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            requires_safety_checker=requires_safety_checker,
        )
        self.controlnet = controlnet

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]],
            video_length: Optional[int],
            init_image: str = None,
            control_image=None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_videos_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "tensor",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
            guess_mode: bool = False,
            control_guidance_start: Union[float, List[float]] = 0.0,
            control_guidance_end: Union[float, List[float]] = 1.0,
            # reference
            attention_auto_machine_weight: float = 1.0,
            gn_auto_machine_weight: float = 1.0,
            style_fidelity: float = 0.5,
            reference_attn: bool = False,
            reference_adain: bool = False,
            **kwargs,
    ):
        controlnet = self.controlnet
        # control_image = control_image[0]
        controlnet_frame = kwargs['controlnet_frame']

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [
                control_guidance_end
            ]

        # 2. Define call parameters
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

        # 3. Encode input prompt
        if prompt_embeds is None:
            text_encoder_lora_scale = (
                cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
            )
            prompt_embeds = self._encode_prompt(
                prompt,
                device,
                num_videos_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=text_encoder_lora_scale,
            )

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # ref diffusers img2img
        denoise_strength = kwargs['denoise_strength']  # hyper-parameters
        timesteps, num_inference_steps = get_timesteps(self.scheduler, num_inference_steps, denoise_strength)
        noise_timestep = timesteps[0:1]
        noise_timestep = noise_timestep.repeat(batch_size * num_videos_per_prompt)

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = prepare_latents(self, init_image, batch_size * num_videos_per_prompt,
                                  num_channels_latents, video_length, height, width, self.unet.dtype,
                                  device, generator, timestep=None, latents=latents)
        latents_dtype = latents.dtype

        # reference init
        if reference_attn or reference_adain:
            ref_image_latents = prepare_latents(self, init_image, batch_size * num_videos_per_prompt,
                                                num_channels_latents, video_length, height, width, self.unet.dtype,
                                                device, generator, timestep=None, latents=latents, without_noise=True)
            ref_image_latents = self.vae.config.scaling_factor * ref_image_latents
            ref_image_latents = torch.cat([ref_image_latents] * 2) if do_classifier_free_guidance else ref_image_latents
            MODE = "write"
            uc_mask = (
                torch.Tensor([1] * batch_size * num_videos_per_prompt * video_length + [
                    0] * batch_size * num_videos_per_prompt * video_length)
                .type_as(ref_image_latents)
                .bool()
            )
            def hacked_basic_transformer_inner_forward(
                    self,
                    hidden_states: torch.FloatTensor,
                    attention_mask: Optional[torch.FloatTensor] = None,
                    encoder_hidden_states: Optional[torch.FloatTensor] = None,
                    encoder_attention_mask: Optional[torch.FloatTensor] = None,
                    timestep: Optional[torch.LongTensor] = None,
                    cross_attention_kwargs: Dict[str, Any] = None,
                    class_labels: Optional[torch.LongTensor] = None,
            ):
                if self.use_ada_layer_norm:
                    norm_hidden_states = self.norm1(hidden_states, timestep)
                elif self.use_ada_layer_norm_zero:
                    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                        hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                    )
                else:
                    norm_hidden_states = self.norm1(hidden_states)

                # 1. Self-Attention
                cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
                if self.only_cross_attention:
                    attn_output = self.attn1(
                        norm_hidden_states,
                        encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                        attention_mask=attention_mask,
                        **cross_attention_kwargs,
                    )
                else:
                    if MODE == "write":
                        self.bank.append(norm_hidden_states.detach().clone())
                        attn_output = self.attn1(
                            norm_hidden_states,
                            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                            attention_mask=attention_mask,
                            **cross_attention_kwargs,
                        )
                    if MODE == "read":
                        if attention_auto_machine_weight > self.attn_weight:
                            attn_output_uc = self.attn1(
                                norm_hidden_states,
                                encoder_hidden_states=torch.cat([norm_hidden_states] + self.bank, dim=1),
                                # attention_mask=attention_mask,
                                **cross_attention_kwargs,
                            )
                            attn_output_c = attn_output_uc.clone()
                            if do_classifier_free_guidance and style_fidelity > 0:
                                attn_output_c[uc_mask] = self.attn1(
                                    norm_hidden_states[uc_mask],
                                    encoder_hidden_states=norm_hidden_states[uc_mask],
                                    **cross_attention_kwargs,
                                )
                            attn_output = style_fidelity * attn_output_c + (1.0 - style_fidelity) * attn_output_uc
                            self.bank.clear()
                        else:
                            attn_output = self.attn1(
                                norm_hidden_states,
                                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                                attention_mask=attention_mask,
                                **cross_attention_kwargs,
                            )
                if self.use_ada_layer_norm_zero:
                    attn_output = gate_msa.unsqueeze(1) * attn_output
                hidden_states = attn_output + hidden_states

                if self.attn2 is not None:
                    norm_hidden_states = (
                        self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                    )

                    # 2. Cross-Attention
                    attn_output = self.attn2(
                        norm_hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=encoder_attention_mask,
                        **cross_attention_kwargs,
                    )
                    hidden_states = attn_output + hidden_states

                # 3. Feed-forward
                norm_hidden_states = self.norm3(hidden_states)

                if self.use_ada_layer_norm_zero:
                    norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

                ff_output = self.ff(norm_hidden_states)

                if self.use_ada_layer_norm_zero:
                    ff_output = gate_mlp.unsqueeze(1) * ff_output

                hidden_states = ff_output + hidden_states

                return hidden_states

            def hacked_mid_forward(self, *args, **kwargs):
                eps = 1e-6
                x = self.original_forward(*args, **kwargs)
                if MODE == "write":
                    if gn_auto_machine_weight >= self.gn_weight:
                        var, mean = torch.var_mean(x, dim=(2, 3), keepdim=True, correction=0)
                        self.mean_bank.append(mean)
                        self.var_bank.append(var)
                if MODE == "read":
                    if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
                        var, mean = torch.var_mean(x, dim=(2, 3), keepdim=True, correction=0)
                        std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                        mean_acc = sum(self.mean_bank) / float(len(self.mean_bank))
                        var_acc = sum(self.var_bank) / float(len(self.var_bank))
                        std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                        x_uc = (((x - mean) / std) * std_acc) + mean_acc
                        x_c = x_uc.clone()
                        if do_classifier_free_guidance and style_fidelity > 0:
                            x_c[uc_mask] = x[uc_mask]
                        x = style_fidelity * x_c + (1.0 - style_fidelity) * x_uc
                    self.mean_bank = []
                    self.var_bank = []
                return x

            def hack_CrossAttnDownBlock2D_forward(
                    self,
                    hidden_states: torch.FloatTensor,
                    temb: Optional[torch.FloatTensor] = None,
                    encoder_hidden_states: Optional[torch.FloatTensor] = None,
                    attention_mask: Optional[torch.FloatTensor] = None,
                    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
                    encoder_attention_mask: Optional[torch.FloatTensor] = None,
            ):
                eps = 1e-6

                # TODO(Patrick, William) - attention mask is not used
                output_states = ()

                for i, (resnet, attn, motion_module) in enumerate(
                        zip(self.resnets, self.attentions, self.motion_modules)):
                    hidden_states = resnet(hidden_states, temb)
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]
                    hidden_states = motion_module(hidden_states, temb,
                                                  encoder_hidden_states=encoder_hidden_states) if motion_module is not None else hidden_states

                    if MODE == "write":
                        if gn_auto_machine_weight >= self.gn_weight:
                            var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                            self.mean_bank.append([mean])
                            self.var_bank.append([var])
                    if MODE == "read":
                        if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
                            var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                            std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                            mean_acc = sum(self.mean_bank[i]) / float(len(self.mean_bank[i]))
                            var_acc = sum(self.var_bank[i]) / float(len(self.var_bank[i]))
                            std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                            hidden_states_uc = (((hidden_states - mean) / std) * std_acc) + mean_acc
                            hidden_states_c = hidden_states_uc.clone()
                            if do_classifier_free_guidance and style_fidelity > 0:
                                hidden_states_c[uc_mask] = hidden_states[uc_mask]
                            hidden_states = style_fidelity * hidden_states_c + (1.0 - style_fidelity) * hidden_states_uc

                    output_states = output_states + (hidden_states,)

                if MODE == "read":
                    self.mean_bank = []
                    self.var_bank = []

                if self.downsamplers is not None:
                    for downsampler in self.downsamplers:
                        hidden_states = downsampler(hidden_states)

                    output_states = output_states + (hidden_states,)

                return hidden_states, output_states

            def hacked_DownBlock2D_forward(self, hidden_states, temb=None, encoder_hidden_states=None):
                eps = 1e-6

                output_states = ()

                for i, resnet, motion_module in zip(range(len(self.resnets)), self.resnets, self.motion_modules):
                    hidden_states = resnet(hidden_states, temb)
                    hidden_states = motion_module(hidden_states, temb,
                                                  encoder_hidden_states=encoder_hidden_states) if motion_module is not None else hidden_states

                    if MODE == "write":
                        if gn_auto_machine_weight >= self.gn_weight:
                            var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                            self.mean_bank.append([mean])
                            self.var_bank.append([var])
                    if MODE == "read":
                        if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
                            var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                            std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                            mean_acc = sum(self.mean_bank[i]) / float(len(self.mean_bank[i]))
                            var_acc = sum(self.var_bank[i]) / float(len(self.var_bank[i]))
                            std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                            hidden_states_uc = (((hidden_states - mean) / std) * std_acc) + mean_acc
                            hidden_states_c = hidden_states_uc.clone()
                            if do_classifier_free_guidance and style_fidelity > 0:
                                hidden_states_c[uc_mask] = hidden_states[uc_mask]
                            hidden_states = style_fidelity * hidden_states_c + (1.0 - style_fidelity) * hidden_states_uc

                    output_states = output_states + (hidden_states,)

                if MODE == "read":
                    self.mean_bank = []
                    self.var_bank = []

                if self.downsamplers is not None:
                    for downsampler in self.downsamplers:
                        hidden_states = downsampler(hidden_states)

                    output_states = output_states + (hidden_states,)

                return hidden_states, output_states

            def hacked_CrossAttnUpBlock2D_forward(
                    self,
                    hidden_states: torch.FloatTensor,
                    res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
                    temb: Optional[torch.FloatTensor] = None,
                    encoder_hidden_states: Optional[torch.FloatTensor] = None,
                    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
                    upsample_size: Optional[int] = None,
                    attention_mask: Optional[torch.FloatTensor] = None,
                    encoder_attention_mask: Optional[torch.FloatTensor] = None,
            ):
                eps = 1e-6
                # TODO(Patrick, William) - attention mask is not used
                for i, resnet, attn, motion_module in zip(range(len(self.resnets)), self.resnets, self.attentions,
                                                          self.motion_modules):
                    # pop res hidden states
                    res_hidden_states = res_hidden_states_tuple[-1]
                    res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                    hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
                    hidden_states = resnet(hidden_states, temb)
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]
                    hidden_states = motion_module(hidden_states, temb,
                                                  encoder_hidden_states=encoder_hidden_states) if motion_module is not None else hidden_states

                    if MODE == "write":
                        if gn_auto_machine_weight >= self.gn_weight:
                            var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                            self.mean_bank.append([mean])
                            self.var_bank.append([var])
                    if MODE == "read":
                        if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
                            var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                            std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                            mean_acc = sum(self.mean_bank[i]) / float(len(self.mean_bank[i]))
                            var_acc = sum(self.var_bank[i]) / float(len(self.var_bank[i]))
                            std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                            hidden_states_uc = (((hidden_states - mean) / std) * std_acc) + mean_acc
                            hidden_states_c = hidden_states_uc.clone()
                            if do_classifier_free_guidance and style_fidelity > 0:
                                hidden_states_c[uc_mask] = hidden_states[uc_mask]
                            hidden_states = style_fidelity * hidden_states_c + (1.0 - style_fidelity) * hidden_states_uc

                if MODE == "read":
                    self.mean_bank = []
                    self.var_bank = []

                if self.upsamplers is not None:
                    for upsampler in self.upsamplers:
                        hidden_states = upsampler(hidden_states, upsample_size)

                return hidden_states

            def hacked_UpBlock2D_forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None,
                                         encoder_hidden_states=None):
                eps = 1e-6
                for i, resnet, motion_module in zip(range(len(self.resnets)), self.resnets, self.motion_modules):
                    # pop res hidden states
                    res_hidden_states = res_hidden_states_tuple[-1]
                    res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                    hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
                    hidden_states = resnet(hidden_states, temb)
                    hidden_states = motion_module(hidden_states, temb,
                                                  encoder_hidden_states=encoder_hidden_states) if motion_module is not None else hidden_states

                    if MODE == "write":
                        if gn_auto_machine_weight >= self.gn_weight:
                            var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                            self.mean_bank.append([mean])
                            self.var_bank.append([var])
                    if MODE == "read":
                        if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
                            var, mean = torch.var_mean(hidden_states, dim=(2, 3), keepdim=True, correction=0)
                            std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                            mean_acc = sum(self.mean_bank[i]) / float(len(self.mean_bank[i]))
                            var_acc = sum(self.var_bank[i]) / float(len(self.var_bank[i]))
                            std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                            hidden_states_uc = (((hidden_states - mean) / std) * std_acc) + mean_acc
                            hidden_states_c = hidden_states_uc.clone()
                            if do_classifier_free_guidance and style_fidelity > 0:
                                hidden_states_c[uc_mask] = hidden_states[uc_mask]
                            hidden_states = style_fidelity * hidden_states_c + (1.0 - style_fidelity) * hidden_states_uc

                if MODE == "read":
                    self.mean_bank = []
                    self.var_bank = []

                if self.upsamplers is not None:
                    for upsampler in self.upsamplers:
                        hidden_states = upsampler(hidden_states, upsample_size)

                return hidden_states

            if reference_attn:
                attn_modules = [module for module in torch_dfs(self.unet) if isinstance(module, BasicTransformerBlock)]
                attn_modules = sorted(attn_modules, key=lambda x: -x.norm1.normalized_shape[0])

                for i, module in enumerate(attn_modules):
                    module._original_inner_forward = module.forward
                    module.forward = hacked_basic_transformer_inner_forward.__get__(module, BasicTransformerBlock)
                    module.bank = []
                    module.attn_weight = float(i) / float(len(attn_modules))

            if reference_adain:
                gn_modules = [self.unet.mid_block]
                self.unet.mid_block.gn_weight = 0

                down_blocks = self.unet.down_blocks
                for w, module in enumerate(down_blocks):
                    module.gn_weight = 1.0 - float(w) / float(len(down_blocks))
                    gn_modules.append(module)

                up_blocks = self.unet.up_blocks
                for w, module in enumerate(up_blocks):
                    module.gn_weight = float(w) / float(len(up_blocks))
                    gn_modules.append(module)

                for i, module in enumerate(gn_modules):
                    if getattr(module, "original_forward", None) is None:
                        module.original_forward = module.forward
                    if i == 0:
                        # mid_block
                        module.forward = hacked_mid_forward.__get__(module, torch.nn.Module)
                    elif isinstance(module, CrossAttnDownBlock2D):
                        module.forward = hack_CrossAttnDownBlock2D_forward.__get__(module, CrossAttnDownBlock2D)
                    elif isinstance(module, DownBlock2D):
                        module.forward = hacked_DownBlock2D_forward.__get__(module, DownBlock2D)
                    elif isinstance(module, CrossAttnUpBlock2D):
                        module.forward = hacked_CrossAttnUpBlock2D_forward.__get__(module, CrossAttnUpBlock2D)
                    elif isinstance(module, UpBlock2D):
                        module.forward = hacked_UpBlock2D_forward.__get__(module, UpBlock2D)
                    module.mean_bank = []
                    module.var_bank = []
                    module.gn_weight *= 2

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0])

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        # torch.Size([32, 77, 768])
        controlnet_prompt_embeds = prompt_embeds
        prompt_embeds = repeat(prompt_embeds, 'b n c -> (b f) n c', f=video_length)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                # torch.Size([2, 4, 16, 64, 64])
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # controlnet(s) inference
                if guess_mode and do_classifier_free_guidance:
                    # Infer ControlNet only for the conditional batch.
                    control_model_input = latents
                    control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                    controlnet_prompt_embeds = controlnet_prompt_embeds.chunk(2)[1]
                else:
                    control_model_input = latent_model_input
                    # controlnet_prompt_embeds = controlnet_prompt_embeds

                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]

                # Got first latent to do controlnet infer
                control_model_input = control_model_input[:, :, 0, :, :]

                for k, (controlnet_, controlnet_frame_, control_image_) in enumerate(
                        zip(self.controlnet, controlnet_frame,
                            control_image)):
                    down_block_res_samples, mid_block_res_sample = controlnet_.forward(
                        control_model_input.to(controlnet_.dtype),
                        t,
                        encoder_hidden_states=controlnet_prompt_embeds,
                        condition_image=control_image_,
                        conditioning_scale=cond_scale,
                        guess_mode=False,
                    )

                    # Broad to (b f) c h w according to controlnet_frame
                    for index, d in enumerate(down_block_res_samples):
                        # 32,320,64,64
                        temp = torch.zeros_like(d).repeat(video_length, 1, 1, 1)
                        for j in range(video_length):
                            temp[j] = d[0].unsqueeze(0) * controlnet_frame_[j]
                            temp[j + video_length] = d[1].unsqueeze(0) * controlnet_frame_[j]
                        down_block_res_samples[index] = temp
                    temp = torch.zeros_like(mid_block_res_sample).repeat(video_length, 1, 1, 1)
                    for j in range(video_length):
                        temp[j] = mid_block_res_sample[0].unsqueeze(0) * controlnet_frame_[j]
                        temp[j + video_length] = mid_block_res_sample[1].unsqueeze(0) * controlnet_frame_[j]
                    mid_block_res_sample = temp
                    if guess_mode and do_classifier_free_guidance:
                        # Infered ControlNet only for the conditional batch.
                        # To apply the output of ControlNet to both the unconditional and conditional batches,
                        # add 0 to the unconditional batch to keep it unchanged.
                        down_block_res_samples = [torch.cat([torch.zeros_like(d), d].to(device, dtype=self.unet.dtype))
                                                  for
                                                  d in down_block_res_samples]
                        mid_block_res_sample = torch.cat(
                            [torch.zeros_like(mid_block_res_sample), mid_block_res_sample]).to(
                            device, dtype=self.unet.dtype)
                    # multi controlnet summary
                    if k == 0:
                        down_block_res_samples_, mid_block_res_sample_ = down_block_res_samples, mid_block_res_sample
                    else:
                        down_block_res_samples_ = [
                            samples_prev + samples_curr
                            for samples_prev, samples_curr in zip(down_block_res_samples_, down_block_res_samples)
                        ]
                        mid_block_res_sample_ += mid_block_res_sample

                # reference
                if reference_attn or reference_adain:
                    noise = randn_tensor(ref_image_latents.shape, generator=generator, device=device,
                                         dtype=ref_image_latents.dtype)
                    ref_xt = self.scheduler.add_noise(ref_image_latents,noise,t.reshape(1, ),)
                    ref_xt = self.scheduler.scale_model_input(ref_xt, t)
                    MODE="write"
                    ref_xt = rearrange(ref_xt, "b c f h w -> (b f) c h w")
                    self.unet(ref_xt, t, encoder_hidden_states=prompt_embeds,
                              cross_attention_kwargs=cross_attention_kwargs, return_dict=False, )
                    # predict the noise residual
                    MODE="read"

                latent_model_input = rearrange(latent_model_input, "b c f h w -> (b f) c h w")

                # predict the noise residual
                # torch.Size([32, 4, 64, 64])
                noise_pred = self.unet(
                    latent_model_input.to(device, dtype=self.unet.dtype),
                    t,
                    encoder_hidden_states=prompt_embeds.to(device, dtype=self.unet.dtype),
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples_,
                    mid_block_additional_residual=mid_block_res_sample_,
                    return_dict=False,
                )[0]
                # torch.Size([2, 4, 16, 64, 64])
                noise_pred = rearrange(noise_pred, "(b f) c h w -> b c f h w", f=video_length)

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.model.to("cpu")
            torch.cuda.empty_cache()

        # Post-processing
        video = decode_latents(self, latents)

        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return video

        return StableDiffusionControlNetAnimateDiffPipelineOutput(videos=video)
