# -- coding: utf-8 --
# @Time : 2023/10/20
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power

from diffusers import ControlNetModel
from diffusers.image_processor import VaeImageProcessor
from cv2box import CVImage
import numpy as np
import torch


MODEL_ZOO = {
    'control_v11p_sd15_canny': {
        'model_path': 'models/ControlNet/control_v11p_sd15_canny/',
        'use_safetensors': False,
    },
    'control_v11p_sd15_normalbae': {
        'model_path': 'models/ControlNet/control_v11p_sd15_normalbae/',
        'use_safetensors': False,
    },
    'control_v11f1e_sd15_tile': {
        'model_path': 'models/ControlNet/control_v11f1e_sd15_tile/',
        'use_safetensors': False,
    },
    'control_v11e_sd15_ip2p': {
        'model_path': 'models/ControlNet/control_v11e_sd15_ip2p/',
        'use_safetensors': False,
    },
    'control_v11p_sd15_inpaint': {
        'model_path': 'models/ControlNet/control_v11p_sd15_inpaint/',
        'use_safetensors': False,
    },

}


class ControlNet:
    def __init__(self, model_name='control_v11p_sd15_canny', cond_scale=1, vae_scale_factor=8, height=512, width=512,
                 device='cuda', dtype=torch.float32):
        self.cond_scale = cond_scale
        self.vae_scale_factor = vae_scale_factor
        self.height = height
        self.width = width
        self.device = device
        self.dtype = dtype
        self.condition_image = None

        self.model = ControlNetModel.from_pretrained(MODEL_ZOO[model_name]['model_path'],
                                                     torch_dtype=dtype,
                                                     use_safetensors=MODEL_ZOO[model_name]['use_safetensors']
                                                     ).to(self.device)
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )

    def preprocess(self, condition_image, do_classifier_free_guidance, guess_mode):
        if self.condition_image is None:
            if isinstance(condition_image, list):
                # inpainting
                from .utils import make_inpaint_condition
                condition_image = make_inpaint_condition(condition_image[0], condition_image[1])
            else:
                condition_image = CVImage(condition_image).pillow()
            self.condition_image = self.control_image_processor.preprocess(condition_image, height=self.height,
                                                                           width=self.width).to(self.device,
                dtype=self.dtype)
            if do_classifier_free_guidance and not guess_mode:
                self.condition_image = torch.cat([self.condition_image] * 2)

    def forward(self, latent, t, condition_image, encoder_hidden_states, conditioning_scale=1, do_classifier_free_guidance=True,
                guess_mode=False):
        """
        Args:
            latent: ([2,4,64,64])
            t: int
            condition_image: str or list(inpainting img+mask
            encoder_hidden_states: ([2, 77, 768])
            conditioning_scale:
            do_classifier_free_guidance:
            guess_mode:
        Returns:

        """
        self.preprocess(condition_image, do_classifier_free_guidance, guess_mode)

        down_block_res_samples, mid_block_res_sample = self.model(
            latent.to(self.dtype),
            t,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=self.condition_image,
            conditioning_scale=conditioning_scale,
            guess_mode=guess_mode,
            return_dict=False,
        )
        return down_block_res_samples, mid_block_res_sample


if __name__ == '__main__':
    condition_image_p = 'resources/for_sd/controlnet/astronaut_canny.png'

    cn = ControlNet(model_name='control_v11p_sd15_inpaint', cond_scale=1, vae_scale_factor=8, height=512, width=512,
                    dtype=torch.float32)

    down_block_res_samples_, mid_block_res_sample_ = cn.forward(torch.rand((2, 4, 64, 64)),
                                                                20,
                                                                condition_image_p,
                                                                torch.rand((2, 77, 768))
                                                                )
    print(down_block_res_samples_[0].shape)
    print(mid_block_res_sample_.shape)
