# AnimateDiff

~~I may stop develop this repo right now,~~ 
AnimateDiff is not designed to do I2V mission at first, 
I spent lots of time to read diffusers source codes, 
this route maybe not the best compared to webui(ldm injection) at the end.

Though it has potential, I believe new motion model trained on bigger-datasets/specific-motion will be released soon.

Still under development.



## TODO

- [x] update diffusers to 0.20.1
- [x] support [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)
- [x] reconstruction codes and make animatediff a diffusers plugin like [sd-webui-animatediff](https://github.com/continue-revolution/sd-webui-animatediff)
- [x] controlnet from [TDS4874](https://github.com/TDS4874/sd-webui-controlnet/tree/animate-diff-support)
- [x] solve/locate color degrade problem, check [TDS_ solution](https://note.com/tds_/n/n5aad9ef8a9b1), It seems that any color problems came from DDIM params.
- [x] controlnet reference mode
- [x] controlnet multi module mode
- [x] ddim inversion from [Tune-A-Video](https://github.com/showlab/Tune-A-Video)
- [x] support [AnimateDiff v2](https://github.com/guoyww/AnimateDiff/commit/108921965da631be96cd5b6c6ea0c9bbb06ecf0b)
- [x] support [AnimateDiff MotionLoRA](https://github.com/guoyww/AnimateDiff/tree/f82a8367ec1471711d342febd8cbef72e4670a12#features)
- [x] support [FreeU](https://github.com/ChenyangSi/FreeU)
- [x] keyframe controlnet apply
- [x] controlnet inpainting mode
- [x] support AnimateDiff v3 wo SparseCtrl
- [ ] keyframe prompts apply

## Experience

### Multi Controlnet

inpainting + canny
<table>
    <tr>
    <td>inpainting + canny</td>
    <td><img src="__assets__/astronaut_mars/An_astronaut_is_riding_a_horse_on_Mars_seed-444264997.png"></td>
    <td><img src="__assets__/astronaut_mars/astronaut_mask.png"></td>
    <td><img src="__assets__/results/multi_controlnet/astronaut_inpaint_canny.gif"></td>
    </tr>
    <tr>
    <td>tail + tail</td>
    <td><img src="__assets__/anime_girl_pairs/tds_1st_image.png"></td>
    <td><img src="__assets__/anime_girl_pairs/tds_2nd_image.png"></td>
    <td><img src="__assets__/results/multi_controlnet/anime_girl_pairs_tail.gif"></td>
    </tr>
</table>

### MotionLoRA I2V results:   

Zoom In / Zoom Out
results from [this old branch](https://github.com/ykk648/AnimateDiff-I2V/tree/d8b30bfff0748c0839e4cfc084aaaa2930627637)
<table>
    <tr>
    <td><img src="__assets__/astronaut_mars/An_astronaut_is_riding_a_horse_on_Mars_seed-444264997.png"></td>
    <td><img src="__assets__/results/motion_lora/astronaut_zoom_out_in.gif"></td>
    </tr>
</table>
<p style="margin-left: 2em; margin-top: -1em">

### Ablation experiment (controlnet/ipadapter)

results from [this old branch](https://github.com/ykk648/AnimateDiff/tree/5bdbfeb3e92dee379f9c543930aa591f89a5b04f)

all / without denoise strength / without ipadapter / without controlnet(first frame)

<table>
    <tr>
    <td><img src="__assets__/a_girl_in_the_wind/a_girl_in_the_wind.png"></td>
    <td><img src="__assets__/results/images_with_control/girl_wind.gif"></td>
    </tr>
</table>
<table>
    <tr>
    <td><img src="__assets__/astronaut_mars/An_astronaut_is_riding_a_horse_on_Mars_seed-444264997.png"></td>
    <td><img src="__assets__/results/images_with_control/astronaut_mars.gif"></td>
    </tr>
</table>
<p style="margin-left: 2em; margin-top: -1em">


### Origin SD1.5 I2V attempt

Below is old results from [this ols branch](https://github.com/ykk648/AnimateDiff/tree/bdfd4578f4db6f148d7533f4ddb209c6b4317c39)

<table>
    <tr>
    <td><img src="__assets__/a_girl_in_the_wind/a_girl_in_the_wind.png"></td>
    <td><img src="__assets__/results/ipadapter/girl_wind.gif"></td>
    </tr>
</table>
<table>
    <tr>
    <td><img src="__assets__/astronaut_mars/An_astronaut_is_riding_a_horse_on_Mars_seed-444264997.png"></td>
    <td><img src="__assets__/results/ipadapter/astronaut_mars.gif"></td>
    </tr>
</table>
<p style="margin-left: 2em; margin-top: -1em">

First image from [pikalabs](https://twitter.com/pika_labs/status/1678892871670464513), second was generated from sd1.5

First used IPAdapter+init-image-denoise, second used only IPAdapter

## ~~Training~~

- 23.8.22: 
Drop local training scripts, using authors repo to do training experiences(I2V).
First, make image injection refer [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter).
Already test in [AI_power](https://github.com/ykk648/AI_power/blob/main/sd_lib/clip_encoder.py).

## First I2V attempt

- 23.8.8: Here are some results of mine, ref [talesofai's folk](https://github.com/talesofai/AnimateDiff/blob/04b2715b39d4a02334b08cb6ee3dfe79f0a6cd7c/animatediff/pipelines/pipeline_animation.py#L288) and [diffusers](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/deepfloyd_if/pipeline_if_img2img.py) to do image latent injection.

<table>
    <tr>
    <td><img src="__assets__/results/animations/model_07/init.jpg"></td>
    <td><img src="__assets__/results/animations/model_07/0802_v14.gif"></td>
    <td><img src="__assets__/results/animations/model_07/0802_v15.gif"></td>
    </tr>
</table>
<p style="margin-left: 2em; margin-top: -1em">
Character Model：<a href="https://civitai.com/models/13237/genshen-impact-yoimiya">Yoimiya</a> (with an initial reference image.)

<table>
    <tr>
    <td><img src="__assets__/results/animations/model_11/miko_init.png"></td>
    <td><img src="__assets__/results/animations/model_11/0803_v14.gif"></td>
    <td><img src="__assets__/results/animations/model_11/0803_v15.gif"></td>
    </tr>
</table>
Character Model：<a href="https://civitai.com/models/8484?modelVersionId=11523">Yae Miko</a> (with an initial reference image.)

<table>
    <tr>
    <td><img src="__assets__/results/animations/model_12/init_image.jpg"></td>
    <td><img src="__assets__/results/animations/model_12/0804_v14.gif"></td>
    <td><img src="__assets__/results/animations/model_12/0804_v15.gif"></td>
    </tr>
</table>
without Character Model, frame 20

- 23.8.9 test [sd-webui-text2video](https://github.com/kabachuha/sd-webui-text2video) noise-add policy, got bad results

## Original README

check [README.md](https://github.com/guoyww/AnimateDiff/blob/main/README.md)