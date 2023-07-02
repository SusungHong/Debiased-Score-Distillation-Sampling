# Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation (D-SDS)
<a href="https://arxiv.org/abs/2303.15413"><img src="https://img.shields.io/badge/arXiv-2305.15413-B31B1B"></a>
<a href="https://susunghong.github.io/Debiased-Score-Distillation-Sampling"><img src="https://img.shields.io/badge/Project%20Page-online-brightgreen"></a>

## 🎭 Why D-SDS?

**Debiased Score Distillation Sampling (D-SDS)** offers a solution when **SDS methods**, such as **DreamFusion, Magic3D, SJC**, etc., don't produce the 3D results you're aiming for. If you've faced issues with artifacts or multiple faces, D-SDS is designed to overcome these challenges through two key mechanisms: Score Debiasing and Prompt Debiasing. For a comprehensive understanding of these processes, we recommend delving into our [paper](https://arxiv.org/abs/2303.15413).

Below are the results with **D-SDS** on **ThreeStudio implementation of DreamFusion**:

| Prompt | SDS (DreamFusion) | Debiased-SDS (Ours) |
|:---------:|:-----------------:|:-------------------:|
| *a colorful toucan with a large beak* | <img src="https://github.com/SusungHong/Debiased-Score-Distillation-Sampling/blob/gh-pages/gif_lowres/toucan_DreamFusion.gif" width="400"/> | <img src="https://github.com/SusungHong/Debiased-Score-Distillation-Sampling/blob/gh-pages/gif_lowres/toucan_ours.gif" width="400"/> |
| *a kangaroo wearing boxing gloves* | <img src="https://github.com/SusungHong/Debiased-Score-Distillation-Sampling/blob/gh-pages/gif_lowres/kangaroo_DreamFusion.gif" width="400"/> | <img src="https://github.com/SusungHong/Debiased-Score-Distillation-Sampling/blob/gh-pages/gif_lowres/kangaroo_ours.gif" width="400"/> |

Below are the results with **D-SDS** on **SJC**:

| SDS (SJC) | Debiased-SDS (Ours) |
|:-----------------:|:-------------------:|
| <img src="https://github.com/SusungHong/Debiased-Score-Distillation-Sampling/blob/gh-pages/gif_lowres/cat_etc_sjc.gif" width="400"/> | <img src="https://github.com/SusungHong/Debiased-Score-Distillation-Sampling/blob/gh-pages/gif_lowres/cat_etc_ours.gif" width="400"/> |

## 🐧 How to Use D-SDS

First, follow the installation instructions for your desired codebase. For example, this could be [ThreeStudio](https://github.com/threestudio-project/threestudio) or [SJC](https://github.com/pals-ttic/sjc).

An amazing project, [ThreeStudio](https://github.com/threestudio-project/threestudio), has already integrated our method in its main branch. To utilize our method:
```
# Sampling with score debiasing
python launch.py --config configs/dreamfusion-if.yaml --train --gpu 0 system.prompt_processor.prompt="a colorful toucan with a large beak" system.guidance.grad_clip=[0,0.5,2.0,10000]

# Sampling with prompt debiasing
python launch.py --config configs/dreamfusion-if.yaml --train --gpu 0 system.prompt_processor.prompt="a colorful toucan with a large beak" system.prompt_processor.use_prompt_debiasing=true prompt_debiasing_mask_ids=[2]

# Sampling with score & prompt debiasing
python launch.py --config configs/dreamfusion-if.yaml --train --gpu 0 system.prompt_processor.prompt="a colorful toucan with a large beak" system.guidance.grad_clip=[0,0.5,2.0,10000] system.prompt_processor.use_prompt_debiasing=true prompt_debiasing_mask_ids=[2]
```

You can also find the manual at [Tips on Improving Quality](https://github.com/threestudio-project/threestudio).

## Acknowledgements

We greatly appreciate the contributions of the great public projects, [SJC](https://github.com/pals-ttic/sjc) and [ThreeStudio](https://github.com/threestudio-project/threestudio).

## Cite as
```
@article{hong2023debiasing,
  title={Debiasing scores and prompts of 2d diffusion for robust text-to-3d generation},
  author={Hong, Susung and Ahn, Donghoon and Kim, Seungryong},
  journal={arXiv preprint arXiv:2303.15413},
  year={2023}
}
```
