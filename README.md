# Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation (D-SDS)
<a href="https://arxiv.org/abs/2303.15413"><img src="https://img.shields.io/badge/arXiv-2305.15413-B31B1B"></a>

Welcome to the official and practical implementation of the [Debiasing Score Distillation Sampling (D-SDS)](https://arxiv.org/abs/2303.15413).

## 🎭 Why Choose D-SDS?

D-SDS offers a solution when SDS methods, such as DreamFusion, Magic3D, SJC, etc., don't produce the 3D results you're aiming for. If you've faced issues with artifacts or multiple faces, D-SDS is designed to overcome these challenges through two key mechanisms: Score Debiasing and Prompt Debiasing. For a comprehensive understanding of these processes, we recommend delving into our [paper](https://arxiv.org/abs/2303.15413).

Below are the results with D-SDS on ThreeStudio implementation of DreamFusion:

|           | SDS (DreamFusion) | Debiased-SDS (Ours, Score) |
|:---------:|:-----------------:|:-------------------:|
| a colorful toucan with a large beak | <img src="https://github.com/SusungHong/Debiased-Score-Distillation-Sampling/assets/5498512/a4090873-8401-4601-b5a9-2f931637a669" width="400"/> | <img src="https://github.com/SusungHong/Debiased-Score-Distillation-Sampling/assets/5498512/e3f9b673-10f6-4844-a22e-f07e049393e1" width="400"/> |
| kangaroo wearing boxing glove | <img src="https://github.com/SusungHong/Debiased-Score-Distillation-Sampling/assets/5498512/067b6980-8e0c-45b0-8951-9816c327b012" width="400"/> | <img src="https://github.com/SusungHong/Debiased-Score-Distillation-Sampling/assets/5498512/138c03ad-b648-4b36-aa12-f8a29ffdfe7a" width="400"/> |

Below are the results with D-SDS on SJC:

|           | SDS (DreamFusion) | Debiased-SDS (Ours, Score + Prompt) |
|:---------:|:-----------------:|:-------------------:|
| a majestic giraffe with a long neck | <img src="https://github.com/SusungHong/Debiased-Score-Distillation-Sampling/assets/5498512/a2b7404e-0caa-4086-96e4-e7c625d7f2e2" width="150"/> | <img src="https://github.com/SusungHong/Debiased-Score-Distillation-Sampling/assets/5498512/9d0cdd3a-5003-4520-8b4d-e69e09f445fb" width="150"/> |
| a small kitten | <img src="https://github.com/SusungHong/Debiased-Score-Distillation-Sampling/assets/5498512/a128db5d-06a3-45be-91b8-c6e5e2c5ecab" width="150"/> | <img src="https://github.com/SusungHong/Debiased-Score-Distillation-Sampling/assets/5498512/4b4a8118-92b0-4021-a28b-60c56724c664" width="150"/> |

## 🐧 Using D-SDS

Incorporating D-SDS into your work is a straightforward process, particularly with our score-debiasing feature that facilitates static and dynamic clipping of 2D-to-3D scores.

A highly commendable project, [ThreeStudio](https://github.com/threestudio-project/threestudio), has already integrated our method in its main branch. To utilize it, you can adjust the setting `system.guidance.grad_clip=[0,0.5,2.0,10000]` while using DeepFloyd-IF for guidance. For example:
```
# Sample with score debiasing
python launch.py --config configs/dreamfusion-if.yaml --train --gpu 0 system.prompt_processor.prompt="a colorful toucan with a large beak" system.guidance.grad_clip=[0,0.5,2.0,10000]
```

For [SJC](https://github.com/pals-ttic/sjc), simply clone the original repository and adjust the setting `/path/to/sjc/run_sjc.py` to ours.

We're currently undertaking code refactoring for prompt debiasing (employing BERT-like LMs to identify contradictions) and will be releasing this updated codebase soon!

## 🔥 Work in process
- [x] Refactoring Score Debiasing
- [ ] Refactoring Prompt Debiasing

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
