# Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation (D-SDS)
<a href="https://arxiv.org/abs/2303.15413"><img src="https://img.shields.io/badge/arXiv-2305.15413-B31B1B"></a>

This is the official and practical implementation of the [Debiased Score Distillation Sampling (D-SDS)](https://arxiv.org/abs/2303.15413), based on [ThreeStudio](https://github.com/threestudio-project/threestudio) and [SJC](https://github.com/pals-ttic/sjc).

## 🎭 Why D-SDS?

**Debiased Score Distillation Sampling (D-SDS)** offers a solution when **SDS methods**, such as **DreamFusion, Magic3D, SJC**, etc., don't produce the 3D results you're aiming for. If you've faced issues with artifacts or multiple faces, D-SDS is designed to overcome these challenges through two key mechanisms: Score Debiasing and Prompt Debiasing. For a comprehensive understanding of these processes, we recommend delving into our [paper](https://arxiv.org/abs/2303.15413).

Below are the results with **D-SDS** on **ThreeStudio implementation of DreamFusion**:

| Prompt | SDS (DreamFusion) | Debiased-SDS (Ours) |
|:---------:|:-----------------:|:-------------------:|
| *a colorful toucan with a large beak* | <img src="https://github.com/SusungHong/Debiased-Score-Distillation-Sampling/assets/5498512/a4090873-8401-4601-b5a9-2f931637a669" width="400"/> | <img src="https://github.com/SusungHong/Debiased-Score-Distillation-Sampling/assets/5498512/e3f9b673-10f6-4844-a22e-f07e049393e1" width="400"/> |
| *a kangaroo wearing boxing gloves* | <img src="https://github.com/SusungHong/Debiased-Score-Distillation-Sampling/assets/5498512/067b6980-8e0c-45b0-8951-9816c327b012" width="400"/> | <img src="https://github.com/SusungHong/Debiased-Score-Distillation-Sampling/assets/5498512/138c03ad-b648-4b36-aa12-f8a29ffdfe7a" width="400"/> |

Below are the results with **D-SDS** on **SJC**:

| SDS (SJC) | Debiased-SDS (Ours) |
|:-----------------:|:-------------------:|
| <img src="https://github.com/SusungHong/Debiased-Score-Distillation-Sampling/assets/5498512/3b1ed9b6-eac4-46cf-9934-bccbf23fb746" width="400"/> | <img src="https://github.com/SusungHong/Debiased-Score-Distillation-Sampling/assets/5498512/caf96dce-44d9-42f1-9815-fe9e91ffd2f8" width="400"/> |

## 🐧 How to Use D-SDS

An awesome project, [ThreeStudio](https://github.com/threestudio-project/threestudio), has already integrated our method in its main branch. To utilize it, you can adjust the setting `system.guidance.grad_clip=[0,0.5,2.0,10000]` while using DeepFloyd-IF for guidance. An example command for this can be found in `run_threestudio_debiasing.sh`:
```
# Sampling with score debiasing
python launch.py --config configs/dreamfusion-if.yaml --train --gpu 0 system.prompt_processor.prompt="a colorful toucan with a large beak" system.guidance.grad_clip=[0,0.5,2.0,10000]
```

For [SJC](https://github.com/pals-ttic/sjc), run the example command ``run_sjc_debiasing.sh`:
```
# Sampling with score debiasing
python run_sjc.py \
--sd.prompt "a colorful toucan with a large beak" \
--n_steps 10000 \
--lr 0.05 \
--sd.scale 100.0 \
--score_thres 8.0 \
--score_dynamic True \
--emptiness_weight 10000 \
--emptiness_step 0.5 \
--emptiness_multiplier 20.0 \
--depth_weight 0
```

We're currently undertaking code refactoring for prompt debiasing (employing BERT-like LMs to identify contradictions) and will be releasing this updated codebase soon!

## 🔥 Work in Progress
- [x] Refactoring Score Debiasing
- [ ] Refactoring Prompt Debiasing for ThreeStudio
- [ ] Refactoring Prompt Debiasing for SJC

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
