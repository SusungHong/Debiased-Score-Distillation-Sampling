import math
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from imageio import imwrite
from pydantic import validator

import sys
sys.path.insert(0, 'sjc/')

from my.utils import (
    tqdm, EventStorage, HeartBeat, EarlyLoopBreak,
    get_event_storage, get_heartbeat, read_stats
)
from my.config import BaseConf, dispatch, optional_load_config
from my.utils.seed import seed_everything

from adapt import ScoreAdapter, karras_t_schedule
from run_img_sampling import GDDPM, SD, StableDiffusion
from misc import torch_samps_to_imgs
from pose import PoseConfig

from run_nerf import VoxConfig
from voxnerf.utils import every
from voxnerf.render import (
    as_torch_tsrs, rays_from_img, ray_box_intersect, render_ray_bundle
)
from voxnerf.vis import stitch_vis, bad_vis as nerf_vis


device_glb = torch.device("cuda")


def tsr_stats(tsr):
    return {
        "mean": tsr.mean().item(),
        "std": tsr.std().item(),
        "max": tsr.max().item(),
    }


class SJC(BaseConf):
    family:     str = "sd"
    gddpm:      GDDPM = GDDPM()
    sd:         SD = SD(
        variant="v1",
        prompt="A high quality photo of a delicious burger",
        scale=100.0
    )
    lr:         float = 0.05
    n_steps:    int = 10000
    vox:        VoxConfig = VoxConfig(
        model_type="V_SD", grid_size=100, density_shift=-1.0, c=3,
        blend_bg_texture=True, bg_texture_hw=4,
        bbox_len=1.0
    )
    pose:       PoseConfig = PoseConfig(rend_hw=64, FoV=60.0, R=1.5)

    score_thres:        float = 8.0
    score_dynamic:      bool = True

    emptiness_scale:    int = 10
    emptiness_weight:   int = 1e4
    emptiness_step:     float = 0.5
    emptiness_multiplier: float = 20.0

    depth_weight:       int = 0

    var_red:     bool = True

    @validator("vox")
    def check_vox(cls, vox_cfg, values):
        family = values['family']
        if family == "sd":
            vox_cfg.c = 4
        return vox_cfg

    def run(self):
        cfgs = self.dict()

        family = cfgs.pop("family")
        model = getattr(self, family).make()

        cfgs.pop("vox")
        vox = self.vox.make()

        cfgs.pop("pose")
        poser = self.pose.make()

        sjc_3d(**cfgs, poser=poser, model=model, vox=vox)


def sjc_3d(
    poser, vox, model: ScoreAdapter,
    lr, n_steps, score_thres, score_dynamic,
    emptiness_scale, emptiness_weight, emptiness_step, emptiness_multiplier,
    depth_weight, var_red, **kwargs
):
    del kwargs

    assert model.samps_centered()
    _, target_H, target_W = model.data_shape()
    bs = 1
    aabb = vox.aabb.T.cpu().numpy()
    vox = vox.to(device_glb)
    opt = torch.optim.Adamax(vox.opt_params(), lr=lr)

    H, W = poser.H, poser.W
    Ks, poses, prompt_prefixes = poser.sample_train(n_steps)

    ts = model.us[30:-10]
    fuse = EarlyLoopBreak(5)

    same_noise = torch.randn(1, 4, H, W, device=model.device).repeat(bs, 1, 1, 1)

    with tqdm(total=n_steps) as pbar, \
        HeartBeat(pbar) as hbeat, \
            EventStorage() as metric:
        for i in range(n_steps):
            if fuse.on_break():
                break

            p = f"{prompt_prefixes[i]} {model.prompt}"
            score_conds = model.prompts_emb([p])

            y, depth, ws = render_one_view(vox, aabb, H, W, Ks[i], poses[i], return_w=True)

            if isinstance(model, StableDiffusion):
                pass
            else:
                y = torch.nn.functional.interpolate(y, (target_H, target_W), mode='bilinear')

            opt.zero_grad()

            with torch.no_grad():
                chosen_σs = np.random.choice(ts, bs, replace=False)
                chosen_σs = chosen_σs.reshape(-1, 1, 1, 1)
                chosen_σs = torch.as_tensor(chosen_σs, device=model.device, dtype=torch.float32)
                # chosen_σs = us[i]

                noise = torch.randn(bs, *y.shape[1:], device=model.device)

                zs = y + chosen_σs * noise
                Ds = model.denoise(zs, chosen_σs, **score_conds)

                if var_red:
                    grad = (Ds - y) / chosen_σs
                else:
                    grad = (Ds - zs) / chosen_σs

                grad = grad.mean(0, keepdim=True)
                
                if score_thres > 0:
                    dynamic_grad = score_thres
                    if score_dynamic:
                        dynamic_grad = score_thres * (0.25 + 0.75 * (i / n_steps))
                    grad = torch.clamp(grad, min=-dynamic_grad, max=dynamic_grad)

            y.backward(-grad, retain_graph=True)

            if depth_weight > 0:
                center_depth = depth[7:-7, 7:-7]
                border_depth_mean = (depth.sum() - center_depth.sum()) / (64*64-50*50)
                center_depth_mean = center_depth.mean()
                depth_diff = center_depth_mean - border_depth_mean
                depth_loss = - torch.log(depth_diff + 1e-12)
                depth_loss = depth_weight * depth_loss
                depth_loss.backward(retain_graph=True)

            emptiness_loss = torch.log(1 + emptiness_scale * ws).mean()
            emptiness_loss = emptiness_weight * emptiness_loss
            if emptiness_step * n_steps <= i:
                emptiness_loss *= emptiness_multiplier
            emptiness_loss.backward()

            opt.step()

            metric.put_scalars(**tsr_stats(y))

            if every(pbar, percent=1):
                with torch.no_grad():
                    if isinstance(model, StableDiffusion):
                        y = model.decode(y)
                    vis_routine(metric, y, depth)

            # if every(pbar, step=2500):
            #     metric.put_artifact(
            #         "ckpt", ".pt", lambda fn: torch.save(vox.state_dict(), fn)
            #     )
            #     with EventStorage("test"):
            #         evaluate(model, vox, poser)

            metric.step()
            pbar.update()
            pbar.set_description(p)
            hbeat.beat()

        metric.put_artifact(
            "ckpt", ".pt", lambda fn: torch.save(vox.state_dict(), fn)
        )
        with EventStorage("test"):
            evaluate(model, vox, poser)

        metric.step()

        hbeat.done()


@torch.no_grad()
def evaluate(score_model, vox, poser):
    H, W = poser.H, poser.W
    vox.eval()
    K, poses = poser.sample_test(100)

    fuse = EarlyLoopBreak(5)
    metric = get_event_storage()
    hbeat = get_heartbeat()

    aabb = vox.aabb.T.cpu().numpy()
    vox = vox.to(device_glb)

    num_imgs = len(poses)

    for i in (pbar := tqdm(range(num_imgs))):
        if fuse.on_break():
            break

        pose = poses[i]
        y, depth = render_one_view(vox, aabb, H, W, K, pose)
        if isinstance(score_model, StableDiffusion):
            y = score_model.decode(y)
        vis_routine(metric, y, depth)

        metric.step()
        hbeat.beat()

    metric.flush_history()

    metric.put_artifact(
        "view_seq", ".mp4",
        lambda fn: stitch_vis(fn, read_stats(metric.output_dir, "view")[1])
    )

    metric.step()


def render_one_view(vox, aabb, H, W, K, pose, return_w=False):
    N = H * W
    ro, rd = rays_from_img(H, W, K, pose)
    ro, rd, t_min, t_max = scene_box_filter(ro, rd, aabb)
    assert len(ro) == N, "for now all pixels must be in"
    ro, rd, t_min, t_max = as_torch_tsrs(vox.device, ro, rd, t_min, t_max)
    rgbs, depth, weights = render_ray_bundle(vox, ro, rd, t_min, t_max)

    rgbs = rearrange(rgbs, "(h w) c -> 1 c h w", h=H, w=W)
    depth = rearrange(depth, "(h w) 1 -> h w", h=H, w=W)
    if return_w:
        return rgbs, depth, weights
    else:
        return rgbs, depth


def scene_box_filter(ro, rd, aabb):
    _, t_min, t_max = ray_box_intersect(ro, rd, aabb)
    # do not render what's behind the ray origin
    t_min, t_max = np.maximum(t_min, 0), np.maximum(t_max, 0)
    return ro, rd, t_min, t_max


def vis_routine(metric, y, depth):
    pane = nerf_vis(y, depth, final_H=256)
    im = torch_samps_to_imgs(y)[0]
    depth = depth.cpu().numpy()
    metric.put_artifact("view", ".png", lambda fn: imwrite(fn, pane))
    metric.put_artifact("img", ".png", lambda fn: imwrite(fn, im))
    metric.put_artifact("depth", ".npy", lambda fn: np.save(fn, depth))


def evaluate_ckpt():
    cfg = optional_load_config(fname="full_config.yml")
    assert len(cfg) > 0, "can't find cfg file"
    mod = SJC(**cfg)

    family = cfg.pop("family")
    model: ScoreAdapter = getattr(mod, family).make()
    vox = mod.vox.make()
    poser = mod.pose.make()

    pbar = tqdm(range(1))

    with EventStorage(), HeartBeat(pbar):
        ckpt_fname = latest_ckpt()
        state = torch.load(ckpt_fname, map_location="cpu")
        vox.load_state_dict(state)
        vox.to(device_glb)

        with EventStorage("test"):
            evaluate(model, vox, poser)


def latest_ckpt():
    ts, ys = read_stats("./", "ckpt")
    assert len(ys) > 0
    return ys[-1]


if __name__ == "__main__":
    seed_everything(0)
    dispatch(SJC)
    # evaluate_ckpt()

'''
Copyright (c) 2022 Score Jacobian Chaining authors

CreativeML Open RAIL-M
dated August 22, 2022

Section I: PREAMBLE

Multimodal generative models are being widely adopted and used, and have the potential to transform the way artists, among other individuals, conceive and benefit from AI or ML technologies as a tool for content creation.

Notwithstanding the current and potential benefits that these artifacts can bring to society at large, there are also concerns about potential misuses of them, either due to their technical limitations or ethical considerations.

In short, this license strives for both the open and responsible downstream use of the accompanying model. When it comes to the open character, we took inspiration from open source permissive licenses regarding the grant of IP rights. Referring to the downstream responsible use, we added use-based restrictions not permitting the use of the Model in very specific scenarios, in order for the licensor to be able to enforce the license in case potential misuses of the Model may occur. At the same time, we strive to promote open and responsible research on generative models for art and content generation.

Even though downstream derivative versions of the model could be released under different licensing terms, the latter will always have to include - at minimum - the same use-based restrictions as the ones in the original license (this license). We believe in the intersection between open and responsible AI development; thus, this License aims to strike a balance between both in order to enable responsible open-science in the field of AI.

This License governs the use of the model (and its derivatives) and is informed by the model card associated with the model.

NOW THEREFORE, You and Licensor agree as follows:

1. Definitions

- "License" means the terms and conditions for use, reproduction, and Distribution as defined in this document.
- "Data" means a collection of information and/or content extracted from the dataset used with the Model, including to train, pretrain, or otherwise evaluate the Model. The Data is not licensed under this License.
- "Output" means the results of operating a Model as embodied in informational content resulting therefrom.
- "Model" means any accompanying machine-learning based assemblies (including checkpoints), consisting of learnt weights, parameters (including optimizer states), corresponding to the model architecture as embodied in the Complementary Material, that have been trained or tuned, in whole or in part on the Data, using the Complementary Material.
- "Derivatives of the Model" means all modifications to the Model, works based on the Model, or any other model which is created or initialized by transfer of patterns of the weights, parameters, activations or output of the Model, to the other model, in order to cause the other model to perform similarly to the Model, including - but not limited to - distillation methods entailing the use of intermediate data representations or methods based on the generation of synthetic data by the Model for training the other model.
- "Complementary Material" means the accompanying source code and scripts used to define, run, load, benchmark or evaluate the Model, and used to prepare data for training or evaluation, if any. This includes any accompanying documentation, tutorials, examples, etc, if any.
- "Distribution" means any transmission, reproduction, publication or other sharing of the Model or Derivatives of the Model to a third party, including providing the Model as a hosted service made available by electronic or other remote means - e.g. API-based or web access.
- "Licensor" means the copyright owner or entity authorized by the copyright owner that is granting the License, including the persons or entities that may have rights in the Model and/or distributing the Model.
- "You" (or "Your") means an individual or Legal Entity exercising permissions granted by this License and/or making use of the Model for whichever purpose and in any field of use, including usage of the Model in an end-use application - e.g. chatbot, translator, image generator.
- "Third Parties" means individuals or legal entities that are not under common control with Licensor or You.
- "Contribution" means any work of authorship, including the original version of the Model and any modifications or additions to that Model or Derivatives of the Model thereof, that is intentionally submitted to Licensor for inclusion in the Model by the copyright owner or by an individual or Legal Entity authorized to submit on behalf of the copyright owner. For the purposes of this definition, "submitted" means any form of electronic, verbal, or written communication sent to the Licensor or its representatives, including but not limited to communication on electronic mailing lists, source code control systems, and issue tracking systems that are managed by, or on behalf of, the Licensor for the purpose of discussing and improving the Model, but excluding communication that is conspicuously marked or otherwise designated in writing by the copyright owner as "Not a Contribution."
- "Contributor" means Licensor and any individual or Legal Entity on behalf of whom a Contribution has been received by Licensor and subsequently incorporated within the Model.

Section II: INTELLECTUAL PROPERTY RIGHTS

Both copyright and patent grants apply to the Model, Derivatives of the Model and Complementary Material. The Model and Derivatives of the Model are subject to additional terms as described in Section III.

2. Grant of Copyright License. Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license to reproduce, prepare, publicly display, publicly perform, sublicense, and distribute the Complementary Material, the Model, and Derivatives of the Model.
3. Grant of Patent License. Subject to the terms and conditions of this License and where and as applicable, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable (except as stated in this paragraph) patent license to make, have made, use, offer to sell, sell, import, and otherwise transfer the Model and the Complementary Material, where such license applies only to those patent claims licensable by such Contributor that are necessarily infringed by their Contribution(s) alone or by combination of their Contribution(s) with the Model to which such Contribution(s) was submitted. If You institute patent litigation against any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the Model and/or Complementary Material or a Contribution incorporated within the Model and/or Complementary Material constitutes direct or contributory patent infringement, then any patent licenses granted to You under this License for the Model and/or Work shall terminate as of the date such litigation is asserted or filed.

Section III: CONDITIONS OF USAGE, DISTRIBUTION AND REDISTRIBUTION

4. Distribution and Redistribution. You may host for Third Party remote access purposes (e.g. software-as-a-service), reproduce and distribute copies of the Model or Derivatives of the Model thereof in any medium, with or without modifications, provided that You meet the following conditions:
Use-based restrictions as referenced in paragraph 5 MUST be included as an enforceable provision by You in any type of legal agreement (e.g. a license) governing the use and/or distribution of the Model or Derivatives of the Model, and You shall give notice to subsequent users You Distribute to, that the Model or Derivatives of the Model are subject to paragraph 5. This provision does not apply to the use of Complementary Material.
You must give any Third Party recipients of the Model or Derivatives of the Model a copy of this License;
You must cause any modified files to carry prominent notices stating that You changed the files;
You must retain all copyright, patent, trademark, and attribution notices excluding those notices that do not pertain to any part of the Model, Derivatives of the Model.
You may add Your own copyright statement to Your modifications and may provide additional or different license terms and conditions - respecting paragraph 4.a. - for use, reproduction, or Distribution of Your modifications, or for any such Derivatives of the Model as a whole, provided Your use, reproduction, and Distribution of the Model otherwise complies with the conditions stated in this License.
5. Use-based restrictions. The restrictions set forth in Attachment A are considered Use-based restrictions. Therefore You cannot use the Model and the Derivatives of the Model for the specified restricted uses. You may use the Model subject to this License, including only for lawful purposes and in accordance with the License. Use may include creating any content with, finetuning, updating, running, training, evaluating and/or reparametrizing the Model. You shall require all of Your users who use the Model or a Derivative of the Model to comply with the terms of this paragraph (paragraph 5).
6. The Output You Generate. Except as set forth herein, Licensor claims no rights in the Output You generate using the Model. You are accountable for the Output you generate and its subsequent uses. No use of the output can contravene any provision as stated in the License.

Section IV: OTHER PROVISIONS

7. Updates and Runtime Restrictions. To the maximum extent permitted by law, Licensor reserves the right to restrict (remotely or otherwise) usage of the Model in violation of this License, update the Model through electronic means, or modify the Output of the Model based on updates. You shall undertake reasonable efforts to use the latest version of the Model.
8. Trademarks and related. Nothing in this License permits You to make use of Licensors’ trademarks, trade names, logos or to otherwise suggest endorsement or misrepresent the relationship between the parties; and any rights not expressly granted herein are reserved by the Licensors.
9. Disclaimer of Warranty. Unless required by applicable law or agreed to in writing, Licensor provides the Model and the Complementary Material (and each Contributor provides its Contributions) on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, including, without limitation, any warranties or conditions of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE. You are solely responsible for determining the appropriateness of using or redistributing the Model, Derivatives of the Model, and the Complementary Material and assume any risks associated with Your exercise of permissions under this License.
10. Limitation of Liability. In no event and under no legal theory, whether in tort (including negligence), contract, or otherwise, unless required by applicable law (such as deliberate and grossly negligent acts) or agreed to in writing, shall any Contributor be liable to You for damages, including any direct, indirect, special, incidental, or consequential damages of any character arising as a result of this License or out of the use or inability to use the Model and the Complementary Material (including but not limited to damages for loss of goodwill, work stoppage, computer failure or malfunction, or any and all other commercial damages or losses), even if such Contributor has been advised of the possibility of such damages.
11. Accepting Warranty or Additional Liability. While redistributing the Model, Derivatives of the Model and the Complementary Material thereof, You may choose to offer, and charge a fee for, acceptance of support, warranty, indemnity, or other liability obligations and/or rights consistent with this License. However, in accepting such obligations, You may act only on Your own behalf and on Your sole responsibility, not on behalf of any other Contributor, and only if You agree to indemnify, defend, and hold each Contributor harmless for any liability incurred by, or claims asserted against, such Contributor by reason of your accepting any such warranty or additional liability.
12. If any provision of this License is held to be invalid, illegal or unenforceable, the remaining provisions shall be unaffected thereby and remain valid as if such provision had not been set forth herein.

END OF TERMS AND CONDITIONS




Attachment A

Use Restrictions

You agree not to use the Model or Derivatives of the Model:
- In any way that violates any applicable national, federal, state, local or international law or regulation;
- For the purpose of exploiting, harming or attempting to exploit or harm minors in any way;
- To generate or disseminate verifiably false information and/or content with the purpose of harming others;
- To generate or disseminate personal identifiable information that can be used to harm an individual;
- To defame, disparage or otherwise harass others;
- For fully automated decision making that adversely impacts an individual’s legal rights or otherwise creates or modifies a binding, enforceable obligation;
- For any use intended to or which has the effect of discriminating against or harming individuals or groups based on online or offline social behavior or known or predicted personal or personality characteristics;
- To exploit any of the vulnerabilities of a specific group of persons based on their age, social, physical or mental characteristics, in order to materially distort the behavior of a person pertaining to that group in a manner that causes or is likely to cause that person or another person physical or psychological harm;
- For any use intended to or which has the effect of discriminating against individuals or groups based on legally protected characteristics or categories;
- To provide medical advice and medical results interpretation;
- To generate or disseminate information for the purpose to be used for administration of justice, law enforcement, immigration or asylum processes, such as predicting an individual will commit fraud/crime commitment (e.g. by text profiling, drawing causal relationships between assertions made in documents, indiscriminate and arbitrarily-targeted use).
'''
