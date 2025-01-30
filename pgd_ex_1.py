import torch
import numpy as np
import os
import sys
import random
import argparse
import json
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.utils import save_image
import torch.nn.functional as F
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim_adv import DDIMSampler
from omegaconf import OmegaConf
from torch.backends import cudnn


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model():
    config = OmegaConf.load("configs/latent-diffusion/cin256-v2.yaml")
    model = load_model_from_config(config, "models/ldm/cin256-v2/model.ckpt")
    return model


parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=6)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--scale', type=float, default=3.0)
parser.add_argument('--ddim-steps', type=int, default=200)
parser.add_argument('--ddim-eta', type=float, default=0.0)
parser.add_argument('--K', type=int, default=5)
parser.add_argument('--s', type=float, default=1.0)
parser.add_argument('--a', type=float, default=0.5)
parser.add_argument('--save-dir', type=str, default='advdiff_results/')
args = parser.parse_args()

goose_class = 99
cat_class = 281


def main():
    os.makedirs(args.save_dir, exist_ok=True)

    model = get_model()
    vic_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(model.device)
    vic_model.eval()
    sampler = DDIMSampler(model, vic_model=vic_model)

    with model.ema_scope():
        uc = model.get_learned_conditioning(
            {model.cond_stage_key: torch.tensor(args.batch_size * [1000]).to(model.device)}
        )
        xc = torch.tensor(args.batch_size * [cat_class])
        c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})

        samples_ddim, _ = sampler.sample(
            S=args.ddim_steps,
            conditioning=c,
            batch_size=args.batch_size,
            shape=[3, 64, 64],
            verbose=False,
            unconditional_guidance_scale=args.scale,
            unconditional_conditioning=uc,
            eta=args.ddim_eta,
            label=xc.to(model.device),
            K=args.K, s=args.s, a=args.a)

        with open("classes.json", "r", encoding="utf-8") as file:
            classes_json_dict = json.load(file)
        adv_dec = model.decode_first_stage(samples_ddim)
        adv_dec = torch.clamp((adv_dec + 1.0) / 2.0, min=0.0, max=1.0)
        adv_dec.requires_grad = True
        optimizer = torch.optim.Adam([adv_dec], lr=0.01)
        flag = False
        max_iter = 500
        epsilon = 0.1
        min_epsilon = 0.02  # 0.05
        epsilon_delta = 0.995  # 0.99
        iter = 0
        while not flag and iter < max_iter:
            optimizer.zero_grad()
            pred = vic_model(adv_dec)
            pred_label = torch.argmax(pred, dim=1)
            print(f"Iteration {iter}, Predicted Labels: {pred_label.tolist()}")
            if (pred_label == goose_class).all():
                flag = True
                print(
                    f"Success. Model predicts {pred_label.tolist()} as {classes_json_dict.get(str(pred_label.tolist()[0]))[1]}.")
                break
            cos_sim = F.cosine_similarity(pred,
                                          torch.nn.functional.one_hot(torch.tensor([goose_class] * args.batch_size),
                                                                      num_classes=1000).float().to(model.device), dim=1)
            loss = -F.cross_entropy(pred, torch.tensor([goose_class] * args.batch_size).to(
                model.device)) - 0.1 * cos_sim.mean()
            print(f"Iteration {iter}, Loss: {loss.item()}")
            grad_list = torch.autograd.grad(loss, adv_dec, retain_graph=True, create_graph=False)
            grad = grad_list[0]
            grad = grad / grad.norm(p=2)
            adv_dec = adv_dec + epsilon * grad
            adv_dec = torch.clamp(adv_dec, 0, 1)
            epsilon = max(min_epsilon, epsilon * epsilon_delta)
            iter += 1
        if not flag:
            print("Failure. Model predicts labels", pred_label.tolist())
            save_image(adv_dec, os.path.join(args.save_dir, 'failed_attack.png'))
        else:
            save_image(adv_dec, os.path.join(args.save_dir, 'success_attack.png'))
        print("Adversarial example saved at", args.save_dir)


if __name__ == '__main__':
    main()