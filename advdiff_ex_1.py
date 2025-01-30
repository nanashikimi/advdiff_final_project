import torch
import numpy as np
import os
import argparse
import json
import matplotlib.pyplot as plt
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.utils import save_image
import torch.nn.functional as F
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim_adv import DDIMSampler
from omegaconf import OmegaConf

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

def project_perturbation(perturbation, epsilon=0.1, p=2):
    if p == 2:
        norm = torch.norm(perturbation.view(perturbation.shape[0], -1), p=2, dim=1, keepdim=True)
        norm = norm.view(perturbation.shape[0], *([1] * (perturbation.dim() - 1)))
        perturbation = perturbation * torch.min(torch.tensor(1.0, device=perturbation.device), epsilon / (norm + 1e-8))
    elif p == np.inf:
        perturbation = torch.clamp(perturbation, -epsilon, epsilon)
    return perturbation

def adversarial_guidance(pred, target_label, s):
    target_one_hot = F.one_hot(torch.tensor([target_label] * pred.shape[0]), num_classes=1000).float().to(pred.device)
    guid = torch.autograd.grad(F.cross_entropy(pred, torch.tensor([target_label] * pred.shape[0]).to(pred.device)), pred, retain_graph=True)[0]
    return s * guid

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=6)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--scale', type=float, default=3.0)
parser.add_argument('--ddim-steps', type=int, default=100)
parser.add_argument('--ddim-eta', type=float, default=0.0)
parser.add_argument('--K', type=int, default=5)
parser.add_argument('--s', type=float, default=1.0)
parser.add_argument('--a', type=float, default=0.5)
parser.add_argument('--save-dir', type=str, default='advdiff_results/')
args = parser.parse_args()

goose_class = 99
cat_class = 281
max_iter = 500
noise_scale = 0.3
epsilon = 0.1

def main(): #predict at least one as "goose"
    os.makedirs(args.save_dir, exist_ok=True)

    model = get_model()
    vic_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(model.device)
    vic_model.eval()
    sampler = DDIMSampler(model, vic_model=vic_model)

    with model.ema_scope():
        xc = torch.tensor(args.batch_size * [cat_class]).to(model.device)
        c = model.get_learned_conditioning({model.cond_stage_key: xc})
        xc_goose = torch.tensor(args.batch_size * [goose_class]).to(model.device)
        c_target = model.get_learned_conditioning({model.cond_stage_key: xc_goose})
        c_adv = c.clone().detach().requires_grad_(True)
        optimizer = torch.optim.AdamW([c_adv], lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
        loss_spisok = []
        flag = False
        iter = 0
        while not flag and iter < max_iter:
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            samples_ddim, _ = sampler.sample(
                S=args.ddim_steps,
                conditioning=c_adv,
                batch_size=args.batch_size,
                shape=[3, 64, 64],
                verbose=False,
                unconditional_guidance_scale=args.scale,
                unconditional_conditioning=c,
                eta=args.ddim_eta,
                label=xc,
                K=args.K, s=args.s, a=args.a)

            adv_dec = model.decode_first_stage(samples_ddim)
            adv_dec = torch.clamp((adv_dec + 1.0) / 2.0, min=0.0, max=1.0)
            pred = vic_model(adv_dec)
            pred_label = torch.argmax(pred, dim=1)
            print(f"Iteration {iter}, Predicted Labels: {pred_label.tolist()}")
            if (goose_class in pred_label.tolist()):
                flag = True
                print(f"Success! Model predicts all as {goose_class}")
                break
            guid = adversarial_guidance(pred, goose_class, args.s)
            target_tensor = torch.tensor([goose_class] * pred.shape[0]).to(model.device)
            loss = -F.cross_entropy(pred, target_tensor) - 0.1 * guid.mean()
            print(f"Iteration {iter}, Loss: {loss.item()}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(c_adv, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                noise = noise_scale * torch.randn_like(c_adv)
                c_adv += project_perturbation(c_adv - c + noise, epsilon=epsilon)
                c_adv.clamp_(-1.0, 1.0)
            loss_spisok.append(loss.item())
            if iter % 10 == 0:
                plt.plot(loss_spisok)
                plt.xlabel("Iteration")
                plt.ylabel("Loss")
                plt.title("Loss Curve")
                plt.savefig(os.path.join(args.save_dir, "loss_curve.png"))
            iter += 1
        save_path = os.path.join(args.save_dir, 'success_attack.png' if flag else 'failed_attack.png')
        save_image(adv_dec, save_path)
        print(f"Adversarial example saved at {save_path}")

if __name__ == '__main__':
    main()
