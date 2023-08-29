import os

import torch
from torch.utils.data import DataLoader

import torchvision

from tqdm import tqdm

import argparse

from model import UNet
from bayesian_flow_torch import BayesianFlow
from utils import count_parameters, strided_sample, plot_images, plot_images_animation


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ep', '--epochs', type=int, default=100)
    parser.add_argument('-b', '--batch_size', type=int, default=512)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-2)
    parser.add_argument('-acc', '--accumulation_steps', type=int, default=1)

    parser.add_argument('-ckpt', '--checkpoint', type=str, required=True)
    parser.add_argument('-d', '--data_path', type=str, required=True)

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = UNet(dropout_prob=0.5)
    model.to(device)

    bayesian_flow = BayesianFlow(model, num_classes=2, beta=3.0, reduced_features_binary=True)

    if os.path.exists(args.checkpoint):
        print(f"Restoring Checkpoint: {args.checkpoint}.")
        checkpoint = torch.load(args.checkpoint)
    else:
        print(f"Starting new training run: {args.checkpoint}.")
        checkpoint = {}

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])

    num_params = count_parameters(model)
    print(f"Total number of parameters: {num_params:,}")

    dataset = torchvision.datasets.MNIST(
        root=args.data_path,
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True
    )

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98)
    )

    if 'optimizer_state_dict' in checkpoint:
        optim.load_state_dict(checkpoint['optimizer_state_dict'])

    global_step = checkpoint.get('global_step', 0)

    for ep in range(checkpoint.get('epochs', 0), args.epochs):
        model.train()

        pbar = tqdm(dataloader)
        pbar.set_description(f"epoch: {ep}")

        for idx, (data, labels) in enumerate(pbar):
            data = data.permute(0, 2, 3, 1)
            data = data.to(device)
            labels = labels.to(device)

            loss = bayesian_flow.discrete_data_continuous_loss(data, labels=labels)

            pbar.set_postfix({
                "loss": loss.item()
            })

            (loss / args.accumulation_steps).backward()

            if ((idx + 1) % args.accumulation_steps == 0) or (idx + 1 == len(dataloader)):
                optim.step()
                optim.zero_grad()
                global_step += 1

        checkpoint = {
            'epochs': ep + 1,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict()
        }
        torch.save(checkpoint, args.checkpoint)

        if (ep + 1) % 5 != 0:
            continue

        num_steps = 1000
        labels = torch.tensor([x if x < 10 else -1 for x in range(16)], dtype=torch.int64, device=device)

        model.eval()
        with torch.no_grad():
            probs_list = bayesian_flow.discrete_data_sample(
                size=(16, 28, 28),
                labels=labels,
                num_steps=num_steps,
                return_all=True,
                device=device
            )
        probs_list = strided_sample(probs_list, 20)
        x_list = [probs[..., :1].clamp(0.0, 1.0).cpu().numpy() for probs in probs_list]

        plot_images(
            images=x_list[-1],
            subplot_shape=(4, 4),
            name=f"Epoch: {ep}, Steps: {num_steps}",
            path=f"figures/epoch-{ep}_steps-{num_steps}.png",
            labels=labels.tolist()
        )

        plot_images_animation(
            images_list=x_list,
            subplot_shape=(4, 4),
            name=f"Epoch: {ep}, Steps: {num_steps}",
            path=f"figures/epoch-{ep}_steps-{num_steps}.gif",
            labels=labels.tolist()
        )


if __name__ == "__main__":
    train()
