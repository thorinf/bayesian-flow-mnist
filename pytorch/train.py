import argparse
import os.path as osp
from enum import Enum

import torch
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm

from bayesian_flow_torch import BayesianFlow
from model import UNet
from utils import update_model_ema, strided_sample, plot_images, plot_images_animation


class TrainType(Enum):
    BINARISED = "binarised"
    CONTINUOUS = "continuous"


def main():
    args = create_argparser().parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = UNet(dropout_prob=0.2)
    model.to(device)

    if args.train_type == TrainType.BINARISED:
        bayesian_flow = BayesianFlow(num_classes=2, beta=3.0, reduced_features_binary=True)

    elif args.train_type == TrainType.CONTINUOUS:
        bayesian_flow = BayesianFlow(sigma=0.001)

    else:
        raise NotImplementedError(f"training not implemented for {args.train_type}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"total parameter count: {num_params:,}")

    ema_model = UNet()
    ema_model.to(device)

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

    if osp.exists(args.checkpoint):
        print(f"restoring Checkpoint: {args.checkpoint}.")
        checkpoint = torch.load(args.checkpoint)
    else:
        print(f"starting new training run: {args.checkpoint}.")
        checkpoint = {}

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])

    if 'optimizer_state_dict' in checkpoint:
        optim.load_state_dict(checkpoint['optimizer_state_dict'])

    if 'ema_model_state_dict' in checkpoint:
        ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
    else:
        ema_model.load_state_dict(model.state_dict())

    for ep in range(checkpoint.get('epochs', 1), args.epochs + 1):
        pbar, n_iter = tqdm(dataloader, total=len(dataloader), desc=f"epoch {ep}"), 0
        model.train()
        for idx, (data, labels) in enumerate(pbar):
            data, labels = data.permute(0, 2, 3, 1).to(device), labels.to(device)

            if args.train_type == TrainType.BINARISED:
                loss = bayesian_flow.discrete_data_continuous_loss(model, data, labels=labels).loss

            elif args.train_type == TrainType.CONTINUOUS:
                data = data * 2 - 1
                loss = bayesian_flow.continuous_data_continuous_loss(model, data, labels=labels).loss

            else:
                raise NotImplementedError(f"training not implemented for {args.train_type}")

            pbar.set_postfix({'loss': loss.item()})

            if not torch.isnan(loss).any():
                (loss / args.accumulation_steps).backward()
                n_iter += 1

            if ((n_iter + 1) % args.accumulation_steps == 0) or (n_iter + 1 == len(dataloader)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                optim.zero_grad()
                update_model_ema(model, ema_model, 0.95)

        checkpoint = {
            'epochs': ep,
            'model_state_dict': model.state_dict(),
            'ema_model_state_dict': ema_model.state_dict(),
            'optimizer_state_dict': optim.state_dict()
        }
        torch.save(checkpoint, args.checkpoint)

        if ep % 5 != 0:
            continue

        num_steps = 1000
        labels = torch.tensor([x if x < 10 else -1 for x in range(16)], dtype=torch.int64, device=device)

        ema_model.eval()
        with torch.no_grad():
            if args.train_type == TrainType.BINARISED:
                probs_list = bayesian_flow.discrete_data_sample(
                    model=ema_model,
                    size=(16, 28, 28),
                    labels=labels,
                    num_steps=num_steps,
                    return_all=True,
                    device=device
                )
                probs_list = strided_sample(probs_list, 20)
                x_list = [probs[..., :1].clamp(0.0, 1.0).cpu().numpy() for probs in probs_list]

            elif args.train_type == TrainType.CONTINUOUS:
                x_hat_list = bayesian_flow.continuous_data_sample(
                    model=ema_model,
                    size=(16, 28, 28, 1),
                    labels=labels,
                    num_steps=num_steps,
                    return_all=True,
                    device=device
                )
                x_hat_list = strided_sample(x_hat_list, 20)
                x_list = [((x + 1) / 2).clamp(0.0, 1.0).cpu().numpy() for x in x_hat_list]

            else:
                raise NotImplementedError(f"training not implemented for {args.train_type}")

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


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ep', '--epochs', type=int, default=50)
    parser.add_argument('-b', '--batch_size', type=int, default=512)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0)
    parser.add_argument('-acc', '--accumulation_steps', type=int, default=1)

    parser.add_argument('-ckpt', '--checkpoint', type=str, required=True)
    parser.add_argument('-d', '--data_path', type=str, required=True)

    parser.add_argument('-t', '--train_type', type=TrainType, default=TrainType.BINARISED.value)
    return parser


if __name__ == "__main__":
    main()
