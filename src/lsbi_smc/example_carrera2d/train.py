"""Step 2: train the fully-connected MVAE on the two-dimensional example.

Reads ``train_data.npz`` and writes ``mvae_best.pth`` (best validation loss).
The observation standardisation statistics (mean/std) are stored inside the
checkpoint so that inference standardises the observation identically.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from lsbi_smc.example_carrera2d.problem import DIM
from lsbi_smc.mvae_fc import MVAEFC


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default="train_data.npz")
    parser.add_argument("--out", type=str, default="mvae_best.pth")
    parser.add_argument("--z-dim", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alp1", type=float, default=5.0, help="weight on the latent-alignment KL")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    npz = np.load(args.data)
    x_sim = npz["x_sim"].astype(np.float32)
    y_sim = npz["y_sim"].astype(np.float32)
    y_sim_n = npz["y_sim_n"].astype(np.float32)

    x_t = torch.from_numpy(x_sim)
    y_t = torch.from_numpy(y_sim)
    yn_t = torch.from_numpy(y_sim_n)

    # standardise observations (shared stats for clean & noisy targets).
    y_mn = y_t.mean()
    y_sd = y_t.std()
    y_t = (y_t - y_mn) / y_sd
    yn_t = (yn_t - y_mn) / y_sd

    dataset = TensorDataset(x_t, y_t, yn_t)
    n_train = int(0.9 * len(dataset))
    n_valid = len(dataset) - n_train
    train_ds, valid_ds = random_split(
        dataset, [n_train, n_valid], generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)

    model = MVAEFC(
        obs_dim=DIM, param_dim=DIM, z_dim=args.z_dim, hidden=args.hidden, depth=args.depth
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_vl = float("inf")
    no_improve = 0
    for epoch in range(args.epochs):
        model.train()
        tr_loss = 0.0
        for x, y, yn in train_loader:
            x, y, yn = x.to(device), y.to(device), yn.to(device)
            optimizer.zero_grad()
            loss, _, _ = model.loss(yn, y, x, alp1=args.alp1)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * y.size(0)
        tr_loss /= len(train_ds)

        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for x, y, yn in valid_loader:
                x, y, yn = x.to(device), y.to(device), yn.to(device)
                loss, _, _ = model.loss(yn, y, x, alp1=args.alp1)
                vl_loss += loss.item() * y.size(0)
        vl_loss /= len(valid_ds)

        print(f"Epoch {epoch:03d}: train = {tr_loss:12.4f} | valid = {vl_loss:12.4f}")

        if vl_loss < best_vl:
            best_vl = vl_loss
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": {
                        "obs_dim": DIM,
                        "param_dim": DIM,
                        "z_dim": args.z_dim,
                        "hidden": args.hidden,
                        "depth": args.depth,
                    },
                    "y_mn": float(y_mn),
                    "y_sd": float(y_sd),
                },
                args.out,
            )
        else:
            no_improve += 1
        if no_improve >= args.patience:
            print(f"early stopping at epoch {epoch}")
            break

    print(f"best validation loss = {best_vl:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
