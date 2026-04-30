import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from lsbi_smc.example_shear4dof.mvae import MVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load .npz file & reshape
npl = np.load("train_data.npz")
x_sim = npl["x_sim"].astype(np.float32)
y_sim = npl["y_sim"].astype(np.float32)
y_sim_n = npl["y_sim_n"].astype(np.float32)
del npl

# convert to Tensor on GPU
ch = [-1]
x_sim_tensor = torch.from_numpy(x_sim)
y_sim_tensor = torch.from_numpy(y_sim[:, ch, :, :])
y_sim_n_tensor = torch.from_numpy(y_sim_n[:, ch, :, :])

# standardize
y_mn, y_sd = y_sim_tensor.mean(), y_sim_tensor.std()
y_sim_tensor = (y_sim_tensor - y_mn) / y_sd
y_sim_n_tensor = (y_sim_n_tensor - y_mn) / y_sd

# create datasets
dataset = TensorDataset(x_sim_tensor, y_sim_tensor, y_sim_n_tensor)
n_total = len(dataset)
n_train = int(0.9 * n_total)
n_valid = n_total - n_train
train_dataset, valid_dataset = random_split(
    dataset, [n_train, n_valid], generator=torch.Generator().manual_seed(42)
)

# create dataloaders
bs = 512
train_loader = DataLoader(
    train_dataset, batch_size=bs, shuffle=True, num_workers=8, pin_memory=True
)
valid_loader = DataLoader(
    valid_dataset, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True
)

# ---------------
# MVAE Training!
# ---------------

# load mvae model
ndof, z_dim = 4, 8
model = MVAE(z_dim=z_dim, ch=1, size=1024, nlabel=ndof, depth=1).to(device)

# Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# training
epochs = 1000
best_vl_loss = float("inf")
patience = 20
epochs_no_improve = 0

for epoch in range(epochs):
    # for tracking loss values -----
    tr_loss, kl_loss, rc_loss, vl_loss = 0, 0, 0, 0

    # training -----
    model.train()
    for x, y, yn in train_loader:
        x = x.to(device)
        y = y.to(device)
        yn = yn.to(device)
        optimizer.zero_grad()
        trl, kll, rcl = model.loss(yn, y, x, alp1=5.0)
        trl.backward()
        optimizer.step()

        # summing up loss values
        tr_loss += trl.item() * y.size(0)
        kl_loss += kll.item() * y.size(0)
        rc_loss += rcl.item() * y.size(0)

    # validation -----
    model.eval()
    with torch.no_grad():
        for x, y, yn in valid_loader:
            x = x.to(device)
            y = y.to(device)
            yn = yn.to(device)
            vll, _, _ = model.loss(yn, y, x, alp1=5.0)

            # summing up loss values
            vl_loss += vll.item() * y.size(0)

    # averaging & print
    tr_loss /= len(train_dataset)
    kl_loss /= len(train_dataset)
    rc_loss /= len(train_dataset)
    vl_loss /= len(valid_dataset)

    print(f"Epoch {epoch:03d}: Train loss = {tr_loss:16.4f} | Valid loss = {vl_loss:16.4f}")

    # plot & save -----
    if vl_loss < best_vl_loss:
        best_vl_loss = vl_loss
        epochs_no_improve = 0
        pth_path = "mvae_best.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            pth_path,
        )
    else:
        epochs_no_improve += 1

    # early stopping
    if epochs_no_improve >= patience:
        break
