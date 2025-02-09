import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch_fft import torch_fft
import numpy as np

class SimpleBinaryClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),  # 对第一个隐藏层输出加dropout
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),  # 对第二个隐藏层输出加dropout
            nn.Linear(hidden_dim, 1)  # 最终输出1 => 二分类可再做sigmoid
        )

    def forward(self, x):
        # x shape=(batch, in_dim)
        x = torch_fft(x)
        mu_X = torch.mean(x)
        std_X = torch.std(x)
        logit = self.net((x-mu_X)/std_X)  # shape=(batch,1)
        return logit

def load_train_data(data_name):
    root_dir = os.path.join(os.path.dirname(__file__), 'Datasets/')
    if data_name[-3:]=='npz':
            data = np.load(root_dir+data_name)['data']
    else:
        data = np.load(root_dir+data_name)
    return data

def normalize_data(X_test):
    # 計算標準差
    stds = np.std(X_test, axis=-1)[:, :, np.newaxis]
    # 以標準差做除法 => 標準化
    X_test = X_test / stds
    # 轉軸
    X_test = np.swapaxes(X_test, 1, 2)
    return X_test

def train_classification(
    model,
    datasets,  
    save_name,
    device,
    epochs=30,
    batch_size=10000,
    lr_base=1e-1
):
    """
    Supervised binary classification training with flexible dataset selection.

    Parameters:
      - model: PyTorch model (e.g., SimpleBinaryClassifier)
      - datasets: Tuple containing dataset filenames (first one is treated as background, the rest as signal)
      - save_name: Filename (without extension) to save/load model weights
      - device: 'cpu' or 'cuda'
      - epochs, batch_size, lr_base: Hyperparameters

    Process:
      1) Load datasets, treating datasets[0] as background (label=0) and others as signal (label=1).
      2) Flatten data to shape (N, in_dim).
      3) Train using BCEWithLogitsLoss.
      4) Save model weights.
    """
    model.to(device)
    
    model_dir = "/home/string-3/Documents/Hackathon/Models/SL_fft/"
    model_path = model_dir + f"{save_name}.pth"

    existed = False
    if os.path.exists(model_path):
        existed = True
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"=> Found existing checkpoint '{model_path}', loaded model for continued training.")
    
    print("existed:", existed)
    model.train()

    # ---- Load datasets dynamically ----
    all_data = []
    all_labels = []

    for i, dataset_name in enumerate(datasets):
        data = normalize_data(load_train_data(dataset_name))
        
        # Use first dataset as background (label=0), others as signal (label=1)
        label = 0 if i == 0 else 1

        tensor_data = torch.flatten(torch.tensor(data, dtype=torch.float32), start_dim=1)
        tensor_labels = torch.full((len(tensor_data),), label, dtype=torch.float32)

        all_data.append(tensor_data)
        all_labels.append(tensor_labels)

    # ---- Merge datasets ----
    X = torch.cat(all_data, dim=0)  # shape=(N_total, in_dim)
    y = torch.cat(all_labels, dim=0) # shape=(N_total,)

    # ---- Move to device ----
    X = X.to(device)
    y = y.to(device).unsqueeze(1)  # shape=(N,1)

    # ---- DataLoader ----
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.BCEWithLogitsLoss()

    epoch_losses = []
    for epoch in range(epochs):
        lr = lr_base
        optimizer = optim.Adam(model.parameters(), lr=lr)

        total_loss = 0.0
        steps = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logit = model(batch_x)
            loss = criterion(logit, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / steps if steps > 0 else 0.0
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, lr={lr:.5f}, loss={avg_loss:.6f}")

    # ---- Save model ----
    model_name = f"{save_name}.pth" if not existed else "retain_model_name.pth"
    torch.save(model.state_dict(), model_dir + model_name)
    print(f"Model saved to {model_name}")
    
    return epoch_losses

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    in_dim = 100*2
    model = SimpleBinaryClassifier(in_dim, hidden_dim=128).to(device)
    
    # example datasets
    datasets = ("background_train.npz", "bbh_for_challenge_train.npy", "sglf_for_challenge_train.npy")
    
    train_classification(
        model,
        datasets,  
        "model_name.pth",
        device,
        epochs=200,
        batch_size=100000,
        lr_base=1e-3
    )
