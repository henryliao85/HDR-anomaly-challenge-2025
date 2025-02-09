import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch_fft import torch_fft

class SimpleBinaryClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # 输出1 => 之后做sigmoid
        )

    def forward(self, x):
        # x shape=(batch, in_dim)
        x = torch_fft(x)
        mu_X = torch.mean(x)
        std_X = torch.std(x)
        logit = self.net((x-mu_X)/std_X)  # shape=(batch,1)
        return logit

def train_classification(
    model,
    train_dataset, 
    save_name,
    device,
    epochs=30,
    batch_size=10000,
    lr_base=1e-1
):
    """
    监督式二分类训练:
      - train_dataset: (dataset0, dataset1, dataset2)
        dataset0 => label 0 (background)
        dataset1 => label 1 (signal)
        dataset2 => label 1 (signal)
      - model: SimpleBinaryClassifier(in_dim, hidden_dim) or your own
      - save_name: 保存/载入模型权重的文件名(不含后缀)
      - device: 'cpu' or 'cuda'
      - epochs, batch_size, lr_base: 超参

    流程:
      1) 合并 dataset0 => label=0, dataset1 + dataset2 => label=1
      2) flatten => shape=(N, in_dim)
      3) DataLoader => BCEWithLogitsLoss => 训练
      4) 保存权重
    """
    model.to(device)

    model_path = f"/home/string-3/Documents/Hackathon/opt/{save_name}.pth"
    existed = False
    if os.path.exists(model_path):
        existed = True
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"=> Found existing checkpoint '{model_path}', loaded model for continued training.")
    print("existed:", existed)
    model.train()

    dataset0, dataset1, dataset2 = train_dataset
    
    # 1) 合并 label=0 (dataset0), label=1 (dataset1+dataset2)
    # 假设 dataset0.shape=(N0, T, ?), dataset1.shape=(N1, T, ?), ...
    # 这里与之前 Flatten usage保持一致 => flatten dimension=1
    # 先转tensor再 flatten
    #   dataset0 => label=0
    data0 = torch.flatten(torch.tensor(dataset0, dtype=torch.float32), start_dim=1)
    label0= torch.zeros(len(data0), dtype=torch.float32)
    #   dataset1+2 => label=1
    data1 = torch.flatten(torch.tensor(dataset1, dtype=torch.float32), start_dim=1)
    data2 = torch.flatten(torch.tensor(dataset2, dtype=torch.float32), start_dim=1)
    data12= torch.cat([data1, data2], dim=0)
    label1= torch.ones(len(data2), dtype=torch.float32)
    
    X = torch.cat([data0, data2], dim=0)  # shape=(N0+N1+N2, in_dim)
    y = torch.cat([label0, label1], dim=0) # shape=(N0+N1+N2,)

    # 2) 放到 device
    X = X.to(device)
    y = y.to(device).unsqueeze(1)  # shape=(N,1)

    # 3) DataLoader
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.BCEWithLogitsLoss()

    epoch_losses = []
    for epoch in range(epochs):
        lr = lr_base / ((epoch+1)**0.5)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        total_loss = 0.0
        steps = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            # forward
            logit = model(batch_x)
            # BCEWithLogitsLoss => 不用手动 sigmoid
            loss = criterion(logit, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / steps if steps>0 else 0.0
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, lr={lr:.5f}, loss={avg_loss:.6f}")

    # 保存
    model_dir = "/home/string-3/Documents/Hackathon/Models/SL_fft/"
    model_name = f"{save_name}.pth" if not existed else "supervied_model_fft_N_v1-1.pth"
    torch.save(model.state_dict(), model_dir+model_name)
    print(f"Model saved to {model_name}")
    return epoch_losses

def predict_probability(model, X_input, device='cpu'):
    """
    推断阶段: X_input shape=(N, in_dim) => 返回 (N,) 概率(0~1), 1代表更像 signal
    """
    model.eval()
    with torch.no_grad():
        X_input = torch.tensor(X_input, dtype=torch.float32, device=device)
        logit = model(X_input)  # shape=(N,1)
        prob = torch.sigmoid(logit).squeeze(1)  # shape=(N,)
    return prob.cpu().numpy()

if __name__ == "__main__":
    import numpy as np
    train_dataset = np.load("train_data.npy")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    in_dim = 100*2
    model = SimpleBinaryClassifier(in_dim, hidden_dim=64).to(device)

    train_classification(
        model,
        train_dataset, 
        "supervied_model_fft_N_v1",
        device,
        epochs=100,
        batch_size=100000,
        lr_base=1e-3)
