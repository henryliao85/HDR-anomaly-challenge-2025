import os
import csv
import datetime
import numpy as np
import pandas as pd
import torch
import netCDF4
from torch.optim import Adam
from sklearn.metrics import f1_score

from dataset import build_training_arrays_from_files, parse_date_from_filename

def fit(model, checkpoint_path, device, data_dir, label_dir):
    """
    Sliding-window training over files from data_dir.
    For each window (of size window_size, sliding by window_step),
    train the model for epochs_per_window epochs using mini-batches.
    """
    model.model.to(device)
    all_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".nc")],
                       key=lambda fp: parse_date_from_filename(os.path.basename(fp))[0])
    total_files = len(all_files)
    window_size = model.cfg.get('window_size', 2400)
    window_step = model.cfg.get('window_step', 2400)
    epochs_per_window = model.cfg.get('epochs_per_window', 150)
    batch_size = model.cfg.get('batch_size', 2400)
    # Loop over sliding windows.
    for start in range(0, total_files - window_size + 1, window_step):
        window_files = all_files[start : start + window_size]
        X_ugvg, X_sla, Y_12 = build_training_arrays_from_files(window_files, label_dir, device)
        N = Y_12.shape[0]
        optimizer = Adam(model.model.parameters(), lr=model.cfg['lr'])
        for epoch in range(epochs_per_window):
            model.model.train()
            permutation = torch.randperm(N)
            epoch_loss = 0.0
            epoch_f1 = 0.0
            for i in range(0, N, batch_size):
                indices = permutation[i:i+batch_size]
                batch_X_ugvg = X_ugvg[indices]
                batch_X_sla = X_sla[indices]
                batch_Y = Y_12[indices]
                optimizer.zero_grad()
                logits, mus, logvars = model.model(batch_X_ugvg, batch_X_sla)  # [B,12]
                probs = torch.sigmoid(logits)
                eps = 1e-8
                f1_vals = []
                for k in range(12):
                    p_k = probs[:, k]
                    y_k = batch_Y[:, k]
                    tp_k = (p_k * y_k).sum()
                    sum_p = p_k.sum()
                    sum_y = y_k.sum()
                    f1_k = 2 * tp_k / (sum_p + sum_y + eps)
                    f1_vals.append(f1_k)
                macro_f1 = sum(f1_vals) / 12.0
                # KL divergence from each VAE branch:
                kl_loss = 0.0
                for mu, logvar in zip(mus, logvars):
                    kl_loss += -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kl_loss = kl_loss / batch_size
                beta = model.cfg.get('vae_beta', 0.001)
                loss = (1.0 - macro_f1) + beta * kl_loss
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(indices)
                epoch_f1 += macro_f1.item() * len(indices)
    # Save the model checkpoint after training
    torch.save(model.model.state_dict(), checkpoint_path)

def predict(model, device, data_dir, label_dir, csv_path = "test.predictions.csv"):
    """
    Predict on data from './data/nc_validate' (the prediction set).
    For each day t in the prediction set, use day(t-1)'s ugosa/vgosa and day(t)'s SLA.
    Only day pairs with day t between 2008-01-01 and 2013-12-31 are used.
    Predictions are saved to CSV and discrete macro-F1 is computed.
    """
    model.model.to(device)
    model.model.eval()
    storage = {}
    fns = sorted(os.listdir(data_dir))
    from os.path import join
    for fn in fns:
        if not fn.endswith(".nc"):
            continue
        day_float, dt_ = parse_date_from_filename(fn)
        if day_float is None:
            continue
        fp = join(data_dir, fn)
        ds = netCDF4.Dataset(fp, mode='r')
        sla = ds.variables["sla"][:].squeeze()
        ugosa = ds.variables["ugosa"][:].squeeze()
        vgosa = ds.variables["vgosa"][:].squeeze()
        ds.close()
        storage[day_float] = {
            'sla': np.ma.filled(sla, 0.0),
            'ugosa': np.ma.filled(ugosa, 0.0),
            'vgosa': np.ma.filled(vgosa, 0.0)
        }
    all_days = sorted(storage.keys())
    day_pairs = []
    for i in range(1, len(all_days)):
        t = all_days[i]
        t_prev = all_days[i-1]
        day_pairs.append((t_prev, t))
    csv_list = [
        "Atlantic_City_1993_2013_training_data.csv",
        "Baltimore_1993_2013_training_data.csv",
        "Eastport_1993_2013_training_data.csv",
        "Fort_Pulaski_1993_2013_training_data.csv",
        "Lewes_1993_2013_training_data.csv",
        "New_London_1993_2013_training_data.csv",
        "Newport_1993_2013_training_data.csv",
        "Portland_1993_2013_training_data.csv",
        "Sandy_Hook_1993_2013_training_data.csv",
        "Sewells_Point_1993_2013_training_data.csv",
        "The_Battery_1993_2013_training_data.csv",
        "Washington_1993_2013_training_data.csv"
    ]
    label_dict = [{} for _ in range(12)]
    for k, csvf in enumerate(csv_list):
        p_ = os.path.join(label_dir, csvf)
        with open(p_, "r", newline='') as f:
            rd = csv.DictReader(f)
            for row in rd:
                dt_str = row["t"]
                anom = int(row["anomaly"])
                dt_ = datetime.datetime.strptime(dt_str, "%Y-%m-%d")
                base = datetime.datetime(1950, 1, 1)
                t_day = (dt_ - base).days
                label_dict[k][float(t_day)] = anom
    # Filter day pairs to only those where day t is in [2008-01-01, 2013-12-31]
    start_dt = datetime.datetime(2008, 1, 1)
    end_dt = datetime.datetime(2013, 12, 31)
    base = datetime.datetime(1950, 1, 1)
    def to_dt(tday):
        return base + datetime.timedelta(days=tday)
    final_pairs = []
    for (t_prev, t) in day_pairs:
        real_dt = to_dt(t)
        if start_dt <= real_dt <= end_dt:
            final_pairs.append((t_prev, t))
    if len(final_pairs) == 0:
        print("No valid day pairs in [2008-01-01..2013-12-31].")
        return
    pred_ugvg = []
    pred_sla = []
    pred_lbl = []
    times_list = []
    for (t_prev, t) in final_pairs:
        if t_prev not in storage or t not in storage:
            continue
        ug = storage[t_prev]['ugosa']
        vg = storage[t_prev]['vgosa']
        sl = storage[t]['sla']
        lbl = np.zeros(12, dtype=np.int64)
        for i_lab in range(12):
            lbl[i_lab] = label_dict[i_lab].get(t, 0)
        pred_ugvg.append(np.stack([ug, vg], axis=0))
        pred_sla.append(sl)
        pred_lbl.append(lbl)
        times_list.append(t)
    M = len(times_list)
    if M == 0:
        print("No valid prediction samples in range.")
        return
    pred_ugvg = np.stack(pred_ugvg, axis=0)  # [M,2,100,160]
    pred_sla = np.stack(pred_sla, axis=0)      # [M,100,160]
    pred_lbl = np.stack(pred_lbl, axis=0)      # [M,12]
    X_ug = torch.tensor(pred_ugvg, dtype=torch.float, device=device)
    X_sl = torch.tensor(pred_sla, dtype=torch.float, device=device)
    batch_size = model.cfg.get('batch_size', 32)
    B = X_ug.size(0)
    all_preds = []
    all_true = []
    for i in range(0, B, batch_size):
        batch_X_ug = X_ug[i:i+batch_size]
        batch_X_sl = X_sl[i:i+batch_size]
        with torch.no_grad():
            logits, _, _ = model.model(batch_X_ug, batch_X_sl)  # [b,12]
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int().cpu().numpy()
        all_preds.append(preds)
        all_true.append(pred_lbl[i:i+batch_size])
    preds_bin = np.concatenate(all_preds, axis=0)
    true_labels = np.concatenate(all_true, axis=0)
    # Save predictions to CSV.
    header = ["time",
              "Atlantic_City", "Baltimore", "Eastport", "Fort_Pulaski", "Lewes",
              "New_London", "Newport", "Portland", "Sandy_Hook", "Sewells_Point",
              "The_Battery", "Washington"]
    rows = []
    for i in range(M):
        t_val = times_list[i]
        real_dt = to_dt(t_val)
        date_str = real_dt.strftime("%Y-%m-%d")
        row = [date_str] + preds_bin[i].tolist()
        rows.append(row)
    df = pd.DataFrame(rows, columns=header)
    df.to_csv(csv_path, index=False)
    macro_f1 = f1_score(true_labels, preds_bin, average='macro')
    print(f"Discrete final F1 (macro) on predictions: {macro_f1:.4f}")
