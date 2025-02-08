import os
import re
import csv
import datetime
import numpy as np
import torch
import netCDF4


def parse_date_from_filename(fn):
    """
    Parses filenames of the form: dt_ena_{YYYYMMDD}_vDT2021.nc.
    Returns (day_float, datetime) where day_float is days since 1950-01-01.
    """
    m = re.search(r"dt_ena_(\d{8})_vDT2021\.nc", fn)
    if not m:
        return None, None
    datestr = m.group(1)
    dt_ = datetime.datetime.strptime(datestr, "%Y%m%d")
    base = datetime.datetime(1950, 1, 1)
    day_float = (dt_ - base).days
    return day_float, dt_

def build_training_arrays_from_files(file_list, label_dir, device):
    """
    Given a list of file paths (from "data/nc3", sorted by time),
    build training arrays.
    For each day t (from t0+1 to t1) use day(t-1)'s ugosa and vgosa (stacked into [2,100,160])
    and day(t)'s SLA (shape [100,160]). Also read labels for day t from CSVs.
    Returns tensors: X_ugvg, X_sla, Y_12.
    """
    storage = {}
    for fp in file_list:
        fn = os.path.basename(fp)
        day_float, _ = parse_date_from_filename(fn)
        if day_float is None:
            continue
        ds = netCDF4.Dataset(fp, mode='r')
        sla = ds.variables['sla'][:].squeeze()       # [100,160]
        ugosa = ds.variables['ugosa'][:].squeeze()     # [100,160]
        vgosa = ds.variables['vgosa'][:].squeeze()     # [100,160]
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
    # Read labels from CSVs.
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
    train_ugvg = []
    train_sla = []
    train_lbl = []
    for (t_prev, t) in day_pairs:
        if t_prev not in storage or t not in storage:
            continue
        ug = storage[t_prev]['ugosa']
        vg = storage[t_prev]['vgosa']
        sl = storage[t]['sla']
        lbl = np.zeros(12, dtype=np.float32)
        for i_lab in range(12):
            lbl[i_lab] = label_dict[i_lab].get(t, 0)
        train_ugvg.append(np.stack([ug, vg], axis=0))  # [2,100,160]
        train_sla.append(sl)                            # [100,160]
        train_lbl.append(lbl)
    train_ugvg = np.stack(train_ugvg, axis=0)
    train_sla = np.stack(train_sla, axis=0)
    train_lbl = np.stack(train_lbl, axis=0)
    X_ug = torch.tensor(train_ugvg, dtype=torch.float, device=device)
    X_sl = torch.tensor(train_sla, dtype=torch.float, device=device)
    Y_12 = torch.tensor(train_lbl, dtype=torch.float, device=device)
    return X_ug, X_sl, Y_12
