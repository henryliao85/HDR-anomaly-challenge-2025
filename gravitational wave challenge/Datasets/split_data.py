import numpy as np
import os
from sklearn.model_selection import train_test_split

def split_and_save_npz(
    path_in,
    path_train_out,
    path_test_out,
    test_size=0.2,
    random_state=42
):
    """
    读取 path_in (npz文件),
    拆分 => train, test,
    分别存到 path_train_out, path_test_out
    """
    with np.load(path_in) as data_in:
        # 这里假设里面只有一个主数组 => data_in["data"]
        arr = data_in["data"]
    arr_train, arr_test = train_test_split(
        arr, test_size=test_size, shuffle=True, random_state=random_state
    )
    # 分别保存
    # np.savez 需要指定包含什么数组
    np.savez(path_train_out, data=arr_train)
    np.savez(path_test_out, data=arr_test)
    print(f"Split {path_in} => {path_train_out}, {path_test_out}, shapes=({arr_train.shape},{arr_test.shape})")

def split_and_save_npy(
    path_in,
    path_train_out,
    path_test_out,
    test_size=0.2,
    random_state=42
):
    """
    读取 path_in (npy文件),
    拆分 => train, test,
    分别存到 path_train_out, path_test_out
    """
    arr = np.load(path_in)
    arr_train, arr_test = train_test_split(
        arr, test_size=test_size, shuffle=True, random_state=random_state
    )
    # 分别保存
    np.save(path_train_out, arr_train)
    np.save(path_test_out, arr_test)
    print(f"Split {path_in} => {path_train_out}, {path_test_out}, shapes=({arr_train.shape},{arr_test.shape})")


if __name__ == "__main__":
    # 假设原文件：
    # dataset0.npz, dataset1.npy, dataset2.npy
    # 拆分后 => dataset0_train.npz, dataset0_test.npz,
    #           dataset1_train.npy, dataset1_test.npy,
    #           dataset2_train.npy, dataset2_test.npy

    root_dir = "/home/string-3/Documents/Hackathon/Datasets/"

    split_and_save_npz(
        root_dir+"background.npz",
        root_dir+"background_train.npz",
        root_dir+"background_test.npz"
    )

    split_and_save_npy(
        root_dir+"bbh_for_challenge.npy",
        root_dir+"bbh_for_challenge_train.npy",
        root_dir+"bbh_for_challenge_test.npy"
    )

    split_and_save_npy(
        root_dir+"sglf_for_challenge.npy",
        root_dir+"sglf_for_challenge_train.npy",
        root_dir+"sglf_for_challenge_test.npy"
    )
