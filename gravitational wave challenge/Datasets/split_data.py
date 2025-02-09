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
    Read path_in (npz file),
    Split into train and test sets,
    Save separately to path_train_out and path_test_out
    """
    with np.load(path_in) as data_in:
        # Assuming there is only one main array => data_in["data"]
        arr = data_in["data"]
    arr_train, arr_test = train_test_split(
        arr, test_size=test_size, shuffle=True, random_state=random_state
    )
    # Save separately
    # np.savez requires specifying which array to include
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
    Read path_in (npy file),
    Split into train and test sets,
    Save separately to path_train_out and path_test_out
    """
    arr = np.load(path_in)
    arr_train, arr_test = train_test_split(
        arr, test_size=test_size, shuffle=True, random_state=random_state
    )
    # Save separately
    np.save(path_train_out, arr_train)
    np.save(path_test_out, arr_test)
    print(f"Split {path_in} => {path_train_out}, {path_test_out}, shapes=({arr_train.shape},{arr_test.shape})")
