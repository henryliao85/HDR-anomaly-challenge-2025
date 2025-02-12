from split_data import split_and_save_npz, split_and_save_npy
from get_synthetic_data import augment_with_noise, simulate_sine_gaussian_1ch
import os

root_dir = os.path.dirname(__file__)
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

status = "train"
for status in ["train", "test"]:
  bg_data = np.load(root_dir+f"background_{status}.npz")['data']
  bbh_data = np.load(root_dir+f"bbh_for_challenge_{status}.npy")
  sg_data = np.load(root_dir+f"sglf_for_challenge_{status}.npy")
  
  noisy_bg_data = augment_with_noise(bg_data, noise_level=1e-3)
  noisy_bbh_data = augment_with_noise(bbh_data, noise_level=1e-3)
  noisy_sg_data = augment_with_noise(sg_data, noise_level=1e-3)
  aug_sg_data = add_sine_gaussian_to_background(bg_data)
  
  np.savez(root_dir+f"noisy_background_{status}.npz", data=noisy_bg_data)
  np.save(root_dir+f"noisy_bbh_for_challenge_{status}.npy", noisy_bbh_data)
  np.save(root_dir+f"noisy_sglf_for_challenge_{status}.npy", noisy_sg_data)
  np.save(root_dir+f"aug_sglf_for_challenge_{status}.npy", aug_sg_data)
