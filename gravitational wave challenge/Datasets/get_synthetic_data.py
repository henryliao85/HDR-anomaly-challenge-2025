import numpy as np

def augment_with_noise(original_data, noise_level=1e-3, num_aug=1, seed=None):
    """
    对形状 (N, C, T) = (N, 2, 200) 数据做随机高斯噪声增广
    
    :param original_data: shape=(N, 2, 200) 的np.array或类似
    :param noise_level: 噪声标准差(相对于1)的量级, 默认0.05
    :param num_aug: 每条原样本要生成多少条新的带噪版本
    :param seed: 可选随机种子
    :return: shape=(N * num_aug, 2, 200) 的新数据
    
    原理:
      1) 把 original_data => (N, 2, 200)
      2) 重复 => (N, num_aug, 2, 200)
      3) 生成与之同形状的高斯噪声 (N, num_aug, 2, 200)
      4) 相加后再 reshape => (N*num_aug, 2, 200)
    """
    if seed is not None:
        np.random.seed(seed)

    N, C, T = original_data.shape  # 这里C=2, T=200

    # 先扩展: (N, 1, C, T) => 再repeat到 (N, num_aug, C, T)
    expanded = np.repeat(original_data[:, np.newaxis, :, :], repeats=num_aug, axis=1)
    # 现在 expanded.shape=(N, num_aug, C, T)

    # 生成同样shape的噪声
    noise = np.random.randn(N, num_aug, C, T).astype(original_data.dtype) * noise_level
    # noise.shape=(N, num_aug, C, T)
    
    new_data = expanded + noise  # (N,num_aug,C,T)
    # reshape => (N * num_aug, C, T)
    new_data = new_data.reshape(N*num_aug, C, T)

    return new_data

def simulate_sine_gaussian_1ch(
    length=200,
    fs=1000,
    A=1.0,
    f0=30.0,
    t0=None,
    sigma=0.05
):
    """
    产生单通道的 sine–Gaussian, 返回 shape=(length,)
    s(t) = A * exp(-(t - t0)^2/(2*sigma^2)) * sin(2πf0 (t - t0))
    """
    if t0 is None:
        t0 = length/(2*fs)   # 把中心放在中点(秒)
    t = np.linspace(0, length/fs, length, endpoint=False)
    gauss_env = np.exp(-0.5 * ((t - t0)/sigma)**2)
    sin_part  = np.sin(2*np.pi*f0*(t - t0))
    s = A * gauss_env * sin_part
    return s.astype(np.float32)

def simulate_sine_gaussian_2ch(
    length=200,
    fs=1000,
    A=1.0,
    f0=30.0,
    t0=None,
    sigma=0.05
):
    """
    对 2 通道 (shape=(2,length))，简单做法：两个通道都一样的 wave
    若想让两个通道不一样，可以再多写一点。
    """
    wave_1ch = simulate_sine_gaussian_1ch(length, fs, A, f0, t0, sigma)  # shape=(length,)
    wave_2ch = np.stack([wave_1ch, wave_1ch], axis=0)   # shape=(2,length)
    return wave_2ch

def add_sine_gaussian_to_background(
    background,
    fs=4096,
    fraction=1.0,
    A_range=(0.3,0.8),
    f0_range=(10,80),
    sigma_range=(0.02,0.1),
    seed=None
):
    """
    :param background: shape=(N,2,200)  => 原本的 background
    :param fs: 采样率
    :param fraction: 0~1, 表示要给多少比例的 background 样本叠加 sine–Gaussian
                     1.0 => 全部都叠加
                     0.5 => 只有一半样本叠加, 另一半保持原状
    :param A_range, f0_range, sigma_range: 幅度/频率/包络随机取值范围
    :param seed: 随机种子
    :return: new_data shape=(N,2,200), 其中部分或全部叠加了 sine–Gaussian
    """
    if seed is not None:
        np.random.seed(seed)

    N, ch, length = background.shape
    new_data = background.copy()

    # 确定要叠加 signal 的样本下标
    num_signals = int(N * fraction)
    signal_indices = np.random.choice(N, size=num_signals, replace=False)

    for idx in signal_indices:
        # 随机生成 sine–Gaussian 的参数
        A = np.random.uniform(*A_range)
        f0= np.random.uniform(*f0_range)
        sigma = np.random.uniform(*sigma_range)

        # 产生 wave => shape=(2,length)
        wave_2ch = simulate_sine_gaussian_2ch(length, fs, A, f0, None, sigma)
        # 叠加
        new_data[idx] += wave_2ch

    return new_data

if __name__ == "__main__":
    root_dir = "/home/string-3/Documents/Hackathon/Datasets/"
    status = "train"

    bg_data = np.load(root_dir+f"background_{status}.npz")['data']
    bbh_data = np.load(root_dir+f"bbh_for_challenge_{status}.npy")
    sg_data = np.load(root_dir+f"sglf_for_challenge_{status}.npy")

    # aug_bg_data = augment_with_noise(bg_data, noise_level=1e-3)
    # aug_bbh_data = augment_with_noise(bbh_data, noise_level=1e-3)
    aug_sg_data = add_sine_gaussian_to_background(bg_data)

    # np.savez(root_dir+f"aug_background_{status}.npz", data=aug_bg_data)
    # np.save(root_dir+f"aug_bbh_for_challenge_{status}.npy", aug_bbh_data)
    np.save(root_dir+f"2new_sglf_for_challenge_{status}.npy", aug_sg_data)