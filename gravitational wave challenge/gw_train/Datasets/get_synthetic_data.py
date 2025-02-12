import numpy as np

def augment_with_noise(original_data, noise_level=1e-3, num_aug=1, seed=None):
    """
    Apply random Gaussian noise augmentation to data with shape (N, C, T) = (N, 2, 200)
    
    :param original_data: np.array or similar with shape=(N, 2, 200)
    :param noise_level: Standard deviation of noise (relative to 1), default is 0.05
    :param num_aug: Number of augmented versions to generate per original sample
    :param seed: Optional random seed
    :return: New data with shape=(N * num_aug, 2, 200)
    
    Process:
      1) Original data => (N, 2, 200)
      2) Expand => (N, num_aug, 2, 200)
      3) Generate Gaussian noise with the same shape (N, num_aug, 2, 200)
      4) Add noise and reshape => (N*num_aug, 2, 200)
    """
    if seed is not None:
        np.random.seed(seed)

    N, C, T = original_data.shape  # Here C=2, T=200

    # Expand: (N, 1, C, T) => Repeat to (N, num_aug, C, T)
    expanded = np.repeat(original_data[:, np.newaxis, :, :], repeats=num_aug, axis=1)
    # Now expanded.shape=(N, num_aug, C, T)

    # Generate noise with the same shape
    noise = np.random.randn(N, num_aug, C, T).astype(original_data.dtype) * noise_level
    # noise.shape=(N, num_aug, C, T)
    
    new_data = expanded + noise  # (N,num_aug,C,T)
    # Reshape => (N * num_aug, C, T)
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
    Generate a single-channel sine-Gaussian signal with shape=(length,)
    s(t) = A * exp(-(t - t0)^2/(2*sigma^2)) * sin(2πf0 (t - t0))
    """
    if t0 is None:
        t0 = length/(2*fs)   # Center the signal in the middle (seconds)
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
    Generate a 2-channel sine-Gaussian signal with shape=(2,length)
    Simple method: both channels have the same wave.
    To make them different, modify the function accordingly.
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
    :param background: shape=(N,2,200)  => Original background data
    :param fs: Sampling rate
    :param fraction: 0~1, indicating the proportion of background samples to mix with sine-Gaussian signals
                     1.0 => All samples have added signals
                     0.5 => Half of the samples have added signals, the other half remain unchanged
    :param A_range, f0_range, sigma_range: Ranges for amplitude, frequency, and Gaussian envelope
    :param seed: Random seed
    :return: new_data shape=(N,2,200), where some or all samples have added sine-Gaussian signals
    """
    if seed is not None:
        np.random.seed(seed)

    N, ch, length = background.shape
    new_data = background.copy()

    # Determine indices of samples to have added signals
    num_signals = int(N * fraction)
    signal_indices = np.random.choice(N, size=num_signals, replace=False)

    for idx in signal_indices:
        # Randomly generate sine-Gaussian parameters
        A = np.random.uniform(*A_range)
        f0= np.random.uniform(*f0_range)
        sigma = np.random.uniform(*sigma_range)

        # Generate wave => shape=(2,length)
        wave_2ch = simulate_sine_gaussian_2ch(length, fs, A, f0, None, sigma)
        # Add to background
        new_data[idx] += wave_2ch

    return new_data
