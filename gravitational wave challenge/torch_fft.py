import torch
#import matplotlib.pyplot as plt

def torch_fft(signal, plot_info=None):
    # 若 plot_info=None, 先設成空dict
    if plot_info is None:
        plot_info = {}

    # 檢查 signal 維度
    if signal.ndim == 1:
        # => shape=(N,) => single_flag=True
        single_flag = True
        # 為方便 FFT 做 batch, unsqueeze(0) => shape=(1,N)
        signal = signal.unsqueeze(0)
    elif signal.ndim == 2:
        # shape=(M,N)
        single_flag = False
    else:
        raise ValueError(f"signal must be 1D or 2D, got shape={signal.shape}")

    # 取最後一維當 N
    N = signal.shape[-1]
    fs = 4096  # 假定採樣率

    # stds = torch.std(signal)
    # signal = signal/stds
    # 執行 torch.fft
    X = torch.fft.fft(signal, dim=-1)  # shape=(M,N), complex

    # 只取前半邊 => shape=(M, N/2)
    half_N = N // 2
    X_half = X[..., :half_N]
    amp_half = torch.abs(X_half) / N   # shape=(M,N/2)

    # log(amp+1)
    amp_half = torch.log1p(amp_half)

    # 若只有一條 => squeeze => (N/2,)
    if single_flag:
        amp_half = amp_half.squeeze(0)  # shape=(N/2,)

    # 若沒有繪圖需求 => 直接回傳
    if 'data type' not in plot_info:
        return amp_half

    # 若有 'data type' 但 signal 有多條 (single_flag=False)，也直接回傳 (不畫圖)
    if not single_flag:
        return amp_half

    # ============= 以下只處理「單條訊號」的繪圖 =============
    data_type = plot_info['data type']

    # 產生時間軸 (N,)（雖然 signal 現在已 squeeze 回 shape=(N/2,)）
    # 你可以從 shape=(1,N) 之前 step 取得 => 其實 N 未改
    t = torch.linspace(0, N/fs, N, device=signal.device)
    t_np = t.cpu().numpy()
    # signal[0] 依舊在 unsqueeze(0) 之後 => shape=(N,), 先 squeeze
    time_signal = signal[0]  # shape=(N,)
    time_signal_np = time_signal.detach().cpu().numpy()

    # 頻率軸 freq_half => shape=(N/2,) (PyTorch tensor)
    freq = torch.fft.fftfreq(N, d=1.0/fs).to(signal.device)
    freq_half = freq[:half_N]
    freq_half_np = freq_half.detach().cpu().numpy()

    # 振幅 => amp_half shape=(N/2,)
    amp_np = amp_half.detach().cpu().numpy()

    繪圖
    plt.figure(figsize=(10,4))
    plt.suptitle(data_type)
    # (a) 時域
    plt.subplot(1,2,1)
    plt.plot(t_np, time_signal_np, label="signal")
    plt.xlabel("Time (sec)")
    plt.ylabel("Amplitude")
    plt.title("Time domain")
    plt.grid(True)

    # (b) 頻域
    plt.subplot(1,2,2)
    plt.plot(freq_half_np, amp_np, label="Amplitude Spectrum")
    plt.xlim(0, fs/2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("log(Amplitude+1)")
    plt.title("Frequency domain")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return amp_half
