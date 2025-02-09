import wget

model_url = "https://huggingface.co/KenLee1130/NSF-HDR-A3D3-Detecting-Anomalous-Gravitational-Wave-Signals/blob/main/supervied_model_fft_N_v3-4.pth"
model_name = "supervied_model_fft_N_v3-4.pth"

wget.download(model_url, model_name)