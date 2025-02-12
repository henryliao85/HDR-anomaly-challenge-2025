import wget
import os

def download_model_weight():
    model_url = "https://huggingface.co/KenLee1130/NSF-HDR-A3D3-Detecting-Anomalous-Gravitational-Wave-Signals/resolve/main/supervied_model_fft_N_v3-4.pth"
    model_name = "supervised_model.pth"

    wget.download(model_url, os.path.join(os.path.dirname(__file__), model_name))
