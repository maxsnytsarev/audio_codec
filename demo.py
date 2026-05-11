import os
import subprocess
import torchaudio
from pathlib import Path
import torch
import torch.nn.functional as F
import gdown


from src.model.FullModel import FullModel

def load_object(path):
    data_object, sr = torchaudio.load(str(path))
    if data_object.shape[0] > 1:
        data_object = data_object.mean(dim=0, keepdim=True)

    if sr != 16000:
        data_object = torchaudio.functional.resample(
            data_object, orig_freq=sr, new_freq=16000
        )
    return data_object

def download_wav(url):
    assert url != "your_wav_url"
    new_dir = "demo_dir"
    demo_file = "demo_dir/demo_real.wav"
    if not os.path.exists(new_dir):
        Path(new_dir).mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["wget", "-c", url, "-O", demo_file])
        path = Path(demo_file)
        wav = load_object(path)
        torchaudio.save("demo_dir/demo_real_16k.wav", wav, 16000)
        print("Download finished")
        return wav
    except Exception as e:
        print("Failed to download file:")
        print(e)

def get_reconstructed_audio(wav):
    if not os.path.exists("demo_dir/demo_real.wav"):
        print("No test wav")
        return None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FullModel(C=32, D=512, N_q=8).to(device)
    check = "demo_dir/last_checkpoint.pth"
    if not os.path.exists(check):
        file_id = "1DAafPFX3gpC1xFy9yrQYu4yoM2Mj7WcS"
        gdown.download(id=file_id, output=check, quiet=False)
        print("Successfully saved weights")
    checkpoint = torch.load(check, map_location=device)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    orig_len = wav.shape[-1]
    if orig_len % 200 != 0:
        cur_len = orig_len + 200 - orig_len % 200
    else:
        cur_len = orig_len
    wav = F.pad(wav, (0, cur_len - orig_len))
    batch = wav.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(data_object=batch)
        fake = out["logits"][0].cpu()
    fake = fake[..., :orig_len]
    torchaudio.save("demo_dir/demo_reconstructed.wav", fake, 16000)
    print(f"Saved reconstructed wav to demo_dir/demo_reconstructed.wav")
    return fake
