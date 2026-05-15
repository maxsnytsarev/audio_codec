# Audio Codec


## About

This repository contains implementation of audio codec presented in [SoundStream](https://arxiv.org/abs/2107.03312)

With this repo you can:
- train the codec model;
- run inference on test audio datasets;
- run a small demo on a single downloadable file;
- download model weights.


## Installation

### Clone the repository 

```bash
git clone https://github.com/maxsnytsarev/audio_codec.git
cd audio_codec   
```

### (If needed) Create virtual environment (e.g. venv)
```bash
python3 -m venv audio_codec
source audio_codec/bin/activate  
```

### Install required packages
```bash
pip install -r requirements.txt
```

### (If needed) Download model weights
```bash
python download_weights.py
```
This will download weights to model_weights/model_weights.pth.
The same weights are also downloaded by demo if missing

## Data preparation

Model expects to have all data in `data/` folder. 
Supported audio formats are `.wav` and `.flac`

Example structure:

```text
data/
├── train-clean-100/
│   ├── sample_001.wav
│   └── sample_002.flac
└── test-clean/
    └── sample_003.wav
```

## Configs
All the configs are in `src/configs/` folder. When you add a dataset - make sure to write the corresponding name in the config

## Training
Default training config is `src/configs/soundstream.yaml`

To train model run
```bash
python train.py 
```
You can also easily manage your own hyperparameters from bash:
```bash
python train.py trainer.n_epochs=20
```

Checkpoints are saved according to the config to `saved` folder. You can also change it from bash if you like:
```bash
python train.py \
  trainer.save_dir="/content/drive/MyDrive/audio_codec_checkpoints" \
 ```

## Inference

If weights are not downloaded the script will do it automatically. 

To run inference:
```bash
python inference.py
```

The inference metrics config is `src/configs/metrics/report_metrics.yaml`

## Demo

You can try the model yourself

Download `DEMO.ipynb` - open it in an empty Colab and follow the instructions. 
You can watch your audio being reconstructed

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
