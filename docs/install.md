# Step-by-step installation instructions

**a. Create a conda virtual environment and activate it.**
```shell
conda create -n pianomotion python=3.10 -y
conda activate pianomotion
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```

**c. (Optional) Install mamba following the [official instructions](https://github.com/state-spaces/mamba).**
```shell
pip install mamba-ssm
```


**d. Clone PianoMotion10M.**
```
git clone https://github.com/agnJason/PianoMotion10M.git
```

**e. Install other requirements.**
```shell
cd /path/to/PianoMotion10M
pip install -r requirement.txt
```

**f. Prepare MANO models.**

Besides, you also need to download the MANO model. Please visit the [MANO website](https://mano.is.tue.mpg.de/) and register to get access to the downloads section. We only require the right hand model. You need to put MANO_RIGHT.pkl under the ./mano folder.

**g. Prepare pretrained models. (Used in training.)**

Download pretrained HuBert([Base](https://huggingface.co/facebook/hubert-base-ls960)/[Large](https://huggingface.co/facebook/hubert-large-ls960-ft)) and Wav2Vec2.0([Base](https://huggingface.co/facebook/wav2vec2-base-960h)/[Large](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self))  to `/path/to/PianoMotion10M/checkpoints`.

**h. Prepare Gesture Autoencoder model. (Used in evaluation.)**

Download pretrained [Gesture Autoencoder model](https://drive.google.com/file/d/1G2Fe_zlJn8I_U_VGldH4SsIa_KauvG3p/view?usp=sharing) to `/path/to/PianoMotion10M/checkpoints`


```
checkpoints
├── gesture_autoencoder_checkpoint_best.bin
├── hubert-base-ls960/
│   ├── pytorch_model.pth
│   ├── config.json
│   ├── ...
├── hubert-large-ls960-ft/
├── wav2vec2-base-960h/
├── wav2vec2-large-960h-lv60-self/
```
