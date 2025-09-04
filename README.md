<h1 align="center">MS-DETR: Towards Effective Video Moment Retrieval and Highlight Detection by Joint Motion-Semantic Learning</h1>

<div align="center">
    <strong>Hongxu Ma</strong><sup> 1,2*</sup>&emsp;
    <strong>Guanshuo Wang</strong><sup> 2*</sup>&emsp;
    <strong>Fufu Yu</strong><sup> 2</sup>&emsp;
    <strong>Qiong Jia</strong><sup> 2</sup>&emsp;
    <strong>Shouhong Ding</strong><sup> 2†</sup>&emsp;
</div>


<div align='center'>
    <sup>1 </sup>Fudan University&emsp; <sup>2 </sup>Tencent Youtu Lab&emsp; 
</div>
<div align='center'>
    <small><sup>*</sup> Both authors contributed equally to this research</small>
    <small><sup>†</sup> Corresponding author</small>
</div>
<div align='center'>
    <a href="https://arxiv.org/abs/2507.12062">Paper (arXiv:2507.12062)</a>
</div>

## Updates & News
### 2025-07-16 Initial Code Release (v0.1.0)
- Includes core training/inference code for MR/HD, evaluation utilities, and example scripts.
- Supports modular ablations via command-line args; see [ms_detr/config.py](ms_detr/config.py) for argument descriptions.
- Coming soon: auxiliary corpus generation code and visualization toolkit (released progressively).
- Paper: [arXiv:2507.12062](https://arxiv.org/abs/2507.12062)
- 
## Prerequisites

We follow the environment requirements of the QD-DETR baseline. Please refer to their repo for background and dataset preparation details: [QD-DETR (CVPR'23)](https://github.com/wjun0830/QD-DETR).

- Tested OS: Linux/macOS
- Python: 3.7
- PyTorch: 1.9.0 (CUDA 11.1 recommended) and matching torchvision
- GPU: NVIDIA GPU with CUDA support (recommended)

### Setup
1. Create a clean Python 3.7 environment (conda example):
```bash
conda create -n msdetr python=3.7 -y
conda activate msdetr
```
2. Install PyTorch 1.9.0 (choose the CUDA/CPU build that matches your system):
```bash
# CUDA 11.1 example
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# or CPU-only
# pip install torch==1.9.0 torchvision==0.10.0
```
3. Install project dependencies:
```bash
pip install -U pip wheel setuptools
pip install numpy scipy tqdm tensorboard scikit-learn matplotlib ftfy regex
```
4. Verify installation:
```bash
python -c "import torch; import numpy as np; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

Notes:
- Keep PyTorch/torchvision versions consistent (e.g., torch==1.9.0 with torchvision==0.10.0).
- If using CUDA, ensure your local CUDA driver is compatible with the selected wheel.
 
## QVHighlights

### Training
Note: Update the data paths in `ms_detr/scripts/train.sh` to your local paths. Download the provided data and features (including auxiliary data) from [data & features link](https://drive.google.com/drive/folders/1piSJBx3ia7NqrxQ3w3lGBRfGhn8hIiOP?usp=drive_link). We will progressively release code to generate these JSONs and features.

Training with (only video) and (video + audio) can be executed by running the shell below (following the setup style of [QD-DETR](https://github.com/wjun0830/QD-DETR)):

```bash
# Only video
bash ms_detr/scripts/train.sh --seed 2018

# Video + audio
bash ms_detr/scripts/train_audio.sh --seed 2018
```

To calculate the standard deviation reported in the paper, we ran with 5 different seeds: `0, 1, 2, 3, 2018` (2018 is the seed used in Moment-DETR).

### Inference Evaluation and Codalab Submission for QVHighlights
Once the model is trained, `hl_val_submission.jsonl` and `hl_test_submission.jsonl` can be produced by running:

```bash
bash ms_detr/scripts/inference.sh results/{direc}/model_best.ckpt 'val'
bash ms_detr/scripts/inference.sh results/{direc}/model_best.ckpt 'test'
```

where `{direc}` is the path to the saved checkpoint directory. For more details on submission, please check [standalone_eval/README.md](standalone_eval/README.md).

### Roadmap
We will progressively release training and evaluation code for additional datasets.
 
## Citation 💖

If you find MS-DETR useful for your project or research, welcome to 🌟 this repo and cite our work using the following BibTeX:
```bibtex
@article{ma2025msdetreffectivevideomoment,
  author  = {Ma, Hongxu and Wang, Guanshuo and Yu, Fufu and Jia, Qiong and Ding, Shouhong},
  title   = {MS-DETR: Towards Effective Video Moment Retrieval and Highlight Detection by Joint Motion-Semantic Learning},
  journal = {arXiv preprint arXiv:2507.12062},
  year    = {2025}
}
```
