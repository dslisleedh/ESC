# ESC-arbitrary-scale

## Installation

```
conda create -n lte python=3.10
conda activate esclte
pip3 install torch torchvision torchaudio
pip install tqdm imageio tensorboardX einops timm
```

## Training

```bash
python train.py --config configs/train-div2k/train_esc-lte.yaml --gpu 0,1,2,3
```

## Testing

<b>Test on DIV2K</b>

```bash
source scripts/test-div2k.sh esc_lte.pth 0   # model_path gpu_id
```

<b>Test on other datasets</b>

```bash
source scripts/test-benchmark.sh esc_lte.pth 0   # model_path gpu_id
```

## Acknowledgements

This code is built on [LIIF](https://github.com/yinboc/liif), [SwinIR](https://github.com/JingyunLiang/SwinIR), and [LTE](https://github.com/jaewon-lee-b/lte). 
We thank the authors for sharing their codes.

## License
Following previous works, only the ESC-arb codes are released under the BSD 3-Clause License.