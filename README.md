# SinGAN (cbo remix)

The original SinGAN ([paper](https://arxiv.org/pdf/1905.01164.pdf), [code](https://github.com/tamarott/SinGAN)) is great, but there were some things I wanted to update:
- Update requirements to reflect version requirements (`torch==1.4.0` and `torchvision==0.5.0`)
- Grayscale input support
- yaml config a-la [detectron2](https://github.com/facebookresearch/detectron2)

See [original readme](ORIGINAL_README.md) for more information on the original
project.

## Installation

```bash
git clone https://github.com/cbosoft/SinGAN
cd SinGAN
pip install -r requirements.txt
```
