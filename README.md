# SinGAN (cbo remix)

The original SinGAN ([paper](https://arxiv.org/pdf/1905.01164.pdf), [code](https://github.com/tamarott/SinGAN)) is great, but there were some things I wanted to update:
- Update requirements to reflect version requirements (`torch==1.4.0` and `torchvision==0.5.0`)
- Grayscale input support
- yaml config a-la [detectron2](https://github.com/facebookresearch/detectron2)

See [original readme](ORIGINAL_README.md) for more information on the original
project.

**Only harmonisation is supported at the moment**. I have yet to update the other operation modes to reflect the new project layout.

## Installation

```bash
git clone https://github.com/cbosoft/SinGAN
cd SinGAN
pip install -r requirements.txt
```

## Usage

Runs are configured by yaml config file. Example config files are in [configs](configs). The [config script](SinGAN/config.py) gives the default values for config values.

If you have a config file, running is very easy:

```bash
python run.py path/to/config.yaml
```

For example, to run the sample training task:

```bash
python run.py configs/sample_train.yaml
```

then, to run the harmonisation task:

```bash
python run.py configs/sample_harmonisation.yaml
```

## Config files

Run configuration is taken care of within a config file to create self-contained and repeatable experiment. This makes it easier to test behaviour and creates a documentation trail (the idea being you copy and edit the config files as you make changes).

Each config file must specify at least two things: the 'mode' of operation, and the image to train on (or that was trained on).

```yaml
mode: train
training:
  image: 'starry_night.png'
```

When training, this is sufficient. When performing generation however, there are more things to consider. For all generation techniques, you will need to pass the training date, to tell SinGAN (remix) which version of the trained model to use:

```yaml
mode: train
training:
  image: 'starry_night.png'
  date: '2021-07-30_08-35-00'
```

### Harmonisation

In harmonisation, you need to give SinGAN a reference and a background image. You can also give it a scale to start the generation at (defaults to zero).

```yaml
mode: train
training:
  image: 'starry_night.png'
  date: '2021-07-30_08-35-00'
harmonisation:
  reference_image: 'starry_night_naive.png'
  background_image: 'starry_night.png'
  start_scale: 6
  dilation_radius: 5
```
