# SinGAN (cbo remix)

The original SinGAN ([paper](https://arxiv.org/pdf/1905.01164.pdf), [code](https://github.com/tamarott/SinGAN)) is great, but there were some things I wanted to update:
- Update requirements to reflect version requirements (`torch==1.4.0` and `torchvision==0.5.0`)
- Grayscale input support
- yaml config a-la [detectron2](https://github.com/facebookresearch/detectron2)

See [original readme](ORIGINAL_README.md) for more information on the original
project.

**Only harmonisation and painting are supported at the moment**. I have yet to update the other operation modes to reflect the new project layout.

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

Run configuration is taken care of within a config file to create self-contained and repeatable experiment (see [yacs](https://github.com/rbgirshick/yacs) for more defense of this approach). This makes it easier to test behaviour and creates a documentation trail (the idea being you create a new copy as you make changes to the config files).

Each config file must specify at least two things: the 'mode' of operation, and the image to train on (or that was trained on).

```yaml
mode: train
training:
  image: 'starry_night.png'
```

When training, this is sufficient. When performing generation however, there are more things to consider. For all generation techniques, you will need to pass the training date, to tell SinGAN (remix) which version of the trained model to use:

```yaml
mode: 'train'
training:
  image: 'starry_night.png'
  date: '2021-07-30_08-35-00'
```

### Harmonisation

In harmonisation, you need to give SinGAN a reference and a background image. You can also give it a scale to start the generation at (defaults to zero). Dilation radius is used to blur the mask when pasting the harmonised image back - a bigger radius will result in a bigger 'halo' effect around the harmonised object.

```yaml
mode: 'harmonisation'
training:
  image: 'starry_night.png'
  date: '2021-07-30_08-35-00'
harmonisation:
  reference_image: 'starry_night_naive.png'
  background_image: 'starry_night.png'
  start_scale: 6
  dilation_radius: 5
```

### Painting

Painting is where you give it a mask (colour matched e.g. so that trees are green and the sky is blue), and SinGAN performs inverse segmentation and forms a plausible original image. You only need to give SinGAN a reference image (the colour-matched mask) and a scale.

```yaml
mode: 'painting'
training:
  image: 'starry_night.png'
  date: '2021-07-30_08-35-00'
harmonisation:
  reference_image: 'starry_night_naive.png'
  start_scale: 6
```
