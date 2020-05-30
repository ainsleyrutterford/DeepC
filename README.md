# CNN Architectures for Coral Density Analysis

Implementations of 2D SegNet (based off of [Yuta Kamikawa's](https://github.com/ykamikawa/tf-keras-SegNet)), 2D U-Net (based off of [zhixuhao's](https://github.com/zhixuhao/unet)), and 3D U-Net architectures in Keras.

## Prerequisits

- `tensorflow<2`
- `keras==2.2.4`

These can be installed by running `pip install -r requirements.txt`.

If you plan on using a GPU to train, the `tensorflow-gpu` corresponding to the `tensorflow` version used is also required.

## Usage

### Training

```
$ python train.py --help
```

### Testing

```
$ python test.py --help
```

### Assessing the accuracy achieved

```
$ python accuracy.py --help
```

### Predicting an entire image

```
$ python predict.py --help
```