# CNN architectures for Coral Density Analysis

Simple implementations of 2D SegNet (based off of [Yuta Kamikawa's](https://github.com/ykamikawa/tf-keras-SegNet)), 2D U-Net (based off of [zhixuhao's](https://github.com/zhixuhao/unet)), and 3D U-Net architectures in Keras.

## Prerequisits

- `numpy<1.17`
- `tensorflow==1.8`
- `keras==2.2.4`

These can be installed by running `pip install -r requirements.txt`.

If you plan on using a GPU to train, `tensorflow-gpu==1.8` is also required.

## Usage

```bash
python train.py
```
