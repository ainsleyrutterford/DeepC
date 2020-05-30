import argparse
import cv2
import glob
import numpy as np
import math
import os

parser = argparse.ArgumentParser()

parser.add_argument("path", type=str, help="Path of the images and labels to produce the sliding window samples from")
parser.add_argument("coords", nargs="+", type=int, help="Coordinates of the top left and top right of the correctly labelled area (top_x top_y bottom_x bottom_y)")
parser.add_argument("--size", type=int, default=256, help="Size of images to output")
parser.add_argument("--stride", type=int, default=20, help="Stride between the start of the output images")
parser.add_argument("--frames", type=int, default=9, help="Number of frames per chunk")

args = parser.parse_args()

image_names = np.array(sorted(glob.glob(os.path.join(args.path, "*.tif"))))
label_names = np.array(sorted(glob.glob(os.path.join(args.path, "*.png"))))

samples = len(image_names) // args.frames

image_names = image_names[ : samples * args.frames]
image_names = image_names.reshape((samples, args.frames))

label_names = label_names[ : samples * args.frames]
label_names = label_names.reshape((samples, args.frames))

top_x, top_y, bottom_x, bottom_y = args.coords

xs = (bottom_x - args.size - top_x) / args.stride
ys = (bottom_y - args.size - top_y) / args.stride

answer = input(f"{math.floor(xs) * math.floor(ys) * 2 * args.frames * samples} images will be created. Continue? (y/n): ")

if answer != "y":
    print("Exiting...")
    exit()

for s in range(samples):
    for f in range(args.frames):
        image = cv2.imread(image_names[s, f], 0)
        label = cv2.imread(label_names[s, f], 0)
        for x in range(math.floor(xs)):
            for y in range(math.floor(ys)):
                x1 = top_x + x * args.stride
                y1 = top_y + y * args.stride
                cropped_image = image[ y1:y1+args.size , x1:x1+args.size ]
                cv2.imwrite(f"images/{s}_{x}_{y}_{f}.png", cropped_image)
                cropped_label = label[ y1:y1+args.size , x1:x1+args.size ]
                cv2.imwrite(f"labels/{s}_{x}_{y}_{f}.png", cropped_label)