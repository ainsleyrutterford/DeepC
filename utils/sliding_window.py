import argparse
import cv2
import math
import os

def get_label_name(image, suffix):
    fullstop   = image.find(".")
    image_name = image[:fullstop]
    extension  = image[fullstop:]
    return image_name, suffix, extension

parser = argparse.ArgumentParser()

parser.add_argument("image", type=str, help="Prefix of image and label to produce the sliding window samples from")
parser.add_argument("coords", nargs="+", type=int, help="Coordinates of the top left and top right of the correctly labelled area (top_x top_y bottom_x bottom_y)")
parser.add_argument("--size", action="store", dest="size", type=int, default=256, help="Size of images to output")
parser.add_argument("--stride", action="store", dest="stride", type=int, default=20, help="Stride between the start of the output images")
parser.add_argument("--suffix", action="store", dest="suffix", type=str, default="_label", help="Suffix that is appended to the label")

args = parser.parse_args()

image_name, suffix, extension = get_label_name(args.image, args.suffix)

image = cv2.imread(args.image, 0)
label = cv2.imread(image_name + suffix + extension, 0)

top_x, top_y, bottom_x, bottom_y = args.coords

xs = (bottom_x - args.size - top_x) / args.stride
ys = (bottom_y - args.size - top_y) / args.stride

answer = input(f"{math.floor(xs) * math.floor(ys) * 4} images will be created. Continue? (y/n): ")

if answer != "y":
    print("Exiting...")
    exit()

for x in range(math.floor(xs)):
    for y in range(math.floor(ys)):
        x1 = top_x + x * args.stride
        y1 = top_y + y * args.stride
        cropped_image = image[ y1:y1+args.size , x1:x1+args.size ]
        cv2.imwrite(f"{image_name}_{x}_{y}_0_{extension}", cropped_image)
        cropped_image = cv2.rotate(cropped_image, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(f"{image_name}_{x}_{y}_1_{extension}", cropped_image)
        cropped_label = label[ y1:y1+args.size , x1:x1+args.size ]
        cv2.imwrite(f"{image_name}_{x}_{y}_0_{suffix}{extension}", cropped_label)
        cropped_label = cv2.rotate(cropped_label, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(f"{image_name}_{x}_{y}_1_{suffix}{extension}", cropped_label)