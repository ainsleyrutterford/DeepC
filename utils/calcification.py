import csv
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from skimage import morphology


# A Slice class that holds all of the information relevant to a slice.
class Slice:
    def __init__(self, name, greys, densities, points, box_width, voxel_size):
        self.image = name + ".tif"
        self.label = name + "_label.tif"
        self.skel = name + "_skel.tif"
        self.greys = greys
        self.densities = densities
        self.points = points
        self.box_width = box_width
        self.voxel_size = voxel_size


# Define the Slice objects for each slice in the calcification directory. The greyscale
# and density values were provided by Dr Kenneth Johnson from the Natural History Museum.
RS0030_1 = Slice(name="calcification/RS0030_1_yz",
                 greys=[49355.9941176471, 44104.6882352941, 39821.1352941176, 32732.8294117647, 26064.9705882353, 21192.1588235294],
                 densities=[1.922, 1.773, 1.654, 1.445, 1.266, 1.13],
                 points=[[1140, 164], [1250, 164]],
                 box_width = 40,
                 voxel_size=0.0762383788824081)

RS0030_2 = Slice(name="calcification/RS0030_1_yz",
                 greys=[49355.9941176471, 44104.6882352941, 39821.1352941176, 32732.8294117647, 26064.9705882353, 21192.1588235294],
                 densities=[1.922, 1.773, 1.654, 1.445, 1.266, 1.13],
                 points=[[1061, 326], [1090, 430]],
                 box_width = 40,
                 voxel_size=0.0762383788824081)

RS0030_3 = Slice(name="calcification/RS0030_1_yz",
                 greys=[49355.9941176471, 44104.6882352941, 39821.1352941176, 32732.8294117647, 26064.9705882353, 21192.1588235294],
                 densities=[1.922, 1.773, 1.654, 1.445, 1.266, 1.13],
                 points=[[876, 282], [924, 372]],
                 box_width = 40,
                 voxel_size=0.0762383788824081)

RS0116_1 = Slice(name="calcification/RS0116_0414",
                 greys=[42451.8791946309, 38889.6577181208, 35743.7852348993, 30638.1208053691, 25961.932885906, 23126.5771812081],
                 densities=[1.922, 1.773, 1.654, 1.445, 1.266, 1.13],
                 points=[[1355, 1334], [1437, 1411]],
                 box_width = 40,
                 voxel_size=0.0956196114420891)

RS0116_2 = Slice(name="calcification/RS0116_0414",
                 greys=[42451.8791946309, 38889.6577181208, 35743.7852348993, 30638.1208053691, 25961.932885906, 23126.5771812081],
                 densities=[1.922, 1.773, 1.654, 1.445, 1.266, 1.13],
                 points=[[1280, 1427], [1360, 1520]],
                 box_width = 40,
                 voxel_size=0.0956196114420891)

RS0116_3 = Slice(name="calcification/RS0116_0414",
                 greys=[42451.8791946309, 38889.6577181208, 35743.7852348993, 30638.1208053691, 25961.932885906, 23126.5771812081],
                 densities=[1.922, 1.773, 1.654, 1.445, 1.266, 1.13],
                 points=[[669, 1512], [623, 1606]],
                 box_width = 20,
                 voxel_size=0.0956196114420891)

RS0128_1 = Slice(name="calcification/RS0128_yz_451",
                 greys=[36217.5648854962, 32658.786259542, 29625.2824427481, 24806.0534351145, 20296.3053435114, 17492.6106870229],
                 densities=[1.922, 1.773, 1.654, 1.445, 1.266, 1.13],
                 points=[[630, 346], [648, 244]],
                 box_width = 40,
                 voxel_size=0.103974558413029)

RS0128_2 = Slice(name="calcification/RS0128_yz_451",
                 greys=[36217.5648854962, 32658.786259542, 29625.2824427481, 24806.0534351145, 20296.3053435114, 17492.6106870229],
                 densities=[1.922, 1.773, 1.654, 1.445, 1.266, 1.13],
                 points=[[763, 468], [720, 566]],
                 box_width = 30,
                 voxel_size=0.103974558413029)

RS0128_3 = Slice(name="calcification/RS0128_yz_451",
                 greys=[36217.5648854962, 32658.786259542, 29625.2824427481, 24806.0534351145, 20296.3053435114, 17492.6106870229],
                 densities=[1.922, 1.773, 1.654, 1.445, 1.266, 1.13],
                 points=[[563, 411], [576, 522]],
                 box_width = 40,
                 voxel_size=0.103974558413029)


# Define a simple quadratic equation for the curve_fit() method to use.
def quad(x, a, b, c):
    return a * (x ** 2) + (b * x) + c


# Returns the point half way between the two points provided
def center_point(x, y):
    return [(x[0] + y[0]) / 2, (x[1] + y[1]) / 2]


# Rotate a point around an origin point by a given angle
def rotate(p, origin=(0, 0), angle=0):
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)


# Finds the average distance between two boundaries.
def euclidean(shape, boundaries):
    euclidean = np.zeros(shape)
    for i in range(len(boundaries) - 1):
        for j in boundaries[i]:
            euclidean[j] = np.inf
            for k in boundaries[i + 1]:
                distance = np.linalg.norm(np.array(j) - np.array(k))
                euclidean[j] = min(euclidean[j], distance)
    return euclidean


def calculate_slice_stats(s):
    # Read in image and skeleton
    image = cv.imread(s.image, -1)
    label = cv.imread(s.skel, -1)

    # Use the scipy curve_fit() method to fit the quadratic function to the data.
    params, params_covariance = optimize.curve_fit(quad, s.greys, s.densities)

    # Find the center point
    a, b = s.points
    c = center_point(a, b)

    # Find the vector from the firs point to the second.
    vector = [b[1] - a[1], b[0] - a[0]]

    # Find angle between the line drawn by the two points and the x axis
    angle_r = np.arctan2(*vector)
    angle_d = np.degrees(angle_r)

    # Rotate the image around the central point
    matrix = cv.getRotationMatrix2D(center=tuple(c), angle=angle_d, scale=1)
    rotated_image = cv.warpAffine(src=image, M=matrix, dsize=image.shape[::-1])
    rotated_label = cv.warpAffine(src=label, M=matrix, dsize=label.shape[::-1])

    # Rotate the two points too.
    a_r, b_r = rotate([a, b], origin=c, angle=-angle_r)

    # Define how wide the rectangular area should be.
    box_width = s.box_width // 2

    # Crop the image and the label to the rectangular area.
    cropped_image = rotated_image[int(c[1] - box_width):int(c[1] + box_width) :, int(a_r[0]):int(b_r[0])]
    cropped_label = rotated_label[int(c[1] - box_width):int(c[1] + box_width) :, int(a_r[0]):int(b_r[0])]
    _, cropped_label = cv.threshold(cropped_label, 50, 255, cv.THRESH_BINARY)

    # Find any very small boundaries that should be removed.
    processed = morphology.remove_small_objects(cropped_label.astype(bool), min_size=6, connectivity=2).astype(int)
    # black out pixels
    cropped_label[np.where(processed == 0)] = 0

    c_image_densities = np.zeros(cropped_image.shape)

    for y in range(cropped_image.shape[0]):
        for x in range(cropped_image.shape[1]):
            c_image_densities[y, x] = quad(cropped_image[y, x], *params)

    # Calculate the density error. Do so by finding a mean density for each horizontal
    # line of pixels and then find the standard deviation of these means.
    density_means = np.zeros(box_width * 2)

    for y in range(box_width * 2):
        split = c_image_densities[y, :]
        density_means[y] = np.mean(split)

    density_error = np.std(density_means)
    mean_density = np.mean(density_means)

    # Use the OpenCV connectedComponents() method to label the individual boundaries.
    num_labels, labels_image = cv.connectedComponents(cropped_label)

    for i in range(num_labels):
        if np.sum(labels_image == i) < 15:
            cropped_label[labels_image == i] = 0
            labels_image[labels_image == i] = 0

    labels = []
    yearly_labels = []

    # Sort the boundaries with the growth surface first.
    for x in range(labels_image.shape[1]-1, -1, -1):
        for y in range(labels_image.shape[0]):
            label = labels_image[y, x]
            if cropped_label[y, x] != 0 and label not in labels:
                labels.append(label)

    for i in range(len(labels)):
        if i % 2 == 0:
            yearly_labels.append(labels[i])

    boundaries = {}

    for i in range(len(yearly_labels)):
        boundaries[i] = []

    # Save the boundary pixels into a boundaries dictionary.
    for y in range(labels_image.shape[0]):
        for x in range(labels_image.shape[1]):
            label = labels_image[y, x]
            if label in yearly_labels:
                index = yearly_labels.index(label)
                boundaries[index].append((y, x))

    euclidean_image = euclidean(cropped_label.shape, boundaries)

    raw_distances = []
    # Find the average distance in pixels between each of the boundaries.
    averages = np.zeros(len(yearly_labels) - 1)
    for i in range(len(yearly_labels) - 1):
        for j in boundaries[i]:
            averages[i] += + euclidean_image[j]
            raw_distances.append(euclidean_image[j])
        averages[i] /= len(boundaries[i])

    # Calculate the extension rate standard error
    extension_error = (np.std(raw_distances) / np.sqrt(len(raw_distances))) * s.voxel_size
    # print(raw_distances)

    # Calculate the linear extension rate and the calcification rate.
    linear_extension_mm = np.mean(averages) * s.voxel_size
    calcification = (linear_extension_mm / 10) * mean_density

    # Calculate the calcification rate standard error
    calcification_error = np.sqrt((density_error / mean_density)**2 + (extension_error / linear_extension_mm)**2)

    return (mean_density, density_error,
            linear_extension_mm, extension_error,
            calcification, calcification_error,
            density_means, raw_distances)


slices = [RS0030_1, RS0030_2, RS0030_3,
          RS0116_1, RS0116_2, RS0116_3,
          RS0128_1, RS0128_2, RS0128_3]

stats = []

print("Calculating stats...")

for s in slices:
    stats.append(calculate_slice_stats(s))

print("Stats calculated.")

with open("stats.csv", mode="w") as f:
    writer = csv.writer(f)
    writer.writerow(["Density", "Linear extension", "Calcification"])
    for i in range(len(slices)):
        row = [f"{stats[i][0]:.2f} ± {stats[i][1]:.2f}",
               f"{stats[i][2]:.2f} ± {stats[i][3]:.2f}",
               f"{stats[i][4]:.2f} ± {stats[i][5]:.2f}"]
        writer.writerow(row)

print("stats.csv written.")

with open("raw_data.csv", mode="w") as f:
    writer = csv.writer(f)
    writer.writerow(["Rectangle", "Voxel_size", "Density_Computer", "Length_in_pixels"])
    for i, s in enumerate(stats):
        densities = s[6]
        distances = s[7]
        for j in range(max(len(densities), len(distances))):
            row = [i+1, slices[i].voxel_size]
            if j >= len(densities):
                row.append("")
            else:
                row.append(densities[j])
            if j >= len(distances):
                row.append("")
            else:
                row.append(distances[j])
            writer.writerow(row)

print("raw_data.csv written.")
