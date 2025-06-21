"""
This file contains the image preprocessing functions for the scenario.
The functions include:
1. network_boundary: to crop the image, remove the scale in the lower left corner
2. road_N, road_E, road_S, road_W: to extract the road from the image
3. merge_roads: to merge the road images into a single image
4. resize_image: to resize the image to 256x64
"""


import cv2
import numpy as np

image_file = '../scenario/Junc1_3turn/1095.png'
# read image
image = cv2.imread(image_file)
cv2.imshow('Original', image)
cv2.waitKey(0)
# convert into matrix
image = np.array(image)
print(image.shape)
# crop the image, remove the scale in the lower left corner
image = image[:, 150:, :]


def network_boundary(image_m):
    # road in black >> cannot use real-world mode
    rows_to_save = np.any(image_m == 0, axis=1)
    cols_to_save = np.any(image_m == 0, axis=0)
    rows_to_save = np.where(rows_to_save)[0]
    cols_to_save = np.where(cols_to_save)[0]  # add for square / same length of roads
    return [min(rows_to_save), max(rows_to_save), min(cols_to_save), max(cols_to_save)+1]


boundary = network_boundary(image)
print(boundary)  # [23, 742, 445, 1164]
image = image[boundary[0]:boundary[1], boundary[2]:boundary[3], :]
cv2.imshow('Cropped', image)
cv2.waitKey(0)
print(image.shape)

# --- only road ---

# each inc+out road from camera
road_N = image[0:303, 317:400, :]
print(road_N.shape)
cv2.imshow('road_N', road_N)
cv2.waitKey(0)

road_E = image[318:401, 416:, :]
print(road_E.shape)
cv2.imshow('road_E', road_E)
cv2.waitKey(0)
# rotate, 270'
road_E_rotated = cv2.flip(cv2.transpose(road_E), 0)
print(road_E_rotated.shape)
cv2.imshow('road_E_rotated', road_E_rotated)
cv2.waitKey(0)

road_S = image[416:, 318:401, :]
print(road_S.shape)
cv2.imshow('road_S', road_S)
cv2.waitKey(0)
# # rotate, 180'
road_S_rotated = cv2.flip(road_S, -1)
print(road_S_rotated.shape)
cv2.imshow('road_S_rotated', road_S_rotated)
cv2.waitKey(0)

road_W = image[318:401, 0:303, :]
print(road_W.shape)
cv2.imshow('road_W', road_W)
cv2.waitKey(0)
# rotate, 90'
road_W_rotated = cv2.flip(cv2.transpose(road_W), 1)
print(road_W_rotated.shape)
cv2.imshow('road_W_rotated', road_W_rotated)
cv2.waitKey(0)

# merge
images = [road_N, road_E_rotated, road_S_rotated, road_W_rotated]
merge_image = np.hstack(images)
print(merge_image.shape)
cv2.imshow('Merged', merge_image)  # (303, 332, 3)
cv2.waitKey(0)

# resize
image = cv2.resize(merge_image, (256, 64), interpolation=cv2.INTER_AREA)
cv2.imshow('Resize', image)
cv2.waitKey(0)
print(image.shape)
cv2.destroyAllWindows()
