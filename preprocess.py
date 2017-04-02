import cv2
import numpy as np
from keras.preprocessing.image import *


# Flip image horizontally, flipping the angle positive/negative
def horizontal_flip(image, steering_angle):
    flipped_image = cv2.flip(image, 1)
    steering_angle = -steering_angle
    return flipped_image, steering_angle


# Shift width/height of the image by a small fraction of the total value, introducing an small angle change
def height_width_shift(image, steering_angle, width_shift_range=50.0, height_shift_range=5.0):
    # translation
    tx = width_shift_range * np.random.uniform() - width_shift_range / 2
    ty = height_shift_range * np.random.uniform() - height_shift_range / 2

    # new steering angle
    steering_angle += tx / width_shift_range * 2 * 0.2

    transform_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    rows, cols, channels = image.shape

    translated_image = cv2.warpAffine(image, transform_matrix, (cols, rows))
    return translated_image, steering_angle


# Increase the brightness by a certain value or randomly
def brightness_shift(image, bright_increase=None):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    if bright_increase:
        image_hsv[:, :, 2] += bright_increase
    else:
        bright_increase = int(30 * np.random.uniform(-0.3, 1))
        image_hsv[:, :, 2] = image[:, :, 2] + bright_increase

    image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
    return image


# Shift range for each channels
def channel_shift(image, intensity=30, channel_axis=2):
    image = random_channel_shift(image, intensity, channel_axis)
    return image


# Rotate the image randomly up to a range_degrees
def rotation(image, range_degrees=5.0):
    # image = random_rotation(image, range_degrees)
    degrees = np.random.uniform(-range_degrees, range_degrees)
    rows, cols = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), degrees, 1.0)
    image = cv2.warpAffine(image, matrix, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
    return image


# Zoom the image randomly up to zoom_range, where 1.0 means no zoom and 1.2 a 20% zoom
def zoom(image, zoom_range=(1.0, 1.2)):
    # image = random_zoom(image, zoom_range)
    # resize
    factor = np.random.uniform(zoom_range[0], zoom_range[1])
    height, width = image.shape[:2]
    new_height, new_width = int(height * factor), int(width * factor)
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # crop margins to match the initial size
    start_row = int((new_height - height) / 2)
    start_col = int((new_width - width) / 2)
    image = image[start_row:start_row + height, start_col:start_col + width]

    return image


# Crop and resize the image
def crop_resize_image(image, cols=160, rows=320, top_crop_perc=0.1, bottom_crop_perc=0.2):
    height, width = image.shape[:2]

    # crop top and bottom
    top_rows = int(height * top_crop_perc)
    bottom_rows = int(height * bottom_crop_perc)
    image = image[top_rows:height - bottom_rows, 0:width]

    # resize to the final sizes even the aspect ratio is destroyed
    image = cv2.resize(image, (cols, rows), interpolation=cv2.INTER_LINEAR)
    return image


# Apply a sequence of random tranformations for a bettwe generalization and to prevent overfitting
def random_transform(image, steering_angle):
    # every second image is flipped horizontally
    if np.random.random() < 0.5:
        image, steering_angle = horizontal_flip(image, steering_angle)

    image, steering_angle = height_width_shift(image, steering_angle)
    image = zoom(image)
    image = rotation(image)
    image = brightness_shift(image)
    image = channel_shift(image)

    return img_to_array(image), steering_angle