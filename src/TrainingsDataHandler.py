import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import random as rnd

# Possible Open Items:
# - Scaling


def get_trainings_data(patch_size, samples_per_img, path=os.path.join(os.path.abspath('..'), 'data')):
    ground, images = load_images(path)
    trainings_data = create_random_features(ground, images, patch_size, samples_per_img)
    return trainings_data


def create_random_features(ground_images, images, patch_size, samples_per_img):
    # dim ground truth same as dim img?
    train = np.zeros((len(images)*samples_per_img, ((patch_size**2)*3)+1))
    count = 0
    for i in range(len(images)):
        img = images[i]
        ground = ground_images[i]
        for j in range(0, samples_per_img):
            rand_x = rnd.randrange(0, img.shape[0]-patch_size)
            rand_y = rnd.randrange(0, img.shape[1]-patch_size)
            feature_patch = img[rand_x:rand_x + patch_size, rand_y:rand_y + patch_size, :]
            label_patch = ground[rand_x:rand_x + patch_size, rand_y:rand_y + patch_size]
            cent = np.mean(label_patch)
            label = cent_to_label(cent)
            train[count, 1:] = feature_patch[:, :, :3].flatten()
            train[count, 0] = label
            count += 1
    return train


def label_to_cent(label):

    if label == 0:
        cent = 0
    elif label == 1:
        cent = 0.25
    elif label == 2:
        cent = 0.25
    elif label == 3:
        cent = 0.5
    elif label == 4:
        cent = 0.75
    else:
        cent = 1
    return cent


def cent_to_label(cent):

    if cent == 0:
        label = 0
    elif 0 < cent <= 0.25:
        label = 1
    elif 0.25 < cent <= 0.5:
        label = 2
    elif 0.5 < cent < 0.75:
        label = 3
    elif 0.75 <= cent < 1:
        label = 4
    else:
        label = 5
    return label


def load_images(path=os.path.join(os.path.abspath('..'), 'data')):
    # default path is working directory/data. ground_truth and images need to be in dame directory.

    ground_truths = sorted(glob.glob(os.path.join(path, 'ground_truths', '*.png')))
    images = sorted(glob.glob(os.path.join(path, 'images', '*.jpg')))
    ground_images = []
    waldo_images = []

    # every img has a ground_truth
    for i in range(0, len(images)):
        ground_images.append(plt.imread(ground_truths[i]).astype(np.float32))
        waldo_images.append(plt.imread(images[i]).astype(np.float32))

    return ground_images, waldo_images

# img = plt.imread(os.path.join(os.path.abspath('..'), 'data', 'ground_truths', '01.png')).astype(np.float32)
# print(np.max(img))
# print((np.min(img)))
