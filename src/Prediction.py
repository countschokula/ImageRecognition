import numpy as np
import src.SVMHandler as Svm
from src.TrainingsDataHandler import label_to_cent
import src.Preprocession


def calculate_prediction_img(img, patch_size, svm_estimator, train_data, preprossecion, svm_handler):
    step_size = np.floor(patch_size/2).astype('int')
    prediction_img = np.zeros(img.shape)
    for i in range(0, img.shape[0]-patch_size, step_size):
        for j in range(0, img.shape[1]-patch_size, step_size):
            patch = img[i:i+patch_size, j:j+patch_size, :3]
            print(patch.size)
            feature = preprossecion.preprocess(patch)
            label = svm_estimator.predict(svm_handler.kernel_method(feature, train_data[:, 1:]).reshape((1, -1)))
            print(label)
            prediction = label_to_cent(label)
            # prediction_img[prediction_img[i:i+patch_size, j:j+patch_size] > 0]\
            #     = (prediction_img[prediction_img[i:i+patch_size, j:j+patch_size] > 0] + prediction)/2
            # prediction_img[prediction_img[i:i + patch_size, j:j + patch_size] == 0]\
            #     = prediction

            prediction_img[i:i + patch_size, j:j + patch_size] = prediction

    return prediction_img


def most_likely_waldo_coordinates(probability_image, patch_size):
    step_size = np.floor(patch_size / 2).astype('int')
    coordinates = (0, 0)
    max_value = 0
    for i in range(0, probability_image.shape[0] - step_size, step_size):
        for j in range(0, probability_image.shape[1] - step_size, step_size):
            current_value = np.sum(probability_image[i:i+patch_size, j:j+patch_size])
            if current_value > max_value:
                coordinates = (i, j)
    return coordinates
