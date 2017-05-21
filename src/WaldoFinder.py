import numpy as np
import matplotlib.pyplot as plt
import src.TrainingsDataHandler as Trainer
import src.SVMHandler as Svm
import os


def where_is_waldo_in_this_image(img, patch_size, samples_per_img):
    data_train = Trainer.get_trainings_data(patch_size, samples_per_img)
    svm_estimator = Svm.grid_cv_optimized_svm(data_train)
    prob_img = calculate_prediction_img(img, patch_size, svm_estimator)
    plt.imshow(prob_img)
    plt.show()
    return most_likely_waldo_coordinates(prob_img, patch_size)


def calculate_prediction_img(img, patch_size, svm_estimator):
    step_size = np.floor(patch_size/2).astype('int')
    prediction_img = np.zeros(img.shape)
    for i in range(0, img.shape[0]-step_size, step_size):
        for j in range(0, img.shape[1]-step_size, step_size):
            patch = img[i:i+patch_size, j:j+patch_size, :]
            label = svm_estimator.predict(patch.flatten().reshape((1, -1)))
            prediction = Trainer.label_to_cent(label)

            prediction_img[prediction_img[i:i+patch_size, j:j+patch_size] > 0]\
                = (prediction_img[prediction_img[i:i+patch_size, j:j+patch_size] > 0] + prediction)/2
            prediction_img[prediction_img[i:i + patch_size, j:j + patch_size] == 0]\
                = prediction

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


image = plt.imread(os.path.join(os.path.abspath('..'), 'data', 'test', '01.jpg')).astype(np.float32)
print(where_is_waldo_in_this_image(image, 10, 1000))
