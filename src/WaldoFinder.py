import numpy as np
import matplotlib.pyplot as plt
import TrainingsDataHandler as Trainer
import SVMHandler as Svm
import Prediction as prob
import os

# Possible Open Items:
# - Scaling
# - Patch Size
# - Progress Print


def where_is_waldo_in_this_image(img, patch_size, samples_per_img):
    data_train = Trainer.get_trainings_data(patch_size, samples_per_img)
    print('- Samples calculated')
    svm_estimator = Svm.grid_cv_optimized_svm(data_train)
    print('- Svm Trained')
    prob_img = prob.calculate_prediction_img(img, patch_size, svm_estimator, data_train)
    print(' - Probabilites calculated')
    plt.imshow(prob_img)
    plt.show()
    return prob.most_likely_waldo_coordinates(prob_img, patch_size)


image = plt.imread(os.path.join(os.path.abspath('..'), 'data', 'test', '01.jpg')).astype(np.float32)
print(where_is_waldo_in_this_image(image, 50, 200))
