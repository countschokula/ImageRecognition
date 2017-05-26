import numpy as np
import matplotlib.pyplot as plt
import src.TrainingsDataHandler as Trainer
import src.SVMHandler as Svm
import src.Prediction as prob
import src.Preprocession as prep
import os

# Possible Open Items:
# - Scaling
# - Patch Size


def where_is_waldo_in_this_image(img, patch_size, samples_per_img , preprocession='None', kernel='rbf'):
    method = 'average'
    kernel = 'rbf'

    preprocession = prep.Preprocession(method, patch_size)
    svm = Svm.SVMHandler(kernel)

    data_train = Trainer.get_trainings_data(patch_size, preprocession, samples_per_img)
    print('- Samples calculated')
    svm_estimator = svm.grid_cv_optimized_svm(data_train)
    print('- Svm Trained')
    prob_img = prob.calculate_prediction_img(img, patch_size, svm_estimator, data_train, preprocession, svm)
    print(' - Probabilites calculated')
    plt.imshow(prob_img)
    plt.show()
    return prob.most_likely_waldo_coordinates(prob_img, patch_size)


image = plt.imread(os.path.join(os.path.abspath('..'), 'data', 'test', '01.jpg')).astype(np.float32)
print(where_is_waldo_in_this_image(image, 50, 200, 'average', 'rbf'))
