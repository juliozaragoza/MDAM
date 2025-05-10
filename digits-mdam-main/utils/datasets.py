import scipy.io.matlab as matlab
from matplotlib import pyplot as plt
import numpy as np
import cv2
import math
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from collections import defaultdict


def load_dataset(path="BigDigits.mat"):
    """Loads dataset"""
    mat_file = path
    mat = matlab.loadmat(mat_file, squeeze_me=True)
    data = mat["data"]
    labels = mat["labs"] - 1
    new_labels = [0 if label == 0 else 1 for label in labels]
    return data, new_labels


def tts(data, labels, train_quantity, random_state=None, under_sampling=False): #under_sampling es para cuando haya más muestras de una clase que de otra
    """Train-Test Split"""#el random state es una semilla para que el split sea reproducible

    train_size = train_quantity / len(labels)

    if random_state:
        np.random.seed(random_state)

    if under_sampling:
        rus = RandomUnderSampler(sampling_strategy="majority")
        X_res, y_res = rus.fit_resample(data, labels)
        X_train, _X_test, y_train, _y_test = train_test_split(
            X_res, y_res, train_size=train_size * 5
        )
        return X_train, data, y_train, labels

    return train_test_split(data, labels, train_size=train_size)


def show_digit(digit, label):
    """Shows a digit given the index in the dataset and its label"""
    # Reshape the digit data to its original 28x28 shape
    digit_image = np.reshape(digit, (28, 28)).T
    # Display the digit
    plt.figure(figsize=(3, 3))
    plt.imshow(digit_image, cmap=plt.cm.gray_r)
    plt.xticks([])
    plt.yticks([])
    plt.title(f'Label: {label} ({"only zeros" if label == 0 else "other numbers"})')
    plt.show()


def show_digits(digits, labels):
    """Shows a grid of digits given a list of digits and their labels"""
    # Calculate the grid size: more columns than rows
    grid_size = int(math.ceil(math.sqrt(len(digits))))
    nrows = grid_size
    ncols = grid_size

    # Create subplots
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))

    for i, ax in enumerate(axs.flat):
        if i < len(digits):
            # Reshape the digit data to its original 28x28 shape
            digit_image = np.reshape(digits[i], (28, 28)).T
            # Display the digit
            ax.imshow(digit_image, cmap=plt.cm.gray_r)
            ax.set_title(
                f'Label: {labels[i]} ({"only zeros" if labels[i] == 0 else "other numbers"})'
            )
        else:
            ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def process_new_image(image_path, label):
    """Loads a new image and processes it to be like the ones in the dataset"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_CUBIC).T
    image = cv2.normalize(image, None, 0, 254, cv2.NORM_MINMAX)
    # Flip the colors
    image = 255 - image
    image = np.reshape(image, (784,))
    return image, label


def find_common_data(models, X_test, y_test):
    """Finds common wrong data predictions between models"""

    def get_incorrect_predictions(model, X_test, y_test):
        """Returns the incorrect predictions and their corresponding labels"""
        predictions = model.predict(X_test)
        incorrect_indices = np.where(predictions != y_test)[0]
        return X_test[incorrect_indices], predictions[incorrect_indices]

    incorrect_predictions = []
    for model in models:
        incorrect_predictions.append(get_incorrect_predictions(model, X_test, y_test))

    common_data = defaultdict(list)
    for x, y in zip(*incorrect_predictions[0]):
        for other_x in incorrect_predictions[1][0]:
            if np.array_equal(x, other_x):
                common_data["x"].append(x)
                common_data["y"].append(y)

    return common_data["x"], common_data["y"]
