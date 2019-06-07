from keras.preprocessing.image import img_to_array, load_img
import os
import glob
import numpy as np
import math
from keras.datasets import mnist


def dataset_load():
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_train = X_train.reshape((X_train.shape[0], 1) + X_train.shape[1:])
    X_train = X_train.reshape(X_train.shape[0],
                                                X_train.shape[2],
                                                X_train.shape[3],
                                                X_train.shape[1])
    print(X_train.shape)
    return X_train
    """
    #mnistから変更

    image_datas = []
    input_dirname = os.path.join('images/dataset2', '*')
    files = glob.glob(input_dirname)
    for picture in files:
        img = load_img(picture, target_size=(256, 256))
        imgArray = img_to_array(img)
        image_datas.append(imgArray)

    return np.array(image_datas)



def combine_images(generated_images):
    image_count = generated_images.shape[0]
    width_count = int(math.sqrt(image_count))
    height_count = int(math.ceil(float(image_count) / width_count))
    width_shape = generated_images.shape[1]
    height_shape = generated_images.shape[2]
    combine_image = np.zeros((height_shape*height_count, width_shape*width_count, 3), dtype=generated_images.dtype)

    index_count=0
    for j in range(height_count):
        for i in range(width_count):
            if image_count<=index_count:
                return combine_image
            combine_image[j*height_shape:(j+1)*height_shape, i*width_shape:(i+1)*width_shape, :] = \
                generated_images[index_count]
            index_count += 1

    return combine_image
