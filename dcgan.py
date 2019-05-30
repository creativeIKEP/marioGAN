from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from data_io import dataset_load, combine_images
import numpy as np
from PIL import Image
import os
import datetime


def generator_model():
    model = Sequential()
    #model.add(Dense(1024, input_shape=(100, ), activation="tanh"))

    model.add(Dense(16*16*256, input_shape=(1000, )))
    model.add(BatchNormalization())
    model.add(Activation("tanh"))

    model.add(Reshape((16, 16, 256), input_shape=(16*16*256,)))

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(128, (5, 5),
                     padding="same",
                     activation="tanh",
                     data_format="channels_last"))

    model.add(UpSampling2D(size=(2, 2)))

    model.add(Conv2D(64, (5, 5),
                     padding="same",
                     activation="tanh",
                     data_format="channels_last"))
    model.add(UpSampling2D(size=(2, 2)))

    model.add(Conv2D(32, (5, 5),
                     padding="same",
                     activation="tanh",
                     data_format="channels_last"))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(3, (5, 5),
                     padding="same",
                     activation="tanh",
                     data_format="channels_last"))
    return model


def discriminator_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5),
                     padding="same",
                     input_shape=(256, 256, 3),
                     activation="tanh",
                     data_format="channels_last"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (5, 5),
                     activation="tanh",
                     data_format="channels_last"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5),
                     activation="tanh",
                     data_format="channels_last"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (5, 5),
                     activation="tanh",
                     data_format="channels_last"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024, activation="tanh"))
    model.add(Dense(1, activation="sigmoid"))
    return model


def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model


def train():
    now_time = datetime.datetime.now()
    folder_name = "{0:%Y-%m-%d_%H-%M}".format(now_time)
    BATCH_SIZE = 50
    half_batch = int(BATCH_SIZE/2)
    epoch_count = 50000


    X_train = dataset_load()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5

    generator = generator_model()
    discriminator = discriminator_model()
    discriminator_on_generator = generator_containing_discriminator(generator, discriminator)

    g_optim = SGD(lr=0.0001, momentum=0.8, nesterov=True)
    d_optim = SGD(lr=0.0001, momentum=0.8, nesterov=True)

    generator.compile(loss="binary_crossentropy", optimizer="SGD")
    discriminator.trainable = True
    discriminator.compile(loss="binary_crossentropy", optimizer=d_optim)
    discriminator_on_generator.compile(
        loss="binary_crossentropy", optimizer=g_optim)

    for epoch in range(epoch_count):
        print("Epoch is", epoch)

        idx = np.random.randint(0, X_train.shape[0], half_batch)
        real_imgs = X_train[idx]
        noise = np.random.uniform(-1, 1, (half_batch, 1000))
        generated_images = generator.predict(noise)

        if epoch % 100 == 0:
            folder_path = "images/output/" + folder_name + "/"
            if not os.path.isdir(folder_path):
                os.makedirs(folder_path)

            image = combine_images(generated_images)
            image = image*127.5+127.5
            Image.fromarray(image.astype(np.uint8)).save(folder_path + str(epoch)+".png")


        d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(generated_images, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.uniform(-1, 1, (BATCH_SIZE, 1000))
        g_loss = discriminator_on_generator.train_on_batch(noise, np.ones((BATCH_SIZE, 1)))
        print("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss, g_loss))

        if epoch % 100 == 0:
            folder_path = 'weight/' + folder_name + "/"
            if not os.path.isdir(folder_path):
                os.makedirs(folder_path)

            generator.save_weights(folder_path + 'generatorWeight.h5')
            discriminator.save_weights(folder_path + 'discriminatorWeight.h5')


if __name__ == "__main__":
    train()
