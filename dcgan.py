from keras.models import Sequential
from keras.layers import Dense, Reshape, Dropout
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD, Adam, RMSprop
from data_io import dataset_load, combine_images
import numpy as np
from PIL import Image
import os
import datetime


def generator_model():
    """
    model = Sequential()
    model.add(Dense(1024, input_shape=(100, ), activation="tanh"))
    model.add(Dense(128 * 7 * 7))
    model.add(BatchNormalization())
    model.add(Activation("tanh"))
    model.add(Reshape((7, 7, 128), input_shape=(7 * 7 * 128,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5),
                     padding="same",
                     activation="tanh",
                     data_format="channels_last"))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, (5, 5),
                     padding="same",
                     activation="tanh",
                     data_format="channels_last"))
    return model
    """
    #mnistから変更

    model = Sequential()

    model.add(Dense(16*16*256, input_shape=(500, )))
    model.add(BatchNormalization())
    model.add(Activation("tanh"))

    model.add(Reshape((16, 16, 256), input_shape=(16*16*256,)))
    #model.add(Dropout((0.5)))

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(128, (5, 5),
                     padding="same",
                     activation="tanh",
                     data_format="channels_last"))

    #model.add(Dropout((0.5)))

    model.add(UpSampling2D(size=(2, 2)))

    model.add(Conv2D(64, (5, 5),
                     padding="same",
                     activation="tanh",
                     data_format="channels_last"))

    #model.add(Dropout((0.5)))

    model.add(UpSampling2D(size=(2, 2)))

    model.add(Conv2D(32, (5, 5),
                     padding="same",
                     activation="tanh",
                     data_format="channels_last"))

    #model.add(Dropout((0.5)))

    model.add(UpSampling2D(size=(2, 2)))

    model.add(Conv2D(3, (5, 5),
                     padding="same",
                     activation="tanh",
                     data_format="channels_last"))

    return model


def discriminator_model():
    """
    model = Sequential()
    model.add(Conv2D(64, (5, 5),
                     padding="same",
                     input_shape=(28, 28, 1),
                     activation="tanh",
                     data_format="channels_last"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5),
                     activation="tanh",
                     data_format="channels_last"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024, activation="tanh"))
    model.add(Dense(1, activation="sigmoid"))
    return model
    """
    #mnistから変更
    model = Sequential()
    model.add(Conv2D(32, (5, 5),
                     padding="same",
                     input_shape=(256, 256, 3),
                     data_format="channels_last"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout((0.3)))
    model.add(AveragePooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (5, 5),
                     padding="same",
                     data_format="channels_last"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout((0.3)))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5),
                     data_format="channels_last"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (5, 5),
                     data_format="channels_last"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout((0.3)))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation="tanh"))
    model.add(Dropout((0.3)))
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
    #mnistから変更
    BATCH_SIZE = 25 * 2
    half_batch = int(BATCH_SIZE/2)
    epoch_count = 500000


    X_train = dataset_load()
    #mnistから変更
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5

    generator = generator_model()
    discriminator = discriminator_model()
    discriminator_on_generator = generator_containing_discriminator(generator, discriminator)

    g_optim = SGD(lr=0.0001, momentum=0.8, nesterov=True)
    d_optim = SGD(lr=0.0001, momentum=0.8, nesterov=True)
    #g_optim = SGD()
    #d_optim = SGD()

    generator.compile(loss="binary_crossentropy", optimizer=g_optim)
    discriminator.trainable = True
    discriminator.compile(loss="binary_crossentropy", optimizer=d_optim)
    discriminator_on_generator.compile(
        loss="binary_crossentropy", optimizer=g_optim)

    for epoch in range(epoch_count):
        epoch += 1
        print("Epoch is", epoch)

        idx = np.random.randint(0, X_train.shape[0], half_batch)
        real_imgs = X_train[idx]
        #mnistから変更
        noise = np.random.uniform(-1, 1, (half_batch, 500))
        generated_images = generator.predict(noise)

        if epoch % 100 == 0:
            folder_path = "images/output/" + folder_name + "/"
            if not os.path.isdir(folder_path):
                os.makedirs(folder_path)

            image = combine_images(generated_images)
            image = image*127.5+127.5
            Image.fromarray(image.astype(np.uint8)).save(folder_path + str(epoch)+".png")


        #d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((half_batch, 1)))
        #d_loss_fake = discriminator.train_on_batch(generated_images, np.zeros((half_batch, 1)))
        d_loss_real = discriminator.train_on_batch(real_imgs, np.random.uniform(0.7, 1.2, half_batch))
        d_loss_fake = discriminator.train_on_batch(generated_images, np.random.uniform(0, 0.3, half_batch))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        #mnistから変更
        noise = np.random.uniform(-1, 1, (BATCH_SIZE, 500))
        g_loss = discriminator_on_generator.train_on_batch(noise, np.ones((BATCH_SIZE, 1)))
        print("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss, g_loss))

        if epoch % 100 == 0:
            folder_path = 'weight/' + folder_name + "/"
            if not os.path.isdir(folder_path):
                os.makedirs(folder_path)

            #generator.save_weights(folder_path + 'generatorWeight.h5')
            #discriminator.save_weights(folder_path + 'discriminatorWeight.h5')
            generator.save(folder_path + 'generator.h5')
            discriminator.save(folder_path + 'discriminator.h5')


if __name__ == "__main__":
    train()
