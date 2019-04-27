import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import argparse
import sys
from keras.layers import merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Softmax
from keras import backend as K
import numpy as np
from keras import initializers
from keras.layers import BatchNormalization
import copy
import keras
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, concatenate
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.layers import Dropout
from keras.layers.merge import Concatenate
from keras import layers
from sklearn.externals import joblib
from tensorflow.python.keras.callbacks import Callback

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
K.set_image_dim_ordering('tf')


def get_options():
    parser = argparse.ArgumentParser(description='UNET for Lung Nodule Detection')

    parser.add_argument('-out_dir', action="store", default='E://data//output_final//',
                        dest="out_dir", type=str)

    parser.add_argument('-epochs', action="store", default=4, dest="epochs", type=int)

    parser.add_argument('-batch_size', action="store", default=2, dest="batch_size", type=int)

    parser.add_argument('-lr', action="store", default=0.001, dest="lr", type=float)
    parser.add_argument('-load_weights', action="store", default=False, dest="load_weights", type=bool)
    parser.add_argument('-filter_width', action="store", default=3, dest="filter_width", type=int)
    parser.add_argument('-stride', action="store", default=3, dest="stride", type=int)
    parser.add_argument('-model_file', action="store", default="", dest="model_file", type=str)  # TODO
    parser.add_argument('-save_prefix', action="store", default="model_",
                        dest="save_prefix", type=str)
    opts = parser.parse_args(args=[])

    return opts


def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    smooth = 0.
    # smooth = 1.
    intersection = K.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)


def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = (target.get_shape()[2] - refer.get_shape()[2]).value
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw / 2), int(cw / 2) + 1
    else:
        cw1, cw2 = int(cw / 2), int(cw / 2)
    # height, the 2nd dimension
    ch = (target.get_shape()[1] - refer.get_shape()[1]).value
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch / 2), int(ch / 2) + 1
    else:
        ch1, ch2 = int(ch / 2), int(ch / 2)

    return (ch1, ch2), (cw1, cw2)



def UNet():

    concat_axis = 3
    inputss = Input((512, 512, 1))
    print(inputss.shape)

    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1')(inputss)
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = Dropout(0.2)(conv4)
    #     pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    #     conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    #     conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    #     up_conv5 = UpSampling2D(size=(2, 2))(conv5)
    #     ch, cw = get_crop_shape(conv4, up_conv5)
    #     crop_conv4 = layers.Cropping2D(cropping=(ch,cw))(conv4)
    #     up6 = layers.concatenate([up_conv5, crop_conv4], axis=concat_axis)
    #     conv6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    #     conv6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up_conv6 = layers.UpSampling2D(size=(2, 2))(conv4)
    ch, cw = get_crop_shape(conv3, up_conv6)
    crop_conv3 = layers.Cropping2D(cropping=(ch, cw))(conv3)
    up7 = layers.concatenate([up_conv6, crop_conv3], axis=concat_axis)
    conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up_conv7 = layers.UpSampling2D(size=(2, 2))(conv7)
    ch, cw = get_crop_shape(conv2, up_conv7)
    crop_conv2 = layers.Cropping2D(cropping=(ch, cw))(conv2)
    up8 = layers.concatenate([up_conv7, crop_conv2], axis=concat_axis)
    conv8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up_conv8 = layers.UpSampling2D(size=(2, 2))(conv8)
    ch, cw = get_crop_shape(conv1, up_conv8)
    crop_conv1 = layers.Cropping2D(cropping=(ch, cw))(conv1)
    up9 = layers.concatenate([up_conv8, crop_conv1], axis=concat_axis)
    conv9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    ch, cw = get_crop_shape(inputss, conv9)
    conv9 = layers.ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(conv9)
    conv10 = layers.Conv2D(1, (1, 1))(conv9)

    print(conv10.shape)


    # conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputss)
    # print(conv1.shape)
    # conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    # print(conv1.shape)
    # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # print(pool1.shape)
    # print('\n')
    #
    # conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    # print(conv2.shape)
    # conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    # print(conv2.shape)
    # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # print(pool2.shape)
    # print('\n')
    #
    # conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    # print(conv3.shape)
    # conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    # print(conv3.shape)
    # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # print(pool3.shape)
    # print('\n')
    #
    # conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    # print(conv4.shape)
    # conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    # print(conv4.shape)
    # drop4 = Dropout(0.5)(conv4)
    # pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    # print(pool4.shape)
    # print('\n')
    # #
    # # conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    # # print(conv5.shape)
    # # conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    # # print(conv5.shape)
    # # drop5 = Dropout(0.5)(conv5)
    # # print('\n')
    # #
    # # up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    # #     UpSampling2D(size=(2, 2))(drop5))
    # # print(up6.shape)
    # # print(drop4.shape)
    # # merge6 = merge([drop4, up6], mode='concat', concat_axis=3)
    # # print('merge: ')
    # # print(merge6.shape)
    # # conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    # # conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    # #
    # up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    #     UpSampling2D(size=(2, 2))(conv4))
    # merge7 = merge([conv3, up7], mode='concat', concat_axis=3)
    # conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    # conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    #
    # up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    #     UpSampling2D(size=(2, 2))(conv7))
    # merge8 = concatenate([conv2, up8], axis=3)
    # conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    # conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    #
    # up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    #     UpSampling2D(size=(2, 2))(conv8))
    # merge9 = concatenate([conv1, up9], axis=3)
    # conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    # conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    # conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    # # conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    # conv10 = Softmax()(conv9)

    print("llllllllllllllllllllllllast")
    Outmodel = Model(inputs=inputss, outputs=conv10)
    Outmodel.summary()
    print(conv10.shape)

    return Outmodel


# def mean_pred(y_true, y_pred):
#     return K.mean(y_pred)

class WeightSave(Callback):
    def __init__(self, options):
        self.options = options

    def on_train_begin(self, logs={}):
        if self.options.load_weights:
            print('LOADING WEIGHTS FROM : ' + self.options.model_file)
            weights = joblib.load( self.options.model_file )
            self.model.set_weights(weights)
    def on_epoch_end(self, epochs, logs = {}):
        cur_weights = self.model.get_weights()
        joblib.dump(cur_weights, self.options.save_prefix + '_script_on_epoch_' + str(epochs) + '_lr_' + str(self.options.lr) + '_WITH_STRIDES_' + str(self.options.stride) +'_FILTER_WIDTH_' + str(self.options.filter_width) + '.weights')

#Callbacks
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
earlystop = EarlyStopping(patience=10)  # when loss don't redunce in next epoch, stop train
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                            patience=2,   # when tried patience's epoch ,do lr deduce
                                            verbose=1,
                                            factor=0.5,  # lr = lr*factor
                                            min_lr=0.00001)
callbacks = [earlystop, learning_rate_reduction]

def train(use_existing):
    print("Loading the options ....")
    options = get_options()
    print("epochs: %d" % options.epochs)
    print("batch_size: %d" % options.batch_size)
    print("filter_width: %d" % options.filter_width)
    print("stride: %d" % options.stride)
    print("learning rate: %f" % options.lr)
    sys.stdout.flush()

    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)
    imgs_train = np.load(options.out_dir + "trainImages.npy").astype(np.float32)
    print(type(imgs_train),'   ',imgs_train.shape)
    imgs_train = np.reshape(imgs_train,(imgs_train.shape[0],imgs_train.shape[2],imgs_train.shape[3],imgs_train.shape[1]))
    print(imgs_train.shape)

    imgs_mask_train = np.load(options.out_dir + "trainMasks.npy").astype(np.float32)
    imgs_mask_train = np.reshape(imgs_mask_train,
                            (imgs_mask_train.shape[0], imgs_mask_train.shape[2], imgs_mask_train.shape[3], imgs_mask_train.shape[1]))
    print(imgs_mask_train.shape)
    # Renormalizing the masks
    imgs_mask_train[imgs_mask_train > 0.] = 1.0

    # Now the Test Data
    imgs_test = np.load(options.out_dir + "testImages.npy").astype(np.float32)
    imgs_test = np.reshape(imgs_test,
                                     (imgs_test.shape[0], imgs_test.shape[2],
                                      imgs_test.shape[3],
                                      imgs_test.shape[1]))
    print(imgs_mask_train.shape)
    imgs_mask_test_true = np.load(options.out_dir + "testMasks.npy").astype(np.float32)
    imgs_mask_test_true = np.reshape(imgs_mask_test_true,
                                 (imgs_mask_test_true.shape[0], imgs_mask_test_true.shape[2], imgs_mask_test_true.shape[3],
                                  imgs_mask_test_true.shape[1]))

    print(imgs_mask_test_true.shape)
    # Renormalizing the test masks
    imgs_mask_test_true[imgs_mask_test_true > 0] = 1.0

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)
    model = UNet()
    # accuracy = mean_pred(imgs_test,imgs_mask_test_true)
    model.compile(optimizer=Adam(lr=options.lr, clipvalue=1., clipnorm=1.), loss=dice_coef_loss, metrics=['accuracy'])
    model.fit(x=imgs_train, y=imgs_mask_train, batch_size=options.batch_size, epochs=options.epochs, verbose=1,validation_data=(imgs_test, imgs_mask_test_true), shuffle=True
            ,callbacks=callbacks)
    keras.callbacks.ModelCheckpoint(
        filepath="E:\\Luna2016-Lung-Nodule-Detection-master\\model\\",
        monitor='accuracy',
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode='auto',
        period=1
    )
    keras.callbacks.CSVLogger("log.csv", separator=',', append=False)
    model.save_weights('mm.h5') # save weight



if __name__ == '__main__':
    model = train(False)
