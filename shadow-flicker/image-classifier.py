import numpy as np
from keras.applications import vgg16, models
from keras.models import Model, Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
import json
from keras import optimizers
import keras
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import glob
import os
import logging


class BaseClassifier(object):
    def __init__(self, augment_images=True, load_model=False, epochs=100, batch_size=8):

        self.load_model = load_model
        self.augment_images = augment_images
        self.epochs = epochs
        self.batch_size = batch_size
        self.logger = logging.basicConfig(level=logging.INFO)

        self.generator_augment_count = 50

        self.parent_dir = os.getcwd()

        self.train_dir = os.path.join(self.parent_dir, 'train')
        self.train_pause_dir = os.path.join(self.train_dir, 'Pause')
        self.train_run_dir = os.path.join(self.train_dir, 'Run')

        self.validation_dir = os.path.join(self.parent_dir, 'validation')
        self.validation_pause_dir = os.path.join(self.validation_dir, 'Pause')
        self.validation_run_dir = os.path.join(self.validation_dir, 'Run')

        self.test_path = os.path.join(self.parent_dir, 'test')
        self.image_dim = (150, 150)
        self.input_shape = (150, 150, 3)

        self.train_image_count = len(os.listdir(self.train_pause_dir)) + len(os.listdir(self.train_run_dir))
        self.validation_image_count = len(os.listdir(self.validation_pause_dir)) + len(os.listdir(self.validation_run_dir))

        self.train_imgs = [img_to_array(load_img(img, target_size=self.image_dim)) for img in glob.glob(self.train_dir + '/*/*')]
        self.train_imgs = np.array(self.train_imgs)
        self.train_imgs_scaled = self.train_imgs.astype('float32')
        self.train_imgs_scaled /= 255

        self.validation_imgs = [img_to_array(load_img(img, target_size=self.image_dim)) for img in glob.glob(self.validation_dir + '/*/*')]
        self.validation_imgs = np.array(self.validation_imgs)
        self.validation_imgs_scaled = self.validation_imgs.astype('float32')
        self.validation_imgs_scaled /= 255


        self.model_name = None
        self.history_file_name = None

        self.model = None
        self.history = None
        self.train_datagen = None
        self.val_datagen = None
        self.train_generator = None
        self.val_generator = None
        self.class_indices = None

        # self.logger.info(f'Training Images: {self.train_image_count}')
        # self.logger.info(f'Validation Images: {self.validation_image_count}')
        # self.logger.info(f'Augment Images: {self.augment_images}')

    def build_model(self):
        pass

    def fit_model(self):
        self.history = self.model.fit_generator(self.train_generator, epochs=self.epochs, steps_per_epoch=self.train_image_count // self.batch_size,
                                                # validation_data=self.val_generator, validation_steps=self.validation_image_count // self.batch_size,
                                                verbose=1)

        with open(self.history_file_name, 'w') as f:
            json.dump(self.history.history, f)

        self.model.save(self.model_name)

    def test_model(self):

        self.test_imgs_paths = [img for img in glob.glob(self.test_path + '/*')]

        print(self.class_indices)
        for img in self.test_imgs_paths:
            test_img = np.array(img_to_array(load_img(img, target_size=self.image_dim)))
            #test_img = np.array(test_img)
            scaled_img = test_img.astype('float32')
            scaled_img = scaled_img / 255

            prediction = self.model.predict_classes(scaled_img.reshape(1, self.input_shape[0], self.input_shape[1], self.input_shape[2]), verbose=0)
            print(img, prediction)

        # self.test_imgs = [img_to_array(load_img(img, target_size=self.image_dim)) for img in self.test_imgs_paths]
        # self.test_imgs = np.array(self.test_imgs)
        # self.test_imgs_scaled = self.test_imgs.astype('float32')
        # self.test_imgs_scaled /= 255

        #
        # predictions = self.model.predict_classes(self.test_imgs_scaled, verbose=0)
        # print(self.test_imgs_scaled, predictions)

    def plot_training_history(self):
        pass

    def create_image_generators(self):
        if self.augment_images:
            self.train_datagen = ImageDataGenerator(rescale=1. / 255, zoom_range=0.3, rotation_range=50,
                                                    width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                                                    horizontal_flip=True, fill_mode='nearest')
        else:
            self.train_datagen = ImageDataGenerator(rescale=1. / 255)

        self.val_datagen = ImageDataGenerator(rescale=1. / 255)

        self.train_generator = self.train_datagen.flow_from_directory(self.train_dir, batch_size=self.batch_size, target_size=self.image_dim, class_mode='binary')
        self.val_generator = self.val_datagen.flow_from_directory(self.validation_dir, batch_size=self.batch_size, target_size=self.image_dim, class_mode='binary')

        self.class_indices = self.train_generator.class_indices

    def run(self):
        self.create_image_generators()
        if self.load_model is False:
            self.build_model()
            self.fit_model()
        else:

            self.model = models.load_model(self.model_name)
        print(self.model_name)
        self.test_model()


class BasicConvolutionNeuralNetwork(BaseClassifier):
    def __init__(self, augment_images, load_model):
        BaseClassifier.__init__(self, augment_images=augment_images, load_model=load_model)
        self.model_name = f'basic-cnn-augment-images.h5' if augment_images else 'basic-cnn-no-augment-images.h5'
        self.history_file_name = f'basic-cnn-augment-images.json' if augment_images else 'basic-cnn-no-augment-images.json'
        # self.logger.info(f'Method: Basic Convolution Neural Network')

    def build_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(16, kernel_size=(3, 3), activation='relu',
                              input_shape=self.input_shape))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                           optimizer=optimizers.RMSprop(lr=1e-4),
                           metrics=['accuracy'])


class TransferLearningFeatureExtraction(BaseClassifier):

    def __init__(self, augment_images, load_model):
        BaseClassifier.__init__(self, augment_images=augment_images, load_model=load_model)
        self.model_name = f'transfer-learning-feature-extraction-augment-images.h5' if augment_images else 'transfer-learning-feature-extraction-no-augment-images.h5'
        self.history_file_name = f'transfer-learning-feature-extraction-augment-images.json' if augment_images else 'transfer-learning-feature-extraction-no-augment-images.json'
        # self.logger.info(f'Method: Transfer Learning Feature Extraction')
        self.read_vgg16_model()
        self.set_trainable_layers()

    def read_vgg16_model(self):
        vgg = vgg16.VGG16(include_top=False, weights='imagenet',
                          input_shape=self.input_shape)

        output = vgg.layers[-1].output
        output = keras.layers.Flatten()(output)
        self.pretrained_model = Model(vgg.input, output)
        # self.input_shape = self.pretrained_model.output_shape[1]

    def set_trainable_layers(self):
        self.pretrained_model.trainable = False
        for layer in self.pretrained_model.layers:
            layer.trainable = False

    def build_model(self):
        input_shape = self.pretrained_model.output_shape[1]
        self.model = Sequential()
        self.model.add(self.pretrained_model)
        self.model.add(Dense(512, activation='relu', input_dim=input_shape))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                           optimizer=optimizers.RMSprop(lr=1e-5),
                           metrics=['accuracy'])


class TransferLearningFineTuned(TransferLearningFeatureExtraction):
    def __init__(self, augment_images, load_model):
        TransferLearningFeatureExtraction.__init__(self, augment_images=augment_images, load_model=load_model)
        self.model_name = f'transfer-learning-fine-tune-augment-images.h5' if augment_images else 'transfer-learning-fine-tune-no-augment-images.h5'
        self.history_file_name = f'transfer-learning-fine-tune-augment-images.json' if augment_images else 'transfer-learning-fine-tune-no-augment-images.json'
        # self.logger.info(f'Method: Transfer Learning Fine Tuning')

    def set_trainable_layers(self):
        self.pretrained_model.trainable = True
        set_trainable = False
        for layer in self.pretrained_model.layers:
            if layer.name in ['block5_conv1', 'block4_conv1']:
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False


class ShadowFlickerClassifier(object):
    def __init__(self, method, augment_images, load_model):

        self.method = method

        self.augment_images = augment_images
        self.load_model = load_model

        assert (self.method in ['basic-cnn', 'transfer-learning-feature-extraction', 'transfer-learning-fine-tuned']), 'Invalid method specified'
        assert (type(self.augment_images) is bool), 'augment_images must be True/False'
        assert (type(self.load_model) is bool), 'load_model must be True/False'

    def get_classifier(self):
        if self.method == 'basic-cnn':
            return BasicConvolutionNeuralNetwork(augment_images=self.augment_images,
                                                 load_model=self.load_model)
        elif self.method == 'transfer-learning-feature-extraction':
            return TransferLearningFeatureExtraction(augment_images=self.augment_images,
                                                     load_model=self.load_model)
        elif self.method == 'transfer-learning-fine-tuned':
            return TransferLearningFineTuned(augment_images=self.augment_images,
                                             load_model=self.load_model)
        else:
            raise ValueError('invalid method specified')


# for i in ['basic-cnn', 'transfer-learning-feature-extraction', 'transfer-learning-fine-tuned']:
#     for j in [True, False]:
#         classifier = ShadowFlickerClassifier(method=i, augment_images=j, load_model=False)
#         classifier.get_classifier().run()


classsifier = ShadowFlickerClassifier(method='transfer-learning-fine-tuned', augment_images=True, load_model=True)
classsifier.get_classifier().run()