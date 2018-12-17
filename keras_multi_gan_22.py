import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import keras
from keras.datasets import cifar10

from keras.layers import Input, MaxPooling2D, LSTM, BatchNormalization, AtrousConv2D
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose
from keras.optimizers import Adam,SGD,RMSprop,Adadelta
from keras.layers.merge import concatenate
from keras import initializers
import cv2
from keras.models import load_model
import h5py
import tensorflow as tf
import os


def get_small_imgs(my_small_img_width, my_dir):
    total_samples = 0
    all_imgs = []
    for filename in os.listdir(my_dir):
        if filename.endswith(".jpeg") or filename.endswith(".jpg") :
            full_file_name = os.path.join(my_dir, filename)
            #low_res_imgs = np.zeros((total_images, my_small_images_dim_y, my_small_images_dim_x, 3))
            image = cv2.imread(full_file_name, 3)  #comes in as 0-255 ints, BGR!

            image = image / 255.0
            image = image - .5
            image = image * 2.0
            #print(image)

            smaller_images_to_sample = ((image.shape[0] // 64) * (image.shape[1] // 64)) * 10#*4#6#7# 10#80
            #total_samples += smaller_images_to_sample
            for i in range(0, smaller_images_to_sample):
                total_samples += 1
                rand_x = random.randint(0,image.shape[0] - my_small_img_width)
                rand_y = random.randint(0,image.shape[1] - my_small_img_width)
                small_img = image[rand_x: rand_x+my_small_img_width, rand_y: rand_y+my_small_img_width, :]

                #small_img = small_img * 255.0  #cv wants 0-255 ints)  BGR!

                all_imgs.append(small_img)

                fn = "/home/foo/data/gan_foo/x_foo_write/" + str(total_samples) + ".jpg"

                if total_samples % 100 == 0:
                    remapped_img = small_img.copy()
                    #remapped_img = remapped_img + 1.0
                    #remapped_img = remapped_img / 2.0
                    remapped_img = remapped_img * 255.0
                    cv2.imwrite(fn, remapped_img)   # write the fileto_disk

    print('Total samples generated: ', total_samples)
    np_x = np.array(all_imgs)
    #print('npshape',np_x[11])
    return  np_x

def clean_cifar(my_total_samples):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    single = []
    for i in range(0,x_train.shape[0]):
        if y_train[i] == 8:
            single.append(x_train[i])
    x_train = np.array(single)

    if x_train.shape[0] < my_total_samples:
        my_total_samples = x_train.shape[0]

    x_train = x_train[0:my_total_samples]
    x_train = x_train / 255.0
    x_train = x_train - .5
    x_train = x_train * 2.0
    channels_swapped = x_train[:,:,:,[2,1,0]]  # OK, the trainign data is now BGR BECAUSE OF CV2.
    print('# samples', x_train.shape[0])
    return channels_swapped

class MULTI_GAN:

    def __init__(self, x_train_, z_vec_dimension=100):
        self.x_train = x_train_
        self.z_vec_dimension = z_vec_dimension
        self.lr = 4#8#2     # lowest resolution to start with
        self.hr = 32    # highest resolution we end with
        self.lr_filters = 256#64#256#128#64#256
        self.discriminator_filters = 128#64#32

        self.relu_slope = .2
        self.kernel_size = 5
        self.kernel_size_discriminator = 5#2
        self.generator_init = 'normal'
        self.discriminator_init = 'normal'
        self.optimizer = Adam(lr=0.00033, beta_1=0.5, beta_2=.999)

        self.current_level = 1#2#1#3#2#1#2#4#3#1#3
        self.total_levels = self.get_total_levels()  # cifar-10 would have 3, (4->8  , 8->16,  16->32)

        self.generator = self.__get_generator()
        self.discriminator = self.__get_discriminator()
        self.gan = self.__get_gan()

        self.batch_size = 16#32#64#128#256#64
        self.label_noise = .03  # instead on real=1, real= (1 - labelnoise) to 1.0

        self.weights_dir = '/output/'
        self.weights_file_prefix = 'multi_gan_gen_layer_'
        self.output_dir = '/output/multi_'

        # self.output_dir = '/home/foo/data/gan_foo/multi/multi_'
        # self.weights_dir = '/home/foo/data/gan_foo/multi_gan_weights/'
        # self.weights_file_prefix = 'multi_gan_gen_layer_'

    def get_batch_at_level(self):
        scaled_down_resolution = int(self.hr / (2 ** (self.total_levels - self.current_level)))
        batch_lr = np.zeros((self.batch_size, scaled_down_resolution, scaled_down_resolution, 3), dtype=float)

        for i in range(0, self.batch_size):
            img_hr = self.x_train[random.randint(0, self.x_train.shape[0] - 1)]  # use np.random here....
            img_hr = img_hr + 1.0
            img_hr = img_hr / 2.0
            img_hr = img_hr * 255.0

            img_lr = cv2.resize(img_hr, (scaled_down_resolution, scaled_down_resolution))
            img_lr = img_lr / 255.0
            img_lr = img_lr - .5
            img_lr = img_lr * 2.0  # TODO: MOVE THESE 3-LINE SCALING LINES INTO A FUNCTION

            batch_lr[i] = img_lr
            # if i == 27:
            #     cv2.imwrite('/home/foo/data/gan_foo/multi/multi.jpg', img_lr)

        return batch_lr

    # def foo(self):
    #     self.current_level = 4
    #     self.generator = self.__get_generator()
    #     self.discriminator = self.__get_discriminator()

    def __get_generator(self):
        generator_input = Input(shape=(self.z_vec_dimension,))

        hid = Dense(self.lr * self.lr * self.lr_filters)(generator_input)

        hid = Reshape((self.lr, self.lr, self.lr_filters))(hid)
        hid = BatchNormalization()(hid)
        hid = LeakyReLU(self.relu_slope)(hid)

        for residual_block in range(1, self.current_level):
            nb_of_filters = self.lr_filters // (2**(residual_block + 1))
            hid = Conv2DTranspose(nb_of_filters, kernel_size=self.kernel_size,
                                  strides=2, kernel_initializer=self.generator_init, padding='same')(hid)
            hid = BatchNormalization()(hid)
            hid = LeakyReLU(self.relu_slope)(hid)

        fake_image = Conv2DTranspose(3, kernel_size=self.kernel_size, strides=2, activation='tanh',
                                     kernel_initializer=self.generator_init, padding='same')(hid)

        generator_ = Model(generator_input, fake_image)
        generator_.compile(optimizer=self.optimizer, loss='binary_crossentropy')
        return generator_

    def __get_discriminator(self):
        scaled_down_resolution = int(self.hr / (2 ** (self.total_levels - self.current_level)))  # TODO: MAKE THIS A DERIVED CLASS PROPERTY

        discriminator_input = Input(shape=(scaled_down_resolution, scaled_down_resolution, 3))  # 3-channel, channels-last
        auxilary_features_input = Input(shape=(4,))
        #aux_dense = Dense(10)(auxilary_features_input)

        kernel_size_ = self.current_level + 3  # this may not be the best way to determine filter nb
        if kernel_size_ > 5:
            kernel_size_ = 5

        hidd = Conv2D(self.discriminator_filters, strides=2, kernel_size=kernel_size_, padding='same',
                      kernel_initializer=self.discriminator_init)(discriminator_input)
        hidd = LeakyReLU(self.relu_slope)(hidd)

        for residual_block in range(0, self.current_level + 1):  # very important
            nb_discriminator_filters = self.discriminator_filters // (2**residual_block)
            hidd = Conv2D(nb_discriminator_filters, strides=2, kernel_size=kernel_size_, padding='same',
                          kernel_initializer=self.discriminator_init)(hidd)
            hidd = LeakyReLU(self.relu_slope)(hidd)

        # hidd = concatenate([hidd,auxilary_features_input])
        hidd = Flatten()(hidd)
#        hidd = concatenate([hidd,aux_dense])
        hidd = concatenate([hidd, auxilary_features_input])

        #hidd = Dense(20, activation='tanh', kernel_initializer=self.discriminator_init)(hidd)

        sig = Dense(1, activation='sigmoid', kernel_initializer=self.discriminator_init)(hidd)

        discriminator_ = Model(inputs=[discriminator_input, auxilary_features_input], outputs=sig)
        discriminator_.compile(optimizer=self.optimizer, loss='binary_crossentropy')
        #print(discriminator_.summary())
        return discriminator_

    def __get_gan(self):
        self.discriminator.trainable = False
        for layer in self.discriminator.layers:
            layer.trainable = False

        gan_input = Input(shape=(self.z_vec_dimension,))
        x_prime = self.generator(gan_input)
        x_aux = Input(shape=(4,))

        gan_out = self.discriminator([x_prime, x_aux])
        gan_ = Model(inputs=[gan_input,x_aux], outputs=gan_out)
        gan_.compile(optimizer=self.optimizer, loss='binary_crossentropy')

        return gan_

    def get_total_levels(self):
        total_levels = 99
        for i in range(1,20):
            if (self.lr * 2**i) == self.hr:
                total_levels = i
                break
        return total_levels

    def extract_minibatch_features(self, my_mini_batch):

        my_features = np.zeros((my_mini_batch.shape[0],4))
        # avg = my_mini_batch.sum() / my_mini_batch.shape[0]
        # print('avg',avg)
        a = np.zeros((my_mini_batch.shape[1],my_mini_batch.shape[2],3))

        m = np.zeros((my_mini_batch.shape[0]))
        for i in range(0, my_mini_batch.shape[0]):
            m[i] = my_mini_batch[i].mean()
            a = a + my_mini_batch[i]

        a = a / my_mini_batch.shape[0] * 1.0 # this it the average of all images in batch
                                                # generators tend to mode collapse and produce
                                            # a bunch of similar images
                                            # this value will be a big clue to the discriminator that
                                            # it's a fake (if it s near zero), forcing the generator to produce a more varied range of image examples

        for i in range(0, my_mini_batch.shape[0]):
            diff = a - my_mini_batch[i]
            diff = diff * diff
            diff = diff.std()
            difm = diff.mean()
            #diff = diff / (my_mini_batch.shape[1] * my_mini_batch.shape[1] * 1.0)

            mstd = my_mini_batch[i].std()

            std = m.std()
            #std = my_mini_batch.std()
            my_features[i] = np.array([diff,std,mstd,difm])
        return my_features

    def write_images_to_disk(self, my_images, my_batch, my_d_loss):
        my_images = my_images + 1.0
        my_images = my_images / 2.0
        my_images = my_images * 255.0

        los = str(my_d_loss) # TODO: MAKE BIG IMAGE HERE

        for i in range(0, my_images.shape[0]):
            level_info = '_lev_' + str(self.current_level) +'_'
            filename = self.output_dir + str(my_batch) + '_' + str(i) + '_' + los + level_info + '.jpg'

            #if random.uniform(0.0,1.0) < .25: #write 25% of the batch
            if i%2 == 0:
                cv2.imwrite(filename, my_images[i])

    def save_weights(self):
        print('Saving weights...')
        number_of_layers_to_save = (self.current_level * 3) + 2


        #for indx, gen_layer in enumerate(self.generator.layers):

        for indx in range(0, number_of_layers_to_save):
            gen_layer = self.generator.layers[indx]
            weights_and_biases = gen_layer.get_weights()
            weights_and_biases_np = np.array(weights_and_biases)
            file_name = self.weights_dir + self.weights_file_prefix + str(indx)
            np.save(file_name, weights_and_biases_np)


    def load_weights(self):
        number_of_layers_to_load = ((self.current_level - 1) * 3) + 2
        for layer_indx in range(0,number_of_layers_to_load):
            np_file_name = self.weights_dir + self.weights_file_prefix + str(layer_indx) + '.npy'
            np_layer = np.load(np_file_name)
            list_of_weights = np_layer.tolist()

            keras_layer = self.generator.layers[layer_indx]

            keras_layer.set_weights(np_layer)
            keras_layer.trainable = False

    def unfreeze_all_weights(self):
        for layer in self.generator.layers:
            layer.trainable = True


    def start_training(self):
        gan_loss_total, disc_loss_total, gan_loss_avg, disc_loss_avg = 0.0, 0.0, 0.0, 0.0

        for batch in range(1, 100000):
            z_vec_noise = np.random.uniform(low=-1, high=1, size=(self.batch_size, self.z_vec_dimension))
            z_vec_noise_double = np.random.uniform(low=-1, high=1, size=(self.batch_size * 2, self.z_vec_dimension))

            batch_real_images = self.get_batch_at_level()
            batch_generated_images = self.generator.predict(z_vec_noise)

            bogus_aux = self.extract_minibatch_features(batch_generated_images)#np.zeros((self.batch_size ,2))
            bogus_aux_double = self.extract_minibatch_features(batch_generated_images)# np.zeros((self.batch_size * 2, 2))

            ones = np.random.uniform(1.0-self.label_noise, 1.0, (self.batch_size, 1))
            ones_double = np.random.uniform(1.0-self.label_noise, 1.0, (self.batch_size * 2, 1))
            zeros = np.random.uniform(0.0, self.label_noise, (self.batch_size, 1))

            gan_loss_on_batch = self.gan.train_on_batch([z_vec_noise, bogus_aux], ones)
            gan_loss_on_batch /= 1.0
            gan_loss_total += gan_loss_on_batch
            gan_loss_avg = gan_loss_total / (batch * 1.0)

            disc_real_loss_on_batch = self.discriminator.train_on_batch([batch_real_images, bogus_aux], ones)
            disc_fake_loss_on_batch = self.discriminator.train_on_batch([batch_generated_images, bogus_aux], zeros)
            disc_loss_on_batch = 0.5 * np.add(disc_real_loss_on_batch, disc_fake_loss_on_batch)  # we have double the amount, so we halve it
            disc_loss_total += disc_loss_on_batch
            disc_loss_avg = disc_loss_on_batch / batch * 1.0


            if batch % 500 == 0:
                print('Level: ', self.current_level, batch)
                print('G/D', gan_loss_on_batch, disc_loss_on_batch)
                print('---------------------')
            if batch % 2500 == 0:
                print('Write images, and save weights...')
                self.write_images_to_disk(batch_generated_images, batch, disc_loss_on_batch)
                self.save_weights()
                self.unfreeze_all_weights()

                if batch % 15000 == 0:

                    if self.current_level < self.total_levels:
                        if self.batch_size > 32:
                            self.batch_size = self.batch_size // 2

                        self.current_level = self.current_level + 1
                        self.generator = self.__get_generator()
                        self.discriminator = self.__get_discriminator()
                        self.gan = self.__get_gan()

                        self.load_weights()
                        print('Level Up!')
                    self.write_images_to_disk(batch_generated_images, batch, disc_loss_on_batch)



            # if gan_loss_on_batch < .28:
            #     #saevweights
            #     #change-_level
            #     # rebuild modesl
            #     # start trainign again


#----------------------------------------------------------

x_train = clean_cifar(25000)
#x_train = get_small_imgs(32, '/home/foo/data/gan_foo/brick_small/')

multi = MULTI_GAN(x_train, 80)

print(multi.start_training())




#1,2,3,4  [5,6,7]  [8,9,10] ..... [last convtrans3]