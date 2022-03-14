
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from net import GAN_net, constrain_net_weights
from keras.utils import plot_model
from keras.models import save_model
import keras
from time import time, gmtime, strftime
import math
import numpy as np
import os
from glob import glob
import cv2
import keras
import tensorflow as tf
from keras import backend as K
from keras.callbacks import TensorBoard

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)

# GPU management
#tfconfig = tf.compat.v1.ConfigProto()
#tfconfig.gpu_options.allow_growth = True
#sess = tf.compat.v1.Session(config=tfconfig)
#K.set_session(sess)

if __name__ == '__main__':

    processing = 'GAN'
    #processing = 'median'

    print('-'*100)
    print('Processing : {}'.format(processing))
    print('-'*100)

    model_name = 'Model_VIPP.h5'.format(processing)           # Output model name
    n_epochs = 40                                                     # Number of epochs

    batch_size = 40                                                    # Training batch size
    val_batch_size = 100
    test_batch_size = 100

    validation_frequency = 100                                          # Training validation frequency

    train_folder = 'G:/Co-Occurr/Dataset_StyleGan2_ZeroZero_NoNorm_Int64/Train_Nets/Dataset_VIPP/train'
    val_folder = 'G:/Co-Occurr/Dataset_StyleGan2_ZeroZero_NoNorm_Int64/Train_Nets/Dataset_VIPP/validation'
    test_folder = 'G:/Co-Occurr/Dataset_StyleGan2_ZeroZero_NoNorm_Int64/Train_Nets/Dataset_VIPP/test'

    # Class textual labels (resp. 0 and 1)
    if processing == 'GAN':
        classes = ['original', 'GAN']

    n_train = 12000                                                    # Number of training blocks
    n_val = 4000                                                       # Number of validation blocks
    n_test = 4000                                                     # Number of test blocks

    size_image = 256                                                    # CNN input image size

    # ------------------------------------------------------------------------------------------------------------------
    # Prepare a list of training and validation images
    # ------------------------------------------------------------------------------------------------------------------

    # List of pristine images
    listfiles0 = glob(os.path.join(train_folder, classes[0], '*.*'))
    listfiles0 = sorted([os.path.basename(x) for x in listfiles0])
    listfiles0 = [os.path.join(train_folder, classes[0], x) for x in listfiles0]

    # List of processed images
    listfiles1 = glob(os.path.join(train_folder, classes[1], '*.*'))
    listfiles1 = sorted([os.path.basename(x) for x in listfiles1])
    listfiles1 = [os.path.join(train_folder, classes[1], x) for x in listfiles1]

    assert(len(listfiles0) == len(listfiles1))
    assert(len(listfiles0) >= n_train)

    listfiles0 = listfiles0[:n_train]
    listfiles1 = listfiles1[:n_train]

    # List of pristine images
    val_listfiles0 = glob(os.path.join(val_folder, classes[0], '*.*'))
    val_listfiles0 = sorted([os.path.basename(x) for x in val_listfiles0])
    val_listfiles0 = [os.path.join(val_folder, classes[0], x) for x in val_listfiles0]

    # List of processed images
    val_listfiles1 = glob(os.path.join(val_folder, classes[1], '*.*'))
    val_listfiles1 = sorted([os.path.basename(x) for x in val_listfiles1])
    val_listfiles1 = [os.path.join(val_folder, classes[1], x) for x in val_listfiles1]

    assert (len(val_listfiles0) == len(val_listfiles1))
    assert (len(val_listfiles0) >= n_val)

    # Choose random images from training set to be used for validation
    val_perm = np.random.permutation(n_val)
    val_listfiles0 = [val_listfiles0[i] for i in val_perm]
    val_listfiles1 = [val_listfiles1[i] for i in val_perm]

    # ------------------------------------------------------------------------------------------------------------------
    # Network
    # ------------------------------------------------------------------------------------------------------------------
    model = GAN_net((size_image, size_image, 6), 2)


    L = 0.01
    D = L / n_epochs
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=L, decay=D, momentum=0.9), metrics=['accuracy'])



    print(model.summary())
    plot_model(model, to_file="StyleGAN_{}net.png".format(processing.title()), show_shapes=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------------------------------------------------------

    # Start timing
    begin_time = time()

    # Determine the number of iterations that complete each epoch (i.e. net has seen all the training set
    max_iterations = int(math.floor(len(listfiles0) / batch_size))


    # Loop epochs
    for ep in range(n_epochs):

        # Shuffle training list before each epoch (same shuffle for both classes)
        rnd_perm = np.random.permutation(n_train)
        listfiles0_perm = [listfiles0[i] for i in rnd_perm]
        listfiles1_perm = [listfiles1[i] for i in rnd_perm]

        # Loop training iterations
        bsize2 = batch_size//2
        for it in range(max_iterations):

            # Prepare training batch
            images0_it = listfiles0_perm[it*bsize2:(it+1)*bsize2]
            images1_it = listfiles1_perm[it*bsize2:(it + 1)*bsize2]

            batch = []
            batch_labels = []
            for c in range(0, bsize2):
                # Pristine
                #img0 = cv2.imread(images0_it[c], cv2.IMREAD_GRAYSCALE)
                img0 = np.load(images0_it[c])
                batch.append(img0/255.)  #comment by ehsan
                #batch.append (img0)      #add by ehsan
                batch_labels.append(0)

                # Processed
                #img1 = cv2.imread(images1_it[c], cv2.IMREAD_GRAYSCALE)
                img1= np.load(images1_it[c])
                batch.append(img1/255.) #comment by ehsan
                #batch.append (img1)   #add by ehsan
                batch_labels.append(1)

            # Train with batch
            #batch_res = np.reshape(np.array(batch), (batch_size, size_image, size_image,1))
            batch_res = np.reshape(np.array(batch), (batch_size, size_image, size_image, 6))
            acc = model.train_on_batch(batch_res, keras.utils.to_categorical(batch_labels, 2))
            # print('Epoch {} Iter {}/{}'.format(ep, it, max_iterations))

            # Force first layer weights
            model = constrain_net_weights(model, p=1)

            # Validate periodically on validation set, again, on batches
            if it > 0 and it % validation_frequency == 0:

                cum_val_acc = 0.0
                cum_val_count = 0
                val_max_iterations = int(math.floor(len(val_listfiles0) / val_batch_size))
                val_bsize2 = val_batch_size//2
                for val_it in range(val_max_iterations):

                    val_images0_it = val_listfiles0[val_it * val_bsize2:(val_it + 1) * val_bsize2]
                    val_images1_it = val_listfiles1[val_it * val_bsize2:(val_it + 1) * val_bsize2]

                    vbatch = []
                    vtrue_class = []

                    for c in range(0, val_bsize2):
                        #img0 = cv2.imread(val_images0_it[c], cv2.IMREAD_GRAYSCALE)
                        img0 = np.load(val_images0_it[c])
                        vbatch.append(img0 / 255.) #comment by ehsan
                        #vbatch.append(img0) #add by ehsan
                        vtrue_class.append(0)

                        #img1 = cv2.imread(val_images1_it[c], cv2.IMREAD_GRAYSCALE)
                        img1 =  np.load(val_images1_it[c])
                        vbatch.append(img1 / 255.) #comment by ehsan
                        #vbatch.append(img1) #add by ehsan
                        vtrue_class.append(1)

                    vbatch_r = np.reshape(np.array(vbatch), (val_batch_size, size_image, size_image,6))
                    vpred_class = np.argmax(model.predict_on_batch(vbatch_r), 1)
                    cum_val_acc += np.sum(vpred_class == vtrue_class)
                    cum_val_count += np.size(vpred_class)

                print('{} - Epoch {} - Iteration {}/{}. Validation Accuracy: {}'
                      .format(strftime("%H:%M:%S", gmtime()), ep, it, max_iterations, cum_val_acc/cum_val_count))

        save_model(model, 'Epoch_#{}_{}'.format(ep, model_name))

    # Save model
    save_model(model, model_name)

    elapsed = time() - begin_time
    print('-'*50)
    print('Training ended after {:5.2f} seconds'.format(elapsed))
    print('-'*50)

    # ------------------------------------------------------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------------------------------------------------------

    listfiles0_test = glob(os.path.join(test_folder, classes[0], '*.*'))
    listfiles0_test = sorted([os.path.basename(x) for x in listfiles0_test])
    listfiles0_test = [os.path.join(test_folder, classes[0], x) for x in listfiles0_test]

    # List of processed images
    listfiles1_test = glob(os.path.join(test_folder, classes[1], '*.*'))
    listfiles1_test = sorted([os.path.basename(x) for x in listfiles1_test])
    listfiles1_test = [os.path.join(test_folder, classes[1], x) for x in listfiles1_test]

    test_max_iterations = int(math.floor(len(listfiles0_test) / test_batch_size))

    cum_test_acc = 0.0
    cum_test_count = 0

    bsize2 = test_batch_size // 2
    for it in range(test_max_iterations):

        test_images0_it = listfiles0_test[it * bsize2:(it + 1) * bsize2]
        test_images1_it = listfiles1_test[it * bsize2:(it + 1) * bsize2]

        tbatch = []
        true_class = []

        for c in range(0, bsize2):
            #img0 = cv2.imread(test_images0_it[c], cv2.IMREAD_GRAYSCALE)
            img0 = np.load(test_images0_it[c])
            tbatch.append(img0 / 255.) #comment by ehsan
            #tbatch.append(img0) #add by ehsan
            true_class.append(0)

           #img1 = cv2.imread(test_images1_it[c], cv2.IMREAD_GRAYSCALE)
            img1 = np.load(test_images1_it[c])
            tbatch.append(img1 / 255.) #comment by ehsan
            #tbatch.append(img1) #add by ehsan
            true_class.append(1)

        tbatch_r = np.reshape(np.array(tbatch), (test_batch_size, size_image, size_image, 6))
        pred_class = np.argmax(model.predict_on_batch(tbatch_r), 1)
        cum_test_acc += np.sum(pred_class == true_class)
        cum_test_count += np.size(pred_class)

    print('-' * 50)
    print("Test accuracy:%6f" % (cum_test_acc / cum_test_count))
    print('-'*50)
