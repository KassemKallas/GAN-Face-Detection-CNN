from glob import glob
import os
import math
import cv2
import numpy as np
from keras.models import load_model
import tensorflow as tf


#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True
#sess = tf.Session(config=config)


if __name__ == '__main__':

    processing = 'GAN'
   # models_folder = 'G:/Co-Occurr/Dataset_StyleGan2/TrainNetwork3Co_USBnet/Model_original_vs_GAN_OrgPaper_3channel.h5'

    models_folder = 'G:/Co-Occurr/Dataset_StyleGan2_ZeroZero/Train_Nets/TrainNetwork6Co_VIPPnet/Model_VIPP.h5'

    models = glob(os.path.join(models_folder))

    test_batch_size = 100
    test_folder = 'G:/Co-Occurr/Dataset_StyleGan2_ZeroZero/Train_Nets/Dataset_VIPP/test/'
    n_test = 4000
    size_image = 256

    # Class textual labels (resp. 0 and 1)
    if processing == 'GAN':
        classes = ['original', 'GAN']
    elif processing == 'resize':
        classes = ['pasd', 'resize08']

    # List of pristine images (test)
    listfiles0_test = glob(os.path.join(test_folder, classes[0], '*.*'))
    listfiles0_test = sorted([os.path.basename(x) for x in listfiles0_test])
    listfiles0_test = [os.path.join(test_folder, classes[0], x) for x in listfiles0_test]

    # List of processed images (test)
    listfiles1_test = glob(os.path.join(test_folder, classes[1], '*.*'))
    listfiles1_test = sorted([os.path.basename(x) for x in listfiles1_test])
    listfiles1_test = [os.path.join(test_folder, classes[1], x) for x in listfiles1_test]

    test_max_iterations = int(math.floor(len(listfiles0_test) / test_batch_size))

    def testNet():

        cum_test_acc = 0.0
        cum_test_count = 0

        bsize2 = test_batch_size // 2
        for it in range(test_max_iterations):

            test_images0_it = listfiles0_test[it * bsize2:(it + 1) * bsize2]
            test_images1_it = listfiles1_test[it * bsize2:(it + 1) * bsize2]

            tbatch = []
            true_class = []

            for c in range(0, bsize2):

                img0 = np.load(test_images0_it[c])
                tbatch.append(img0)
                true_class.append(0)

                img1 = np.load(test_images1_it[c])
                tbatch.append(img1)
                true_class.append(1)

            tbatch_r = np.reshape(np.array(tbatch), (test_batch_size, size_image, size_image, 6))

            tbatch_r_3_channels = tbatch_r[:, :, :, 0:6]


            pred_class = np.argmax(model.predict_on_batch(tbatch_r), 1)

            cum_test_acc += np.sum(pred_class == true_class)
            cum_test_count += np.size(pred_class)

        acc = np.float( cum_test_acc / cum_test_count)
        with open('Stamm_accuracy_{}_vision.csv'.format(processing), 'a') as file:
            file.write('{},{}\n'.format(m, acc))

        print('-' * 50)
        print('Model {} - Test accuracy:{}'.format(m, acc))
        print('-' * 50)

        return


    cnt = 1
    for m in models:
        if cnt > 0:
            model = load_model(m)
            testNet()
        cnt +=1

