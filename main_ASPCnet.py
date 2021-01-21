# -*- coding: utf-8 -*-

'''
__author__ = 'Cimy'
__mtime__  = '2021/01/20'
 If you have any question, please contact me fell free.
 e-mail: wangjp85@mail2.sysu.edu.cn
'''

import os
import datetime
import scipy.io as scio
from keras import utils, callbacks
from keras.models import Model
from keras.optimizers import Adam
# from Capsule_Keras import *
from util import createPatches, DiscriminantAnalysis, report, random_sample, DrawResult, applyPCA
from keras.utils import multi_gpu_model
from keras.layers import *
from ASPcaps_layer import ConvertToCaps, FlattenCaps, ASPCaps_layer, CapsuleLayer, CapsToScalars
from ASP_layer import ASP_layer
import argparse
import matplotlib.pyplot as plt


def creat_model_aspcaps(x_train, num_classes):
    img_rows, img_cols, num_dim = x_train.shape[1], x_train.shape[2], x_train.shape[3]
    input_layer = Input((img_rows, img_cols, num_dim))
    layer_01 = ASP_layer(filters=128, kernel_size=args.KS, dilation=args.DR_1, stride=1)(input_layer)
    layer_02 = Conv2D(filters=128, kernel_size=(1, 1), strides=(2, 2), activation='relu', padding='same')(layer_01)
    layer_03 = ASP_layer(filters=256, kernel_size=args.KS, dilation=args.DR_1, stride=1)(layer_02)
    layer_04 = Conv2D(filters=256, kernel_size=(1, 1), strides=(2, 2), activation='relu', padding='same')(layer_03)
    layer_05 = BatchNormalization(momentum=args.momentum)(layer_04)
    layer_06 = ConvertToCaps()(layer_05)
    layer_07 = ASPCaps_layer(32, 4, kernel_size=(args.KS, args.KS), strides=(1, 1),
                             dilation_rate=(args.DR_1, args.DR_1))(layer_06)
    layer_08 = ASPCaps_layer(32, 4, kernel_size=(args.KS, args.KS), strides=(1, 1),
                             dilation_rate=(args.DR_1, args.DR_1))(layer_07)
    layer_09 = FlattenCaps()(layer_07)
    layer_10 = CapsuleLayer(num_capsule=num_classes, dim_capsule=16, routings=3, channels=0, name='digit_caps')(
        layer_09)
    output_layer = CapsToScalars(name='capsnet')(layer_10)

    if args.numGPU > 1:
        model = Model(inputs=input_layer, outputs=output_layer)
        parallel_model = multi_gpu_model(model, gpus=args.numGPU)
    else:
        parallel_model = Model(inputs=input_layer, outputs=output_layer)
    return parallel_model


parser = argparse.ArgumentParser(description="ASPCNet")
parser.add_argument("--numGPU", type=int, default="1", action="store", help="The total numbers of GPU to use")
parser.add_argument("--GPUid", type=int, default="0", action="store", help="The specific numbers of GPU to use")
parser.add_argument("--dir", type=str, default="./data", action="store", help="input(default: ./data)")
parser.add_argument("--itera", type=int, default="5", action="store", help="The numbers of repeating the experiments")
parser.add_argument("--WindowsSize", type=int, default="23", action="store", help="The Patch size for each pixel")
parser.add_argument("--N_C", type=int, default="15", action="store", help="The reducing demision")
parser.add_argument("--batch_size", type=int, default="64", action="store",
                    help="the minimal training number in each iteration")
parser.add_argument("--epochs", type=int, default="500", action="store", help="the training epoch")
parser.add_argument("--KS", type=int, default="3", action="store", help="The convolution kernel size")
parser.add_argument("--DR_1", type=int, default="3", action="store", help="The Dilation Rate")
parser.add_argument("--epsilon", type=float, default="1e-08", action="store", help="epsilon of adam optimizer", )
parser.add_argument("--beta_2", type=float, default="0.999", action="store",
                    help="the second parameter of adam optimizer")
parser.add_argument("--beta_1", type=float, default="0.9", action="store", help="the first parameter of adam optimizer")
parser.add_argument("--learning_rate", type=float, default="0.001", action="store", help="learning rate")
parser.add_argument("--patience", type=int, default="40", action="store", help="early stopping step")
parser.add_argument("--momentum", type=float, default="0.9", action="store", help="BN layer")
parser.add_argument("--dropout", type=float, default="0.2", action="store", help="dropout layer")
parser.add_argument("--save_model", type=str, default="True", action="store",
                    help="save the model wight and graph when the training and test finished")
parser.add_argument("--clear_model", type=str, default="True", action="store",
                    help="clear the model graph when the training and test finished")
args = parser.parse_args()

CUDA_VISIBLE_DEVICES = args.GPUid
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPUid)

for root, dirs, files in os.walk(args.dir):
    print('||======The Training set is localed in [%s/%s]======||' % (root, files[0]))
labels = scio.loadmat(root + '/' + files[0])['ground']
data1 = np.array(scio.loadmat(root + '/' + files[0])['img'])
data, pca = applyPCA(data1, numComponents=args.N_C)
patchesData, patchesLabels = createPatches(data, labels, windowSize=args.WindowsSize)
patchesLabels = patchesLabels.astype(np.int32)

num_classes = labels.max()
result_index = np.zeros((num_classes + 4, args.itera))

train_sample = (200, 200, 200, 200, 200, 200, 200, 200, 200)
validate_sample = [0, 0, 0, 0, 0, 0, 0, 0, 0]

for it in range(args.itera):
    trainIndex, valIndex, testIndex = random_sample(train_sample, validate_sample, patchesLabels)
    x_train, x_test, x_val = patchesData[trainIndex, :, :, :], \
                             patchesData[testIndex, :, :, :], \
                             patchesData[valIndex, :, :, :]
    y_train, y_test, y_val = utils.to_categorical(patchesLabels[trainIndex] - 1, num_classes), \
                             utils.to_categorical(patchesLabels[testIndex] - 1, num_classes), \
                             utils.to_categorical(patchesLabels[valIndex] - 1, num_classes)
    true_label = patchesLabels[testIndex]

    parallel_model = creat_model_aspcaps(x_train, num_classes)

    parallel_model.compile(loss=lambda y_true, y_pred_t: y_true * K.relu(0.9 - y_pred_t) ** 2 +
                                                         0.25 * (1 - y_true) * K.relu(y_pred_t - 0.1) ** 2,
                           optimizer=Adam(lr=args.learning_rate,
                                          beta_1=args.beta_1,
                                          beta_2=args.beta_2,
                                          epsilon=args.epsilon),
                           metrics=['accuracy'])

    print('||======= total training sample is %d=======||' % (sum(train_sample)))
    parallel_model.summary()
    start_time = datetime.datetime.now()
    callback = callbacks.EarlyStopping(monitor='acc',  # monitor='val_acc',
                                       min_delta=0,
                                       patience=args.patience,
                                       verbose=1,
                                       mode='auto',
                                       restore_best_weights=True)
    history = parallel_model.fit(x_train,
                                 y_train,
                                 verbose=2,
                                 batch_size=args.batch_size,
                                 validation_data=(x_val, y_val),
                                 epochs=args.epochs,
                                 callbacks=[callback])

    end_time_train = datetime.datetime.now()
    print('====================================================')
    print('||======= Training Time for % s' % (end_time_train - start_time), '======||')
    print('====================================================')

    y_pred = np.argmax(parallel_model.predict(x_test), axis=1) + 1

    classification, confusion, accuracy_matrix = report(true_label, y_pred)
    end_time_test = datetime.datetime.now()
    print('====================================================')
    print('||======= Test Time for % s' % (end_time_test - start_time), '======||')
    print('====================================================')
    Result_all = np.argmax(parallel_model.predict(patchesData), axis=1) + 1
    DrawResult(labels, Result_all, testIndex, y_pred, background=1, imageID='PU', dpi=800)

    result_index[0:-1, it] = accuracy_matrix
    result_index[-1, it] = ((end_time_train - start_time).total_seconds())
    print('WindowsSize = %d, batch_size = %d, N_C = %d, it = %d' % (args.WindowsSize, args.batch_size, args.N_C, it))

    # if args.save_model == 'True':
    #     parallel_model.save('aspc_model.hdf5')
    if args.clear_model == 'True':
        K.clear_session()

    print(result_index)
