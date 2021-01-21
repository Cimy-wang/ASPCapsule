# -*- coding: utf-8 -*-

'''
__author__ = 'Cimy'
__mtime__  = '2021/01/20'
 If you have any question, please contact me fell free.
 e-mail: wangjp85@mail2.sysu.edu.cn
'''

from keras.layers import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import random
from sklearn import preprocessing


def createPatches(X, y, windowSize, removeZeroLabels=False):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = np.pad(X, ((margin, margin), (margin, margin), (0, 0)), 'symmetric')
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]), dtype='float')
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]), dtype='float32')
    patchIndex = 0
    for c in range(margin, zeroPaddedX.shape[1] - margin):
        for r in range(margin, zeroPaddedX.shape[0] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c -
                                                           margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    return patchesData, patchesLabels


def DiscriminantAnalysis(X, Y, numComponents):
    Y = Y.reshape(-1)
    newX = np.reshape(X, (-1, X.shape[2]))
    model_lda = LinearDiscriminantAnalysis(n_components=numComponents)
    newX = model_lda.fit(newX, Y).transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX


def report(true_label, y_pred):
    classification = classification_report(true_label, y_pred)
    confusion = confusion_matrix(true_label, y_pred)
    oa = np.trace(confusion) / sum(sum(confusion))
    ca = np.diag(confusion) / confusion.sum(axis=1)
    Pe = (confusion.sum(axis=0) @ confusion.sum(axis=1)) / np.square(sum(sum(confusion)))
    K = (oa - Pe) / (1 - Pe)
    aa = sum(ca) / len(ca)
    List = []
    List.append(np.array(oa)), List.append(np.array(aa)), List.append(np.array(K))
    List = np.array(List)
    accuracy_matrix = np.concatenate((ca, List), axis=0)
    # ==== Print table accuracy====
    ind = [list(range(1, len(ca) + 1, 1)) + ['OA', 'AA', 'KA']][0]
    target_names = [u'%s' % l for l in ind]
    last_line_heading = 'avg / total'
    name_width = max(len(cn) for cn in target_names)
    width = max(name_width, len(last_line_heading), 2)
    headers = ["accuracy"]
    head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
    report = head_fmt.format(u'', *headers, width=width)
    report += u'\n\n'
    rows = zip(target_names, accuracy_matrix)
    row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' u'\n'
    for row1 in rows:
        report += row_fmt.format(*row1, width=width, digits=5)
    report += u'\n'
    return classification, confusion, accuracy_matrix



def random_sample(train_sample, validate_sample, patchesLabels):
    num_classes = np.max(patchesLabels)
    dataList = patchesLabels 
    TrainIndex = []  
    TestIndex = []
    ValidateIndex = []

    for i in range(num_classes):
        train_sample_temp = train_sample[i]
        validate_sample_temp = validate_sample[i]
        index = np.where(patchesLabels == (i + 1))[0]
        Train_Validate_Index = random.sample(range(0, int(index.size)), train_sample_temp+validate_sample_temp)
        TrainIndex = np.hstack((TrainIndex, index[Train_Validate_Index[0:train_sample_temp]])).astype(np.int32)
        ValidateIndex = np.hstack((ValidateIndex, index[Train_Validate_Index[train_sample_temp:100000]])).astype(np.int32)
        Test_Index = [index[i] for i in range(0, len(index), 1) if i not in Train_Validate_Index]
        TestIndex = np.hstack((TestIndex, Test_Index)).astype(np.int32)

    return TrainIndex, ValidateIndex, TestIndex  



def DrawResult(labels, Result_all, testIndex, y_pred, background=1, imageID='PU', dpi = 800):
    # ID=1:Pavia University
    # ID=2:Indian Pines
    # ID=7:Houston
    global palette
    global row
    global col
    labels = (labels.swapaxes(0, 1))
    labels = labels.reshape(labels.size)
    num_class = int(labels.max())
    if imageID == 'PU':  # PaviaU
        row = 610
        col = 340
        palette = np.array([[192, 192, 192], [  0, 255,   0], [  0, 255, 255], [  0, 128,   0], [255,   0, 255],
                            [165,  82,  41], [128,   0, 128], [255,   0,   0], [255, 255,   0]])
    elif imageID == 'IN':  # Indian
        row = 145
        col = 145
        palette = np.array([[140,  67,  46], [  0,   0, 255], [255, 100,   0], [  0, 255, 123], [164,  75, 155],
                            [101, 174, 255], [118, 254, 172], [ 60,  91, 112], [255, 255,   0], [255, 255, 125],
                            [255,   0, 255], [100,   0, 255], [  0, 172, 254], [  0, 255,   0], [171, 175,  80],
                            [101, 193, 60]])
    elif imageID == 'SA':  # Salinas
        row = 512
        col = 217
        palette = np.array([[140,  67,  46], [  0,   0, 255], [255, 100,   0], [  0, 255, 123], [164,  75, 155],
                            [101, 174, 255], [118, 254, 172], [ 60,  91, 112], [255, 255,   0], [255, 255, 125],
                            [255,   0, 255], [100,   0, 255], [  0, 172, 254], [  0, 255,   0], [171, 175,  80],
                            [101, 193, 60]])
    palette = palette * 1.0 / 255
    X_result = np.zeros((labels.shape[0], 3))

    if background == 1:
        Result_all = Result_all
    else:
        Result_all = labels
        Result_all[testIndex] = y_pred

    for i in range(1, num_class + 1):
        X_result[np.where(Result_all == i), 0] = palette[i - 1, 0]
        X_result[np.where(Result_all == i), 1] = palette[i - 1, 1]
        X_result[np.where(Result_all == i), 2] = palette[i - 1, 2]
    X_result = np.reshape(X_result, (col, row, 3))
    X_result_1 = X_result.swapaxes(0, 1)
    plt.figure(dpi=dpi)
    plt.axis("off")
    plt.imshow(X_result_1)
    plt.show()
    return X_result

def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX, pca

