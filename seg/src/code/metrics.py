
'''
Created on 2019骞�3鏈�21鏃�

@author: vcc
'''
import keras.backend as k

import numpy
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score


class Metrics(Callback):
    '''
    瀵规ā鍨嬬殑璇勪及鏂瑰紡
    dice绯绘暟
    f-score
    '''

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (numpy.asarray(self.model.predict(
            self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        return
    
    @staticmethod
    def dice(y_true, y_pred):
        '''
        dice绯绘暟锛岄槻姝㈠垎姣嶄负闆讹紝涓婁笅鍚屾椂鐩稿姞鍊硷細1e-7
        '''
        label = k.flatten(y_true)
        predict = k.flatten(y_pred)
        intersection = k.sum(label * predict)
        return (2.*intersection + k.epsilon()) / (k.sum(label) + k.sum(predict) + k.epsilon())
    
    @staticmethod
    def dice_loss(y_true, y_pred):
        return - Metrics.dice(y_true, y_pred)
    
