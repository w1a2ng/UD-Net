'''
Created on 2019年4月3日

@author: vcc
'''

import models
from reader import Reader
from process import Process

if __name__ == '__main__':
    reader = Reader()
    reader.scan_pic('../data/source/test')
    # 这里的predictmask为空列表，用于存放模型输出结果
    imglist , predictmask = reader.read_boost()
    unet = models.unet(pretrained_weights='../product/model/unet_weights_best.h5',batch_norm=True)
    predictmask = unet.predict_class(imglist)
    