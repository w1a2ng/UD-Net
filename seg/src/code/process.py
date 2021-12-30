'''
Created on 2019年3月18日

@author: vcc
'''

import numpy as np
import cv2
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import random
from reader import Reader


class Process:
    '''
    图像数据加工
    '''
    
    def __init__(self):
        '''
        Constructor
        '''
    
    def shift(self, img, width_value=None, height_value=None, randint=None):
        '''
        随机上下左右平移，空缺处填零
        '''
        width = img.shape[1]
        height = img.shape[0]
        new_img = np.zeros(img.shape)
        
        def shift_pixel(length, value):
            shift = int(length * value)
            shift = [0, shift]
            shift.sort()
            return shift[randint]
        
        if width_value != None:
            if randint == 0:
                width_value = -width_value
            w_shift = shift_pixel(width, width_value)
            for i in range(height):
                for j in range(width):
                    if (j + w_shift) < 0 or (j + w_shift) >= width:
                        continue
                    new_img[i, j + w_shift] = img[i, j]
            img = new_img
        
        if height_value != None:
            if randint == 0:
                height_value = -height_value
            h_shift = shift_pixel(height, height_value)
            for i in range(height):
                if (i + h_shift) < 0 or (i + h_shift) >= height:
                    continue
                new_img[i + h_shift] = img[i]
        return new_img
    
    def rotate_zoom(self, img, angle=0, zoom_rate=1):
        '''
        旋转缩放,有插值
        '''
        width = img.shape[1]
        height = img.shape[0]
        
        matRotate = cv2.getRotationMatrix2D((height * 0.5, width * 0.5), angle, zoom_rate)
        new_img = cv2.warpAffine(img, matRotate, (height, width))
        return new_img
    
    def resize(self, img):
        width = img.shape[1]
        height = img.shape[0]
        return cv2.resize(img, (height // 2, width // 2), interpolation=cv2.INTER_NEAREST)
    
    def unet_dataEnhance(self, img, mask, rotation_range=30, width_shift_range=0.05, height_shift_range=0.1, zoom_range=0.8, seed=5):
        '''
        简易随机数据增强
        '''
        img_list = []
        mask_list = []
        for i in range(10):
            randint = random.randint(0, 1)
            choose = random.randint(0, 1)
            if choose == 0:
                img_list.append(self.shift(img, width_shift_range, randint=randint))
                mask_list.append(self.shift(mask, width_shift_range, randint=randint))
            elif choose == 1:
                img_list.append(self.shift(img, height_value=height_shift_range, randint=randint))
                mask_list.append(self.shift(mask, height_value=height_shift_range, randint=randint))
#             elif choose == 2:
#                 angle = random.randint(-rotation_range, rotation_range)
#                 img_list.append(self.rotate_zoom(img, angle))
#                 mask_list.append(self.rotate_zoom(mask, angle))
#             elif choose == 3:
#                 zoom = random.randint(zoom_range * 100, zoom_range * 150) / 100
#                 img_list.append(self.rotate_zoom(img, zoom_rate=zoom))
#                 mask_list.append(self.rotate_zoom(mask, zoom_rate=zoom))
        random.seed(seed)
        random.shuffle(img_list)
        random.seed(seed)
        random.shuffle(mask_list)
        return img_list, mask_list
    
    def info_throwout(self, img, threshold_value=-10, usefilter=None):
        '''
        丢弃一部分图像信息强化特征
        img: 图像array矩阵
        threshold_value: 按照阈值丢弃全局信息,并二值化
        filter: 选择滤波器去除图像噪音，可选参数bilater（双边滤波）和graussian（高斯滤波）
        '''
#         if np.max(img) > 1:
#             pass
#         else:
#             if threshold_value < 1 and threshold_value > 0:
#                 img[img <= (threshold_value / 2)] = 0.
#                 img[img <= threshold_value] /= 2.
        _, img = cv2.threshold(img, threshold_value, 1, cv2.THRESH_BINARY)
        
        if usefilter == None:
            pass
        elif usefilter == 'bilater':
            img = cv2.bilateralFilter(img, 3, 140, 140)
        elif usefilter == 'graussian':
            img = cv2.GaussianBlur(img, (3, 3), 1.8)
        
        return img
             
    def histogram_equalization(self, img , clipLimit=2.0):
        '''
        直方图均衡化
        '''
        if img.dtype != np.int16:
            img = img.astype(np.uint16)
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(4, 4))
        img = clahe.apply(img)
        
        # 提升亮度
        
        return img 
    
    def data_processing(self, *img_mask):
        '''
        图像预处理，归一化
        '''
        img = self.info_throwout(img_mask[0], -10)
        img = self.mask_interests(img_mask[0], img)
        
        if len(np.unique(img_mask[1])) == 2 or len(np.unique(img_mask[1])) == 1:
            pass
        else:
            print(img_mask[1])
            print(np.unique(img_mask[1]))
            raise Exception('掩模图应当为二值图！')
        
        minnum = np.min(img)
        if minnum < 0:
            if minnum > -1000:
                img[img == 0] = -1000
                img += 1000
            else:
                img[img == 0] = min - 1000
                img += (min - 1000)
        else:
            raise Exception('未经处理/处理失败的图像')
        
        img = self.histogram_equalization(img).astype('float32')
        img = (img_mask[0] - np.min(img_mask[0])) / (np.max(img_mask[0]) - np.min(img_mask[0]))
        mask = img_mask[1] / 255.
        if len(img.shape) == 2:
            img = img.reshape((img.shape[0], img.shape[1], 1))
        if len(mask.shape) == 2:
            mask = mask.reshape((mask.shape[0], mask.shape[1], 1))
        
#         mask = to_categorical(mask, num_classes=len(np.unique(img_mask[1])))
        return img, mask
    
    def unet_generator(self, imglist, masklist, batch_size=32 , seed=5):
#         dataGenerator = ImageDataGenerator(
#             rotation_range=30,
#             width_shift_range=0.05,
#             height_shift_range=0.1,
#             zoom_range=0.05,
#             shear_range=0.05,
#             validation_split = 0.1
#             )
#         image_gen = dataGenerator.flow(imglist, 0, batch_size = batch_size, seed=seed)
#         mask_gen = dataGenerator.flow(imglist, 0, batch_size = batch_size, seed=seed)
        reader = Reader()
        img_list = []
        mask_list = []
        data = []
        label = []
        list_len = len(imglist)
        start = 0
        if len(masklist) == list_len:
            while True:
                if start > list_len:
                    break
                for i in range(start, list_len):
                    if len(img_list) > batch_size and len(mask_list) > batch_size:
                        img_batch = img_list[:batch_size]
                        mask_batch = mask_list[:batch_size]
                        img_list = img_list[batch_size:]
                        mask_list = mask_list[batch_size:]
                        start = i
                        break
                    img = reader.read_dcm(imglist[i])
                    mask = reader.read_png(masklist[i])
                    img = self.resize(img)
                    mask = self.resize(mask)
                    img_list.append(img)
                    mask_list.append(mask)
                    nimglist, nmasklist = self.unet_dataEnhance(img, mask, seed=seed)
                    img_list.extend(nimglist)
                    mask_list.extend(nmasklist)
                    
                unet_gen = zip(img_batch, mask_batch)
                
                for each in unet_gen:
                    img , mask = self.data_processing(each[0], each[1])
                    data.append(img)
                    label.append(mask)
                yield (np.array(data), np.array(label))
                data.clear()
                label.clear()
        else:
            raise Exception("读入文件数量不一致")
    
    def mask_interests(self, img, mask):
        '''
        通过掩膜图切割图象
        '''
        if len(mask.shape) == len(img.shape) == 2:
        
#             for i in mask:
#                 for j in mask[i]:
#                     if mask[i][j] == 0.:
#                         img[i][j] = 0.
            
            return img * mask
        elif len(mask.shape) == len(img.shape) == 3:
            if mask.shape[2] == img.shape[2] == 1:
                return img * mask

    def to_3d(self, imglist, label, length=10):
        '''
        将切割后的1d序列分离为3d序列
        imglist格式:[[array1,array2,..],[array1,array2,..],[array1,array2,..],]
        '''
        for eachlist in imglist:
            for eachimg in eachlist:
                shape = eachimg.shape
                if len(shape) == 2:
                    eachimg.reshape((shape[0], shape[1], 1))
                    
                elif len(shape) == 3:
                    if shape[2] == 1:
                        eachimg
                else:
                    raise Exception('输入图像维度错误')
