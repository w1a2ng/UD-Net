'''
Created on 2019年3月17日

@author: vcc
'''
import dicom
import cv2
import numpy as np
import os
from multiprocessing.pool import ThreadPool


class Reader:
    '''
    读取各文件夹下的图像文件进行操作
    '''
    dcm_paths = []
    png_paths = []

    def __init__(self):
        '''
        Constructor
        '''
        
    def scan_pic(self, path='../data/source/train'):
        '''
        扫面目录下面所有dcm和png掩膜文件
        此方法对顺序敏感，如果成对出现文件名需一一对应，掩膜需要有_mask标识
        '''
        if os.path.exists(path):
            
            for file in os.listdir(path):
                
                if os.path.isfile(os.path.join(path,file)):
                    fname = os.path.splitext(file)
                    
                    if fname[-1].lower() == '.dcm':
                        self.dcm_paths.append(os.path.join(path, file))
                        
                    elif fname[-1].lower() == '.png':
                        
                        if fname[0].find('_mask') != -1:
                            self.png_paths.append(os.path.join(path, file))
                else:
                    self.scan_pic(os.path.join(path, file))

    def read_dcm(self, path):
        '''
        读取ct图像，返回ct array
        '''
        f = dicom.read_file(path)
        img = f.pixel_array * f.RescaleSlope + f.RescaleIntercept
        return img
    
    def read_png(self, path):
        '''
        读取png图像，返回img
        '''
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        img = img.astype(np.float32)
        return img
    
    def read_boost(self):
        '''
        读取预读取路径图像，返回图像掩膜
        '''
        img = []
        mask = []
        
        if len(self.dcm_paths) == 0 and len(self.png_paths) == 0:
            self.scan_pic()
        
        if len(self.dcm_paths) != 0 and len(self.png_paths) != 0:
            if len(self.dcm_paths) == len(self.png_paths):
                
                with ThreadPool(2) as p:
                    img = p.map(self.read_dcm, self.dcm_paths)
                    mask = p.map(self.read_png, self.png_paths)
                
        return img,mask

if __name__ == '__main__':
    reader = Reader()
    reader.scan_pic()