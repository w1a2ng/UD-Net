'''
Created on 2019年3月28日

@author: vcc
'''
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from reader import Reader
from process import Process
import models


def train_unet(batch_size, epoch):
    spe = 594 // batch_size
    reader = Reader()
    process = Process()
    model = models.unet((256,256,1),batch_norm=True)
    checkpoint = ModelCheckpoint('../product/model/unet_weights_best.h5', save_best_only=True, save_weights_only=True)
    earlystop = EarlyStopping(patience=4)
    reader.scan_pic()
#     img, mask = reader.read_boost()
    img ,mask = reader.dcm_paths , reader.png_paths
    train_img, val_img, train_mask, val_mask = train_test_split(img, mask, test_size=0.2, random_state=42)
    generator = process.unet_generator(train_img, train_mask, batch_size=batch_size)
    validation_data = process.unet_generator(val_img, val_mask, batch_size=batch_size)
    return model.fit_generator(generator, steps_per_epoch=spe, epochs=epoch, callbacks=[checkpoint,earlystop], validation_data = validation_data, validation_steps=spe)

if __name__ == '__main__':
    history = train_unet(1, 2)
