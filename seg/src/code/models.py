from keras.models import Model,Sequential
from keras.layers import Conv2D,MaxPooling2D,Input,concatenate,UpSampling2D,Dropout,Dense,BatchNormalization,Activation,TimeDistributed
from keras.optimizers import Adam,SGD
from keras.utils import plot_model
from keras.applications import DenseNet121
from keras.layers.pooling import GlobalMaxPooling2D
import metrics
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
'''
模型模块：
u-net： 用于对医学图像进行语义分割，
densenet121： 对分割后的语义提取局部特征分类
'''

def unet(input_size=(512, 512, 1),pretrained_weights=None,batch_norm=False,activation ='relu'):
    
    def conv_usebn(input_layer,kernel,batch_norm = batch_norm):
        conv = Conv2D(kernel, 3, padding='same', kernel_initializer='he_normal')(input_layer)
        if batch_norm:
            conv = BatchNormalization()(conv)
        return conv
    
    def block_compress(input_layer,kernel,activation = activation,batch_norm = batch_norm):
        conv = conv_usebn(input_layer, kernel, batch_norm)
        conv = Activation(activation)(conv)
        conv = conv_usebn(conv, kernel, batch_norm)
        conv = Activation(activation)(conv)
        pool = MaxPooling2D(pool_size=(2, 2))(conv)
        return conv,pool
    
    def block_decompress(input_layer,up_layer,kernel,activation = activation,batch_norm = batch_norm):
        up = Conv2D(512, 2, activation=activation, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(input_layer))
        merge = concatenate([up_layer, up], axis=3)
    
        conv = conv_usebn(merge, kernel, batch_norm)
        conv = conv_usebn(conv, kernel, batch_norm)
        return conv
    
    inputs = Input(input_size)
    
    conv1,pool1 = block_compress(inputs, 64, activation, True)
    conv2,pool2 = block_compress(pool1, 128, activation, True)
    conv3,pool3 = block_compress(pool2, 256, activation, True)
    conv4,pool4 = block_compress(pool3, 512, activation, True)
      
    conv = conv_usebn(pool4, 1024, True)
    conv = conv_usebn(conv, 1024, True)
      
    conv = block_decompress(conv, conv4, 512, activation, True)
    conv = block_decompress(conv, conv3, 256, activation, True)
#     conv = conv_usebn(pool2, 256, True)
#     conv = conv_usebn(conv, 256, True)
    conv = block_decompress(conv, conv2, 128, activation, True)
    conv = block_decompress(conv, conv1, 64, activation, True)
    
    conv = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    conv = Conv2D(1, 1, activation='sigmoid')(conv)
    
    model = Model(inputs=inputs , outputs=conv)
      
    model.compile(optimizer=Adam(lr=1e-4), loss=metrics.dice_loss, metrics=['accuracy',Metrics.dice])

    model.summary()
    
    if pretrained_weights != None:
        model.load_weights(pretrained_weights)
    
    return model


def densenet121(input_shape=(512,512,3),use_pretrained=True):
    '''
    相比resnet小训练集的情况下可以防止过拟合
    网络参数更少，计算成本更低
    '''
    if use_pretrained:
        model = DenseNet121(weights='imagenet',include_top=False,pooling='avg')
        for layer in model.layers:
            layer.trainable = False
    else:
        model = DenseNet121(weights=None,include_top=False,pooling='avg')
     
    inputs = Input((None,input_shape[0],input_shape[1],input_shape[2]))
#     model = Flatten()(model)
#     model = GlobalMaxPooling2D()(model)
    model = TimeDistributed(model)(inputs)
    model = Bidirectional(LSTM(512,dropout=0.3,activation='relu'))(model)
    model = Dense(1024,activation='relu')(model)
    model = Dropout(0.2)(model)
    model = Dense(512,activation='relu')(model)
    out = Dense(2,activation='sigmoid')(model)
     
    model = Model(inputs=inputs , outputs=out)
    
    model.compile(optimizer = Adam(lr=1e-4),loss='binary_crossentropy', metrics=['accuracy'])    
    model.summary()

    return model

def cnn_3d():
    return

if __name__ == '__main__':
#     model = unet()
#     plot_model(model, to_file='../product/model/unet.png', show_shapes=True)
    model = densenet121(use_pretrained=False)
    plot_model(model, to_file='../product/model/densenet121.png', show_shapes=True)