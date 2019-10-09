from keras.models import Model
from keras.layers import Conv2D,BatchNormalization,MaxPooling2D,Input,ZeroPadding2D,Permute,Reshape,Dense,LeakyReLU,Activation,Bidirectional,LSTM,TimeDistributed
from keras.activations import relu
from crnn.util import resizeNormalize,strLableConverter
import numpy as np
import tensorflow as tf

graph=tf.get_default_graph();

def crnn_keras(imgH,nc,nClass,nH,leackRelu=False,lstmFlag=True):

    data_format="channels_first"
    imgInput=Input(shape=(1,imgH,None),name="imgInput")
    ks=[3,3,3,3,3,3,2]
    ps=[1,1,1,1,1,1,0]
    ss=[1,1,1,1,1,1,1]
    nm=[64,128,256,256,512,512,512]
    def convRelu(i,batchNormalization=False,x=None):
        nIn=nc if i==0 else nm[i-1]
        nOut=nm[i]
        if leackRelu:
            activation=LeakyReLU(alpha=0.2)
        else:
            activation=Activation(relu,name='relu{0}'.format(i))

        x=Conv2D(filters=nOut,kernel_size=ks[i],strides=ss[i],padding="valid" if ps[i]==0 else "same",data_format=data_format,use_bias=True,name="cnn.conv{0}".format(i))(x)

        if batchNormalization:
            x=BatchNormalization(epsilon=1e-5,axis=1,momentum=0.1,name="cnn.batchnorm{0}".format(i))(x)
        x=activation(x)
        return x
    x=imgInput
    x=convRelu(0,batchNormalization=False,x=x)
    x=MaxPooling2D(pool_size=(2,2),name="cnn.pooling{0}".format(0),data_format=data_format)(x)

    x=convRelu(1,batchNormalization=False,x=x)
    x=MaxPooling2D(pool_size=(2,2),name="rnn.pooling{0}".format(1),data_format=data_format)(x)

    x=convRelu(2,batchNormalization=True,x=x)
    x=convRelu(3,batchNormalization=False,x=x)
    x=ZeroPadding2D(padding=(0,1),data_format=data_format)(x)
    x=MaxPooling2D(pool_size=(2,2),strides=(2,1),name="rnn.pooling{0}".format(2),data_format=data_format)(x)

    x=convRelu(4,batchNormalization=True,x=x)
    x=convRelu(5,batchNormalization=False,x=x)
    x=ZeroPadding2D(padding=(0,1),data_format=data_format)(x)
    x=MaxPooling2D(pool_size=(2,2),strides=(2,1),name="cnn.pooling{0}".format(3),data_format=data_format)(x)
    x=convRelu(6,batchNormalization=True,x=x)

    x=Permute(dims=(3,2,1))(x)
    x=Reshape(target_shape=(-1,512))(x)

    out=None
    if lstmFlag:
        x=Bidirectional(LSTM(units=nH,return_sequences=True,use_bias=True,recurrent_activation="sigmoid"))(x)
        x=TimeDistributed(Dense(nH))(x)
        x=Bidirectional(LSTM(units=nH,return_sequences=True,use_bias=True,recurrent_activation="sigmoid"))(x)
        out=TimeDistributed(Dense(nClass))(x)
    else:
        out=Dense(nClass,name="linear")(x)
    return Model(imgInput,out)
model=crnn_keras(32,128,1290,256,True,True)
model.summary()


