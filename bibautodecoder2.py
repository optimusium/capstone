# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:07:03 2019

@author: boonping
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 13:51:31 2019

@author: boonping
"""

#Section 0: Import Libraries
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.model_selection import train_test_split


from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten,Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D,MaxPooling2D,UpSampling2D
from tensorflow.keras.layers import add,Lambda,subtract,multiply,concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical,plot_model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import optimizers
from tensorflow.keras import backend
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import IPython


from matplotlib import font_manager as fm
fpath       = os.path.join(os.getcwd(), "ipam.ttf")
prop        = fm.FontProperties(fname=fpath)

#Section 1: Gray Plot Setting to print out wafer resistance map

                            # Set up 'ggplot' style
plt.style.use('ggplot')     # if want to use the default style, set 'classic'
plt.rcParams['ytick.right']     = True
plt.rcParams['ytick.labelright']= True
plt.rcParams['ytick.left']      = False
plt.rcParams['ytick.labelleft'] = False
plt.rcParams['font.family']     = 'Arial'

# Create a functin do plot gray easily
def grayplt(img,title=''):
    '''
    plt.axis('off')
    if np.size(img.shape) == 3:
        plt.imshow(img[:,:,0],cmap='gray',vmin=0,vmax=1)
    else:
        plt.imshow(img,cmap='gray',vmin=0,vmax=1)
    plt.title(title, fontproperties=prop)
    '''
    
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')

    # Show the image
    if np.size(img.shape) == 3:
        ax.imshow(img[:,:,0],cmap='hot',vmin=0,vmax=1)
    else:
        ax.imshow(img,cmap='hot',vmin=0,vmax=1)
    print(title)
    plt.show()

# .............................................................................
#Section 2: Unlabelled data processing 
#2.1 Source: 1090 unlabelled resistance data. Each line represnts 42x42 locations with resistance measurement. In 1D 
raw_data = pd.read_excel('bibtest.xlsx',sheet_name = "bibtest")
raw_data=raw_data.drop(labels=["session","SourceDip","HeatTrap","Other"],axis=1)
sdarray=raw_data
#2.2 8:2 train/test data split. Ignore the label data as autoencoder training is unsupervised.
sdarray_train,sdarray_test,label_train,label_test = train_test_split(sdarray,sdarray,test_size = 0.1)

#2.3 convert array to numpy array. still in 1D for each entry
sdarray_train_np=sdarray_train.as_matrix()
sdarray_test_np=sdarray_test.as_matrix()

#2.4 reshape into 2D array for each entry (shape=(entry,42,42))
twoDarray_train = sdarray_train_np.reshape((sdarray_train_np.shape[0], 20, 16))
twoDarray_test = sdarray_test_np.reshape((sdarray_test_np.shape[0], 20, 16))


trDat=twoDarray_train   
tsDat=twoDarray_test
trDat2=twoDarray_train   
tsDat2=twoDarray_test

#print(twoDarray_train[0])
#raise

#2.5 Scaling. 500ohm is the maximum resistance value possible. Change to float type
trDat       = trDat.astype('float32')/1 #/255
tsDat       = tsDat.astype('float32')/1 #/255
for i in range(len(trDat)):
    temp=(trDat[i]-np.min(trDat[i]))/(np.max(trDat[i])-np.min(trDat[i]) )
    trDat[i]=temp
for i in range(len(tsDat)):
    temp=(tsDat[i]-np.min(tsDat[i]))/(np.max(tsDat[i])-np.min(tsDat[i]) )
    tsDat[i]=temp
trDat2       = trDat2.astype('float32')/1 #/255
tsDat2       = tsDat2.astype('float32')/1 #/255

#2.6 Reshape
# Retrieve the row size of each image
# Retrieve the column size of each image
imgrows     = tsDat.shape[1]
imgclms     = tsDat.shape[2]

trDat       = trDat.reshape(trDat.shape[0],
                            imgrows,
                            imgclms,
                            1)
tsDat       = tsDat.reshape(tsDat.shape[0],
                            imgrows,
                            imgclms,
                            1)
trDat2       = trDat2.reshape(trDat2.shape[0],
                            imgrows,
                            imgclms,
                            1)
tsDat2       = tsDat2.reshape(tsDat2.shape[0],
                            imgrows,
                            imgclms,
                            1)

# fix random seed for reproducibility
seed        = 29
np.random.seed(seed)

modelname   = 'autoencoder'

#Section 5: define the deep learning model
#reference: https://www.datacamp.com/community/tutorials/autoencoder-classifier-python
def createModel():
    #5.1 Set input shape which is 1 channel of 42x42 2D image
    inputShape=(20,16,1)
    inputs      = Input(shape=inputShape)
    inputs2   = Conv2D(16, (2,2), padding="same",strides=(1,1),kernel_initializer='he_normal', activation='relu')(inputs)
    inputs2   = Conv2D(32, (2,2), padding="same",strides=(1,1),kernel_initializer='he_normal', activation='relu')(inputs2)
    
    #5.2 Auto encoder
    #This is to process the input so the neccessary feature that can be used for image reconstruction can be gathered.
    #5.2.1 First hidden layer. 64 neurons, downsize the image to (21,21). Conv2D is chosen over maxPooling as it allow the filter kernel to be trained.
    
    '''
    x           = Conv2D(64, (8,1), padding="same",strides=(1,2),kernel_initializer='he_normal', activation='relu')(inputs)
    encoded     = Conv2D(40, (4,1), padding="same",strides=(1,2),kernel_initializer='he_normal', activation='relu')(x)
    x           = UpSampling2D(size=(1, 2))(encoded)
    x           = Conv2D(40, (4,1), padding="same",kernel_initializer='he_normal', activation='relu')(x)
    x           = UpSampling2D(size=(1, 2))(x)
    x           = Conv2D(64, (8,1), padding="same",kernel_initializer='he_normal', activation='relu')(x)
    x           = Dense(1, name="horizontal")(x)
    '''

    '''    
    #horizontal
    x = AveragePooling2D(pool_size=(1, 8), strides=(1,8))(inputs)
    x = Lambda(lambda x: x - 0.375)(x)
    x = Activation("relu")(x)
    x = Lambda(lambda x: x *1024)(x)
    x = Activation("tanh")(x)
    x = UpSampling2D(size=(1, 8))(x)

    #vertical
    x2 = AveragePooling2D(pool_size=(5, 1), strides=(5,1))(inputs)
    x2 = Lambda(lambda x2: x2 - 0.33)(x2)
    x2 = Activation("relu")(x2)
    x2 = Lambda(lambda x2: x2 *1024)(x2)
    x2 = Activation("tanh")(x2)
    x2 = UpSampling2D(size=(5, 1))(x2)

    #heat trap
    x3 = AveragePooling2D(pool_size=(4, 4), strides=(4,4))(inputs)
    x3 = Lambda(lambda x3: x3 - 0.25)(x3)
    x3 = Activation("relu")(x3)
    x3 = Lambda(lambda x3: x3 *1024)(x3)
    x3 = Activation("tanh")(x3)
    x3 = UpSampling2D(size=(4, 4))(x3)
    '''    

    #heat trap
    #x3 = AveragePooling2D(pool_size=(4, 4), strides=(4,4))(inputs2)  
    x3=inputs2
    x3      = Conv2D(32, (4,4), padding="same",strides=(4,4),kernel_initializer='he_normal', activation='relu')(x3)
    x3_2 = Lambda(lambda x3: x3 - 0.2)(x3)
    x3_2 = Activation("relu")(x3_2)
    x3_2 = Lambda(lambda x3_2: x3_2 + 0.2)(x3_2)
    x3_2 = Lambda(lambda x3_2: x3_2 * 8)(x3_2)
    x3 = multiply([x3,x3_2])
     
    #x3 = Activation("tanh")(x3)
    x3 = UpSampling2D(size=(4, 4))(x3)
    #x3 = multiply([x3,inputs])
    x3 = Lambda(lambda x3: x3*0.5)(x3)

    #x3a = AveragePooling2D(pool_size=(2, 2), strides=(2,2))(inputs2)
    x3a=inputs2
    x3a      = Conv2D(32, (2,2), padding="same",strides=(2,2),kernel_initializer='he_normal', activation='relu')(x3a)
    
    x3a_2 = Lambda(lambda x3a: x3a - 0.2)(x3a)
    x3a_2 = Activation("relu")(x3a_2)
    x3a_2 = Lambda(lambda x3a_2: x3a_2 + 0.2)(x3a_2)
    x3a_2 = Lambda(lambda x3a_2: x3a_2 * 8)(x3a_2)
    x3a = multiply([x3a,x3a_2])
    
    #x3a = Activation("tanh")(x3a)
    x3a = UpSampling2D(size=(2, 2))(x3a)
    #x3a = multiply([x3a,inputs])
    x3a = Lambda(lambda x3a: x3a *0.5)(x3a)


    x3 = add([x3,x3a], name="cluster")
    
    #vertical
    #x2 = subtract([inputs,x3])
    x2= inputs2
    #x2 = AveragePooling2D(pool_size=(5, 1), strides=(5,1))(x2)
    x2 = Conv2D(32, (5,1), padding="same",strides=(5,1),kernel_initializer='he_normal', activation='relu')(x2)
    
    x2_2 = Lambda(lambda x2: x2 - 0.2)(x2)
    x2_2 = Activation("relu")(x2_2)
    x2_2 = Lambda(lambda x2_2: x2_2 + 0.2)(x2_2)
    x2_2 = Lambda(lambda x2_2: x2_2 * 8)(x2_2)
    x2 = multiply([x2,x2_2])
    
    #x2 = Activation("tanh")(x2)
    x2 = UpSampling2D(size=(5, 1), name='vertical')(x2)
    #x2= multiply([x2,inputs], name='vertical')
    #x2 = add([x2,x2a])

    #horizontal
    #x = subtract([inputs,x3])
    #x = subtract([x,x2])
    x = inputs2
    #x = AveragePooling2D(pool_size=(1, 8), strides=(1,8))(x)
    x = Conv2D(32, (1,8), padding="same",strides=(1,8),kernel_initializer='he_normal', activation='relu')(x)    
    #x = Lambda(lambda x: x *8)(x)
    x_2 = Lambda(lambda x: x - 0.2)(x)
    x_2 = Activation("relu")(x_2)
    x_2 = Lambda(lambda x_2: x_2 + 0.2)(x_2)
    x_2 = Lambda(lambda x_2: x_2 * 8)(x_2)    
    x = multiply([x,x_2])
    #x = Activation("relu")(x)
    
    
    
    x = UpSampling2D(size=(1, 8), name="horizontal")(x)
    #x = multiply([x,inputs])
    #x = add([x,xa])





    y = subtract([inputs2,x])    
    y = subtract([y,x2])    
    y = subtract([y,x3]) 
    
    #y = Lambda(lambda y: y * 8)(y)
    #y = Lambda(lambda y: y - 0.125)(y)
    y = Activation("relu", name="pepper")(y)
    #y = Lambda(lambda y: y *1024)(y)
    #y = Activation("tanh")(y)
    #y = multiply([y,inputs])
    
    
    #=======================================================

    '''
    x           = Conv2D(32, (1,8), padding="same",strides=(1,8),kernel_initializer='he_normal', activation='relu')(x)
    x           = Lambda(lambda x: x - 0.1)(x)
    x           = Activation("relu")(x)
    x           = Lambda(lambda x: x * 1.5)(x)
    #x           = Lambda(lambda x: x * 8)(x)
    x_code      = Conv2D(16, (2,1), padding="same",strides=(1,1),kernel_initializer='he_normal', activation='relu')(x)
    x           = UpSampling2D(size=(1,8), name="horizontal" )(x_code)
    #x2           = Dense(1)(x2)
    
    x2           = Conv2D(32, (5,1), padding="same",strides=(5,1),kernel_initializer='he_normal', activation='relu')(x2)
    x2           = Lambda(lambda x2: x2 - 0.1)(x2)
    x2           = Activation("relu")(x2)
    x2           = Lambda(lambda x2: x2 * 1.5)(x2)
    #x2           = Lambda(lambda x2: x2 * 8)(x2)
    x2_code      = Conv2D(16, (2,1), padding="same",strides=(1,1),kernel_initializer='he_normal', activation='relu')(x2)
    x2           = UpSampling2D(size=(5, 1), name="vertical" )(x2_code)
    #x2           = Dense(1)(x2)

    x3      = Conv2D(32, (4,4), padding="same",strides=(4,4),kernel_initializer='he_normal', activation='relu')(x3)
    x3      = Lambda(lambda x3: x3 - 0.1)(x3)
    x3      = Activation("relu")(x3)
    x3_code      = Lambda(lambda x3: x3 * 1.5)(x3)
    #x3_code = Lambda(lambda x3: x3 * 16)(x3)
    #x3           = Dropout(0.3)(x3)
    x3           = UpSampling2D(size=(4, 4) )(x3_code)
    x3           = Dense(1)(x3)
    
    x3b      = Conv2D(32, (6,6), padding="same",strides=(4,4),kernel_initializer='he_normal', activation='relu')(x3)
    x3b      = Lambda(lambda x3b: x3b - 0.1)(x3b)
    x3b      = Activation("relu")(x3b)
    x3b_code      = Lambda(lambda x3b: x3b * 1.5)(x3b)
    #x3b_code = Lambda(lambda x3b: x3b * 16)(x3b)
    #x3b          = Dropout(0.3)(x3b)
    x3b          = UpSampling2D(size=(4, 4) )(x3b_code)
    x3b          = Dense(1)(x3b)

    x3c     = Conv2D(32, (2,2), padding="same",strides=(2,2),kernel_initializer='he_normal', activation='relu')(x3)
    x3c      = Lambda(lambda x3c: x3c - 0.1)(x3c)
    x3c      = Activation("relu")(x3c)
    x3c_code      = Lambda(lambda x3c: x3c * 1.5)(x3c)
    #x3c_code = Lambda(lambda x3c: x3c * 4)(x3c)
    #x3c          = Dropout(0.3)(x3c)
    x3c          = UpSampling2D(size=(2, 2) )(x3c_code)
    x3c          = Dense(1)(x3c)
    
    x3           = add([x3,x3b,x3c]) 
    x3           = Lambda(lambda x3: x3 /3, name="cluster")(x3)

    
    y           = Conv2D(32, (2,2), padding="same",strides=(1,1),kernel_initializer='he_normal', activation='relu')(y)
    y           = Conv2D(16, (2,2), padding="same",strides=(1,1),kernel_initializer='he_normal', activation='relu', name="pepper")(y)
    #y           = Dropout(0.3)(y)
    #y           = Conv2D(1, (1,1), padding="same",strides=(1,1),kernel_initializer='he_normal', activation='relu')(y)
    #y           = Dense(1)(y)
    
    #y = Lambda(lambda y: y - 0.1)(y)
    #y = Activation("relu")(y)
    #y = Lambda(lambda y: y *1024)(y)
    #y = Activation("tanh")(y)
    #y           = multiply([y,inputs])
    #y           = subtract([y,x], name="pepper")
    '''


    
    
    
    
   
    #outputs = add([x,x2,x3,y])
    #outputs = Lambda(lambda outputs: outputs /4)(outputs)
    
    outputs = concatenate([x,x2,x3])
    outputs   = Conv2D(16, (2,2), padding="same",strides=(1,1),kernel_initializer='he_normal', activation='relu')(outputs)
    outputs   = Conv2D(8, (2,2), padding="same",strides=(1,1),kernel_initializer='he_normal', activation='relu')(outputs)
    #outputs = Conv2D(1, (2,2), padding="same",strides=(1,1),kernel_initializer='he_normal', activation='relu')(outputs)

    '''
    xx=Flatten()(x_code)
    xx2=Flatten()(x2_code)
    xx3a=Flatten()(x3_code)
    xx3b=Flatten()(x3b_code)
    xx3c=Flatten()(x3c_code)
    
    classifier=concatenate([xx,xx2,xx3a,xx3b,xx3c])
    classifier=Dense(64,kernel_initializer='he_normal', activation='relu')(classifier)
    classifier=Dense(8,kernel_initializer='he_normal', activation='softmax')(classifier)
    '''

    model       = Model(inputs=inputs,outputs=outputs) 
    model.compile(loss='mean_squared_error', optimizer = optimizers.RMSprop(), metrics=['accuracy'])
    model2       = Model(inputs=inputs,outputs=outputs) 
    model2.compile(loss='mean_squared_error', optimizer = optimizers.RMSprop(), metrics=['accuracy'])
    #model2       = Model(inputs=inputs,outputs=classifier) 
    #model2.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam() , metrics=['accuracy'])    
    return model,model2   

model,model2       = createModel()
modelGo,modelGo2   = createModel()
model.summary() 
model2.summary() 
#raise

def lrSchedule(epoch):
    lr  = 0.75e-3
    if epoch > 195:
        lr  *= 1e-4
    elif epoch > 180:
        lr  *= 1e-3
        
    elif epoch > 160:
        lr  *= 1e-2
        
    elif epoch > 140:
        lr  *= 1e-1
        
    elif epoch > 120:
        lr  *= 2e-1
    elif epoch > 60:
        lr  *= 0.5
        
    print('Learning rate: ', lr)
    
    return lr

LRScheduler     = LearningRateScheduler(lrSchedule)

                            # Create checkpoint for the training
                            # This checkpoint performs model saving when
                            # an epoch gives highest testing accuracy
filepath        = modelname + ".hdf5"
checkpoint      = ModelCheckpoint(filepath, 
                                  monitor='val_acc', 
                                  verbose=0, 
                                  save_best_only=True, 
                                  mode='max')

                            # Log the epoch detail into csv
csv_logger      = CSVLogger(modelname +'.csv')
callbacks_list  = [checkpoint,csv_logger,LRScheduler]

model_train=model.fit(trDat, 
            trDat, 
            validation_data=(tsDat, tsDat), 
            epochs=30, 
            batch_size=10,
            callbacks=callbacks_list)

#7.1 Plotting loss curve for autoencoder.
loss=model_train.history['loss']
val_loss=model_train.history['val_loss']
epochs = range(30)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

modelGo.load_weights(filepath)
modelGo.compile(loss='mean_squared_error', 
                optimizer=optimizers.RMSprop() ,
                metrics=['accuracy'])



predicts_img    = modelGo.predict(tsDat)
intermediate_layer_model = Model(inputs=modelGo.input,
                                 outputs=modelGo.get_layer("horizontal").output)
intermediate_output = intermediate_layer_model.predict(tsDat)

intermediate_layer_modela = Model(inputs=modelGo.input,
                                 outputs=modelGo.get_layer("vertical").output)
intermediate_outputa = intermediate_layer_modela.predict(tsDat)

intermediate_layer_modelb = Model(inputs=modelGo.input,
                                 outputs=modelGo.get_layer("cluster").output)
intermediate_outputb = intermediate_layer_modelb.predict(tsDat)

'''
intermediate_layer_model2 = Model(inputs=modelGo.input,
                                 outputs=modelGo.get_layer("pepper").output)
intermediate_output2 = intermediate_layer_model2.predict(tsDat)
'''
#for i in range(len(tsDat)):
for i in range(0,len(tsDat),4):
#i=0
#if 1:
    grayplt(tsDat[i],title='input')
    #grayplt(tsDat2[i]/np.max(tsDat2[i]),title='input')
    #print(tsDat2[i])
    #print(tsDat[i])
    grayplt(intermediate_output[i],title='horizontal')
    #print(np.max(intermediate_output[i]),np.min(intermediate_output[i]))
    grayplt(intermediate_outputa[i],title='vertical')
    grayplt(intermediate_outputb[i],title='cluster')
    #grayplt(intermediate_output2[i],title='pepper')
    
    grayplt(predicts_img[i],title='reconstructed')
    #grayplt(predicts_img[i]*( np.max(tsDat2[i])-np.min(tsDat2[i]) )+np.min(tsDat2[i]),title='reconstructed')
    #grayplt( (predicts_img[i]*( np.max(tsDat2[i])-np.min(tsDat2[i]) )+np.min(tsDat2[i]))/np.max(tsDat2[i]),title='reconstructed')
#print(tsDat[0])
#print(predicts_img[0])