import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv1D, Flatten, BatchNormalization, LeakyReLU, Input, Dropout, Dense, Add, Dropout
from tensorflow.keras import Model, datasets, models
from tensorflow.keras.optimizers import Adam

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

data = pd.read_csv('cust_satis_train.csv')

data.head()

x = data.drop(labels=['ID','TARGET'], axis=1)

y = data['TARGET']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, stratify=y)

# To Remove constants and Quasi constants

# We will remove those coloumns which has varaince less than 1 % so automatically columns which are constant and 
# quasi constant wiil be removed

filt = VarianceThreshold(0.01)

x_train = filt.fit_transform(x_train)
x_test = filt.transform(x_test)         # fit_tarnsform is not used in x_test to avoid over fitting

# Remove duplicate features for that we need to transpose x_train and x_test 

x_train_T = x_train.T
x_test_T = x_test.T

x_train_T = pd.DataFrame(x_train_T)
x_test_T = pd.DataFrame(x_test_T)

duplicate_features = x_train_T.duplicated()

features_to_keep = [not index for index in duplicate_features]


x_train = x_train_T[features_to_keep].T
x_test = x_test_T[features_to_keep].T

x_train

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

y_train = np.array(y_train)
y_test = np.array(y_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# CNN MODEL

# Convolutional Neural Network

init = tf.random_normal_initializer(0.,0.2)

def fraud():
    I = Input(shape=x_train[0].shape)
    
    C1 = Conv1D(32, 2, kernel_initializer=init)(I)
    B1 = BatchNormalization()(C1)
    L1 = LeakyReLU()(B1)
    D1 = Dropout(0.5)(L1)
    
    C2 = Conv1D(64, 2, kernel_initializer=init)(D1)
    B2 = BatchNormalization()(C2)
    L2 = LeakyReLU()(B2)
    D2 = Dropout(0.5)(L2)
    
    F3 = Flatten()(D2)   
    DE3 = Dense(64)(F3)
    L3 = LeakyReLU()(DE3)
    D3 = Dropout(0.5)(L3)

    
    out = Dense(1, activation='sigmoid')(D3)
    
    model = Model(inputs=I, outputs=out)
    
    return model
    

model = fraud()
model.summary()

model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

train = model.fit(x_train, y_train, validation_split=0.1, batch_size=10, epochs=10)

# Plots to display loss and accuracy

plt.figure()
plt.plot(train.history['accuracy'])
plt.plot(train.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.figure()
plt.plot(train.history['loss'])
plt.plot(train.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()