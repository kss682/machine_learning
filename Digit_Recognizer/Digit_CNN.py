import time
import numpy as np
import pandas as pd
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU




#data preprocessing
df_trainingdata=pd.read_csv('train.csv')
df_testingdata=pd.read_csv('test.csv')


training_data=df_trainingdata.drop(['label'],axis=1)
training_label=df_trainingdata['label']


X_train=np.array(training_data)
y_train=np.array(training_label)
X_test=np.array(df_testingdata)

X_train=X_train.reshape(-1,28,28,1)
X_test=X_test.reshape(-1,28,28,1)

print("Training data shape:",X_train.shape)
print("Testing data shape:",X_test.shape)


X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
X_train=X_train/255.0
X_test=X_test/255.0


y_train=to_categorical(y_train)

xtrain,xtest,ytrain,ytest=train_test_split(X_train,y_train,test_size=0.2,random_state=12)


#Model
Digit_Model=Sequential()
Digit_Model.add(Conv2D(32,kernel_size=(3,3),activation='linear',input_shape=(28,28,1),padding='same'))
Digit_Model.add(LeakyReLU(alpha=0.1))
Digit_Model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
Digit_Model.add(Dropout(0.25))
Digit_Model.add(Conv2D(64,kernel_size=(3,3),activation='linear',input_shape=(28,28,1)))
Digit_Model.add(LeakyReLU(alpha=0.1))
Digit_Model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
Digit_Model.add(Dropout(0.25))
Digit_Model.add(Conv2D(128,kernel_size=(3,3),activation='linear',input_shape=(28,28,1)))
Digit_Model.add(LeakyReLU(alpha=0.1))
Digit_Model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
Digit_Model.add(Dropout(0.25))
Digit_Model.add(Flatten())
Digit_Model.add(Dense(128,activation='linear'))
Digit_Model.add(LeakyReLU(alpha=0.1))
Digit_Model.add(Dense(10,activation='softmax'))



Digit_Model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
start=time.time()
Digit_Model.fit(xtrain,ytrain,batch_size=64,epochs=10,verbose=1,validation_data=(xtest,ytest))
end=time.time()
print("Time taken :",end-start)

test_eval=Digit_Model.evaluate(xtest,ytest)
print("Losses :",test_eval[0])
print("Accuracy :",test_eval[1])


predictions=Digit_Model.predict(X_test)
predictions=np.argmax(np.round(predictions),axis=1)

df=pd.DataFrame(predictions)
df.to_csv('result.csv')





