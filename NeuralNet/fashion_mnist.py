"""
The is a sample work using ConvNet for the fashion_mnist datasets.
This is based on datacamp tutorial on ConvNet.  

"""


from keras.datasets import fashion_mnist
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#loading the data
(X_train,y_train),(X_test,y_test)=fashion_mnist.load_data()


#Analyse the data
print("Training data shape:",X_train.shape, y_train.shape)

print("Testing data shape:", X_test.shape,y_test.shape)

classes=np.unique(y_train)
nclass=len(classes)

print("Total classes:",nclass)
print("The classes:", classes)

plt.figure(figsize=[5,5])

# Display the first image in training data
plt.subplot(121)
plt.imshow(X_train[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(y_train[0]))

# Display the first image in testing data
plt.subplot(122)
plt.imshow(X_test[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(y_test[0]))

plt.show()

#data preprocessing
X_train=X_train.reshape(-1,28,28,1)
X_test=X_test.reshape(-1,28,28,1)


X_train=X_train.astype('float32')
X_train=X_train.astype('float32')
X_train=X_train/255.0
X_test=X_test/255.0


y_train=to_categorical(y_train)
y_test=to_categorical(y_test)



xtrain,xtest,ytrain,ytest=train_test_split(X_train,y_train,test_size=0.2,random_state=12)


#Model
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU



batch_size = 64
epochs = 20
num_classes = 10




fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,1),padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2, 2),padding='same'))
fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(Dense(num_classes, activation='softmax'))




fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])



fashion_train = fashion_model.fit(xtrain,ytrain, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(xtest, ytest))


test_eval = fashion_model.evaluate(X_test,y_test, verbose=0)

print("Test loss:",test_eval[0])
print("Test Accuracy:",test_eval[1] )





