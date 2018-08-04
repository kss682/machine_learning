import pandas as pd
import numpy as np
from sklearn import model_selection
import tensorflow as tf
from tensorflow import keras

#data to train
df=pd.read_csv("train.csv")
df.fillna(value=-99999,inplace=True)
df_image=df.drop(["label"],1)
df_label=df["label"]

image=np.array(df_image)
label=np.array(df_label)

image=image/255.0
X_train,X_test,y_train,y_test=model_selection.train_test_split(image,label,test_size=0.2)


#data to test
df_test=pd.read_csv('test.csv')
df_test.fillna(value=-99999,inplace=True)
image_test=np.array(df_test)
image_test=image_test/255.0



#MODEL
model=keras.models.Sequential([
	keras.layers.Flatten(),
	keras.layers.Dense(500,activation='sigmoid'),
	keras.layers.Dense(500,activation='sigmoid'),
	keras.layers.Dense(10,activation='softmax'),
])

model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),loss=keras.losses.sparse_categorical_crossentropy,metrics=[keras.metrics.categorical_accuracy])
model.fit(X_train,y_train,epochs=10,batch_size=1000)

print(model.evaluate(X_test,y_test))

prediction=model.predict(image_test)


value=[np.argmax(prediction[i]) for i in range(len(prediction))]
print(value[:10])


#to csv
df_result=pd.DataFrame(value)
df_result.to_csv('result.csv')


