from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier
from sklearn import preprocessing,model_selection
import numpy as np



iris=datasets.load_iris()
X=iris.data
y=iris.target

scale=preprocessing.MinMaxScaler()

X=scale.fit_transform(X)

X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.2)

clf=AdaBoostClassifier(n_estimators=50,learning_rate=1,random_state=None)
model=clf.fit(X_train,y_train)

print(model.score(X_test,y_test))



