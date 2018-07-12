import pandas as pd
import numpy as np
from sklearn import datasets,model_selection,preprocessing,svm


#data processing
iris=datasets.load_iris()
X=iris.data
y=iris.target
X=np.array(X)
y=np.array(y)
X=preprocessing.scale(X)

X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.2)
#classifier
for kernel in ['linear','rbf','poly']:

	clf=svm.SVC(kernel=kernel,random_state=0,gamma=10,C=1)
	clf.fit(X_train,y_train)
	print(clf.score(X_test,y_test))

