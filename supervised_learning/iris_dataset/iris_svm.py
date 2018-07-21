
import numpy as np
from sklearn import datasets,model_selection,preprocessing
from sklearn import svm

#data processing
iris=datasets.load_iris()
X=iris.data
y=iris.target
X=np.array(X)
y=np.array(y)
X=preprocessing.MinMaxScaler().fit_transform(X)

X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.2)
#classifier
for kernel in ['linear','sigmoid','rbf']:

        clf=svm.SVC(kernel=kernel,gamma=0.1,C=1)
        clf.fit(X_train,y_train)
        print(clf.score(X_test,y_test))


