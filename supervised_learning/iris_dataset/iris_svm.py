
import numpy as np
from sklearn import datasets,preprocessing
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

#data processing
iris=datasets.load_iris()
X=iris.data
y=iris.target
X=np.array(X)
y=np.array(y)


#X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.2)
#classifier
for kernel in ['linear','sigmoid','rbf']:

        clf=make_pipeline(preprocessing.MinMaxScaler(),svm.SVC(kernel=kernel,gamma=0.1,C=1))
        
        print(cross_val_score(clf,X,y,cv=5).mean())


