
import numpy as np
from sklearn import datasets,model_selection,preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier

#data processing
iris=datasets.load_iris()
X=iris.data
y=iris.target
X=np.array(X)
y=np.array(y)
X=preprocessing.MinMaxScaler().fit_transform(X)

X_train,X_test,y_train,y_test=model_selection.train_test_split(X[:140],y[:140],test_size=0.2)
#classifier
MLPC=MLPClassifier(activation='logistic',solver='lbfgs',random_state=1)
ABC=AdaBoostClassifier(n_estimators=50,learning_rate=1,algorithm='SAMME.R',random_state=None)
RFC=RandomForestClassifier(n_estimators=10,criterion='gini',n_jobs=-1)

classifier=None
score=0.95
for clf in [MLPC,ABC,RFC]:
	model=clf.fit(X_train,y_train)
	if model.score(X_test,y_test)>score:
		score=model.score(X_test,y_test)
		print(score)
		print(clf)		
		classifier=clf
if classifier==None:
	classifier=svm.SVC(gamma=0.1,C=1)
	

classifier.fit(X_train,y_train)
	


print(classifier.predict(X[140:]))
print(classifier)
	
