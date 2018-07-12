import  pandas as pd 
import numpy as np
from sklearn import preprocessing, model_selection
from sklearn.ensemble import RandomForestClassifier
#to change sex to binary
def gentobin(df):
        if df['Sex']=='male':
                return 1
        else:
                return 0
#to change embarked to 1  2 3
def chgembark(df):
	if df['Embarked']=='C':
		return 1
	elif df['Embarked']=='Q':
		return 2
	else:
		return 3

df_train=pd.read_csv('train.csv')
df_train=df_train.drop(['PassengerId','Name','Ticket','Cabin'],1)

df_train['Sex-']=df_train.apply(gentobin,axis=1)
df_train=df_train.drop(['Sex'],1)

df_train['Embarked-']=df_train.apply(chgembark,axis=1)
df_train=df_train.drop(['Embarked'],1)
df_train.fillna(value=-99999, inplace=True)
#print(df_train)

#data set to be trained
df_test=pd.read_csv('test.csv')
df_test=df_test.drop(['PassengerId','Name','Ticket','Cabin'],1)

df_test['Sex-']=df_test.apply(gentobin,axis=1)
df_test=df_test.drop(['Sex'],1)

df_test['Embarked-']=df_test.apply(chgembark,axis=1)
df_test=df_test.drop(['Embarked'],1)
df_test.fillna(value=-99999, inplace=True)
#print(df_test)


X=np.array(df_train.drop(['Survived'],1))
X_predict=np.array(df_test)
y=np.array(df_train['Survived'])

X=preprocessing.scale(X)
X_predict=preprocessing.scale(X_predict)

X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.2)

#classifieer to be choosen
clf=RandomForestClassifier(criterion='entropy', random_state=0,n_jobs=-1)
clf.fit(X_train,y_train)
confidence=clf.score(X_test,y_test)
print(confidence)

predict_sur=clf.predict(X_predict)
for i in predict_sur:
	if i>=0.5:
		i=1
	else:
		i=0
print(predict_sur)
df=pd.DataFrame(predict_sur)
df.to_csv("result.csv")

#from matplotlib import pyplot as plt

#plt.scatter(X_predict,predict_sur)
#plt.xlabel('features')
#plt.ylabel('survived')
#plt.show()
