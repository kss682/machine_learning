import numpy as np


class NN():
	#def __init__(self):
	#	pass


	def sigmoid(self,x,deriv=False):
		if(deriv==True):
			return x*(1-x)
		else:
			return 1/(1+np.exp(-x))



	def layers(self,X_train,y_train):
		X_train=np.array(X_train)	
		y_train=np.array(y_train)
		hiddenlayerweights=np.random.randint(1,size=(X_train.shape[1],50))
		hiddenlayerbias=np.ones([X_train.shape[0],50])
		outputlayerweights=np.random.randint(1,size=(50,1))
		outputlayerbias=np.ones([X_train.shape[0],1])

		


		#optimization using gradient descent
		for i in range(100000000):
			l1=np.add(np.dot(X_train,hiddenlayerweights),hiddenlayerbias)
			l1=self.sigmoid(l1)
			output=np.add(np.dot(l1,outputlayerweights),outputlayerbias)
			output=self.sigmoid(output)
			#print(y_train.shape)
			#print(output.shape)			
			output_error=y_train-output
		
			if i%100000==0:
				print(np.mean(np.abs(output_error)))

			output_delta=output_error*self.sigmoid(output,deriv=True)
			#print(output_delta.shape)
			#print(self.outputlayerweights.shape)
			layer_error=output_delta.dot(outputlayerweights.transpose())
			layer_delta=layer_error*self.sigmoid(l1,deriv=True)

			hiddenlayerweights=np.add(hiddenlayerweights,np.dot(X_train.transpose(),layer_delta))
			outputlayerweights=np.add(outputlayerweights,np.dot(l1.transpose(),output_delta))	
			
	
X=[[0,0,0,0,0],[0,0,0,0,1],[0,0,0,1,0]]
y=[[0],[1],[2]]
clf=NN()
clf.layers(X,y)


	
