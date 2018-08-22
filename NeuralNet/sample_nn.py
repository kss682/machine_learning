import numpy as np


def sigmoid(x,derive=False):
	if derive==True:
		return x*(1-x)
	else:

		return 1/(1+np.exp(-x))



X=np.random.randint(2,size=(5,8))

Y=np.random.randint(2,size=(5,1))


hiddenweights=np.random.randn(8,20)
hiddenbias=np.ones([5,20])
outputweights=np.random.randn(20,1)
outputbias=np.ones([5,1])	





for i in range(60000):

	



	l0=X
	l1=sigmoid(np.dot(l0,hiddenweights)+hiddenbias)
	l2=sigmoid(np.dot(l1,outputweights)+outputbias)

	
	l2_error=Y-l2

	
	if i%10000==0:
		print("error"+str(np.mean(np.abs(l2_error))))

	#shape of l2(5,1)
	
	l2_delta=l2_error*sigmoid(l2,derive=True)
	

	l1_error=l2_delta.dot(outputweights.transpose())

	l1_delta=l1_error*sigmoid(l1,derive=True)

	hiddenweights+=l0.transpose().dot(l1_delta)
	outputweights+=l1.transpose().dot(l2_delta)



	




	


print(l2)







	
	



