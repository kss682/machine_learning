import tensorflow as tf
from sklearn import datasets,preprocessing

iris=datasets.load_iris()
X=iris.data
y=iris.target
X=preprocessing.scale(X)


X_train=X[:140]
X_test=X[140:]
y_train=y[:140]
y_test=y[140:]


nodes_1=50
n_class=3
batch_size=20

x=tf.placeholder('float',[None,4])
y=tf.placeholder('float',[None])
def neural_net(data):
	hidden_layer={'weights':tf.Variable(tf.zeros([4,nodes_1])),'bias':tf.Variable(tf.zeros([nodes_1]))}
	output_layer={'weights':tf.Variable(tf.zeros([nodes_1,n_class])),'bias':tf.Variable(tf.zeros([n_class]))}
	


	l1=tf.add(tf.matmul(data,hidden_layer['weights']),hidden_layer['bias'])
	l1=tf.nn.relu(l1)


	output=tf.add(tf.matmul(l1,output_layer['weights']),output_layer['bias'])
	output=tf.nn.relu(output)

	return output


def model(x):
	prediction=neural_net(x)

	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=tf.reshape(y,[-1,1])) )
	optimizer = tf.train.AdamOptimizer().minimize(cost)
    

	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		
		loss=0
		for i in range(int(len(X_train)/batch_size)):
			_,c=sess.run([optimizer,cost],feed_dict={x:X_train[i*batch_size:i*batch_size+20],y:y_train[i*batch_size:i*batch_size+20]})
			loss+=c


		print("loss ", loss)
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		print(correct)		
		#accuracy=tf.reduce_mean(tf.cast(correct, 'float'))
		#print('Accuracy:',accuracy.eval({x:X_test, y:y_test}))		
model(x)			
		
			
