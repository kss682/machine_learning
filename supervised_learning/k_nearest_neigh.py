import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

class knn():                
	K=3  #the no of nearest neighbours taken

	def distance(self,data,group):   #function to  find nearest neighbours
		distance=[]
		for i in data:
			for j in data[i]:
				euclidd=np.linalg.norm(np.array(j)-np.array(group))
				distance.append([euclidd,i])
		distance.sort()
		count_k=0
		count_r=0	
		for i in distance[:self.K]:		
			if i[1]=='k':
				count_k+=1
			else:
				count_r+=1
		if count_k>=count_r:
			return 'k'
		else:
			return 'r'	

		

	def plot(self,data,group):
		for i in data:
        		for j in data[i]:
                		plt.scatter(j[0],j[1],s=100,color=i)
		
		group_color=self.distance(data,group)
		plt.scatter(group[0],group[1],s=100,color=group_color)
		plt.show()
	







datasets={'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features=input()
new_features=[int(x) for x in new_features.split(" ")]

clf=knn()
clf.plot(datasets,new_features)

