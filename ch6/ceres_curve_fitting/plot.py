import numpy as np
import matplotlib.pyplot as plt

data=np.loadtxt('data.txt')
x=data[:,0]
y=data[:,1]
a=0.891943
b=2.17039
c=0.944142
y1=np.exp(a*x*x+b*x+c)
y2=np.exp(x*x+2*x+1)
plt.plot(x,y1,'r--',x,y2,'b-')
plt.plot(x,y,'xk')
plt.legend(['Ceres_pose', 'True_pose', 'Gaussian_pose'], loc='best') 
plt.show()


