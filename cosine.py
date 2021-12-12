import numpy as np 
import matplotlib.pyplot as plt 

 

x = np.arange(0, 2*np.pi, 0.1)
n = 2
a = n*(np.pi)
h1 = 2*np.sin(x)*np.sin(a)
#h2 = -np.cos(x+a)



plt.plot(x, h1)
plt.ylabel('h')
plt.xlabel('x')
plt.legend(['cos(x)'], loc = 1)
plt.show()
