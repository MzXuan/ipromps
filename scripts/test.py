import numpy as np
import matplotlib.pyplot as plt

alpha = 60
dt = 0.01
x = 1.0
t = 0.0
step = np.linspace(0,1,100)
y1 = x*np.exp(-alpha*step)
y2 = x*np.exp(alpha*step)

plt.plot(step,y1)
plt.plot(step,y2)

plt.show()

