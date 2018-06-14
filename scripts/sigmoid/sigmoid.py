#import section
from matplotlib import pylab
import pylab as plt
import numpy as np

#sigmoid = lambda x: 1 / (1 + np.exp(-x))
def sigmoid(x):
    # return (1 / (1 + np.exp(-x)))
    i = -3
    k = 3
    return (1 / (1 + np.exp(-k*(x-i))))

def sigmoid_1(x):
    i = -3
    k = 3
    return (1 / (1 + np.exp(k*(x-i))))

mySamples = []
mySigmoid = []


x = plt.linspace(-5,5,101)
y = plt.linspace(-5,5,101)
x_t = np.linspace(0,1,101)

# prepare the plot, associate the color r(ed) or b(lue) and the label 
plt.plot(x_t, sigmoid_1(x), 'r')
plt.plot(x_t, sigmoid(y), 'b')


plt.show()
