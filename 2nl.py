import numpy as np
import math
from sklearn import preprocessing
# sigmoid function

def nn(x, w):
    """Output function y = x * w"""
    return np.dot(x,w)
def loss(y, t):
    """MSE loss function"""
    return np.mean((t - y)**2)    
def gradient(w, x, t):
    """Gradient function. (Remember that y = nn(x, w) = x * w)"""
    return 2 * np.dot(x.T ,(nn(x, w) - t))
def delta_w(w_k, x, t, learning_rate):
    """Update function delta w"""
    return -learning_rate * np.mean(gradient(w_k, x, t))

# input dataset
learning_rate = 0.5
nb_of_iterations = 10000
X = np.array([[ 8656346,
7678491,
7212995,
7059951,
2291288,
1695000,
435301,
394505,
332547,
232384,
65586,
659997,
289337,
10578455]])

print "input : ********************************\n",X

print "***************************************"
X = preprocessing.normalize(X)

print "Trasformed Input : ********************************\n",X

print "***************************************"
# output expected           
y = np.array([[20743128.0,10638475.0,3021499.0,1064225.0,12101366.0]])

y = preprocessing.normalize(y)
print len(y[0])
print "transformed Expected Output *************************************** \n",y
print "***************************************"

#  weights matrix
syn0 = np.array([[1,0,0 ,0,0],
 [0,1,0 ,0,0],
 [0.5,0.2,0.1 ,0,0.2],
 [0.4,0.1,0.15,0.05,0.3],
 [0.7,0.05,0.1 ,0.05,0.1],
 [0.4,0.3,0.1 ,0.1,0.1],
 [0.3,0.3,0.2 ,0.1,0.1],
 [0.1,0.3,0.15 ,0.15,0.3],
 [0.6,0.2,0.075 ,0.025,0.1],
 [0.2,0.4,0.15 ,0.15,0.1],
 [0.7,0,0.2,0.05,0.05],
 [0.5,0.1,0.4 ,0,0],
 [0.3,0.1,0 ,0.1,0.5],
 [0.25,0,0.025 ,0.025,0.7]])
print "weights *************************************** \n",syn0

print "***************************************"
for iter in xrange(nb_of_iterations):

    # forward propagation
    l0 = X
   
    l1 = nn(l0,syn0)
    #dimension l1 : (1,5)
    print "output ********************************\n",l1
    print "***************************************"
    # how much did we miss?
    l1_error = loss(l1,y)
    
    print "error *************************************** \n",l1_error
    print "***************************************"

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    #l1_delta = l1_error * nonlin(l1,True)
    l1_delta = delta_w(syn0, l0, y, learning_rate)
    #dimension l1_delta (1,5)
    print "delta *************************************** \n", l1_delta 
    print "***************************************"     
    
    # update weights
    #syn0 += np.dot(l0.T,l1_delta)
    
    syn0 += np.dot(l0.T,l1_delta)

    

print "Matrix of weights After Training:"
print syn0
print l1

