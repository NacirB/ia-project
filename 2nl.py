import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
# input dataset
X = np.array([ 8656346,
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
10578455])

 
# output expected           
y = np.array([20743128,10638475,3021499,1064225,12101366])

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = [[1,0,0 ,0,0],
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
 [0.25,0,0.025 ,0.025,0.7]]

print(syn0)
for iter in xrange(1):

    # forward propagation
    l0 = X
    l1 = np.dot(l0,syn0)

    # how much did we miss?
    l1_error = np.power(y-l1,1)

    print(l1_error)

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    #l1_delta = l1_error * nonlin(l1,True)

    # update weights
    #syn0 += np.dot(l0.T,l1_delta)

print "Output After Training:"
print l1

