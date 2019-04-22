import numpy as np
import matplotlib.pyplot as plt

def segment_X(X,E,tau):
    'X is the original series'
    'E is the embedding dimension'
    'tau is the time delay'
    'This function gets all segments of length E in X and calls that A.'
    'It also returns the next value of each segment and calls that B'
    A = []
    B = np.empty(len(X)-(E-1)*tau)
    B[:]=np.nan
    j=0
    for i in range(tau*(E-1),len(X)):
        A +=[X[i-tau*(E-1):i+tau:tau][::-1]]
        if i +tau<len(X):
            B[j] = X[i+tau]
        j+=1
    return np.array(A),B

A,B =segment_X([1,2,3,4,5,6,7,8],2,2)


def dist_neighbours(A,index):
    y = A[index]
    distance_vector = np.linalg.norm(A-y,axis =1)
    distance_vector[index] = np.inf
    return distance_vector
def simplex_predictor(A,B,mapfunction,n=None):
    if n is None:
        n = len(A[0])
    y = A[-1]
    del_indices = np.where(np.isnan(B)) #get indices to delete in A
    A = np.delete(A,del_indices,axis =0)
    distance_vector = np.linalg.norm(A-y,axis =1)
    indices = np.argpartition(distance_vector,n) #finds the n closest points after removing the point itself

    print(indices) #there's a bug here : bug kinda fixed
    return mapfunction(B[index]),indices

'''
d= simplex_predictor(A,B,np.mean)
for a in A:
    x,y =a
    plt.plot(x,y,'r.')
plt.show()
'''

def exp_map(B,distance_vector,indices):
    weights = np.exp(-distance_vector[indices])
    weights = weights/np.sum(weights)
    return np.dot(weights,B[indices])
    


def conv_cross_map(X,Y,E,tau,mapfunc,n=None):
    if n is None:
        n = E #possible bug here
    manfX,futX = segment_X(X,E,tau)
    manfY,futY = segment_X(Y,E,tau)
    del_indices = np.where(np.isnan(futX)) #get indices to delete in manfX
    manfX = np.delete(manfX,del_indices,axis =0)
    PredY = np.zeros(len(manfX)) #len(futY)-len(manfX) gives the points for which the future isn't known
    for i in range(len(manfX)):
        distance_vector = dist_neighbours(manfX,i)
        indices = np.argpartition(distance_vector,n)
        PredY[i] = mapfunc(futY,distance_vector,indices)
    return PredY,futY
theta = np.linspace(0,6*np.pi,300)
X = np.sin(theta)
Y = np.cos(theta)

PredY,futY= conv_cross_map(X,Y,6,2,exp_map)
plt.plot(theta[12:],2.8*PredY,'r')
plt.plot(theta[12:],futY[:-2],'b')
plt.show()



        
        
        
    
    


