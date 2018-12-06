import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import FeatureEditor as fe

np.random.seed(3)

sigmoid = lambda z:1/(1+np.exp(-z))

def sigmoid_prime(z):
    a = sigmoid(z)
    return np.multiply(a,1-a)

Relu = lambda z: np.maximum(0,z)

Relu_prime = lambda z: np.where(z>=0,1,0)

Leaky_Relu = lambda z: np.maximum(0.01*z,z)

Leaky_Relu_prime = lambda z: np.where(z>=0,1,0.01)

getshape = lambda arr: [item.shape for item in arr]

def init_weights(layer_dist=[],multiplier=0.01):
    w=[]
    b=[]
    for dims in zip(layer_dist,layer_dist[1:]):
        w.append(np.random.randn(dims[1],dims[0])*multiplier)
        b.append(np.zeros((dims[1],1)))
    return w,b

def cost(y,a):
    m = y.shape[1]
    return np.sum(np.dot(y,np.log(a).T)+np.dot(1-y,np.log(1-a).T))/(-m)

def forward_prop(w,b,g,x,cache={}):
    cache['A']=[x]
    cache['Z']=[]
    i=0
    for weights,bias in zip(w,b):
        cache['Z'].append( np.dot(weights,cache['A'][-1])+bias )
        cache['A'].append( g[i](cache['Z'][-1]) )
        i+=1
    return cache

def backward_prop(w,b,gprime,y,cache):
    assert(len(w)==len(cache['Z'])==len(gprime))
    m = y.shape[1]
    #init cache for differentials
    cache['dZ']=list(np.zeros(len(w)))
    cache['dW']=list(np.zeros(len(w)))
    cache['db']=list(np.zeros(len(w)))
    cache['dA']=list(np.zeros(len(w)))
    #activation differential for last layer
    cache['dA'][-1] = -(y/cache['A'][-1])+((1-y)/(1-cache['A'][-1]))
    #start backward propagation from output to input layer
    for l in reversed(range(len(w))):
        cache['dZ'][l]= np.multiply( cache['dA'][l],gprime[l](cache['Z'][l]))
        cache['dW'][l]= np.dot( cache['dZ'][l], cache['A'][l].T)/m
        cache['db'][l]= np.sum( cache['dZ'][l],axis=1).reshape(b[l].shape)/m
        if l!=0: #no need to calculate dA for input layer
            cache['dA'][-1]= np.dot( w[l].T,cache['dZ'][l])
    return cache

def gradient_descent(w,b,g,gprime,x,y,cache={},iterations=250,alpha=0.1,lambd=0):
    costs=[]
    for i in range(iterations):
        cache=forward_prop(w,b,g,x)
        cache=backward_prop(w,b,gprime,y,cache)
        costs.append(cost(y,cache['A'][-1]))
        #update weights & bias
        for l in range(len(w)):
            w[l]= w[l]-(alpha*cache['dW'][l])
            b[l]= b[l]-(alpha*cache['db'][l])
    return w,b,costs,cache

def cal_accuracy(w,b,g,x,y,data_type='training'):
    m = y.shape[1]
    cache = forward_prop(w,b,g,x,cache={})
    predication=np.where(cache['A'][-1]>=0.5,1,0)
    accuracy = np.sum(np.where(predication==y,1,0))
    print(data_type+' accuracy=',(accuracy/m)*100,'%')

def train_nn(x,y,g,gprime,layer_dist=[],iterations=15000,alpha=12,lambd=0,show_plt=True):
    #prep the neural network 
    w,b = init_weights(layer_dist)
    cache = forward_prop(w,b,g,x,cache={})
    print('initial cost=',cost(y,cache['A'][-1]))
    ##cache = backward_prop(w,b,gprime,y,cache)
    w,b,costs,cache= gradient_descent(w,b,g,gprime,x,y,cache={},iterations=iterations,alpha=alpha,lambd=lambd)
    print('final cost=',cost(y,cache['A'][-1]))
    if show_plt:
        plt.plot(costs)
        plt.show()
    return w,b,cache


def prepare_dataset():
    file = r'C:\Users\niraj_shivajikale\Desktop\Machine Learning\Datasets\Credit Approval\crx.csv'
    df= pd.read_csv(file)
    fe.LabelCoding_Series(df,['Result','A1','A4','A5','A6','A7','A9','A10','A12','A13'])
    return df


df= prepare_dataset()

#x_all =np.matrix(np.random.rand(2,m)%1)*10
#y_all = np.matrix(np.where(np.square(x_all[0,:])+np.square(x_all[1,:])<=25,1,0))
#x_all = np.insert(x_all,0,1,axis=0)

#split=int(m*0.65)
#x_train = x_all[:,:split]
#y_train = y_all[:,:split]
#x_test = x_all[:,split:]
#y_test = y_all[:,split:]

#g = [Relu,sigmoid]
#gprime = [Relu_prime,sigmoid_prime]

#w,b,cache=train_nn(x_train,y_train,g,gprime,layer_dist=[3,2,1],iterations=15000,alpha=12,lambd=0,show_plt=True)
#cal_accuracy(w,b,g,x_train,y_train,data_type='training')
#cal_accuracy(w,b,g,x_test,y_test,data_type='testing')
