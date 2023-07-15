'''
               Max West
    https://arxiv.org/abs/2212.00264
'''

from pennylane import numpy as np
from numba import vectorize
from numba.types import int64

def load_data(dataset, rearrange, num_classes, num_test):

    x_train, y_train = np.load("../im_data/" + dataset + "_xtrain.npy"), np.load("../im_data/" + dataset + "_ytrain.npy")
    x_test,  y_test  = np.load("../im_data/" + dataset + "_xtest.npy"),  np.load("../im_data/" + dataset + "_ytest.npy")
    
    if dataset == "fmnist":
        
        num_channels = 1

        x_train = x_train.astype(np.float32)[:,0,:,:]
        x_test  = x_test.astype(np.float32)[:,0,:,:]
        
        x_train = np.pad(x_train, pad_width=((0, 0), (2, 2), (2, 2)), mode='constant', constant_values=0.0)
        x_test = np.pad(x_test, pad_width=((0, 0), (2, 2), (2, 2)), mode='constant', constant_values=0.0)
        
        x_train = x_train.reshape((x_train.shape[0], -1))
        x_test  = x_test.reshape((x_test.shape[0], -1))
    
    if dataset == "cifar10":
    
        num_channels = 3
        
        x_train = np.transpose(x_train, (0,3,1,2))
        x_test  = np.transpose(x_test, (0,3,1,2))  
    
        x_train = x_train.reshape((x_train.shape[0], -1))
        x_test  = x_test.reshape((x_test.shape[0], -1))
        
        x_train = np.pad(x_train, ((0, 0), (512, 512)))
        x_test  = np.pad(x_test,  ((0, 0), (512, 512)))
    
    def get_subset(x, y):

        y = np.argmax(y, axis=1)
        subset = np.where(y < num_classes)
        x = x[subset]
        y = y[subset]
        x /= np.linalg.norm(x, axis=1, keepdims=True)

        return x, y

    x_train, y_train = get_subset(x_train, y_train)
    x_test,  y_test  = get_subset(x_test, y_test)
    
    x_test = x_test[:num_test]
    y_test = y_test[:num_test]
    
    if rearrange:
    
        _map = np.argsort(permute(np.arange(x_train.shape[1]), 32, num_channels))
        
        tmp_train = np.zeros_like(x_train, requires_grad=False)
        tmp_test  = np.zeros_like(x_test,  requires_grad=False)
        
        tmp_train[:,:] = x_train[:,_map]
        tmp_test[:,:]  = x_test[:,_map]
        
        x_train = tmp_train
        x_test  = tmp_test

    print('x_train.shape: ', x_train.shape, 'y_train.shape: ', y_train.shape, 'x_test.shape: ', x_test.shape, 'y_test.shape: ', y_test.shape, flush=True)

    return x_train, y_train, x_test, y_test


@vectorize([int64(int64, int64, int64)], target='parallel')
def permute(i,n,c):
    
    # maps i to the index related by reflection (see Figure 3 of https://arxiv.org/abs/2212.00264)
    # where the image size is (c,n,n)
    # the three channel version is a bit hacky on account of the retrospectively awkward zero padding
    
    if c == 1:
        if (i%n) < n/2:
            result = int(n*(i//n)/2 + (i%n))
        else:
            result = int(n**2 - (n*(i//n)/2 + n-(i%n)) )

    if c == 3:

        if i >= n**2 / 2 and i < 3*n**2 + n**2 /2:

            if (i - n**2/2) % n < n/2:
                result = int(n**2/2 + ((i - n**2/2)//n)*n/2 + ((i - n**2/2)%n))
            else:
                result = int(3*n**2 + n**2/2 - (n*((i - n**2/2)//n)/2 + n - ((i - n**2/2)%n)))

        else:
            result = i

    return result

#n = 8
#perm = permute(np.arange(n**2), n, 1)
#print('\n')
#for i in range(n**2):
#    print(f'{perm[i]: <4}', end='')
#    if not (i+1)%n: print('', flush=True)

def cost(weights, X, Y, circuit, batch_size):
    
    output = 5 * np.array(circuit(weights, X)).transpose()
    log_probs = output - np.log(np.sum(np.exp(output), axis=1,keepdims=True))
    
    return -np.sum(log_probs[np.arange(batch_size),Y._value.astype(int)]) / batch_size 

