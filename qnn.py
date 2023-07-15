'''
               Max West
    https://arxiv.org/abs/2212.00264
'''

import pennylane as qml
import matplotlib.pyplot as plt
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer
import multiprocessing as mp
from functools import partial
from time import time
from utils import *

np.random.seed(959)

num_iterations = 781
batch_per_it   = 2
batch_size     = 32
num_processes  = 4
rearrange      = 1
equivariant    = 1
num_layers     = 50
num_classes    = 10 
num_test       = 500
dataset        = "fmnist"
num_qubits     = 10 if dataset == "fmnist" else 12
opt            = AdamOptimizer(stepsize=1e-3)

weights = np.random.randn(num_layers, num_qubits, 3 - equivariant, requires_grad=True)
print("weights shape: ", weights.shape)

x_train, y_train, x_test, y_test = load_data(dataset, rearrange, num_classes, num_test)
dev = qml.device("default.qubit", wires=num_qubits)

@qml.qnode(dev)
def circuit(weights, x):
        
    qml.QubitStateVector(x, wires=range(num_qubits))
    
    if equivariant:
        
        for w in weights:
            for i in range(num_qubits):
                qml.RX(w[i,0], wires=i)
                qml.IsingYY(w[i,1], wires=[i, (i+1) % num_qubits])
                    
        return [ qml.expval(qml.PauliZ(i) @ qml.PauliZ((i+1) % num_qubits)) for i in range(num_classes) ]
        
    else:
        
        for w in weights:
            for i in range(num_qubits):
                qml.Rot(*w[i], wires=i)
    
            for i in range(num_qubits - 1):
                qml.CZ(wires=[i, i+1])
    
        return [ qml.expval(qml.PauliZ(i)) for i in range(num_classes) ]

def acc(weights, data, labels, num_processes, i):
    
    box    = int(len(data)/num_processes)
    X = data[i*box:(i+1)*box] 
    preds  = np.argmax(np.array(circuit(weights, X)).transpose(), axis=1) 
    labels = labels[i*box:(i+1)*box]
    return len(labels[labels==preds])

if __name__ == '__main__':
    
    pool = mp.Pool(processes=num_processes)
    
    j = 0
    test_acc_history  = []

    for it in range(num_iterations):
        t = time()   
            
        for i in range(batch_per_it):
            
            batch_index = np.array(range(batch_size*j,batch_size*(j+1)))
            X_train_batch = x_train[batch_index]
            Y_train_batch = y_train[batch_index]
            weights, _, _, _, _ = opt.step(cost, weights, X_train_batch, Y_train_batch, circuit, batch_size)
            j += 1

        train_time = time()-t
        t = time()
        
        multi_test  = partial(acc, weights, [ x for x in x_test ],  y_test,  num_processes)
        test_acc_history.append( sum(pool.map(multi_test,  [ i for i in range(num_processes) ])) / len(y_test))
        
        if max(test_acc_history) == test_acc_history[-1]:
            np.save(dataset + "_" + str(num_layers) + "_weights_best" + "_rearranged" * rearrange + "_equivariant" * equivariant, weights)

        print( "Iter: {:5d} | Acc test: {:0.7f} | "
        	"".format(it + 1, test_acc_history[-1]), flush=True)
        
        print("Train time, test time: ", round(train_time, 3), round(time()-t, 3), flush=True)
    
    pool.close()
    pool.join()

    np.save(dataset + "_" + str(num_layers) + "_weights_final" + "_rearranged" * rearrange + "_equivariant" * equivariant, weights)
    print("test_acc", test_acc_history)
