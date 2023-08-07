import numpy as np

def sigmoid(x):#                            SIGMOID ACTIVATION FUNCTION
    return 1/(1 + np.exp(-x))

def out(x):#                                DERIVATVE FOR SIGMOID ACTIVATION FUNCTION
    return sigmoid(x)*(1- sigmoid(x))


data = [[1,1],[1,0],[0,1],[0,0]]
target = [0,1,0,0]
#SETTING UP THE WEIGHTS AND BIAS
w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()

#NUMBER OF ITERATION
epoch = 1000
gamma = 0.005

for i in range(epoch):
    pick = np.random.randint(0 , len(data))
    picked_data = data[pick]

    z = w1*picked_data[0] + w2*picked_data[1] + b
    pred = sigmoid(z)

    cost = (pred - target[pick])**2
    print("EPOCH {} COST {}".format(i , cost))
#CALCULATING THE DERIVATIVES    
    dcost_dpred = (pred - target[pick])*2
    dpred_dz = out(z)
    dz_dw1 = picked_data[0]
    dz_dw2 = picked_data[1]
    dz_db = 1
#BACK PROPAGATION
    
    w1 = w1 - gamma*dcost_dpred*dpred_dz*dz_dw1
    w2 = w2 - gamma*dcost_dpred*dpred_dz*dz_dw2
    b = b - gamma*dcost_dpred*dpred_dz*dz_db

