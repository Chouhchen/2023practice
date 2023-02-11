#!/usr/bin/env python
# coding: utf-8

# In[1]:


print ("DNN practice")


# In[2]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as pit


# In[4]:


# read data file
data= pd.read_csv ('data.csv')


# In[5]:


# data sets: n = number of parameters (colums), m = number of datasets (rows)

data = np.array(data)
m, n = data.shape
# shuffle the datasets
np.random.shuffle(data)

#take the first 1000 datasets?  switch collumn and row to make dataset into array
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]


#take the rest datasets as training set switch collumn and row to make dataset into array
data_train = data [1000:m].T
Y_train = data_dev[0]
X_train = data_dev[1:n]


# In[6]:


# intial the neural network
# a= number of layers  and number of nodes per layers of the neural network 
# inital weights and biaess by generate raumdon output between 0 and 1, than subtract 0.5 here to get a random ooutput between -0.5 and 0.5

def init_params():
    W1 = np.ramdon.rand(a, n) - 0.5
    b1 = np.ramdon.rand(a, 1) - 0.5
    W2 = np.ramdon.rand(a, a) - 0.5
    b2 = np.ramdon.rand(10, 1) - 0.5
    return W1, b1, W2, b2

# Use RelU as active function for the nodes, it is a function to output each elements in Z if Z > 0 and 0 if Z < = 0
def ReLU(Z):
    return np.maximum(0, Z)

# at the final layer of hidden layer, map all the result into the properbility (0 to 1) of each node of this layer.
def softmax(Z):
    return np.exp(Z) / np.sum(exp(Z))

#forward propagation 
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.doc(X) + b1
    A1 = ReLU(Z1)
    Z2 = w2.dot(Z2) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# create matrix of zeros and then create indexed array, Y.size is the number of training samples
def one_hot(Y):
    one_hot_Y = np.zeros(Y.size, Y.max()+1)
    one_hot_Y[np.arange(Y.size), Y] = 1 # got to the row is accessing, specified the label in y and lable the column axis to 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_ReLU(Z):
    return Z > 0
    
#backword propagation 
def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, 2)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dw2 = 1 / m * dZ2.dot(X.T)
    db1 = 1/ m * np.sum(dZ1, 2)    
    return dW1, db1, dW2, db2

#output object parameters

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accurcy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_paratms()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10  == 0:    #  report the accuracy rate at every 10 iteration.
            print("Iterantion: ", i)
            print("Accuracy: ", get_accuracy(get_predictions(A2), Y))
            
    return W1, b1, W2, b2

    


# In[7]:


# train 100 iterations with learning rate of 0.1
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 100, 0.1 )


# In[8]:


# make prediction 
def make_predicitions(X, W1, b1, W2, b2):
    Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

# index = the index of the data to be tested

def test_predictions(index, W1, b1, W2, B2):
    current_data= X_train[:, index, None]
    prediction = make_preditions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ",prediction)
    print("Lable: ", label)

# visualize output data to be texted into a image file of 28 *28 with 255 pixel.

current_data = current_data.reshape(len(current_data), -1) /255
plt.grey()
plt.imshow(current_image, interpolations='nearest')
plt.show()

# prompt user to input the index number of the data to be tested

print('Enter the index number of the data you want to test:')
index = input()


# In[ ]:




