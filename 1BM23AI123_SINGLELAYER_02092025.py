#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1/(1+np.exp(-z))

def tanh(z):
    return np.tanh(z)

def relu(z):
    return np.maximum(0,z)

class Neuron:
    def __init__(self, weights,bias,activation="sigmoid"):
        self.weights = np.array(weights)
        self.bias = bias
        self.activation = activation
        
    def forward(self,input):
        z = np.dot(inputs,self.weights)+self.bias
        if self.activation == "sigmoid":
            return sigmoid(z)
            
        elif self.activation == "tanh":
            return tanh(z)
        
        elif self.activation == "relu":
            return relu(z)
        else:
            raise ValueError("Unknown activation function")
                

inputs = np.array([0.5,-1.2,3.0])
weights = [0.4,-0.6,0.3]
bias = 0.1

for act in ["sigmoid","tanh","relu"]:
    neuron = Neuron(weights,bias,activation=act)
    output = neuron.forward(inputs)
    print(f"Activation = {act:7s} --> Output = {output:4f}")

x = np.linspace(-10, 10, 400)  

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(x, sigmoid(x), label='sigmoid')
plt.title('Sigmoid Activation')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(x, tanh(x), label='tanh', color='orange')
plt.title('Tanh Activation')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(x, relu(x), label='ReLU', color='green')
plt.title('ReLU Activation')
plt.grid(True)

plt.tight_layout()
plt.show()



# In[ ]:




