#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Week 1: Consider a neural network that takes two inputs, has one hidden layer with two nodes, and
# an output layer with one node. Let's start by randomly initializing the weights and the biases in the
# network. print the weights and biases.

import numpy as np
n = 2 # number of inputs
num_hidden_layers = 1 # number of hidden layers
m = 2 # number of nodes in each hidden layer
num_nodes_output = 1 # number of nodes in the output layer
weights1 = np.around(np.random.uniform(size=n*m), decimals=2) # initialize the weights
biases1 = np.around(np.random.uniform(size=m), decimals=2) # initialize the biases
print("Weights between Input & Hidden layer",weights1)
print("Bias in Hidden layer",biases1)
weights2 = np.around(np.random.uniform(size=m*num_nodes_output), decimals=2) # initialize the weights
biases2 = np.around(np.random.uniform(size=num_nodes_output), decimals=2) # initialize the biases
print("Weights between Hidden layer & Output",weights2)
print("Bias in Output layer",biases2)


# In[6]:


# Week 2: Consider the week 1 network and compute the following
# • The weighted sum.
# • Assuming a sigmoid activation function, let's compute the activation of the first node.
# • compute the activation of the second node
# • compute the weighted sum of these inputs to the node in the output layer
# • compute the output of the network as the activation of the node in the output layer

import numpy as np
n = 2 # number of inputs
num_hidden_layers = 1 # number of hidden layers
m = 2 # number of nodes in each hidden layer
num_nodes_output = 1 # number of nodes in the output layer
weights = np.around(np.random.uniform(size=6), decimals=2) # initialize the weights
biases = np.around(np.random.uniform(size=6), decimals=2) # initialize the biases
print(weights)
print(biases)
x_0 = 0.5 # input 1
x_1 = 0.85 # input 2
print('x0 is {} and x1 is {}'.format(x_0, x_1))
z_11 = x_0 * weights[0] + x_1 * weights[1] + biases[0]
z_12 = x_0 * weights[2] + x_1 * weights[3] + biases[1]
print('The weighted sum of the inputs at the first node in the hidden layer is{}'.format(np.around(z_11, decimals=2)))
print('The weighted sum of the inputs at the second node in the hidden layer is{}'.format(np.around(z_12, decimals=2)))
a_11 = 1.0 / (1.0 + np.exp(-z_11))
a_12 = 1.0 / (1.0 + np.exp(-z_12))
print('The activation of the first node in the hidden layer is {}'.format(np.around(a_11,decimals=2)))
print('The activation of the second node in the hidden layer is {}'.format(np.around(a_12,decimals=2)))
z_2 = a_11 * weights[4] + a_12 * weights[5] + biases[2]
print('The weighted sum of the inputs at the node in the output layer is {}'.format(np.around(z_2,decimals=2)))
a_2 = 1.0 / (1.0 + np.exp(-z_2))
print('The output of the network for x1 = 0.5 and x2 = 0.85 is {}'.format(np.around(a_2,decimals=2)))


# In[9]:


# Week 3: Initialize a network with the following specification
# • Takes 5 inputs
# • has three hidden layers
# • has 3 nodes in the first layer, 2 nodes in the second layer, and 3 nodes in the third layer
# • has 1 node in the output layer

import numpy as np # import the Numpy library
num_inputs=5
num_hidden_layers=3
num_nodes_hidden=[3, 2, 3]
num_nodes_output=1
inputs = np.around(np.random.uniform(size=num_inputs), decimals=2)
def initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):
    num_nodes_previous = num_inputs
    network = {}
    for layer in range(num_hidden_layers + 1):
        if layer == num_hidden_layers:
            layer_name = 'output'
            num_nodes = num_nodes_output
        else:
            layer_name = 'layer_{}'.format(layer + 1)
            num_nodes = num_nodes_hidden[layer]
        network[layer_name] = {}
        for node in range(num_nodes):
            node_name = 'node_{}'.format(node+1)
            network[layer_name][node_name] = {
            'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
            'bias': np.around(np.random.uniform(size=1), decimals=2),
            }
        num_nodes_previous = num_nodes
    return network
small_network = initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden,
num_nodes_output)
print(small_network)


# In[11]:


# Week 4a: Consider the Week 3 network and do the following
# • Change the activation of the network from sigmoid to tanh and observe the performance of the
# network
# • Compute the activation of every node in first hidden node
# • Compute the activation of every node in second hidden node
# • Compute the activation of every node in third hidden node
# Code
# Activation function is sigmoid 1/1+e^-z
import numpy as np # import the Numpy library
num_inputs=5
num_hidden_layers=3
num_nodes_hidden=[3, 2, 3]
num_nodes_output=1
inputs = np.around(np.random.uniform(size=num_inputs), decimals=2)
def initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):
    num_nodes_previous = num_inputs # number of nodes in the previous layer
    network = {}
    # loop through each layer and randomly initialize the weights and biases associated with eachlayer
    for layer in range(num_hidden_layers + 1):
        if layer == num_hidden_layers:
            layer_name = 'output' # name last layer in the network output
            num_nodes = num_nodes_output
        else:
            layer_name = 'layer_{}'.format(layer + 1) # otherwise give the layer a number
            num_nodes = num_nodes_hidden[layer]
        # initialize weights and bias for each node
        network[layer_name] = {}
        for node in range(num_nodes):
            node_name = 'node_{}'.format(node+1)
            network[layer_name][node_name] = {
            'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
            'bias': np.around(np.random.uniform(size=1), decimals=2),
            }
        num_nodes_previous = num_nodes
    return network # return the network
def compute_weighted_sum(inputs, weights, bias):
    return np.sum(inputs * weights) + bias
def node_activation(weighted_sum):
    return 1.0 / (1.0 + np.exp(-1 * weighted_sum))
small_network = initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden,num_nodes_output)
print('The inputs to the network are {}'.format(inputs))
node_weights = small_network['layer_1']['node_1']['weights']
node_bias = small_network['layer_1']['node_1']['bias']
weighted_sum = compute_weighted_sum(inputs, node_weights, node_bias)
node_output = node_activation(compute_weighted_sum(inputs, node_weights, node_bias))
def forward_propagate(network, inputs):
    layer_inputs = list(inputs) # start with the input layer as the input to the first hidden layer
    for layer in network:
        layer_data = network[layer]
        layer_outputs = []
        for layer_node in layer_data:
            node_data = layer_data[layer_node]
            # compute the weighted sum and the output of each node at the same time
            node_output = node_activation(compute_weighted_sum(layer_inputs, node_data['weights'],
            node_data['bias']))
            layer_outputs.append(np.around(node_output[0], decimals=2))
            if layer != 'output':
                print('The outputs of the nodes in hidden layer number {} is {}'.format(layer.split('_')[1],
            layer_outputs))
        layer_inputs = layer_outputs # set the output of this layer to be the input to next layer
        network_predictions = layer_outputs
    return network_predictions
predictions = forward_propagate(small_network, inputs)
print('The predicted value by the network for the given input is {}'.format(np.around(predictions,decimals=2)))


# In[17]:


# Week 4b: Consider the Week 3 network and do the following
# • Change the activation of the network from sigmoid to tanh and observe the performance of the
# network
# • Compute the activation of every node in first hidden node
# • Compute the activation of every node in second hidden node
# • Compute the activation of every node in third hidden node
# Code
# Activation function is tanh (e^z-e^-z)/(e^z+e^-z)

import numpy as np # import the Numpy library
num_inputs=5
num_hidden_layers=3
num_nodes_hidden=[3, 2, 3]
num_nodes_output=1
inputs = np.around(np.random.uniform(size=num_inputs), decimals=2)
small_network = initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden,num_nodes_output)
def initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):
    num_nodes_previous = num_inputs # number of nodes in the previous layer
    network = {}
# loop through each layer and randomly initialize the weights and biases associated with eachlayer
    for layer in range(num_hidden_layers + 1):
        if layer == num_hidden_layers:
            layer_name = 'output' # name last layer in the network output
            num_nodes = num_nodes_output
        else:
            layer_name = 'layer_{}'.format(layer + 1) # otherwise give the layer a number
            num_nodes = num_nodes_hidden[layer]
        # initialize weights and bias for each node
        network[layer_name] = {}
        for node in range(num_nodes):
            node_name = 'node_{}'.format(node+1)
            network[layer_name][node_name] = {
            'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
            'bias': np.around(np.random.uniform(size=1), decimals=2),
            }
        num_nodes_previous = num_nodes
    return network # return the network
def compute_weighted_sum(inputs, weights, bias):
    return np.sum(inputs * weights) + bias
def node_activation(weighted_sum):
    return (np.exp(weighted_sum)-np.exp(-1 * weighted_sum)) / (np.exp(weighted_sum)+np.exp(-1 * weighted_sum))

print('The inputs to the network are {}'.format(inputs))
node_weights = small_network['layer_1']['node_1']['weights']
node_bias = small_network['layer_1']['node_1']['bias']
weighted_sum = compute_weighted_sum(inputs, node_weights, node_bias)
node_output = node_activation(compute_weighted_sum(inputs, node_weights, node_bias))
def forward_propagate(network, inputs):
    layer_inputs = list(inputs) # start with the input layer as the input to the first hidden layer
    for layer in network:
        layer_data = network[layer]
        layer_outputs = []
        for layer_node in layer_data:
            node_data = layer_data[layer_node]
            # compute the weighted sum and the output of each node at the same time
            node_output = node_activation(compute_weighted_sum(layer_inputs, node_data['weights'],node_data['bias']))
            layer_outputs.append(np.around(node_output[0], decimals=2))
        if layer != 'output':
            print('The outputs of the nodes in hidden layer number {} is {}'.format(layer.split('_')[1],layer_outputs))
        layer_inputs = layer_outputs # set the output of this layer to be the input to next layer
    network_predictions = layer_outputs
    return network_predictions
predictions = forward_propagate(small_network, inputs)
print('The predicted value by the network for the given input is {}'.format(np.around(predictions,decimals=2)))


# In[18]:


# Week 5: Consider the Week 3 network and do the following
# • Change the activation of the network from tanh to relu and observe the performance of the
# network
# • Compute the activation of every node in first hidden node
# • Compute the activation of every node in second hidden node
# • Compute the activation of every node in third hidden node
# Code
# Activation function is relu max(0,z)
import numpy as np # import the Numpy library
num_inputs=5
num_hidden_layers=3
num_nodes_hidden=[3, 2, 3]
num_nodes_output=1
inputs = np.around(np.random.uniform(size=num_inputs), decimals=2)
small_network = initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden,num_nodes_output)
def initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):
    num_nodes_previous = num_inputs # number of nodes in the previous layer
    network = {}
    # loop through each layer and randomly initialize the weights and biases associated with each layer
    for layer in range(num_hidden_layers + 1):
        if layer == num_hidden_layers:
            layer_name = 'output' # name last layer in the network output
            num_nodes = num_nodes_output
        else:
            layer_name = 'layer_{}'.format(layer + 1) # otherwise give the layer a number
            num_nodes = num_nodes_hidden[layer]
        # initialize weights and bias for each node
        network[layer_name] = {}
        for node in range(num_nodes):
            node_name = 'node_{}'.format(node+1)
            network[layer_name][node_name] = {
            'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
            'bias': np.around(np.random.uniform(size=1), decimals=2),
            }
        num_nodes_previous = num_nodes
    return network # return the network
def compute_weighted_sum(inputs, weights, bias):
    return np.sum(inputs * weights) + bias
def node_activation(weighted_sum):
    return (np.maximum(0,weighted_sum))

print('The inputs to the network are {}'.format(inputs))
node_weights = small_network['layer_1']['node_1']['weights']
node_bias = small_network['layer_1']['node_1']['bias']
weighted_sum = compute_weighted_sum(inputs, node_weights, node_bias)
node_output = node_activation(compute_weighted_sum(inputs, node_weights, node_bias))
def forward_propagate(network, inputs):
    layer_inputs = list(inputs) # start with the input layer as the input to the first hidden layer
    for layer in network:
        layer_data = network[layer]
        layer_outputs = []
        for layer_node in layer_data:
            node_data = layer_data[layer_node]
            # compute the weighted sum and the output of each node at the same time
            node_output = node_activation(compute_weighted_sum(layer_inputs, node_data['weights'],
            node_data['bias']))
            layer_outputs.append(np.around(node_output[0], decimals=2))
        if layer != 'output':
            print('The outputs of the nodes in hidden layer number {} is {}'.format(layer.split('_')[1],layer_outputs))
        layer_inputs = layer_outputs # set the output of this layer to be the input to next layer
    network_predictions = layer_outputs
    return network_predictions
predictions = forward_propagate(small_network, inputs)
print('The predicted value by the network for the given input is {}'.format(np.around(predictions,decimals=2)))


# In[ ]:




