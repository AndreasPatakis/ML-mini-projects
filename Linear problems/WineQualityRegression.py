import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import pandas as pd
import random as rand


def J(W,Y,X,pos):
    def hypothesis(pos):
        result = 0
        for i in range(len(W)):
            result += W[i]*X[pos][i]
        return result
    result = 0
    for i in range(len(X)):
        result += X[i][pos] * (Y[i] - hypothesis(i))
    return result

#Gradient Descent Algorithm
def GradientDescent(a,max_iter,W,X,Y):
    current_weights = W
    step_size = 1.0
    m = len(Y)  #Num of traings sets
    n = len(W)  #Num of features
    for k in range(max_iter):
        for i in range(n):
            sum = (-2/m)*(J(W,Y,X,i))
            step_size = a * sum
            current_weights[i] = current_weights[i] - step_size
        W = current_weights
    return W




data = []
with open("./WineQualityDataset/winequality-red.csv","r") as data_csv:
    for line in data_csv:
        data.append(line[:-1])

#First row is the name of its feature
data = data[1:]
#Matrix X for features m = 4898
X = []
#Vector Y for observed values
Y = []

#Settings X and Y
for line in data:
    values = line.split(';')
    X.append([float(values[0]),float(values[1]),float(values[2]),float(values[3]),float(values[4]),float(values[5]),
    float(values[6]),float(values[7]),float(values[8]),float(values[9]),float(values[10])])
    Y.append(int(values[11]))


#Some useless plotting
for i in range(len(X)):
    plt.scatter(X[i][0],Y[i],color="pink")

#plt.yticks(np.arange(0,11,1))
#plt.xticks(np.arange(0,15,1))

#Adding a constant 1 for Xo
X = np.vstack(X)
m = len(X)
X = np.array([np.ones(m),X[:,0]]).T

#Initializing the matrix for weights
W = np.array([1.0,1.0])
#Initializing vector of observed values
Y = np.array(Y).T

#Feature scaling? Or maybe not
"""
for i in range(len(X)):
    X[i][1] = X[i][1]/len(X)
    Y[i] = Y[i]/len(Y)
"""

#Initializing some usefull information for Gradient Descent
learning_rate = 0.01
min_step = 0.0001
max_iteration = 300

#Gradient Descent approach
W = GradientDescent(learning_rate,max_iteration,W,X,Y)
#Normal Equation approach
#W = np.linalg.inv(X.T @ X) @ (X.T @ Y)

print(W)

line_x = np.linspace(0,17)
line_y = W[0] + W[1]*line_x

plt.plot(line_x,line_y)
plt.show()
