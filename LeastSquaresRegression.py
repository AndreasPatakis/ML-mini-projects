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
def GradientDescent(a,min_step,max_iter,W,X,Y):
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
with open("./datasetForRegression.csv","r") as datacsv:
    for line in datacsv:
        data.append(line[:-1])

X = []
Y = []
W = [0.0,0.0]


for i in range(len(data)):
    values = data[i].split(",")
    X.append(float(values[0]))
    Y.append(float(values[1]))


X = np.array([np.ones(len(X)),X]).T


for i in range(len(X)):
    plt.scatter(X[i][1],Y[i],color="pink")



#Initializing some usefull information for Gradient Descent
learning_rate = 0.0001
min_step = 0.0000001
max_iteration = 1000

W = GradientDescent(learning_rate,min_step,max_iteration,W,X,Y)

print(W)

line_x = np.linspace(0,80)
line_y = W[0] + W[1]*line_x

plt.plot(line_x,line_y)
plt.show()
