import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import pandas as pd
import random as rand



def gradientD(X,Y,learning_rate):
    result = []
    m = len(X)
    w0 = 0
    w1 = 0
    for i in range(1000):
        y_predicted = w0 + w1*X
        SumB = (-2/m)*sum(Y-y_predicted)
        SumA = (-2/m)*sum(X*(Y-y_predicted))
        print(SumA)
        input("Continue..")
        w0 = w0 - learning_rate * SumB
        w1 = w1 - learning_rate * SumA
    result.append(w0)
    result.append(w1)
    return result


data = []
with open("./datasetForRegression.csv","r") as datacsv:
    for line in datacsv:
        data.append(line[:-1])

X = []
Y = []


for i in range(len(data)):
    values = data[i].split(",")
    X.append(float(values[0]))
    Y.append(float(values[1]))

X = np.array(X)
Y = np.array(Y)

for i in range(len(X)):
    plt.scatter(X[i],Y[i],color="pink")

w = gradientD(X,Y,0.0001)

line_x = np.linspace(0,80)
line_y = w[0] + w[1]*line_x


print(w)
plt.plot(line_x,line_y)
plt.show()
