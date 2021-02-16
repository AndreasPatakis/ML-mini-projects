import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import pandas as pd
import random as rand

datafile = []

with open("./Haberman's Survival Data Set/haberman.data") as data:
    for line in data:
        datafile.append(line[:-1])

#Setting Features X
X = []
#Classes Y (1 = survived, 2 = died)
Y = []


for record in datafile:
    data = record.split(',')
    feature = [int(data[0]),int(data[1]),int(data[2])]
    X.append(feature)
    Y.append(int(data[3]))

#Changing data so that our classes will have the observed values 1 = survived, -1 = died.
for i in range(len(Y)):
    if(Y[i] == 2):
        Y[i] = -1

for i in range(len(X)):
    plt.scatter(X[i][0],Y[i],color = "pink")

plt.xticks(np.arange(20,80,1))

m = len(X)
X = np.vstack(X)
X = np.array([(np.ones(m)),X[:,0],X[:,1],X[:,2]]).T

Y = np.array(Y).T


W = np.linalg.inv(X.T @ X) @ (X.T @ Y)

print(W)
