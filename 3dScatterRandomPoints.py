import numpy as np
import random as rand
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import csv

m = 400
class_i = [[]for x in range(3)]
class_b = [[]for x in range(3)]
#Points for feature A
for i in range(m):
    x1 = rand.randint(20,60)
    x2 = rand.randint(20,60)
    x3 = rand.randint(20,40)
    class_i[0].append([x1,x2,x3,1])


#Points for feature B
for i in range(m):
    x1 = rand.randint(60,100)
    x2 = rand.randint(80,120)
    x3 = rand.randint(20,40)
    class_i[1].append([x1,x2,x3,2])


#Points for feature C
for i in range(m):
    x1 = rand.randint(100,140)
    x2 = rand.randint(20,60)
    x3 = rand.randint(50,80)
    class_i[2].append([x1,x2,x3,3])


m = 20
#Points for feature A
for i in range(m):
    x1 = rand.randint(20,60)
    x2 = rand.randint(20,60)
    x3 = rand.randint(20,40)
    class_b[0].append([x1,x2,x3,1])


#Points for feature B
for i in range(m):
    x1 = rand.randint(60,100)
    x2 = rand.randint(80,120)
    x3 = rand.randint(20,40)
    class_b[1].append([x1,x2,x3,2])


#Points for feature C
for i in range(m):
    x1 = rand.randint(100,140)
    x2 = rand.randint(20,60)
    x3 = rand.randint(50,80)
    class_b[2].append([x1,x2,x3,3])



with open("3dPointsTrain.csv", 'w', newline='') as myfile:
     wr = csv.writer(myfile)
     for c in class_i:
         for point in c:
             wr.writerow(point)

with open("3dPointsTest.csv", 'w', newline='') as myfile:
     wr = csv.writer(myfile)
     for c in class_b:
         for point in c:
             wr.writerow(point)






cc = [[]for x in range(3)]
cc[0] = np.array(class_i[0])
cc[1] = np.array(class_i[1])
cc[2] = np.array(class_i[2])

colors = ["red","green","pink"]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(class_i)):
    ax.scatter(cc[i][:,0],cc[i][:,1],cc[i][:,2], color = colors[i])

plt.show()
