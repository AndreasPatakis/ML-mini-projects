from Neural_Networks_Framework import Neural_Network

# Execution example

train = []
with open("./3dPointsTrain.csv","r") as datacsv:
    for line in datacsv:
        train.append(line[:-1].split(","))

test = []
with open("./3dPointsTest.csv","r") as datacsv:
    for line in datacsv:
        test.append(line[:-1].split(","))

test_int = []
train_int = []
for list in test:
    test_int.append([float(item) for item in list])
for list in train:
    train_int.append([float(item) for item in list])

y_test = []
y_train = []
for list in test_int:
    if(list[-1] == 1):
        y_test.append([1,0,0])
    elif(list[-1] == 2):
        y_test.append([0,1,0])
    else:
        y_test.append([0,0,1])

for list in train_int:
    if(list[-1] == 1):
        y_train.append([1,0,0])
    elif(list[-1] == 2):
        y_train.append([0,1,0])
    else:
        y_train.append([0,0,1])


train_int = [item[:-1] for item in train_int]
test_int = [item[:-1] for item in test_int]




#Neural_Network(Num_Of_Inputs_Features,Nodes_Of_Each_Layer,Learning_Rate)
net1 = Neural_Network(3,[6,3],0.2)
#.train(Inputs,Outputs,Num_Of_Iterations,Batch)
net1.train(train_int,y_train,50,20)
net1.test(test_int,y_test)
