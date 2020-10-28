import csv
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import networkx as nx
import matplotlib.pyplot as plt

'''
want to discover patterns and knowledge through the use of bayes classification
how do we want to classify data?
    2 classes - 1) has a co-author
                2) do NOT have a co-author
P(class | data) = [P(data|class) * P(class)]/ P(data)
'''
def seperate_into_classes(matrix):
    length = len(matrix)

    ones_class = np.any(matrix > 1, axis=1)
    print(ones_class)



'''
@param - takes in filename
@return - an adjacency matrix
'''
def readFile(filename):
    graph = nx.read_edgelist(filename)
    adj_matrix = nx.adjacency_matrix(graph)
    return adj_matrix.toarray()




def main():
    #plt.figure('CA-GrQc')
    adj_matrix = readFile('CA-GrQc.txt')
    seperate_into_classes(adj_matrix)

''''
    xValues = []
    yValues = []
    with open('CA-GrQc.txt', 'r') as data:  # using the with keyword ensures files are closed properly
        for line in data.readlines():
            parts = line.split('\t')  # change this to whatever the deliminator is
            parts[1] = parts[1].replace("\n", "")

            xValues.append(float(parts[0]))
            yValues.append(float(parts[1]))


    X = np.asarray(xValues)
    Y = np.asarray(yValues)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=24)
    X_train = X_train.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)

    clf = GaussianNB()
    clf.fit(X_train, y_train)

'''
if __name__ == "__main__":
    main()
