import csv
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


def main():
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

if __name__ == "__main__":
    main()
