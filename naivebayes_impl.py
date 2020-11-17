from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import plot_confusion_matrix, ConfusionMatrixDisplay, confusion_matrix
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas
import numpy
import datetime

dataset = load_breast_cancer()

def setupData():
    x = dataset['data']
    y = dataset['target']
    column_names = dataset['feature_names']
    x = pandas.DataFrame(x, columns = column_names)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 34)
    return x_train, x_test, y_train, y_test

def getStats(x_train, y_train):
    means = x_train.groupby(y_train).apply(numpy.mean)
    stand_devs = x_train.groupby(y_train).apply(numpy.std)
    class_probs = x_train.groupby(y_train).apply(lambda x: len(x)) / x_train.shape[0]
    return means, stand_devs, class_probs

def getClassPredictions(x_test, y_train, means, stand_devs, class_probs):
    y_prediction = []
    for element in range(x_test.shape[0]):
        probs = {}
        for c in numpy.unique(y_train):
            probs[c] = class_probs.iloc[c]
            for index, param in enumerate(x_test.iloc[element]):
                probs[c] *= norm.pdf(param, means.iloc[c, index], stand_devs.iloc[c, index])
        y_prediction.append(pandas.Series(probs).values.argmax())
    return y_prediction

def printResults(means, stand_devs, class_probs):
    print("Mean value for each class:\n##################################################################################")
    print(means)
    print()
    print("Standard deviations for each class:\n##################################################################################")
    print(stand_devs)
    print()
    print("Class probabilities:\n##################################################################################")
    print(class_probs)
    

def testModel(x_train, y_train, x_test, y_test, class_predictions):
    model = GaussianNB()
    model.fit(x_train, y_train)
    y_prediction = model.predict(x_test)
    score = accuracy_score(y_test, class_predictions) * 100
    score = round(score, 2)
    print()
    print("Accuracy score using predictions against test data:\n##################################################################################")
    print("Accuracy score: " + str(score) + "%")
    score = accuracy_score(y_test, y_prediction) * 100
    score = round(score, 2)
    print("Accuracy score using sklearn model: " + str(score) + "%")
    score = precision_score(y_test, class_predictions, average='micro') * 100
    score = round(score, 2)
    print()
    print("Precision score using predictions against test data:\n##################################################################################")
    print ("Precision Score: " + str(score) + "%")
    score = precision_score(y_test, y_prediction, average='micro') * 100
    score = round(score, 2)
    print ("Precision Score with sklearn model: " + str(score) + "%")
    print()
    print("Jaccard score using predictions against test data:\n##################################################################################")
    score = jaccard_score(y_test, class_predictions, average='micro') * 100
    score = round(score, 2)
    print ("Jaccard Score: " + str(score) + "%")
    score = jaccard_score(y_test, y_prediction, average='micro') * 100
    score = round(score, 2)
    print ("Jaccard Score using sklearn: " + str(score) + "%")
    return y_prediction

def graphResults(x_test, class_predictions, sklearn_predictions):
    plt.scatter(x_test, sklearn_predictions)
    plt.title('Predictions of Breast Tumors as Cancerous with Tumor Radius')
    plt.xlabel("Tumor Radius")
    plt.ylabel("Melignant (0) or Benign (1)")
    plt.savefig('graphs/predictions_sklearn.png')

    plt.scatter(x_test, class_predictions)
    plt.savefig('graphs/predictions_impl.png')

def graphConfusionMatrix(X_test, y_test, class_predictions):
    numpy.set_printoptions(precision=2)
    cm = confusion_matrix(y_test, class_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=dataset.target_names)

    disp = disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig('graphs/predictions_confusion_matrix.png')

def main():
    start = datetime.datetime.now()
    x_train, x_test, y_train, y_test = setupData()
    means, stand_devs, class_probs = getStats(x_train, y_train)
    class_predictions = getClassPredictions(x_test, y_train, means, stand_devs, class_probs)
    end = datetime.datetime.now()
    runtime = (end - start).total_seconds()
    print("Algorithm Runtime: " + str(runtime) + " seconds\n")
    printResults(means, stand_devs, class_probs)
    sklearn_predictions = testModel(x_train, y_train, x_test, y_test, class_predictions)
    graphResults(x_test['mean radius'], class_predictions, sklearn_predictions)
    graphConfusionMatrix(x_test, y_test, class_predictions)

main()