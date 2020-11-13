from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import load_wine
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, plot_confusion_matrix, ConfusionMatrixDisplay, confusion_matrix
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas
import numpy

dataset = load_wine()

def setupData():
    x = dataset['data']
    y = dataset['target']
    column_names = dataset['feature_names']
    x = pandas.DataFrame(x, columns = column_names)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 34)
    return x_train, x_test, y_train, y_test

def getStats(x_train, x_test, y_train, y_test):
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

def printResults(y_test, means, stand_devs, class_probs, class_predictions):
    print("Mean value for each class:\n##################################################################################")
    print(means)
    print()
    print("Standard deviations for each class:\n##################################################################################")
    print(stand_devs)
    print()
    print("Class probabilities:\n##################################################################################")
    print(class_probs)
    print()
    score = accuracy_score(y_test, class_predictions) * 100
    score = round(score, 2)
    print("Accuracy using predictions against test data:\n##################################################################################")
    print("Accuracy = " + str(score) + "%")

def testModel(x_train, y_train, x_test, y_test):
    model = GaussianNB()
    model.fit(x_train, y_train)
    y_prediction = model.predict(x_test)
    score = accuracy_score(y_test, y_prediction) * 100
    score = round(score, 2)
    print("Accuracy using sklearn model: " + str(score) + "%")
    return y_prediction

def graphResults(x_test, class_predictions, sklearn_predictions):
    plt.scatter(x_test, sklearn_predictions, s=35, c='b', marker='s', label='Sklearn')
    plt.scatter(x_test, class_predictions, s=15, c='r', label='Our implementation')
    plt.legend()
    plt.title('Predictions with Alcohol Content')
    plt.xlabel("Alcohol Percentage")
    plt.ylabel("Cultivars")
    plt.savefig('predictions_sklearn.png')


def graphConfusionMatrix(X_test, y_test, class_predictions):
    numpy.set_printoptions(precision=2)
    cm = confusion_matrix(y_test, class_predictions)
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=dataset.target_names)

    disp = disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")

    plt.show()
def main():
    x_train, x_test, y_train, y_test = setupData()
    means, stand_devs, class_probs = getStats(x_train, x_test, y_train, y_test)
    class_predictions = getClassPredictions(x_test, y_train, means, stand_devs, class_probs)
    printResults(y_test, means, stand_devs, class_probs, class_predictions)
    sklearn_predictions = testModel(x_train, y_train, x_test, y_test)
    graphResults(x_test['alcohol'], class_predictions, sklearn_predictions)

    graphConfusionMatrix(x_test, y_test, class_predictions)


main()