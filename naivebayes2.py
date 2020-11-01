from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import norm
import pandas
import numpy

dataset = load_iris()
x = dataset['data']
y = dataset['target']
column_names = dataset['feature_names']
x = pandas.DataFrame(x, columns = column_names)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 34)
means = x_train.groupby(y_train).apply(numpy.mean)
stand_devs = x_train.groupby(y_train).apply(numpy.std)
class_probs = x_train.groupby(y_train).apply(lambda x: len(x)) / x_train.shape[0]

y_prediction = []
for element in range(x_test.shape[0]):
    probs = {}
    for c in numpy.unique(y_train):
        probs[c] = class_probs.iloc[c]
        for index, param in enumerate(x_test.iloc[element]):
            probs[c] *= norm.pdf(param, means.iloc[c, index], stand_devs.iloc[c, index])
    y_prediction.append(pandas.Series(probs).values.argmax())

score = accuracy_score(y_test, y_prediction)
print(score)