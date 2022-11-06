import numpy as np
import pandas as pd
from sklearn import metrics, neighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('car.data')
print(df.head())

X = df[['buying','maint','safety']].values

y = df[['class']]

#converting the data into numbers
le = LabelEncoder()
for i in range(len(X[0])):
    X[:, i] = le.fit_transform(X[:, i])


#converting y into numbers using map function
label_mapping = {
    'unacc':0,
    'acc':1,
    'good':2,
    'vgood':3
}
y['class'] = y['class'].map(label_mapping)
y = np.array(y)


#Model

knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

knn.fit(X_train, y_train)


prediction = knn.predict(X_test)
accuracy = metrics.accuracy_score(y_test, prediction)

print('predictions: ', prediction)
print('accuracy: ', accuracy)

#Testing the model
print('actual value: ', y[50])
print('predicted value', knn.predict(X)[50])

