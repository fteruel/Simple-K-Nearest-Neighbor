from scipy.spatial import distance
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt




def k_NN_sklearn(X_train, y_train, X_test, K):
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


    classifier = KNeighborsClassifier(K, metric='minkowski', p = 2)
    #minkowski with p=2 for euclidean distance
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    return y_pred



def prediction(X_train, y_train, X_test, K=3):

    distances = []
    y_pred= []
    for j in range( len(X_test)):  
        for i in range( len(X_train)):
            eucliDist = distance.euclidean(X_train[i], X_test[j])
            distances.append([eucliDist,y_train[i]])



        votes = [i[1] for i in sorted(distances)[:K]]
        vote_result = Counter(votes).most_common(1)[0][0]

        y_pred.append(vote_result) 

    return y_pred

def plot(X_train, y_train, X_test):
    
    colors = ['red', 'green']
    for i in range(len(X_train)):
        x = X_train[i][0] 
        y = X_train[i][1]
        color = colors[y_train[i]]

        plt.scatter(x,y,c=color, alpha= 0.5)

    for i in range( len(X_test)):
        x = X_test[i][0] 
        y = X_test[i][1]
        plt.scatter(x,y,c='black')

    plt.title('Simple K Nearest Neighbor')
    plt.xlabel('x')
    plt.ylabel('y')

    print (f1_score(y_test, y_pred))






