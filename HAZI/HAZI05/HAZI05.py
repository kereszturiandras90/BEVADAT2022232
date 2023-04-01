#import numpy as np
import random
import seaborn as sns
import pandas as pd
from typing import Tuple
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
import math 

csv_path = "diabetes.csv"

class KNNClassifier:
    k_neighbors = 0

    def __init__(self, k:int, test_split_ratio :float) -> None:
        self.k = k
        self.test_split_ratio = test_split_ratio
        KNNClassifier.k_neighbors = k

    @staticmethod
    def load_csv(csv_path:str) -> Tuple[pd.array, pd.array]:
     #random.seed(42)
     dataset = pd.read_csv(csv_path, sep=",", index_col=None, header=None)
     #print(dataset)
     dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
     #print(dataset)
     x, y = dataset.iloc[:,:8], dataset.iloc[:,-1]
     #x = x.values
     #y = y.values
     x = x.apply(lambda x: x if not isinstance(x, str) else None).dropna()
     y = y.apply(lambda y: y if not isinstance(y, str) else None).dropna()
     print(x)
     print(y)
     return x, y 

    def train_test_split(self, features: pd.array, labels: pd.array):
        test_size = int(len(features) * self.test_split_ratio)
        train_size = int(len(features) - test_size)
        #print(test_size)
        #print(train_size)
        assert len(features) == test_size + train_size, "Size mismatch!"

        #features = features.iloc[1:] 
        #labels = labels.iloc[1:]
        #shuffled_df = pd.concat([features, labels], axis=1).sample(frac=1, random_state=42).reset_index(drop=True)
        self.x_train,self.y_train = features.iloc[:train_size,:],labels.iloc[:train_size]
        self.x_test,self.y_test = features.iloc[train_size:train_size+test_size,:], labels.iloc[train_size:train_size + test_size]
        #print(self.x_test)
        
        return self.x_train, self.x_test, self.y_train, self.y_test

    def euclidean(self, element_of_x:pd.array) -> pd.array:
     #element_of_x = element_of_x.apply(lambda x: x if not isinstance(x, str) else None).dropna()
     #self.x_train = self.x_train.apply(lambda x: x if not isinstance(x, str) else None).dropna()
     print(element_of_x)
     print(self.x_train)
     return math.sqrt(sum((self.x_train.astype(float) - element_of_x.astype(float))**2,axis=1))
     #return pd.Series(self.x_train.sub(element_of_x, axis='columns').pow(2).sum(axis=1)).apply(pd.np.sqrt)

    def predict(self, x_test:pd.array) -> pd.array:
     labels_pred = []
     #print(x_test)
     for x_test_element in x_test:
        distances = self.euclidean(x_test_element)
        distances = pd.array(sorted(zip(distances,self.y_train)))
        label_pred = mode(distances[:self.k,1],keepdims=False).mode
        labels_pred.append(label_pred)
     self.y_preds = pd.array(labels_pred, dtype=int) 
     return pd.array(labels_pred, dtype=int)

    def accuracy(self) -> float:
     true_positive = (self.y_test == self.y_preds).sum()
     return true_positive / len(self.y_test) * 100

    def confusion_matrix(self) -> pd.array:
        conf_matrix = confusion_matrix(self.y_test,self.y_preds)
        #sns.heatmap(conf_matrix,annot=True)
        return conf_matrix

p1 = KNNClassifier(3,0.5)
x,y = p1.load_csv("diabetes.csv")
p1.train_test_split(x,y)
p1.predict(p1.x_test)   



    
    




