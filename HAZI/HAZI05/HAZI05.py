#import numpy as np
import random
import seaborn as sns
import pandas as pd
from typing import Tuple
from scipy.stats import mode
from sklearn.metrics import confusion_matrix


csv_path = "diabetes.csv"

class KNNClassifier:
    k_neighbors = 0

    def __init__(self, k:int, test_split_ratio :float) -> None:
        self.k = k
        self.test_split_ratio = test_split_ratio
        KNNClassifier.k_neighbors = k

    @staticmethod
    def load_csv(csv_path:str) -> Tuple[pd.array, pd.array]:
     dataset = pd.read_csv(csv_path, sep=",", index_col=None, header=None, skiprows=[0] )
     dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
     x, y = dataset.iloc[:,:8], dataset.iloc[:,-1]
     return x, y 

    def train_test_split(self, features: pd.array, labels: pd.array):
        test_size = int(len(features) * self.test_split_ratio)
        train_size = int(len(features) - test_size)
        assert len(features) == test_size + train_size, "Size mismatch!"

        self.x_train,self.y_train = features.iloc[:train_size,:],labels.iloc[:train_size]
        self.x_test,self.y_test = features.iloc[train_size:train_size+test_size,:], labels.iloc[train_size:train_size + test_size]

    def euclidean(self, element_of_x:pd.array) -> pd.array:
     return ((self.x_train - element_of_x)**2).sum(axis=1).apply(lambda x: x**0.5)

    def predict(self, x_test:pd.array) -> pd.array:
     labels_pred = []
     for x_test_element in x_test:
        distances = self.euclidean(x_test_element)
        distances = pd.array(sorted(zip(pd.array((distances,self.y_train)))))
        label_pred = mode(distances[:self.k,1],keepdims=False).mode
        labels_pred.append(label_pred)
     self.y_preds = pd.array(labels_pred, dtype=int) 
     return pd.array(labels_pred, dtype=int)

    def accuracy(self) -> float:
     print(self.y_preds)
     print(self.y_test)
     true_positive = (self.y_test == self.y_preds).sum()
     return true_positive / len(self.y_test) * 100

    def confusion_matrix(self) -> pd.array:
        conf_matrix = confusion_matrix(self.y_test,self.y_preds)
        return conf_matrix

  



    
    




