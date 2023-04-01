import numpy as np
import random
import seaborn as sns
import pandas as pd
from typing import Tuple
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
import math 

csv_path = "iris.csv"

class KNNClassifier:
    k_neighbors = 0

    def __init__(self, k:int, test_split_ratio :float) -> None:
        self.k = k
        self.test_split_ratio = test_split_ratio
        KNNClassifier.k_neighbors = k

    @staticmethod
    def load_csv(csv_path:str) -> Tuple[pd.DataFrame, pd.Series]:
     random.seed(42)
     dataset = pd.read_csv(csv_path, header=0)
     dataset = dataset.sample(frac=1).reset_index(drop=True)
     x, y = dataset.iloc[:,:4], dataset.iloc[:,-1]
     return x, y 

    def train_test_split(self, features: pd.DataFrame, labels: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        test_size = int(len(features) * self.test_split_ratio)
        train_size = len(features) - test_size
        assert len(features) == test_size + train_size, "Size mismatch!"
        
        shuffled_df = pd.concat([features, labels], axis=1).sample(frac=1, random_state=42).reset_index(drop=True)
        self.x_train, self.y_train = shuffled_df.iloc[:train_size, :-1], shuffled_df.iloc[:train_size, -1]
        self.x_test, self.y_test = shuffled_df.iloc[train_size:train_size + test_size, :-1], shuffled_df.iloc[train_size:train_size + test_size, -1]
        
        return self.x_train, self.x_test, self.y_train, self.y_test

    def euclidean(self, element_of_x:pd.DataFrame) -> pd.DataFrame:
     return math.sqrt(sum((self.x_train - element_of_x)**2,axis=1))

    def predict(self, x_test:pd.DataFrame) -> pd.DataFrame:
     labels_pred = []
     for x_test_element in x_test:
        distances = euclidean(x_test_element)
        distances = pd.DataFrame(sorted(zip(distances,self.y_train)))
        label_pred = mode(distances[:self.k,1],keepdims=False).mode
        labels_pred.append(label_pred)
     self.y_preds = pd.DataFrame(labels_pred).astype(int)   
     return pd.DataFrame(labels_pred).astype(int) 

    def accuracy(self) -> float:
     true_positive = (self.y_test == self.y_preds).sum()
     return true_positive / len(self.y_test) * 100

    def confusion_matrix(self) -> np.ndarray:
        conf_matrix = confusion_matrix(self.y_test.to_numpy(),self.y_preds.to_numpy())
        #sns.heatmap(conf_matrix,annot=True)
        return conf_matrix




    
    




