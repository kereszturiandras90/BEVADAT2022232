import math
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
    def load_csv(csv_path:str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        dataset = pd.read_csv(csv_path, header=None, skiprows=[0])
        dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        x, y = dataset.iloc[:, :9], dataset.iloc[:, -1]
        return x, y

    def train_test_split(self, features: pd.array, labels: pd.array) -> tuple:
        test_size = int(len(features) * self.test_split_ratio)
        train_size = len(features) - test_size
        assert len(features) == test_size + train_size, "Size mismatch!"

        self.x_train, self.y_train = features.iloc[:train_size,:], labels.iloc[:train_size]
        self.x_test, self.y_test = features.iloc[train_size:train_size+test_size,:], labels.iloc[train_size:train_size + test_size]

        return (self.x_train, self.y_train, self.x_test, self.y_test)

    def euclidean(self, element_of_x:pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame.sub(self.x_train, element_of_x, axis=1).pow(2).sum(axis=1).apply(math.sqrt)

    def predict(self, x_test:pd.DataFrame) -> pd.DataFrame:
        labels_pred = []
        for _, x_test_element in x_test.iterrows():
            distances = self.euclidean(x_test_element)
            distances = pd.DataFrame(sorted(zip(distances,self.y_train)))
            label_pred = mode(distances[:self.k],keepdims=False).mode
            labels_pred.append(label_pred)
        self.y_preds = pd.DataFrame(labels_pred, dtype=int)
        self.y_preds = self.y_preds[1].values
        return self.y_preds

    def accuracy(self) -> float:
        print(self.y_test)
        print(self.y_preds)
        true_positive = (self.y_test == self.y_preds).sum()
        return true_positive / len(self.y_test) * 100

    def confusion_matrix(self):
        conf_matrix = confusion_matrix(self.y_test, self.y_preds)
        return conf_matrix
    
    def best_k(self) -> Tuple:
        results = []
        for k in range(1,21):
           self.k = k
           self.predict(self.x_test)
           acc = self.accuracy()
           result = tuple((k, acc))
           results.append(result)
        best = max(results,key=lambda item:item[1])
        return best[0]

    
    




