import numpy as np
import seaborn as sns
from typing import Tuple
from scipy.stats import mode
from sklearn.metrics import confusion_matrix

class KNNClassifier:

    def __init__(self, k:int, test_split_ratio :float) -> None:
        self.k = k
        self.test_split_ratio = test_split_ratio

    def load_csv(csv_path:str) -> Tuple[nd.array,nd.array ]:
     np.random.seed(42)
     dataset = np.genfromtxt(csv_path, delimiter=',')
     np.random.shuffle(dataset)
     x,y = dataset[:,:-1], dataset[:,-1]
     return x,y    

    def train_test_split(self, features: np.array, labels: np.array):
     test_split_ratio = self.test_split_ratio
     test_size = int(len(features)*test_split_ratio)
     train_size = len(features) -test_size
     assert len(features) == test_size + train_size, "Size match!"

     self.x_train, self.y_train = features[:train_size, :], labels[:train_size]
     self.x_test, self.y_test = features[:train_size:, :], labels[train_size:]
    

    def eucledian(element_of_x: np.array) -> np.array:
     return np.sum((points - element_of_x)**2, axis=1)


    def plot_confusion_matrix():
      conf_matrix = confusion_matrix(self.y_test, self.y_preds)
      sns.heatmap(confusion_matrix, annot=True)
      return conf_matrix   


    def accuracy(y_test:np.ndarray,y_preds:np.ndarray) -> float:
       true_positive = (y_test == y_preds).sum()
       return true_positive / len(y_test) * 100 

    def predict(x_test: np.array) -> np.array:
     for x_test_element in x_test:
      distances = eucledian(self.x_train, x_test_element)
      distances = np.array(sorted(zip(distances, self.y_train)))

      label_pred = mode(distances[:self.k,1], keepdims = false).mode
      label_pred.append(label_pred)
      print(distances)

      self.y_preds = np.array(labels_pred, dtype=np.int64)

    def accuracy():
     true_positive = (self.y_test == self.y_preds).sum()
     return true_positive / len(self.y_test) * 100     







