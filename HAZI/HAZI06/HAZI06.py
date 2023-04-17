import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from NJCleaner import NJCleaner
from DecisionTreeClassifier import DecisionTreeClassifier

base_csv_path = "2018_03.csv"
clean_csv_path = "NJexp.csv"

nj_cleaner = NJCleaner(base_csv_path)
nj_cleaner.prep_df(clean_csv_path)

data = pd.read_csv(clean_csv_path)

X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1, 1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)

classifier = DecisionTreeClassifier(min_samples_split=2, max_depth=4)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)
print(accuracy_score(Y_test, Y_pred))

"""
A  tanítás során az okozott nehézségethogy a max depth értéke először túl nagy volt, ami errorhoz 
vezetett. Ezt kellett csak javítani. A legjobb paraméter megtalálásához a grid search-öt vettem igénybe.
A min_samples_split range 1-5 volt, a max_depth range pedig 1-4 volt.Észrevettem, az hogy a min_sample_split nem 
befolyásolta az eredményeket, csak a max_depth. Ennek a legjobb eredményei 4-nél voltak.
A fit-elések eredménye:

min_samples_split, max_depth, accuracy:
1, 4, 0.7849166666666667
2, 4, 0.7849166666666667
3, 4, 0.7849166666666667
4, 4, 0.7849166666666667
5, 4, 0.7849166666666667
1, 3, 0.7839166666666667
2, 3, 0.7839166666666667
3, 3, 0.7839166666666667
4, 3, 0.7839166666666667
5, 3, 0.7839166666666667
1, 2, 0.7823333333333333
"""