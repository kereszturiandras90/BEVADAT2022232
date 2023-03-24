import pandas as pd
import matplotlib.pyplot as plt

def dict_to_dataframe(data):


    #dt = pd.read_csv(data).to_dict()
    MyDataFrame = pd.DataFrame.from_dict(data)
    return MyDataFrame


#stats = {"country": ["Brazil", "Russia", "India", "China", "South Africa"],
#      "capital": ["Brasilia", "Moscow", "New Dehli", "Beijing", "Pretoria"],
#      "area": [8.516, 17.10, 3.286, 9.597, 1.221],
#      "population": [200.4, 143.5, 1252, 1357, 52.98] }
#print(dict_to_dataframe(stats))
# Készíts egy függvényt ami a bemeneti DataFrame-ből vissza adja csak azt az oszlopot amelynek a neve a bemeneti string-el megegyező.\n","\n","Egy példa a bemenetre: test_df, 'area'\n","Egy példa a kimenetre: test_df\n","return type: pandas.core.series.Series\n","függvény neve: get_column\n","'''"]}

def get_column(df, colname):
       new_df = df.copy()
       return new_df[colname]

#print(get_column(dict_to_dataframe(stats), 'area'))


def get_top_two(df):
    new_df = df.copy()
    return new_df.nlargest(2,'area')

#print(get_top_two(dict_to_dataframe(stats)))

def population_density(df):
    new_df = df.copy()
    new_df["density"] = new_df["population"] / new_df["area"]
    return new_df

#print(population_density(dict_to_dataframe(stats)))   


def plot_population(df):
    new_df = df.copy()
    countries = list(new_df['country'])
    populations = list(new_df['population'])
  
    fig = plt.figure(figsize = (10, 5))
 

    plt.bar(countries, populations, color ='maroon',
        width = 0.4)
 
    plt.xlabel("Country")
    plt.ylabel("Population (millions)")
    plt.title("Population of Countries")
    plt.show()
    return plt

#plot_population(dict_to_dataframe(stats))

def plot_area(df):
    new_df = df.copy()

    labels = list(new_df['country'])
    sizes = list(new_df['area'])
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels)
    plt.title("Area of Countries")
    plt.show()
    return plt

#plot_area(dict_to_dataframe(stats))

