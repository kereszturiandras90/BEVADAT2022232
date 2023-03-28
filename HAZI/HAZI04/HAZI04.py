import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np

def csv_to_df(path):
    df = pd.read_csv(path)
    return df
   

def capitalize_columns(df):
  new_df = df.copy()  
  
  new_df = new_df.rename(columns=lambda x: x.capitalize() if "e" not in x else x)  
  return new_df
   


def math_passed_count(df):
  new_df = df.copy() 
  osszeg = sum(df['math score']>49)
  return osszeg


def did_pre_course(df):
    new_df = df.copy() 
    temp = new_df[new_df["test preparation course"] == "completed"]
    return temp



def average_scores(df):
    new_df = df.copy()
    agg_functions = {
    'mean points':
    ['mean']
}
    new_df['mean points']= new_df['math score'] + new_df['reading score'] + new_df['writing score']
    new_df.drop(['math score'], axis=1)
    new_df.drop(['reading score'], axis=1)
    new_df.drop(['writing score'], axis=1)
    
  
    return new_df.groupby(['parental level of education']).agg(agg_functions)
    
     



def add_age(df):
    new_df = df.copy()
    np.random.seed(42)   
    new_df['age'] = np.random.randint(18, 67, new_df.shape[0])
    return new_df


def female_top_score(df):
    new_df = df.copy()
    new_df['sum'] = new_df['reading score'] + new_df['writing score'] + new_df['math score']
    new_df = new_df[new_df["gender"] == "female"].sort_values(by='sum', ascending = False)
    new_df = new_df[['math score', 'reading score', 'writing score']].head(1)
    ered = tuple(new_df.itertuples(index=False,name=None))

    return ered[0]



def add_grade(df):
    new_df = df.copy()
    point_to_grad = {(0.0000, 0.5999): "F", (0.6000, 0.6999): "D", (0.7000, 0.7999): "C", (0.8000, 0.8999): "B", (0.9000, 1.0000): "A"}
    def grade_range(x):
     for key in point_to_grad:
        if x >= key[0] and x <= key[1]:
            return point_to_grad[key]
      

    new_df['grade']=((new_df['reading score'] + new_df['writing score'] + new_df['math score'] ) / 300)
    new_df['grade']=new_df['grade'].apply(grade_range)
    return new_df
 

def math_bar_plot(df):
    new_df = df.copy()
    sex = (['female', 'male'])
    MathAvg = (new_df[(new_df['gender']=="female")])['math score'].mean(), (new_df[(new_df['gender']=="male")])['math score'].mean()
    fig, ax = plt.subplots()
 

    ax.bar(sex , MathAvg)
 
    ax.set_xlabel("Gender")
    ax.set_ylabel("Math Score")
    ax.set_title("Average Math Score by Gender")
    return fig
 

def writing_hist(df):

    fig, ax = plt.subplots()

    new_df = df.copy()
    ax = new_df['writing score'].hist()
    ax.set_title('Distribution of Writing Scores')
    ax.set_ylabel('Number of Students')
    ax.set_xlabel('Writing Score')

    return fig
  

def ethnicity_pie_chart(df):
    new_df = df.copy()
    fig, ax = plt.subplots()

    ngroupA = new_df['race/ethnicity'].value_counts()['group A']
    ngroupB = new_df['race/ethnicity'].value_counts()['group B']
    ngroupC = new_df['race/ethnicity'].value_counts()['group C']
    ngroupD = new_df['race/ethnicity'].value_counts()['group D']
    ngroupE = new_df['race/ethnicity'].value_counts()['group E']

    
    race = 'groupA', 'groupB', 'groupC',  'groupD', 'groupE'
    number = [ngroupA, ngroupB, ngroupC, ngroupD, ngroupE]

    ax.pie(number, labels=race, autopct='%1.1f%%')
    ax.axes.get_yaxis().set_visible(False)
    ax.set_title('Proportion of Students by Race/Ethnicity')

    return fig


  