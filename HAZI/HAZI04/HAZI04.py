import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np

def csv_to_df(path):
    df = pd.read_csv(path)
    return df

#print(csv_to_df("StudentsPerformance.csv"))    

def capitalize_columns(df):
  new_df = df.copy()  
  
  new_df = new_df.rename(columns=lambda x: x.capitalize() if "e" in x else x)
 # for s in new_df.columns:
 #   if "e" not in s:
 #     new_df.columns[s] = s.capitalize()   
  return new_df  

#print(capitalize_columns(csv_to_df("StudentsPerformance.csv")))

def math_passed_count(df):
  new_df = df.copy() 
  osszeg = sum(df['math score']>49)
  return osszeg

#print(math_passed_count(csv_to_df("StudentsPerformance.csv"))) 

def did_pre_course(df):
    new_df = df.copy() 
    temp = new_df[new_df["test preparation course"] == "completed"]
    return temp

#print(did_pre_course(csv_to_df("StudentsPerformance.csv"))) 


def average_scores(df):
    new_df = df.copy()
    agg_functions = {
    'math score':
    ['mean'],
    'reading score':
    ['mean'],
    'writing score':
    ['mean'],
}
    return new_df.groupby(['parental level of education']).agg(agg_functions)
     

print(average_scores(csv_to_df("StudentsPerformance.csv"))) 


def add_age(df):
    new_df = df.copy()
    random.seed(42)   
    new_df['age'] = np.random.randint(18, 66, new_df.shape[0])

    return new_df

print(add_age(csv_to_df("StudentsPerformance.csv"))) 

def female_top_score(df):
    new_df = df.copy()
    new_df['sum'] = new_df['reading score'] + new_df['writing score'] + new_df['math score']
    new_df = new_df[new_df["gender"] == "female"].sort_values(by='sum', ascending = False)
    new_df = new_df[['reading score', 'writing score', 'math score']].head(1)
    ered = tuple(new_df.itertuples(index=False,name=None))

    return ered

print(female_top_score(csv_to_df("StudentsPerformance.csv"))) 


def add_grade(df):
    new_df = df.copy()
    point_to_grad = {(0.00, 0.59): "F", (0.60, 0.69): "D", (0.70, 0.79): "C", (0.80, 0.89): "B", (0.90, 1.00): "A"}
    def grade_range(x):
     for key in point_to_grad:
        if x >= key[0] and x <= key[1]:
            return point_to_grad[key]
      

    new_df['grade']=((new_df['reading score'] + new_df['writing score'] + new_df['math score'] ) / 300)
    new_df['grade']=new_df['grade'].apply(grade_range)
    return new_df

print(add_grade(csv_to_df("StudentsPerformance.csv")))     