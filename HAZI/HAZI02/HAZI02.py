import numpy as np


# Írj egy olyan fügvényt, ami megfordítja egy 2d array oszlopait

def column_swap(array : np.array) -> np.array:
 return array[:, ::-1]

arr1 = np.array([[1,2],[3,4]])
print(column_swap(arr1))

# Készíts egy olyan függvényt ami összehasonlít két array-t és adjon vissza egy array-ben, hogy hol egyenlőek \n"

def compare_two_array(a : np.array, b : np.array) -> np.array:
 return np.where(a == b)


arr = compare_two_array(np.array([1,2,3,4,5]),np.array([1,3,2,4,5]))

print(arr)




def encode_Y(array: np.array, class_number) -> np.array:
       one_hot = np.zeros((array.size, class_number))
       one_hot[np.arange(array.size), array] = 1

       return one_hot
    
arr = encode_Y(np.array([1, 0, 3]), 4)
print(arr)

def decode_Y(array :np.array) -> np.array:
        return np.argmax(array, axis=1)

print(decode_Y(arr))


def eval_classification(classes : list, pred : np.array) -> str:
        x = np.argmax(pred)
        return classes[x]

n = eval_classification( ['alma', 'körte', 'szilva'], np.array([0.2, 0.2, 0.6]))
print(n) 


    
def replace_odd_numbers(array: np.array) -> np.array:
       array[array % 2 != 0] = -1
       return array
   
print(replace_odd_numbers(np.array([1,2,3,4,5,6])))


def replace_values(array: np.array, value):
    array[array < value] = -1
    array[array >= value] = 1
   
    return array

print(replace_values(np.random.rand(3, 5, 5), 0.5))
a = replace_values(np.array([1,2,3,4,5,6]), 3)
print(a)
print(type(a))

def array_multi(array: np.array) -> int:
    return np.prod(array)
    
print(array_multi(np.array([[1,2,3,4], [1,2,3,4]])))


def array_multi_2d(array: np.array) -> np.array:
    return np.prod(array, axis=1)

print(array_multi_2d(np.array([[1,2,3,4], [1,2,3,3]])))


def add_border(array : np.array) -> np.array:
    return np.pad(array, pad_width=1, mode='constant', constant_values=0)

print(add_border(np.array([[1,2],[3,4]])))


def list_days(start : str, end : str) -> np.array:
    return np.arange(start, end, dtype='datetime64[D]')

print(list_days('2000-02', '2000-03'))

def get_act_date():
    return np.datetime64('today', 'D')

print(get_act_date())



