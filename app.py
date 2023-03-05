#Készíts egy olyan függvényt ami paraméterként egy listát vár amiben egész számok vannak, 
#és el kell döntenie,hogy van-e benne páratlan szám. A visszatérésí érték egy bool legyen (True:van benne,False:nincs benne)
#Egy példa a bemenetre: [1,2,3,4,4,5]
#Egy példa a kimenetre: True
#return type: bool
#függvény neve legyen: contains_odd

def contains_odd(args):
 Contains= False

 for szam in args:
     if(szam % 2 != 0):
            Contains= True
          #  print("contains")
     
 return Contains
 
Numbers = [12,11,4,4]

#contains_odd(Numbers)
print(contains_odd(Numbers))


#Készíts egy függvényt ami paraméterként egy listát vár amiben egész számok vannak,
#és eldönti minden eleméről, hogy páratlan-e. A kimenet egy lista legyen amiben True/False értékek vannak.
#Egy példa a bemenetre: [1,2,3,4,5]
#Egy példa a kimenetre: [True,False,True,False,True]
#return type: list
#függvény neve legyen: is_odd


def is_odd(args):
 

 
 iterator = map(lambda szam: szam % 2 != 0, args)
          
     
 return iterator
 
Numbers = [1,2,1,4]

#contains_odd(Numbers)
print(list(is_odd(Numbers)))



#Készíts egy függvényt ami paraméterként 2 db listát vár, és kiszámolja a listák elemenként vett összegét.
#A függvény egy listával térjen vissza amiben a megfelelő indexen lévő lista_1 és lista_2 elemek összege van.
#Egy példa a bemenetekre: input_list_1:[1,2,3,4], input_list_2:[1,2,3,4]
#Egy példa a kimenetre: [2,3,4,8]
#return type: list
#függvény neve legyen: element_wise_sum

def element_wise_sum(ls1,ls2):
 

 
 res_list = []
 for i in range(0, len(ls1)):
    res_list.append(ls1[i] + ls2[i])
          
     
 return res_list

test_list1 = [1, 3, 4, 6, 8]
test_list2 = [4, 5, 6, 2, 10]

print(list(element_wise_sum(test_list1, test_list2)))


#Készíts egy függvényt ami paraméterként egy dictionary-t vár és egy listával tér vissza
#amiben a kulcs:érték párok egy Tuple-ben vannak.
#Egy példa a bemenetere: {"egy":1,"ketto":2,"harom":3}
#Egy példa a kimenetre: [("egy",1),("ketto",2),("harom",3)]
#return type: list
#függvény nevel egyen: dict_to_list


def dict_to_list(dict):
 

 
 result = [(k, v) for k, v in dict.items()]
 
          
     
 return result

test_dict = {"egy":1,"ketto":2,"harom":3}


print(list(dict_to_list(test_dict)))

#Ha végeztél a feladatokkal akkor ezt a jupytert alakítsd át egy .py file-ra 
#ha vscode-ban dolgozol: https://stackoverflow.com/questions/64297272/best-way-to-convert-ipynb-to-py-in-vscode
#ha jupyter lab-ban dolgozol: https://stackoverflow.com/questions/52885901/how-to-save-python-script-as-py-file-on-jupyter-notebook