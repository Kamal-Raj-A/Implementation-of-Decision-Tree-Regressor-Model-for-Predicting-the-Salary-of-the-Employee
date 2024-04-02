# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. calculate Mean square error,data prediction and r2. 


## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: KAMAL RAJ A
RegisterNumber:  212223040082
*/
import pandas as pd
data=pd.read_csv("/content/Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:
## data.head():
![image](https://github.com/Kamal-Raj-A/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145742556/6736e094-f866-40fb-83e6-ad704452f737)

## data.info():
![image](https://github.com/Kamal-Raj-A/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145742556/d459fd01-8274-4242-b586-6d055b444425)

## data.isnull().sum():
![image](https://github.com/Kamal-Raj-A/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145742556/be97c78d-c30d-4ac3-86a0-173b82f16ad2)

## data.head():
![image](https://github.com/Kamal-Raj-A/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145742556/bceee393-1d94-42ec-bdb0-d58d88aade28)

## x.head():
![image](https://github.com/Kamal-Raj-A/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145742556/4360a9b8-74d8-47d1-a035-bc1cf3c30b1c)

## mse:
![image](https://github.com/Kamal-Raj-A/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145742556/2ef294d5-5160-436b-a289-ccdfd5a3e995)

## r^2:
![image](https://github.com/Kamal-Raj-A/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145742556/b5b749bf-d710-45ea-9953-3ae2b08cd327)

## dt.predict([[5,6]]):
![image](https://github.com/Kamal-Raj-A/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145742556/741bf19a-c948-4cee-9adb-63bdfff3c6bb)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
