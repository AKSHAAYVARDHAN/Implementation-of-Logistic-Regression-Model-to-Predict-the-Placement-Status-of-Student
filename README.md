# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the required packages and print the present data.
2.Print the placement data and salary data.
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices.
5.Display the results.
```

## Program:
```
/*

/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Akshaay Vardhan
RegisterNumber: 212224220007
*/

import pandas as pd
data=pd.read_csv("C:/Users/admin/Downloads/Midhun/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:

### Top 5 Elements:
<img width="1221" height="226" alt="image" src="https://github.com/user-attachments/assets/318dcfba-ba3b-49c2-b12f-48868d2f9da8" />

<img width="1091" height="240" alt="image" src="https://github.com/user-attachments/assets/f78c63d1-aa5c-4255-8c30-af82db8223db" />

<img width="982" height="497" alt="image" src="https://github.com/user-attachments/assets/af979f7e-9ffd-4af3-a211-8a051422bc3b" />

### Data Duplicate:

<img width="61" height="48" alt="image" src="https://github.com/user-attachments/assets/807916c8-79d7-438f-b881-3e80492a3cc2" />

### Print Data:

<img width="982" height="502" alt="image" src="https://github.com/user-attachments/assets/ec48645f-9c0a-43d5-ab32-84f832809497" />

### Data_Status

<img width="922" height="510" alt="image" src="https://github.com/user-attachments/assets/47012c76-6e85-41a2-bae4-7f1ea57e7a10" />

### Y_Prediction:


### Confusion Array:

<img width="762" height="71" alt="image" src="https://github.com/user-attachments/assets/d4ae3664-5e90-4a66-a7bc-c88cbf1af16a" />

### Accuracy Array:

<img width="210" height="51" alt="image" src="https://github.com/user-attachments/assets/55ee9c38-4ede-4235-9293-29da8fb30ace" />

### Classification Report:

<img width="582" height="176" alt="image" src="https://github.com/user-attachments/assets/da21a8f6-cdc6-4319-8e2f-681e4e0119ca" />

### Prediction:

<img width="303" height="33" alt="image" src="https://github.com/user-attachments/assets/7debc1b8-54ac-4340-bd2a-38d037c752d3" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
