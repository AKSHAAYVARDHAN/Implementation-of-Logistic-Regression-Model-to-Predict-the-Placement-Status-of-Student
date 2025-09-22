# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the required packages and print the present data
2.Print the placement data and salary data.
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices.
```

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Akshaay Vardhan S
RegisterNumber: 212224220007

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

### TOP 5 ELEMENTS
<img width="1221" height="226" alt="image" src="https://github.com/user-attachments/assets/48985837-0d66-4418-9231-9f23adb196f4" />
<img width="1091" height="240" alt="image" src="https://github.com/user-attachments/assets/8ba9cccd-3358-4af6-895e-c2b7f250a654" />
<img width="982" height="497" alt="image" src="https://github.com/user-attachments/assets/3c10e7f1-9909-492b-9805-67945f809b0a" />

### DATA DUPLICATE
<img width="61" height="48" alt="image" src="https://github.com/user-attachments/assets/6d844123-cfe8-4e4d-a898-cd682668a28e" />

### PRINT DATA
<img width="982" height="502" alt="image" src="https://github.com/user-attachments/assets/7fbb042d-3946-4119-801e-f981172bf135" />

### DATA_STATUS
<img width="922" height="510" alt="image" src="https://github.com/user-attachments/assets/cb3de336-d90e-4076-912d-a61204182c45" />

### Y_PREDICTION ARRAY
<img width="586" height="263" alt="image" src="https://github.com/user-attachments/assets/6be7c222-0cb8-493f-b4b0-f437869cdb1c" />

### CONFUSION ARRAY
<img width="762" height="71" alt="image" src="https://github.com/user-attachments/assets/2599c7b0-8269-4ad9-afe5-267351b949c7" />

### ACCURACY VALUE
<img width="210" height="51" alt="image" src="https://github.com/user-attachments/assets/be711bd0-89bb-48e4-a4ed-f86c8e00dcce" />

### CLASSFICATION REPORT
<img width="582" height="176" alt="image" src="https://github.com/user-attachments/assets/559b1136-f2f2-4902-ad2f-4da8af812220" />


### PREDICTION
<img width="303" height="33" alt="image" src="https://github.com/user-attachments/assets/1c47d9a1-19fe-4cff-be0e-c70242bd9489" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
