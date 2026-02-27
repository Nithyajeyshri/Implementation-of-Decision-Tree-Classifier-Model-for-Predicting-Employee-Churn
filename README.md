# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import pandas
2. Import Decision tree classifier
3. Fit the data in the model
4. Find the accuracy score 
## Program:
```
import pandas as pd
data = pd.read_csv("Employee.csv")
print("data.head():")
print(data.head())
print("data.info():")
print(data.info())
print("isnull() and sum():")
print(data.isnull().sum())
print("data value counts():")
print(data["left"].value_counts())
data.columns = data.columns.str.strip()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
data["Departments"] = le.fit_transform(data["Departments"])
print("data.head() after encoding:")
print(data.head())
x = data[[
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_montly_hours",
    "time_spend_company",
    "Work_accident",
    "promotion_last_5years",
    "Departments",
    "salary"
]]

print("x.head():")
print(x.head())
y = data["left"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy value:", accuracy)
print("Data Prediction:")
print(dt.predict([[0.5, 0.8, 9, 260, 6, 0, 0, 2, 1]]))
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
plot_tree(dt, feature_names=x.columns, class_names=['Stayed', 'Left'], filled=True)
plt.show()
```

## Output:
<img width="745" height="810" alt="Screenshot 2026-02-27 155919" src="https://github.com/user-attachments/assets/ca0c5605-7c0d-4f6f-a4a5-d04802225d30" />
<img width="475" height="412" alt="Screenshot 2026-02-27 155934" src="https://github.com/user-attachments/assets/4dccfb4e-2137-4c10-96d9-f52575d2d52a" />
<img width="814" height="938" alt="Screenshot 2026-02-27 155949" src="https://github.com/user-attachments/assets/2c62883a-aa4e-4739-8836-d985e13fe790" />
<img width="1775" height="106" alt="Screenshot 2026-02-27 160005" src="https://github.com/user-attachments/assets/9853239d-d7d4-4096-8aa2-154d1bc8f04c" />
<img width="1214" height="809" alt="Screenshot 2026-02-27 160013" src="https://github.com/user-attachments/assets/6727a89d-5200-4481-ba40-b48c3bdd2223" />

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
