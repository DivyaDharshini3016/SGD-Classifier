# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import necessary libraries and load data

2.Split dataset into Training and Testing sets

3.Train the model using Stochastic Gradient Descent(SGD)

4.Make predictions and evaluate accuracy

5.Generate confusion matrix

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Divya Dharshini S
RegisterNumber: 212224240039

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
iris=load_iris()
df=pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target']=iris.target
print(df.head())
X = df.drop('target',axis=1)
y=df['target']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
sgd_clf=SGDClassifier(max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train,y_train)
y_pred=sgd_clf.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}")
cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(cm)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt='d', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()  
*/

```

## Output:

<img width="614" height="225" alt="Screenshot 2025-09-22 105716" src="https://github.com/user-attachments/assets/73de6ec2-6797-45c9-9888-9dc15c6893fe"/>
<img width="218" height="91" alt="Screenshot 2025-09-22 105725"src="https://github.com/user-attachments/assets/07ea31d6-1fb0-4f10-b54b-916fff975655"/>
<img width="662" height="402" alt="Screenshot 2025-09-22 105745" src="https://github.com/user-attachments/assets/1da51e9d-59ff-42c2-9373-63d78ef4306e"/>

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
