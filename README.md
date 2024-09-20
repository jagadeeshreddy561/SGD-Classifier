# Implementation of Logistic Regression Using SGD Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
step1: start the program.
step2: Import Necessary Libraries and Load Data.
step3: Split Dataset into Training and Testing Sets.
step4: Train the Model Using Stochastic Gradient Descent (SGD).
step5: Make Predictions and Evaluate Accuracy.
step6: Generate Confusion Matrix.
step7: End the program.
```
## Program:
```
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Mallu Jagadeeswar Reddy
RegisterNumber:  212222240059
```
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
iris = load_iris()
df = pd.DataFrame(data = iris.data , columns = iris.feature_names)
df['target'] = iris.target
print(df.head())
x = df.drop('target',axis=1)
y = df['target']
x_train , x_test,y_train,y_test = train_test_split(x,y,test_size =0.2,random_state=42)
sgd_clf = SGDClassifier(max_iter = 1000 , tol=1e-3)
sgd_clf.fit(x_train,y_train)
y_pred = sgd_clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy : {accuracy:.3f}")
cm = confusion_matrix(y_test,y_pred)
print("confusion Matrix:")
print(cm)

```

## Output:
![image](https://github.com/user-attachments/assets/e5288dd4-e39d-462e-968c-f7d805bac741)
![image](https://github.com/user-attachments/assets/53b2e610-6f57-4ce2-8471-c0463260a069)
![image](https://github.com/user-attachments/assets/380f529c-8916-49a3-b6be-390695898b43)
![image](https://github.com/user-attachments/assets/16821deb-2d63-4f85-b74b-f61f637dc518)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
