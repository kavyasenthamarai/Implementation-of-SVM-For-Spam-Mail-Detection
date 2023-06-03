# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
-Collect a labeled dataset of emails, distinguishing between spam and non-spam.

-Preprocess the email data by removing unnecessary characters, converting to lowercase, removing stop words, and performing stemming or lemmatization.

-Extract features from the preprocessed text using techniques like Bag-of-Words or TF-IDF.

-Split the dataset into a training set and a test set.

-Train an SVM model using the training set, selecting the appropriate kernel function and hyperparameters.

-Evaluate the trained model using the test set, considering metrics such as accuracy, precision, recall, and F1 score.

-Optimize the model's performance by tuning its hyperparameters through techniques like grid search or random search.

-Deploy the trained and fine-tuned model for real-world use, integrating it into an email server or application.

-Monitor the model's performance and periodically update it with new data or adjust hyperparameters as needed.


## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: KAVYA K
RegisterNumber: 212222230065  
*/
```
```
import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:

1. Result output

![image](https://github.com/kavyasenthamarai/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118668727/41f74835-e2ec-4920-8d8a-8b86d2584826)

2. data.head()
3. 
![image](https://github.com/kavyasenthamarai/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118668727/6f032c01-1a77-4638-bb29-d09a42388e62)

3. data.info()
4. 
![image](https://github.com/kavyasenthamarai/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118668727/0d28a1f7-307f-48ff-b81d-1ffbae547532)

4. data.isnull().sum()

![image](https://github.com/kavyasenthamarai/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118668727/c1fc4c0b-b529-4ad4-a800-3dff8015e29e)

5. Y_prediction value

![image](https://github.com/kavyasenthamarai/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118668727/a129bd1a-3a1a-424f-9da6-40088300ac25)

6. Accuracy value

![image](https://github.com/kavyasenthamarai/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118668727/ac15167e-bbd0-42b0-b3bf-c8fb44187844)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
