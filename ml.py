from sklearn.ensemble import RandomForestClassifier
import pandas as  pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

df= pd.read_csv('total.csv')
#print(df)

X = df.iloc[:,1:]
Y = df.iloc[:,2]
l = ['complement'] * (250- X.shape[1]) 
for index,col in enumerate(l):
    X[col+str(index)] = 0

X = X.values
Y = Y.values

x_train,x_test,y_train,y_test=train_test_split(X ,Y,test_size=0.2,random_state = np.random.randint(1,1000, 1)[0] )

model =RandomForestClassifier()

model.fit(x_train,y_train)
predictions = model.predict(x_test)
print(classification_report(y_test,predictions))