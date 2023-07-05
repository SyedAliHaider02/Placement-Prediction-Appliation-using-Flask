import pandas as pd
import numpy as np
import pickle

df=pd.read_csv('Placement_Data_Full_Class.csv')

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['gender']=encoder.fit_transform(df['gender'])
df['ssc_b']=encoder.fit_transform(df['ssc_b'])
df['hsc_b']=encoder.fit_transform(df['hsc_b'])
df['hsc_s']=encoder.fit_transform(df['hsc_s'])
df['degree_t']=encoder.fit_transform(df['degree_t'])
df['workex']=encoder.fit_transform(df['workex'])
df['specialisation']=encoder.fit_transform(df['specialisation'])
df['status']=encoder.fit_transform(df['status'])

df['salary'] = df['salary'].fillna(df['salary'].median())

X=df.drop('status',axis=1)
y=df['status']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=26)
rf.fit(X_train,y_train)

pickle.dump(rf,open('placement.pkl','wb'))