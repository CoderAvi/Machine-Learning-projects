import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
data = pd.read_csv('500_Person_Gender_Height_Weight_Index.csv')
print(data.describe())
def give_names_to_indices(ind):
    if ind==0:
        return 'Extremely Weak'
    elif ind==1:
        return 'Weak'
    elif ind==2:
        return 'Normal'
    elif ind==3:
        return 'OverWeight'
    elif ind==4:
        return 'Obesity'
    elif ind==5:
        return 'Extremely Obese'
data['Index'] = data['Index'].apply(give_names_to_indices)
sns.lmplot('Height','Weight',data,hue='Index',size=7,aspect=1,fit_reg=False)
people = data['Gender'].value_counts()
categories = data['Index'].value_counts()
# STATS FOR MEN
data[data['Gender']=='Male']['Index'].value_counts()
# STATS FOR WOMEN
data[data['Gender']=='Female']['Index'].value_counts()
data2 = pd.get_dummies(data['Gender'])
data.drop('Gender',axis=1,inplace=True)
data = pd.concat([data,data2],axis=1)
y=data['Index']
data =data.drop(['Index'],axis=1)
scaler = StandardScaler()
data = scaler.fit_transform(data)
data=pd.DataFrame(data)
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=101)
param_grid = {'n_estimators':[100,200,300,400,500,600,700,800,1000]}
grid_cv = GridSearchCV(RandomForestClassifier(random_state=101),param_grid,verbose=3)
grid_cv.fit(X_train,y_train)
print(grid_cv.best_params_)
# weight category prediction
pred = grid_cv.predict(X_test)
print(classification_report(y_test,pred))
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print('Acuuracy is --> ',accuracy_score(y_test,pred)*100)
print('\n')
def lp(details):
    gender = details[0]
    height = details[1]
    weight = details[2]
    
    if gender=='Male':
        details=np.array([[np.float(height),np.float(weight),0.0,1.0]])
    elif gender=='Female':
        details=np.array([[np.float(height),np.float(weight),1.0,0.0]])
    
    y_pred = grid_cv.predict(scaler.transform(details))
    return (y_pred[0])
    
#Live predictor
your_details = ['Male',175,80]
print(lp(your_details))
