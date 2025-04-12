!pip install scikit-learn --upgrade

import sklearn

sklearn.__version__
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
df=pd.read_csv('healthcare-dataset-stroke-data.csv')
df
df.info()
df.isnull().sum()
df=df.fillna(df['bmi'].mean())
df.isnull().sum()
np.unique(df["work_type"])
df=df.iloc[:,1:]
df
sns.countplot(x = 'gender',data=df)
plt.show()
sns.countplot(x ='smoking_status', data = df)
plt.show()
sns.countplot(x ='Residence_type', data = df)
plt.show()
sns.countplot(x ='work_type', data = df)
plt.show()
plt.figure(figsize=(10,6))
sns.barplot(x="smoking_status", y="stroke", data=df)
plt.xlabel("\nSmoking status", fontweight="bold")
plt.ylabel("Stroke (mean)\n", fontweight="bold")
plt.show()
plt.figure(figsize=(10,6))
sns.barplot(x="work_type", y="stroke", data=df)
plt.xlabel("\nWork type", fontweight="bold")
plt.ylabel("Stroke (mean)\n", fontweight="bold")
plt.show()
some_attri=['gender','hypertension','heart_disease','ever_married','work_type','Residence_type','smoking_status', 'stroke']
fig, axis = plt.subplots(4, 2, figsize=(14,20))
axis = axis.flatten()
for i, col_name in enumerate(some_attri):
    sns.countplot(x=col_name, data=df, ax=axis[i], hue =df['stroke'])
plt.show()
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (16,10))
fig.patch.set_facecolor('#faf9f7')

for i in (ax1, ax2, ax3):
    i.set_facecolor('#faf9f7')

sns.kdeplot(
    df['age'][df['stroke'] == 0],
    ax = ax1,
    color = "#c8c14f",
    fill = True,
    alpha = 0.5,
    linewidth = 1.5,
    ec = 'black'
)
sns.kdeplot(
    df['age'][df['stroke'] == 1],
    ax = ax1,
    color = "#cd34b5",
    fill = True,
    alpha = 0.5,
    linewidth = 1.5,
    ec = 'black'
)
ax1.legend(['No Stroke', 'Stroke'], loc = 'upper left')
ax1.set_xlabel('Age', fontsize = 14, labelpad = 10)
ax1.set_ylabel('Density', fontsize = 14, labelpad = 10)

sns.kdeplot(
    df['avg_glucose_level'][df['stroke'] == 0],
    ax = ax2,
    color = "#c8c14f",
    fill = True,
    alpha = 0.5,
    linewidth = 1.5,
    ec = 'black'
    )

sns.kdeplot(
    df['avg_glucose_level'][df['stroke'] == 1],
    ax = ax2,
    color = "#cd34b5",
    fill = True,
    alpha = 0.5,
    linewidth = 1.5,
    ec = 'black'
)
ax2.legend(['No Stroke', 'Stroke'])
ax2.set_xlabel('Average Glucose Levels', fontsize = 14, labelpad = 10)
ax2.set_ylabel('')

sns.kdeplot(
    df['bmi'][df['stroke'] == 0],
    ax = ax3,
    color = "#c8c14f",
    fill = True,
    alpha = 0.5,
    linewidth = 1.5,
    ec = 'black'
)

sns.kdeplot(
    df['bmi'][df['stroke'] == 1],
    ax = ax3,
    color = "#cd34b5",
    fill = True,
    alpha = 0.5,
    linewidth = 1.5,
    ec = 'black'
)

ax3.legend(['No Stroke', 'Stroke'])
ax3.set_xlabel('BMI', fontsize = 14, labelpad = 10)
ax3.set_ylabel('')

plt.suptitle('Density of Age, Glucose, and BMI by Stroke', fontsize = 16, fontweight = 'bold')
for i in (ax1, ax2, ax3):
    for j in ['top', 'left', 'bottom', 'right']:
        i.spines[j].set_visible(False)

fig.tight_layout()
fig, ax = plt.subplots(figsize=(10,6))
fig.patch.set_facecolor('#faf9f7')
ax.set_facecolor('#faf9f7')

labels = ['Stroke', 'No Stroke']
colors = ["#f1d295", "#ea5f94"]
sizes = df['stroke'].value_counts()

plt.pie(sizes, explode = [0, 0.15], labels = labels, colors = colors,
           autopct = '%1.1f%%', shadow = True, startangle = 130,
           wedgeprops = {'ec': 'black'}, textprops = {'fontweight': 'medium'}
)
plt.axis('equal')
plt.title('Percentage of Strokes')
labelencoder=LabelEncoder()
for i in df.columns:
  if df.dtypes[i]!=int and df.dtypes[i]!=float:
    df[i]=labelencoder.fit_transform(df[i])

df

plt.subplots(figsize=(10,8))
sns.heatmap(data=df.corr(),annot=True, cmap = "YlGnBu")
plt.show()
X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values
!pip install imbalanced-learn --upgrade
# from imblearn.over_sampling import RandomOverSampler
import imblearn
imblearn.__version__
from imblearn.over_sampling import RandomOverSampler
oversampler = RandomOverSampler()
X_os,Y_os = oversampler.fit_resample(X,Y)
len(X_os)
acc=[]
f1=[]
precision=[]
recall=[]
roc_auc=[]
x_train,x_test,y_train,y_test=train_test_split(X_os,Y_os,test_size=0.3,stratify=Y_os)

model1 = DecisionTreeClassifier()
model1.fit(x_train,y_train)
y_pred1 = model1.predict(x_test)

acc1 = accuracy_score(y_test, y_pred1)
f1_1 = f1_score(y_test, y_pred1)
pre1 = precision_score(y_test, y_pred1)
rec1 = recall_score(y_test, y_pred1)
roc1 = roc_auc_score(y_test, y_pred1)

print("accuracy_score of Model-1 =", acc1)
print("f1_score of Model-1 =", f1_1)
print("precision_score of Model-1 =", pre1)
print("recall_score of Model-1 =", rec1)
print("roc_auc score of Model-1 =", roc1)
model2 = RandomForestClassifier(random_state=42)
model2.fit(x_train,y_train)
y_pred2 = model2.predict(x_test)

acc2 = accuracy_score(y_test, y_pred2)
f1_2 = f1_score(y_test, y_pred2)
pre2 = precision_score(y_test, y_pred2)
rec2 = recall_score(y_test, y_pred2)
roc2 = roc_auc_score(y_test, y_pred2)

print("accuracy_score of Model-2 =", acc2)
print("f1_score of Model-2 =", f1_2)
print("precision_score of Model-2 =", pre2)
print("recall_score of Model-2 =", rec2)
print("roc_auc score of Model-2 =", roc2)
model3 = XGBClassifier()
model3.fit(x_train,y_train)
y_pred3 = model3.predict(x_test)

acc3 = accuracy_score(y_test, y_pred3)
f1_3 = f1_score(y_test, y_pred3)
pre3 = precision_score(y_test, y_pred3)
rec3 = recall_score(y_test, y_pred3)
roc3 = roc_auc_score(y_test, y_pred3)

print("accuracy_score of Model-3=", acc3)
print("f1_score of Model-3=", f1_3)
print("precision_score of Model-3=", pre3)
print("recall_score of Model-3=", rec3)
print("roc_auc score of Model-3=", roc3)
model4 = LGBMClassifier()
model4.fit(x_train,y_train)
y_pred4 = model4.predict(x_test)

acc4 = accuracy_score(y_test, y_pred4)
f1_4 = f1_score(y_test, y_pred4)
pre4 = precision_score(y_test, y_pred4)
rec4 = recall_score(y_test, y_pred4)
roc4 = roc_auc_score(y_test, y_pred4)

print("accuracy_score of Model-4=", acc4)
print("f1_score of Model-4=", f1_4)
print("precision_score of Model-4=", pre4)
print("recall_score of Model-4=", rec4)
print("roc_auc score of Model-4=", roc4)
model5 = LogisticRegression()
model5.fit(x_train,y_train)
y_pred5 = model5.predict(x_test)

acc5 = accuracy_score(y_test, y_pred5)
f1_5 = f1_score(y_test, y_pred5)
pre5 = precision_score(y_test, y_pred5)
rec5 = recall_score(y_test, y_pred5)
roc5 = roc_auc_score(y_test, y_pred5)

print("accuracy_score of Model-5=", acc5)
print("f1_score of Model-5=", f1_5)
print("precision_score of Model-5=", pre5)
print("recall_score of Model-5=", rec5)
print("roc_auc score of Model-5=", roc5)
model6 = SVC(kernel = 'rbf')

model6.fit(x_train,y_train)
y_pred6 = model6.predict(x_test)

acc6 = accuracy_score(y_test, y_pred6)
f1_6 = f1_score(y_test, y_pred6)
pre6 = precision_score(y_test, y_pred6)
rec6 = recall_score(y_test, y_pred6)
roc6 = roc_auc_score(y_test, y_pred6)

print("accuracy_score of Model-6=", acc6)
print("f1_score of Model-6=", f1_6)
print("precision_score of Model-6=", pre6)
print("recall_score of Model-6=", rec6)
print("roc_auc score of Model-6=", roc6)
model7 = DecisionTreeClassifier()

params = {"max_depth" : [3,5,7,9,11,13,15,17,19,21,23,25,27,29],
          "min_samples_leaf":[1,3,4,5,6,7,8,9],
          "max_leaf_nodes":[None,10,20,30,40,50,60,70] }

clf = GridSearchCV(estimator=model7,
                   param_grid=params,
                   scoring='accuracy',
                   verbose=2, cv = 2)

clf.fit(x_train, y_train)

print("Best parameters:", clf.best_params_)
print("Highest Accuracy: ", (-1)*(-clf.best_score_))
model7 = DecisionTreeClassifier(max_depth = 27, max_leaf_nodes = None, min_samples_leaf = 1)
model7.fit(x_train,y_train)
y_pred7 = model7.predict(x_test)

acc7 = accuracy_score(y_test, y_pred7)
f1_7 = f1_score(y_test, y_pred7)
pre7 = precision_score(y_test, y_pred7)
rec7 = recall_score(y_test, y_pred7)
roc7 = roc_auc_score(y_test, y_pred7)

print("accuracy_score of Model-7 =", acc7)
print("f1_score of Model-7 =", f1_7)
print("precision_score of Model-7 =", pre7)
print("recall_score of Model-7 =", rec7)
print("roc_auc score of Model-7 =", roc7)
model8 = RandomForestClassifier()

params = {'max_depth': [5,9,11,13,15,17,19,21,23,25],
          'n_estimators': [10,30,50,70,90,100,200,300],
          'max_leaf_nodes': [10,30,50,70,None]}

clf = GridSearchCV(estimator=model8,
                   param_grid=params,
                   scoring='accuracy',
                   verbose=2, cv = 2)

clf.fit(x_train, y_train)

print("Best parameters:", clf.best_params_)
print("Highest Accuracy: ", -1*(-clf.best_score_))
model8 = RandomForestClassifier(max_depth = 25,  max_leaf_nodes = None, n_estimators = 50)
model8.fit(x_train,y_train)
y_pred8 = model8.predict(x_test)

acc8 = accuracy_score(y_test, y_pred8)
f1_8 = f1_score(y_test, y_pred8)
pre8 = precision_score(y_test, y_pred8)
rec8 = recall_score(y_test, y_pred8)
roc8 = roc_auc_score(y_test, y_pred8)

print("accuracy_score of Model-8 =", acc8)
print("f1_score of Model-8 =", f1_8)
print("precision_score of Model-8 =", pre8)
print("recall_score of Model-8 =", rec8)
print("roc_auc score of Model-8 =", roc8)
model9 = XGBClassifier()

params = {'max_depth': [7,9,11,13,15,17,19,21,23,25],
          'learning_rate': [0.1, 0.3, 0.5],
          'n_estimators': [30,50,70,90,100,200],
          'subsample': [0.3, 0.5, 0.7]}

clf = GridSearchCV(estimator=model9,
                   param_grid=params,
                   scoring='accuracy',
                   verbose=2, cv = 2)

clf.fit(x_train, y_train)

print("Best parameters:", clf.best_params_)
print("Highest Accuracy: ", -1*(-clf.best_score_))
model9 = XGBClassifier(max_depth = 23, learning_rate = 0.3, n_estimators = 200, subsample = 0.7)
model9.fit(x_train,y_train)
y_pred9 = model9.predict(x_test)

acc9 = accuracy_score(y_test, y_pred9)
f1_9 = f1_score(y_test, y_pred9)
pre9 = precision_score(y_test, y_pred9)
rec9 = recall_score(y_test, y_pred9)
roc9 = roc_auc_score(y_test, y_pred9)

print("accuracy_score of Model-9 =", acc9)
print("f1_score of Model-9 =", f1_9)
print("precision_score of Model-9 =", pre9)
print("recall_score of Model-9 =", rec9)
print("roc_auc score of Model-9 =", roc9)

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
model10 = LGBMClassifier(verbose= -100)

params = {
          'max_depth': [5,7,9,11,13,15,17,19,21,23,25],
          'learning_rate': [0.1, 0.3, 0.5],
          'n_estimators': [10,30,50,70,90,100,150,200],
          'subsample': [0.3, 0.5, 0.7]}

clf = GridSearchCV(estimator=model10,
                   param_grid=params,
                   scoring='accuracy',
                   verbose=2,cv=2)

clf.fit(x_train, y_train)

print("Best parameters:", clf.best_params_)
print("Highest Accuracy: ", (-1)*(-clf.best_score_))
model10 = LGBMClassifier(max_depth = 21, learning_rate = 0.3, n_estimators = 200, subsample = 0.3)

model10.fit(x_train,y_train)
y_pred10 = model10.predict(x_test)

acc10 = accuracy_score(y_test, y_pred10)
f1_10 = f1_score(y_test, y_pred10)
pre10 = precision_score(y_test, y_pred10)
rec10 = recall_score(y_test, y_pred10)
roc10 = roc_auc_score(y_test, y_pred10)

print("accuracy_score of Model-10 =", acc10)
print("f1_score of Model-10 =", f1_10)
print("precision_score of Model-10 =", pre10)
print("recall_score of Model-10 =", rec10)
print("roc_auc score of Model-10 =", roc10)
model11 = LogisticRegression()

params = {'penalty' : ['l1','l2'],
          'C'       : np.logspace(-3,3,7),
          'solver'  : ['newton-cg', 'lbfgs', 'liblinear']}

clf = GridSearchCV(estimator=model11,
                   param_grid=params,
                   scoring='accuracy',
                   verbose=2, cv = 2)

clf.fit(x_train, y_train)

print("Best parameters:", clf.best_params_)
print("Highest Accuracy: ", (-1)*(-clf.best_score_))
model11 = LogisticRegression(C = 100, penalty = 'l2', solver = 'lbfgs')
model11.fit(x_train,y_train)
y_pred11 = model11.predict(x_test)

acc11 = accuracy_score(y_test, y_pred11)
f1_11 = f1_score(y_test, y_pred11)
pre11 = precision_score(y_test, y_pred11)
rec11 = recall_score(y_test, y_pred11)
roc11 = roc_auc_score(y_test, y_pred11)

print("accuracy_score of Model-11 =", acc11)
print("f1_score of Model-11 =", f1_11)
print("precision_score of Model-11 =", pre11)
print("recall_score of Model-11 =", rec11)
print("roc_auc score of Model-11 =", roc11)
model12 = SVC()

params = {'C': [0.1, 1, 10, 100],
          'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
          'kernel': ['rbf','sigmoid']}

clf = GridSearchCV(estimator=model12,
                   param_grid=params,
                   scoring='accuracy',
                   verbose=2, cv = 2)

clf.fit(x_train, y_train)

print("Best parameters:", clf.best_params_)
print("Highest Accuracy: ", -1*(-clf.best_score_))
model12 = SVC(C = 1, gamma = 1, kernel = 'rbf')
model12.fit(x_train,y_train)
y_pred12 = model12.predict(x_test)

acc12 = accuracy_score(y_test, y_pred12)
f1_12 = f1_score(y_test, y_pred12)
pre12 = precision_score(y_test, y_pred12)
rec12 = recall_score(y_test, y_pred12)
roc12 = roc_auc_score(y_test, y_pred12)

print("accuracy_score of Model-12 =", acc12)
print("f1_score of Model-12 =", f1_12)
print("precision_score of Model-12 =", pre12)
print("recall_score of Model-12 =", rec12)
print("roc_auc score of Model-12 =", roc12)
acc_score = [acc7,acc8, acc9, acc10, acc11, acc12]
f1_score  = [f1_7,f1_8, f1_9, f1_10, f1_11, f1_12]
pre_score = [pre7,pre8, pre9, pre10, pre11, pre12]
rec_score = [rec7,rec8, rec9, rec10, rec11, rec12]
roc_score = [roc7,roc8, roc9, roc10, roc11, roc12]
names = ['Decision Tree', 'Random Forest', 'XGB Classifier', 'LGBM Classifier', 'Logistic Regression', 'SVC']
plt.figure(figsize = (15,6))

plt.plot(names, acc_score, marker = 'x', color = 'orange', linestyle = 'dashed')
plt.xlabel('Models')
plt.ylabel('Accuracy Score')
plt.show()

plt.figure(figsize = (15,6))

plt.plot(names, f1_score, marker = 'x', color = 'red', linestyle = 'dashed')
plt.xlabel('Models')
plt.ylabel('F1- Score')
plt.show()
plt.figure(figsize = (15,6))

plt.plot(names, pre_score, marker = 'x', color = 'purple', linestyle = 'dashed')
plt.xlabel('Models')
plt.ylabel('Precision Score')
plt.show()
plt.figure(figsize = (15,6))

plt.plot(names, rec_score, marker = 'x', color = 'blue', linestyle = 'dashed')
plt.xlabel('Models')
plt.ylabel('Recall Score')
plt.show()

plt.figure(figsize = (15,6))

plt.plot(names, roc_score, marker = 'x', color = 'green', linestyle = 'dashed')
plt.xlabel('Models')
plt.ylabel('ROC-AUC Score')
plt.show()
import pickle
pickle.dump(model8, open('RandomForest.pkl', 'wb'))
model8.predict([[1,51.0,	0,	0,	1,	2,	0,	166.29,	25.600000,	1	]])

