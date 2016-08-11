import csv
import pandas as pd
import numpy as np
import datetime
import matplotlib
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
import math
 
train = pd.read_csv('D:/Train_pjb2QcD.csv')
 
##Exploratory analysis
'''
train.describe(include='all').transpose()
train.describe(include=[np.number]).transpose()
train.describe(include=['O']).transpose()
for cols in train._get_numeric_data():
     
    ax =train[cols].plot(kind='box')
    fig = ax.get_figure()
    fig.savefig(cols+'.png')
    fig.clf()
     
train.corr()
 
corrDf = train.corr()
for row in corrDf.iterrows():
    for col in corrDf.columns:
        if row[0]!=col:
            if row[1][col] >0.5:
                print row[0],",",col,",",row[1][col]
'''
 
test = pd.read_csv('D:/Test_wyCirpO.csv')
train = pd.read_csv('D:/Train_pjb2QcD.csv')
train['isTrainDataSet'] = 1
test['isTrainDataSet'] = 0
dataset = train.append(test)
#print train.shape,test.shape,dataset.shape
 
 
##########################################################################fEATURE eNGINEERING###########################################################################
dateColumns = ['Manager_DoB','Manager_DOJ','Applicant_BirthDate','Application_Receipt_Date']
for colName in dateColumns:
    dataset[colName]= pd.to_datetime(dataset[colName],coerce=True)
dataset['applicantAge'] = ((dataset['Application_Receipt_Date']-dataset['Applicant_BirthDate'])/ np.timedelta64(1, 'D'))/365
dataset['managerExperience'] = ((dataset['Application_Receipt_Date']-dataset['Manager_DOJ'])/ np.timedelta64(1, 'D'))/365
dataset['managerAge'] =((dataset['Application_Receipt_Date']-dataset['Manager_DoB'])/ np.timedelta64(1, 'D'))/365
for colname in  dataset.columns:
    print colname,dataset[colname].unique()," no:",len(dataset[colname].unique())
dataset['Manager_Joining_Designation']=dataset['Manager_Joining_Designation'].map(lambda x:float(str(x)[-1]) if (str(x)[-1]!='n')& (str(x)[-1]!='r') else float('nan'))
dataset['Manager_Current_Designation']=dataset['Manager_Current_Designation'].map(lambda x:float(str(x)[-1]) if (str(x)[-1]!='n')& (str(x)[-1]!='r') else float('nan'))
dataset['Manager_Promotion']=dataset['Manager_Current_Designation']-dataset['Manager_Joining_Designation']
dataset['PinDifference'] =abs(dataset['Office_PIN']-dataset['Applicant_City_PIN'])
 
'''WOW Feature'''
 
trset=dataset[dataset['isTrainDataSet'] == 1]
gp= trset[['ID','Application_Receipt_Date','Business_Sourced']].groupby(trset.Application_Receipt_Date)
#counting the number of applicants for each date on traindata
ct=gp['ID'].count()
#counting the number of Business_Sourced entries for each date on traindata
srcCnt=gp['Business_Sourced'].sum()
ratio=srcCnt/ct
avg=sum(ratio)/len(ratio)
###avg=0.34701933974315452
###Using a ratio of 0.45 considering a buffer of entries with 0 Business_Sourced which comes in the first half everyday
lst=[]
for date in trset['Application_Receipt_Date'].unique():
    c=ct[date]
    v=int(math.ceil(c*0.45))
    for i in range(v):
        lst.append(1)
    for i in range(c-v):
        lst.append(0)
    print date    
trset['wow']=lst
 
#trset['Day']=trset['Application_Receipt_Date'].map(lambda x : x.weekday())
 
#compare=trset[['ID','Application_Receipt_Date','Business_Sourced','wow','Day']]
#
#from sklearn import metrics
#y=compare['Business_Sourced']
#pred=compare['wow']
#fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
#metrics.auc(fpr, tpr)
"""For test"""
tstset=dataset[dataset['isTrainDataSet'] == 0]
gp= tstset[['ID','Application_Receipt_Date','Business_Sourced']].groupby(tstset.Application_Receipt_Date)
ct=gp['ID'].count()
lst=[]
for date in tstset['Application_Receipt_Date'].unique():
    c=ct[date]
    v=int(math.ceil(c*0.45))
    for i in range(v):
        lst.append(1)
    for i in range(c-v):
        lst.append(0)
    print date
tstset['wow']=lst
dataset = trset.append(tstset)
 
dataset['Day']=dataset['Application_Receipt_Date'].map(lambda x : x.weekday())
##Some more feature engineering
 
columnsToBeDropped = ['Applicant_City_PIN','Applicant_BirthDate','Application_Receipt_Date','Manager_Business2','Manager_DOJ',
                      'Manager_DoB','Manager_Num_Products']
dataset = dataset.drop(columnsToBeDropped,1)
 
for colname in  dataset.columns:
    print colname,dataset[colname].unique()," no:",len(dataset[colname].unique())
     
dataset.Applicant_Occupation.fillna("Others",inplace=True)
dataset.Applicant_Occupation.unique()
dataset.Applicant_Marital_Status.fillna("applicantMaritalStatusMissing",inplace=True)
dataset.Applicant_Marital_Status.unique()
dataset.Applicant_Gender.fillna("applicantGenderMissing",inplace=True)
dataset.Applicant_Gender.unique()
dataset.Applicant_Qualification.fillna("Others",inplace=True)
dataset.Applicant_Qualification.unique()
dataset.loc[(dataset['Applicant_Qualification']=='Associate/Fellow of Institute of Company Secretories of India' )
        | (dataset['Applicant_Qualification']==
        'Associate/Fellow of Institute of Institute of Costs and Works Accountants of India'),'Applicant_Qualification'] ='Associate / Fellow of Institute of Chartered Accountans of India'
 
dataset.Manager_Gender.fillna("Manager_GenderMissing",inplace=True)
dataset.Manager_Status.fillna("Manager_StatusMissing",inplace=True)
columnsToBe1HotEncoded = ['Applicant_Gender','Applicant_Marital_Status','Applicant_Occupation', 'Applicant_Qualification' ,
                         'Manager_Gender','Manager_Status']
d = pd.get_dummies(dataset[columnsToBe1HotEncoded])
df_new = pd.concat([dataset, d], axis=1)
 
df_new= df_new.drop(columnsToBe1HotEncoded,1)
 
df_new.columns
df_new=df_new.fillna(-9999)
#################################################################################################################################################################################################
'''Split'''
trainSet=df_new[df_new.isTrainDataSet==1]
trainSet=trainSet.drop(['isTrainDataSet','ID'],1)
Xo=trainSet.drop(['Business_Sourced'],1)
Xo=Xo[Xo['wow']==1]
Xi=Xo[Xo['wow']==0]
y=trainSet['Business_Sourced'][trainSet['wow']==1]
yi=trainSet['Business_Sourced'][trainSet['wow']==0]
 
#trainSet['Business_Sourced'].describe()
 
testSet=df_new[df_new.isTrainDataSet!=1]
testSet=testSet.drop(['isTrainDataSet','ID'],1)
XTest=testSet.drop(['Business_Sourced'],1)
XTesto=XTest[XTest['wow']==1]
XTesti=XTest[XTest['wow']==0]
'''First model'''
#clf=GradientBoostingClassifier(loss='deviance',n_estimators=300,learning_rate=0.4,max_depth=10 ,max_features='auto',verbose=1,min_samples_split=2,min_samples_leaf=1)
#clf = RandomForestClassifier(n_estimators=100, max_depth=None,min_samples_split=1, random_state=0)
clf = ExtraTreesClassifier(n_estimators=100, max_depth=10,min_samples_split=2, random_state=0)
model=clf.fit(Xo,y)
pred=model.predict(XTesto)
predo=pd.Series(pred,index=XTesto.index)
predi=pd.Series(np.zeros(shape=(XTesti.shape[0])),index=XTesti.index)
finalPred=predo.append(predi)
finalPred=finalPred.sort_index()
 
subm=pd.DataFrame(df_new[df_new.isTrainDataSet!=1]['ID'],columns=['ID'])
subm['Business_Sourced']=finalPred
subm.to_csv('F:/FinalSmartRecruitETonWow100estmaxdepth10.csv',index=False)    
###############################################################################Grid Search and Stacking################################################################################################
'''GRID'''
#max_depth = np.arange(1, 100, 10)
##n_estimators = np.arange(10, 40, 5)
#param_grid = dict(max_depth=max_depth)
#
#cv = StratifiedShuffleSplit(y, 3, test_size=0.5, random_state=0)
#
#grid = GridSearchCV(clf, param_grid=param_grid, cv=cv,scoring='roc_auc',verbose=5)
#
#choose=grid.fit(Xo,y)
#bestParam=choose.best_params_
#bestScore=choose.best_score_ 
 
 
'''Stage 1 Prob'''
#X=Xo.as_matrix()
#clf = ExtraTreesClassifier(n_estimators=200, max_depth=None,min_samples_split=1, random_state=0)
##clf =GradientBoostingClassifier(loss='deviance',n_estimators=200,learning_rate=0.4,max_depth=8 ,max_features='auto',verbose=1,min_samples_split=2,min_samples_leaf=1)
##clf = RandomForestClassifier(n_estimators=200, max_depth=None,min_samples_split=1, random_state=0)
#cv = StratifiedShuffleSplit(y, 4, test_size=0.5, random_state=0)
#predDF=pd.DataFrame()
#i=0
#pred = np.zeros(shape=(X.shape[0]))
#for train_index, test_index in cv:
#    print("TRAIN:", train_index, "TEST:", test_index)
#    X_A, X_B = X[train_index], X[test_index]
#    y_A, y_B = y[train_index], y[test_index]
#    modelOnB=clf.fit(X_B,y_B)
#    PredOnA=modelOnB.predict_proba(X_A)
#    modelOnA=clf.fit(X_A,y_A)
#    PredOnB=modelOnA.predict_proba(X_B)
#    #pred=np.array()
#    pred[train_index]=PredOnA
#    pred[test_index]=PredOnB
#    colname='col'+str(i)
#    print pred
#    print colname
#    predDF[colname]=pred
#    i=i+1
#
#fprob=predDF.mean(axis=1)  
#
#
#model=clf.fit(Xo,y)
#Testfprob=model.predict(XTest)
#XTest['1ststage']=Testfprob
#Xo['1ststage']=fprob
#
##clf2 = RandomForestClassifier(n_estimators=200, max_depth=None,min_samples_split=1, random_state=0)
#clf =GradientBoostingClassifier(loss='deviance',n_estimators=200,learning_rate=0.4,max_depth=8 ,max_features='auto',verbose=1,min_samples_split=2,min_samples_leaf=1)
##clf2 = ExtraTreesClassifier(n_estimators=200, max_depth=None,min_samples_split=1, random_state=0)
#model=clf2.fit(Xo,y)
#pred=model.predict(XTest)
#subm=pd.DataFrame(df_new[df_new.isTrainDataSet!=1]['ID'],columns=['ID'])
#subm['Business_Sourced']=pred
#subm.to_csv('F:/ETmaxdepthnoneonGBmaxdep8.csv',index=False)
 
 
""""""""""""
