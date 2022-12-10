from cProfile import label
from socket import herror
import pandas as pd
import numpy as np
#import time
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#start_time = time.time()

def preprocessData(df,fileType):
    df.columns=['age','type_employer','fnlwgt','education','education_num','marital','occupation','relationship','race','sex','capital_gain','capital_loss','hr_per_week','country','income']
    if fileType == 'test file':
        df.drop(columns=['income'],axis=1,inplace=True)
    df.replace('?',np.nan,inplace=True)
    df.dropna(axis=0,inplace=True)
    df.drop(columns=['fnlwgt','education','relationship'],axis=1,inplace=True)
    for col in ["capital_gain","capital_loss"]:
        df[col] = np.where((df[col] > 0 ),'yes','no')
    df['country'] = np.where((df['country'] != 'United-States' ),'Other','United-States')    
    ageGroups =[[0,25,'young age (<=25)'],[26,45,'adult age ([26,45])'],[46,65,'senior age ([46,65])'],[66,90,'old age ([66,90])']]
    for ageGroup in ageGroups:
        df[ageGroup[2]] = np.where((df['age'] >= ageGroup[0]),df['age'],np.inf)
        df[ageGroup[2]] = np.where((df[ageGroup[2]] <= ageGroup[1]),1,0)
    df.drop(columns=['age'],axis=1,inplace=True) 
    hoursPerWeekThreshold = 40
    hoursPerWeekLevels = ['part-time (<40)','full-time(=40)','over-time(>40)']
    for hoursPerWeekLevel in hoursPerWeekLevels:
        if(hoursPerWeekLevel.find("part-time")!=-1):
            df[hoursPerWeekLevel] = np.where((df['hr_per_week'] < hoursPerWeekThreshold),1,0)
        elif(hoursPerWeekLevel.find("full-time")!=-1):
            df[hoursPerWeekLevel] = np.where((df['hr_per_week'] == hoursPerWeekThreshold),1,0)
        elif(hoursPerWeekLevel.find("over-time")!=-1):
            df[hoursPerWeekLevel] = np.where((df['hr_per_week'] > hoursPerWeekThreshold),1,0)
    df.drop(columns=['hr_per_week'],axis=1,inplace=True)
    workClasses = {
        'Gov':['Federal-gov','Local-gov','State-gov'],
        'Not-working':['Without-pay','Never-worked'],
        'Private':['Private'], 
        'Self-employed':['Self-emp-inc','Self-emp-not-inc']
    }
    colIndex = df.columns.get_loc('type_employer')
    for index in range(0,df.loc[:,'type_employer'].size):
        if(df.iloc[index,colIndex] in workClasses['Gov']):
            df.iat[index,colIndex] = 'Gov'
        elif(df.iloc[index,colIndex] in workClasses['Not-working']):
            df.iat[index,colIndex] = 'Not-working'
        elif(df.iloc[index,colIndex] in workClasses['Self-employed']):
            df.iat[index,colIndex] ='Self-employed'
    for workClass in workClasses.keys():
            df['type_employer='+(workClass)] = np.where((df['type_employer'] == workClass ),1,0)
    df.drop(columns=['type_employer'],axis=1,inplace=True)

    maritalStatus = {
        'Married':['Married-AF-spouse', 'Married-civ-spouse'],
        'Never-married':['Never-married'],
        'Not-married':['Married-spouse-absent','Separated','Divorced','Widowed']
    }
    colIndex = df.columns.get_loc('marital')
    for index in range(0,df.loc[:,'marital'].size):
        if(df.iloc[index,colIndex] in maritalStatus['Married']):
            df.iat[index,colIndex] = 'Married'
        elif(df.iloc[index,colIndex] in maritalStatus['Not-married']):
            df.iat[index,colIndex] = 'Not-married'
    for maritalStatusKey in maritalStatus.keys():
            df['marital_status='+(maritalStatusKey)] = np.where((df['marital'] == maritalStatusKey),1,0)
    df.drop(columns=['marital'],axis=1,inplace=True)
    occupations={
        'Exec-managerial':['Exec-managerial'],
        'Prof-specialty':['Prof-specialty'],
        'Other':['Tech-support','Adm-clerical', 'Priv-house-serv','Protective-serv','Armed-Forces','Other-service'],
        'ManualWork':['Craft-repair','Farming-fishing','Handlers-cleaners','Machine-op-inspct','Transport-moving'],
        'Sales':['Sales']
    }
    colIndex = df.columns.get_loc('occupation')
    for index in range(0,df.loc[:,'occupation'].size):
        if(df.iloc[index,colIndex] in occupations['Other']):
            df.iat[index,colIndex] = 'Other'
        elif(df.iloc[index,colIndex] in occupations['ManualWork']):
            df.iat[index,colIndex] = 'ManualWork'
    for occupationsKey in occupations.keys():
            df['occupation='+(occupationsKey)] = np.where((df['occupation'] == occupationsKey),1,0)
    df.drop(columns=['occupation'],axis=1,inplace=True)
    print("Input data:["+str(df.shape[0])+","+str(df.shape[1])+"]")
    le = LabelEncoder() # label encoder 
    if fileType == 'input file':
         df['income']=le.fit_transform(df['income'])
    df['sex']=le.fit_transform(df['sex'])
    df['race']=le.fit_transform(df['race'])
    df['capital_gain']=le.fit_transform(df['capital_gain'])
    df['capital_loss']=le.fit_transform(df['capital_loss'])
    df['country']=le.fit_transform(df['country'])
    df = pd.get_dummies(df,drop_first=True)
    #df.to_excel("test234.xlsx")
    scaler = StandardScaler()
    train_col_sacle = df_input[['education_num','race','sex','capital_gain','capital_loss','country']]
    train_scaler_col = scaler.fit_transform(train_col_sacle)
    train_scaler_col = pd.DataFrame(train_scaler_col,columns=train_col_sacle.columns)
    df['education_num']= train_scaler_col['education_num']
    df['race']= train_scaler_col['race']
    df['sex']= train_scaler_col['sex']
    df['capital_gain']= train_scaler_col['capital_gain']
    df['capital_loss']= train_scaler_col['capital_loss']
    df['country']= train_scaler_col['country']

df_input = pd.DataFrame(pd.read_csv('adult.input',header=None))
df_test = pd.DataFrame(pd.read_csv('adult.test',header=None))

preprocessData(df_input,'input file')
preprocessData(df_test,'test file')
#df_input.to_excel("output.xlsx")
var_columns = [c for c in df_input.columns if c not in ['income']]
X = df_input.loc[:,var_columns]
y = df_input.loc[:,'income']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=3)
model_tree = DecisionTreeClassifier(criterion='entropy',min_samples_leaf=12)#entropy,gini
model_tree.fit(X_train, y_train)
y_train_pred = model_tree.predict(X_train)
y_valid_pred = model_tree.predict(X_valid)
y_test_pred = model_tree.predict(df_test)
df_test_pred = pd.DataFrame(y_test_pred)
df_test_pred = np.where((df_test_pred==0),'<=50K','>50K')
pd.DataFrame(df_test_pred).to_csv('pred_dt_3.csv', header=None, index=False)
dt_report = classification_report(y_valid,y_valid_pred)
print("dt classification_report" ,'\n',dt_report)
#print("Process finished --- %s seconds ---" % (time.time() - start_time))  