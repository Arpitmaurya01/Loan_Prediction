import pandas as pd
import joblib
from data_transformation import Train_Test

import data_load

data=data_load



data.df_test.drop("Loan_ID",axis=1,inplace=True)

num_val=[fea for fea in data.df_test.columns if data.df_test[fea].dtypes!="O"]
cat_val=[fea for fea in data.df_test.columns if data.df_test[fea].dtypes=="O"]

if data.df_test.isnull().sum().sum()!=0:
    data.df_test["Gender"].fillna(data.df_train["Gender"].mode()[0],inplace=True)               #fill missing value
    data.df_test["Married"].fillna(data.df_train["Married"].mode()[0],inplace=True)
    data.df_test["Dependents"].fillna(data.df_train["Dependents"].mode()[0],inplace=True)
    data.df_test["Education"].fillna(data.df_train["Education"].mode()[0],inplace=True)
    data.df_test["Self_Employed"].fillna(data.df_train["Self_Employed"].mode()[0],inplace=True)

    data.df_test.fillna(data.df_train[num_val].mean(),inplace=True)

    data.df_test["Gender"]=data.df_test["Gender"].map({"Female":0,"Male":1})           # Encoading
    data.df_test["Married"]=data.df_test["Married"].map({"No":0,"Yes":1})
    data.df_test["Dependents"]=data.df_test["Dependents"].map({"0":0,"1":1,"2":2,"3+":3})
    data.df_test["Education"]=data.df_test["Education"].map({"Graduate":0,"Not Graduate":1})
    data.df_test["Self_Employed"]=data.df_test["Self_Employed"].map({"No":0,"Yes":1})
    data.df_test["property_Area"]=data.df_test["property_Area"].map({"Rural":0,"Semiurban":1,"Urban":2})


print(data.df_test.isnull().sum())
sc=Train_Test.sc

data.df_test=sc.transform(data.df_test)


model=joblib.load("artifacts\\best_model.pkl")

test_predict=model.predict(data.df_test)
predicted_values=pd.DataFrame(test_predict)

predicted_values.to_csv('artifacts/test_prediction.csv')