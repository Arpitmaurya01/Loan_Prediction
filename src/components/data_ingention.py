import pandas as pd
import numpy as np
import data_load
from src.logger import logging
from src.exeption import CustomException
import sys

data=data_load

class Data_Clean():


    logging.info("Data Preprocessing is Strated")

    try:

        data.df_train.drop("Loan_ID",axis=1,inplace=True)

        cat_val=[fea for fea in data.df_train.columns if data.df_train[fea].dtypes=="O"]
        num_val=[fea for fea in data.df_train.columns if data.df_train[fea].dtypes!="O"]

        if data.df_train.isnull().sum().sum()!=0:
            data.df_train["Gender"].fillna(data.df_train["Gender"].mode()[0],inplace=True)
            data.df_train["Married"].fillna(data.df_train["Married"].mode()[0],inplace=True)
            data.df_train["Dependents"].fillna(data.df_train["Dependents"].mode()[0],inplace=True)
            data.df_train["Education"].fillna(data.df_train["Education"].mode()[0],inplace=True)
            data.df_train["Self_Employed"].fillna(data.df_train["Self_Employed"].mode()[0],inplace=True)

            data.df_train.fillna(data.df_train[num_val].mean(),inplace=True)

        data.df_train["Gender"]=data.df_train["Gender"].map({"Female":0,"Male":1})
        data.df_train["Married"]=data.df_train["Married"].map({"No":0,"Yes":1})
        data.df_train["Dependents"]=data.df_train["Dependents"].map({"0":0,"1":1,"2":2,"3+":3})
        data.df_train["Education"]=data.df_train["Education"].map({"Graduate":0,"Not Graduate":1})
        data.df_train["Self_Employed"]=data.df_train["Self_Employed"].map({"No":0,"Yes":1})
        data.df_train["property_Area"]=data.df_train["property_Area"].map({"Rural":0,"Semiurban":1,"Urban":2})
        data.df_train["Loan_Status"]=data.df_train["Loan_Status"].map({"N":0,"Y":1})

        clean_data=data.df_train
    except Exception as e:
        raise CustomException(e,sys)

    logging.info("Data Preprocessing is completed")
    


