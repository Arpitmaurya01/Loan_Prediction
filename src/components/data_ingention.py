import pandas as pd
import numpy as np
import data_load

data=data_load

class Data_Clean():
    data.df_train.drop("Loan_ID",axis=1,inplace=True)

    cat_val=[fea for fea in data.df_train.columns if data.df_train[fea].dtypes=="O"]
    num_val=[fea for fea in data.df_train.columns if data.df_train[fea].dtypes!="O"]

    if data.df_train.isnull().sum().sum()!=0:
        data.df_train.fillna(data.df_train[cat_val].mode(),inplace=True)
        data.df_train.fillna(data.df_train[num_val].mean(),inplace=True)

    clean_data=pd.get_dummies(data.df_train,drop_first=True)

    print(clean_data)



