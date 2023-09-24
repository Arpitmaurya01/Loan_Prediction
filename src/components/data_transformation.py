from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_ingention import Data_Clean

from src.logger import logging
from src.exeption import CustomException
import sys

data=Data_Clean

class Train_Test():
    logging.info("Train test splite is starting")

    try:
        X=data.clean_data.drop(['Loan_Status'],axis=1)
        y=data.clean_data['Loan_Status']

        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

    
        sc=StandardScaler()

        X_train=sc.fit_transform(X_train)
        X_test=sc.transform(X_test)

    except Exception as e:
        raise CustomException(e,sys)
    
    logging.info("Train test splite is compleated")