import pandas as pd
from src.logger import logging
#Load Data
df_train=pd.read_csv("artifacts\\training_set.csv")

df_test=pd.read_csv("artifacts\\testing_set.csv")

logging.info("Data is loaded")


class Data_Load():
    
    print(df_train.shape)
    print(df_test.shape)
    print(df_train.info())
    print(df_test.info())

logging.info("printing of Description of Data is done")
