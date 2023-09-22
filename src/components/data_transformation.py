from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_ingention import Data_Clean

data=Data_Clean

class Train_Test():
    X=data.clean_data.drop(['Loan_Status_Y'],axis=1)
    y=data.clean_data['Loan_Status_Y']

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

    print(X_train)