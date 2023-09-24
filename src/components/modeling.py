from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from data_transformation import Train_Test

from src.logger import logging
from src.exeption import CustomException
import sys

data=Train_Test


class model():
    def lr():
        model1=LogisticRegression()                             # Logistic Regression
        model1.fit(data.X_train,data.y_train)

        train_pred1=model1.predict(data.X_train) #train prediction
        test_pred1=model1.predict(data.X_test)   #test prediction

        train_acc1=accuracy_score(data.y_train,train_pred1)
        test_acc1=accuracy_score(data.y_test,test_pred1)

        print("Logistic Regression : ","Train Accuracy = ",train_acc1,"Test Accuracy = ",test_acc1)
        logging.info("Logistic Regression is done")
    def knn():
         model2=KNeighborsClassifier()                # KNN
         model2.fit(data.X_train,data.y_train)

         train_pred2=model2.predict(data.X_train) #train prediction
         test_pred2=model2.predict(data.X_test)   #test prediction

         train_acc2=accuracy_score(data.y_train,train_pred2)
         test_acc2=accuracy_score(data.y_test,test_pred2)

         print("KNN : ","Train Accuracy = ",train_acc2,"Test Accuracy = ",test_acc2)
         logging.info("KNN is done")

    def svc():
         model2=SVC()                # Support Vector machine
         model2.fit(data.X_train,data.y_train)

         train_pred2=model2.predict(data.X_train) #train prediction
         test_pred2=model2.predict(data.X_test)   #test prediction

         train_acc2=accuracy_score(data.y_train,train_pred2)
         test_acc2=accuracy_score(data.y_test,test_pred2)

         print("SVC : ","Train Accuracy = ",train_acc2,"Test Accuracy = ",test_acc2)
         logging.info("SVC is done")


    def dt():
         model2=DecisionTreeClassifier()                # Decision Treee
         model2.fit(data.X_train,data.y_train)

         train_pred2=model2.predict(data.X_train) #train prediction
         test_pred2=model2.predict(data.X_test)   #test prediction

         train_acc2=accuracy_score(data.y_train,train_pred2)
         test_acc2=accuracy_score(data.y_test,test_pred2)

         print("Decision Tree : ","Train Accuracy = ",train_acc2,"Test Accuracy = ",test_acc2)
         logging.info("Decision tree  is done")   
    def rf():
         model2=RandomForestClassifier()                # Random Forest
         model2.fit(data.X_train,data.y_train)

         train_pred2=model2.predict(data.X_train) #train prediction
         test_pred2=model2.predict(data.X_test)   #test prediction

         train_acc2=accuracy_score(data.y_train,train_pred2)
         test_acc2=accuracy_score(data.y_test,test_pred2)

         print("Random Forest : ","Train Accuracy = ",train_acc2,"Test Accuracy = ",test_acc2)
         logging.info("Random Forest is done")
    def ad_boost():
          model2=AdaBoostClassifier()                # Adaboost
          model2.fit(data.X_train,data.y_train)

          train_pred2=model2.predict(data.X_train) #train prediction
          test_pred2=model2.predict(data.X_test)   #test prediction

          train_acc2=accuracy_score(data.y_train,train_pred2)
          test_acc2=accuracy_score(data.y_test,test_pred2)

          print("Adaboost : ","Train Accuracy = ",train_acc2,"Test Accuracy = ",test_acc2)
          logging.info("Adaboost is done")
         
    def Grad_boost():
          model2=GradientBoostingClassifier()                # Gradient Boost
          model2.fit(data.X_train,data.y_train)

          train_pred2=model2.predict(data.X_train) #train prediction
          test_pred2=model2.predict(data.X_test)   #test prediction

          train_acc2=accuracy_score(data.y_train,train_pred2)
          test_acc2=accuracy_score(data.y_test,test_pred2)

          print("Gradient Boost : ","Train Accuracy = ",train_acc2,"Test Accuracy = ",test_acc2)
          logging.info("gradient boost is done")

    def xg_boost():
        model2=XGBClassifier()                # XG boost
        model2.fit(data.X_train,data.y_train)

        train_pred2=model2.predict(data.X_train) #train prediction
        test_pred2=model2.predict(data.X_test)   #test prediction

        train_acc2=accuracy_score(data.y_train,train_pred2)
        test_acc2=accuracy_score(data.y_test,test_pred2)

        print("XG_boost classifier : ","Train Accuracy = ",train_acc2,"Test Accuracy = ",test_acc2)
        logging.info("XG boost is done")

         
         
model.lr()
model.knn()
model.svc()
model.dt()
model.rf()
model.ad_boost()
model.Grad_boost()
model.xg_boost()
        
        
