import pandas as pd
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
from sklearn.metrics import classification_report
from src.exeption import CustomException

from data_transformation import Train_Test
import joblib

from src.logger import logging

import sys

data=Train_Test

models = {

                "Logistic Regression": LogisticRegression(),
                "KNN"              : KNeighborsClassifier(),
                "SVC"              :SVC(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBoost Classifier": XGBClassifier(),
                
                
            }



class model():
     def __init__(self):

          pass

     logging.info("Model Training is startted")

     try:
          def modeling():

               best=[]

          

               for i in range(len(models)):


                    model2=list(models.values())[i]                       
                    model2.fit(data.X_train,data.y_train)
                    train_pred2=model2.predict(data.X_train) #train prediction
                    test_pred2=model2.predict(data.X_test)   #test prediction
                    train_acc2=accuracy_score(data.y_train,train_pred2)
                    test_acc2=accuracy_score(data.y_test,test_pred2)

                    print(list(models.keys())[i] , "Train Accuracy = ",train_acc2,"Test Accuracy = ",test_acc2)
                    con_mat=confusion_matrix(data.y_test,test_pred2)
                    classif_report=classification_report(data.y_test,test_pred2)
                    #print("Confusion Matrix :", con_mat)
                    #print("classification_report :", classif_report)









          
     except Exception as e:
          CustomException(e,sys)
     logging.info("Model Training is Completed")
     



par={
     "Logistic regression":{'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
                         'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
                         'max_iter' : [100, 1000,2500, 5000]},

     "KNN"     :{'n_neighbors':list(range(1,31)),
               "p":[1,2]},

     "SVC"         :{'C':[1.5,1.6,2,2.5],
               'kernel':["linear","rbf"],
               'gamma':[0.1,0.5,0.6,1]},

     "Decision Tree":{'splitter' : ['best', 'random'],
               'criterion' : ['gini', 'entropy'],
               'max_depth': [4,5,6,7,8],
               'min_samples_split': [4,5,6,7,8,9,10],
               'min_samples_leaf': [3,4,5,6,7,8,9,10]},

     "Random Forest":{'n_estimators':list(range(1,51)),
               "criterion" : ['gini' , 'entropy'],
               'max_depth': [5,6,7]},

     "AdaBoost Classifier":{'n_estimators':list(range(1,100)),
          "learning_rate":[0.001,0.01,0.1,0.005,0.05]},

     "Gradient Boosting":{'n_estimators':list(range(1,51)),
               "learning_rate":[0.5,0.01,0.1],
               "max_depth" : [4,5,6]},

     "XGBoost Classfier":{'n_estimators':list(range(1,51)),
          'learning_rate':[0.1,0.3,1],
          'max_depth':[2,4,5,6]},


}
class HyperParameterTuning():
     def __init__(self):
          pass

     logging.info("Hyperparameter tuning is starting")
    

     try:
           def tuning():
                 

               for i in range(len(list(models))):
                                            
                    est=list(models.values())[i]
                    parm= list(par.values())[i]
          

                    esc=GridSearchCV(est,parm,cv=5)                   #Hyperparameter Tunning
                    esc.fit(data.X_train,data.y_train)

                    est.set_params(**esc.best_params_)                # Find Best parameter

                    est.fit(data.X_train,data.y_train)

                    train_pred=est.predict(data.X_train)
                    test_pred=est.predict(data.X_test)

                    train_accuracy=accuracy_score(data.y_train,train_pred)
                    test_accuracy=accuracy_score(data.y_test,test_pred)

                    print("{}'s Train Accuracy : {} and Test Accuracy : {}".format(list(models.keys())[i],train_accuracy,test_accuracy))

     except Exception as e:
          CustomException(e,sys) 

     logging.info("Hyperparameter Tuning is completed")

#HyperParameterTuning.tuning()

class Best_Model():
                    
                    est=XGBClassifier()
                    parm= {'n_estimators':list(range(1,51)),
                                             'learning_rate':[0.1,0.3,1],
                                             'max_depth':[2,4,5,6]}
          

                    esc=GridSearchCV(est,parm,cv=5)                   #Hyperparameter Tunning
                    esc.fit(data.X_train,data.y_train)

                    est.set_params(**esc.best_params_)                # Find Best parameter

                    est.fit(data.X_train,data.y_train)

                    train_pred=est.predict(data.X_train)
                    test_pred=est.predict(data.X_test)

                    train_accuracy=accuracy_score(data.y_train,train_pred)
                    test_accuracy=accuracy_score(data.y_test,test_pred)

                    print(train_accuracy)
                    print(test_accuracy)

                    joblib.dump(est,"artifacts/best_model.joblib")
     
     


Best_Model