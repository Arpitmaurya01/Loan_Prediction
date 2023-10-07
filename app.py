
import numpy as np
import joblib
import streamlit as st

#Loading the saved model

loaded_model=joblib.load("artifacts\\best_model.pkl")

#creating a function for prediction

def loan_prediction(input_data):

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


    prediction = loaded_model.predict(input_data_reshaped)
    if prediction==0:
        return "NO"
    else:
        return "Yes"


# Getting Input data from User

def main():

    st.title(" Loan Prediction ")

    Gender=st.selectbox('Gender',["Female","Male"])
    Married=st.selectbox("Married",["No","Yes"])
    Dependents=st.selectbox('Dependents',["0","1","2","3+"])
    Education=st.selectbox('Education',["Not Gradute","Gradute"])
    Self_Employed=st.selectbox('Self_Employed',["No","Yes"])
    ApplicantIncome=st.number_input('ApplicantIncome',min_value=0,max_value=100000)
    CoapplicantIncome=st.number_input('CoapplicantIncome',min_value=0,max_value=50000)
    LoanAmount=st.number_input('LoanAmount',min_value=0,max_value=1000)
    Loan_Amount_Term=st.number_input('Loan_Amount_Term',min_value=0,max_value=480)
    Credit_History=st.selectbox('Credit_History',["0","1"])
    property_Area=st.selectbox('property_Area',["Rural","Semiurban","Urban"])

    loan=[]

    if st.button("Loan Predict"):
        data=[Gender,Married,Dependents,Education,Self_Employed,
              ApplicantIncome,CoapplicantIncome,LoanAmount,
              Loan_Amount_Term,Credit_History,property_Area]
        
        data1=[]

        if data[0]=="Female":
            data1.append(0)
        if data[0]=="Male":
            data1.append(1)

        if data[1]=="No":
            data1.append(0)
        if data[1]=="Yes":
            data1.append(1)

        if data[2]=="0":
            data1.append(0)
        if data[2]=="1":
            data1.append(1)
        if data[2]=="2":
            data1.append(2)
        if data[2]=="3+":
            data1.append(3)
        
        if data[3]=="Not Gradute":
            data1.append(0)
        if data[3]=="Gradute":
            data1.append(1)

        if data[4]=="No":
            data1.append(0)
        if data[4]=="Yes":
            data1.append(1)

        data1.append(data[5])
        data1.append(data[6])
        data1.append(data[7])
        data1.append(data[8])

        if data[9]=="0":
            data1.append(0)
        if data[9]=="1":
            data1.append(1)
        
        if data[10]=="Rural":
            data1.append(0)
        if data[10]=="Semiurban":
            data1.append(1)
        if data[10]=="Urban":
            data1.append(2)


        loan=loan_prediction(data1)

    st.success(loan)

if __name__== '__main__':
    main()
        


        



