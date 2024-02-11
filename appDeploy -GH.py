import numpy as np
import pandas as pd
import streamlit as st 
import pickle

model = pickle.load(open('rfc_model.pkl', 'rb'))
cols=['Age','JobLevel', 'Revenu_mensuel','StockOptionLevel','TotalWorkingYears','YearsInCurrentRole','YearsWithCurrManager','JobRole_Sales Representative','État_civil_Single','Heures_supplémentaires_Yes']

def main(): 
    st.title("Prédiction de l'Attrition d'un Employé")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">App Prediction Attrition </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    
    age = st.text_input("Age","0") 
    JobLevel = st.selectbox("JobLevel ",["1","2","3","4","5"]) 
    Revenu_mensuel = st.text_input("Revenu_mensuel", "0") 
    StockOptionLevel = st.selectbox("StockOptionLevel",["1","2","3","4"]) 
    TotalWorkingYears = st.text_input("TotalWorkingYears","0") 
    YearsInCurrentRole = st.text_input("YearsInCurrentRole","0") 
    YearsWithCurrManager = st.text_input("YearsWithCurrManager","0") 
    JobRole_SalesRepresentative = st.selectbox("JobRole_Human Resources",["0","1"])
    État_civil_Single = st.selectbox("État_civil_Single",["0","1"])
    Heures_supplémentaires_Yes = st.selectbox("Heures_supplémentaires_Yes",["0","1"])
    
    if st.button("Predict"): 
        features = [[age,JobLevel,Revenu_mensuel,StockOptionLevel,TotalWorkingYears,YearsInCurrentRole,YearsWithCurrManager,JobRole_SalesRepresentative,État_civil_Single,Heures_supplémentaires_Yes]]
        data = {'age': int(age), 'JobLevel': int(JobLevel), 'Revenu_mensuel': int(Revenu_mensuel),'StockOptionLevel': int(StockOptionLevel),'TotalWorkingYears': int(TotalWorkingYears),'YearsInCurrentRole': int(YearsInCurrentRole),'YearsWithCurrManager': int(YearsWithCurrManager),'JobRole_Sales Representative': int(JobRole_SalesRepresentative ),'État_civil_Single': int(État_civil_Single),'Heures_supplémentaires_Yes': int(Heures_supplémentaires_Yes) }
        print(data)
        df=pd.DataFrame([list(data.values())], columns=['Age','JobLevel', 'Revenu_mensuel','StockOptionLevel','TotalWorkingYears','YearsInCurrentRole','YearsWithCurrManager','JobRole_Sales Representative','État_civil_Single','Heures_supplémentaires_Yes'])
            
        features_list = df.values.tolist()      
        prediction = model.predict(features_list)
    
        output = int(prediction[0])
        if output == 1:
            text = "Attrition positive: Cet employé a l'intention de quitter l'entreprise"
        else:
            text = "Attrition négative: Cet employé n'a pas l'intention de quitter l'entreprise"

        st.success(' {}'.format(text))


if __name__=='__main__': 
    main()