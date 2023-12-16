import pandas as pd
import streamlit as st
import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta


url_api = "http://localhost:3000/ml-predict"

toi = ['ileocolic', 'colocolic', 'ileoileocolic', 'ileoileal','jejunojejunal', 'appendicocolic']
gender = ['Male', 'Female']
pre_ab = ['present', 'absent']
locin = ['RUQ', 'LUQ', 'RLQ', 'LLQ', 'rectum']
method = ['Pneumatic reduction', 'Hydrostatic reduction']
yes_no = ['Yes', 'No']

def call_api(url, data):
    response = requests.post(url, json=data)
    return response.json()

def main():
    st.title("Machine Learning Model Aid Prediction for Failed Nonoperative Reduction of Intussusception")
    
    sample_data = {
        "data": {
            'Type of intussusception': st.selectbox("Type of intussusception", toi),
            'Gender': st.selectbox("Gender", gender),
            'Age': float(st.text_input("Age", 11)),
            'Body weight': float(st.text_input("Body weight", 10.0)), 
            'Duration of symptoms': float(st.text_input("Duration of symptoms", 24)), 
            'Vomiting': st.selectbox("Vomiting", pre_ab), 
            'Bloody stool/Rectal bleeding': st.selectbox("Bloody stool/Rectal bleeding", pre_ab), 
            'Abdominal distension': st.selectbox("Abdominal distension", pre_ab), 
            'Body Temperature': float(st.text_input("Body Temperature", 36.7)), 
            'Palpable abdominal mass': st.selectbox("Palpable abdominal mass", pre_ab), 
            'Location of intussusception': st.selectbox("Location of intussusception", locin), 
            'Method of reduction': st.selectbox("Method of reduction", method), 
            'Small bowel obstruction from Plain film Abdomen ': st.selectbox("Small bowel obstruction from Plain film Abdomen", yes_no),
            'a thick peripheral hypoechoic rim': st.selectbox("a thick peripheral hypoechoic rim", pre_ab), 
            'free intraperitoneum fluid': st.selectbox("free intraperitoneum fluid", pre_ab),
            'fluid trapped within intussusceptum': st.selectbox("fluid trapped within intussusceptum", pre_ab),
            'enlarged lymph node in intussusception': st.selectbox("enlarged lymph node in intussusception", pre_ab), 
            'pathologic lead point': st.selectbox("pathologic lead point", pre_ab),
            'absence of blood flow in the intussusception': st.selectbox("absence of blood flow in the intussusception", pre_ab)
        }
    }
    
    # Show the JSON data
    st.subheader("JSON Data to Send:")
    st.json(sample_data)
    print(sample_data)
    
    # Button to call the API
    if st.button("Call API"):
        st.info("Calling API...")

        # Call the API with basic authentication
        response_data = call_api(url_api, sample_data)

        # Show the API response
        # st.subheader("API Response:")
        # st.json(response_data)

        print("\n\n")
        print("*"*100)
        print(response_data)
        print("*"*100)
        st.header("Results")
        st.write("**Success:**",response_data['Success']*100,"% ")
        st.write("**Failed:**",response_data['Failed']*100,"% ")
        
if __name__ == "__main__":
    main()