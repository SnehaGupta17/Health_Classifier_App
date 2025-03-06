import pandas as pd 
import numpy as np 
import joblib
import streamlit as st 
from PIL import Image  
from sklearn.preprocessing import StandardScaler


classifier = joblib.load('lr_classifier.pkl') 
scaler = joblib.load('lr_scaler.pkl')

def welcome(): 
    return 'welcome all'

# defining the function which will make the prediction using  
# the data which the user inputs 
def prediction(Age, BirthAsphyxia, HypDistrib, CO2Report,GruntingReport,LVHreport,LVH,ChestXray_Oligaemic,XrayReport_Oligaemic,Disease_TGA):   
    input_data = np.array([[Age, BirthAsphyxia, HypDistrib, CO2Report, GruntingReport, LVHreport, LVH, ChestXray_Oligaemic,XrayReport_Oligaemic,Disease_TGA]], dtype=int)
    # input_data = pd.DataFrame([[Age, BirthAsphyxia, HypDistrib, CO2Report, GruntingReport, LVHreport, LVH, ChestXray_Oligaemic, XrayReport_Oligaemic, Disease_TGA]],
    #                           columns=['Age', 'BirthAsphyxia', 'HypDistrib', 'CO2Report', 'GruntingReport', 'LVHreport', 'LVH', 'ChestXray_Oligaemic', 'XrayReport_Oligaemic', 'Disease_TGA'])

    scaled_input_data = scaler.transform(input_data)
   
    prediction = classifier.predict(scaled_input_data) 
    # print(prediction) 
    return prediction [0]

# this is the main function in which we define our webpage  
def main(): 
      # giving the webpage a title 
    # st.title("Synthetic Infant Health Data") 
      
    # here we define some of the front end elements of the web page like  
    # the font and background color, the padding and the text to be displayed 
    html_temp = """ 
    <div style ="background-color:hotpink;padding:13px"> 
    <h1 style ="color:black;text-align:center;">👶🏻Synthetic Infant Health Report Classifier App⚕️ </h1> 
    </div> <br>
    <p> Selections for the following report:<br>
        For Age:<br> ('0-3_days' --> 0),<br>('4-10_days' --> 1),<br>('11-30_days' --> 2)<br>
        For other inputs: 0 for No, 1 for Yes </p><br>
        Select appropriately.<br>
    """
      
    # this line allows us to display the front end aspects we have  
    # defined in the above code 
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # the following lines create text boxes in which the user can enter  
    # the data required to make the prediction 
    # Age = st.text_input("Age", "Type Here") 
    # BirthAsphyxia = st.text_input("BirthAsphyxia", "Type Here") 
    # HypDistrib = st.text_input("HypDistrib", "Type Here") 
    # CO2Report = st.text_input("CO2Report", "Type Here") 
    # LVHreport = st.text_input("LVHreport", "Type Here") 
    # RUQO2 = st.text_input("RUQO2", "Type Here") 
    Age = st.number_input("Age", min_value=0, max_value=2, step=1, value=0)  # Age as integer input
    BirthAsphyxia = st.number_input("BirthAsphyxia", min_value=0, max_value=1, step=1, value=0)  # Binary input (0 or 1)
    HypDistrib = st.number_input("HypDistrib", min_value=0, max_value=1, step=1, value=0)  # Binary input (0 or 1)
    LVHreport = st.number_input("LVHreport", min_value=0, max_value=1, step=1, value=0)  # Binary input (0 or 1)
    CO2Report = st.number_input("CO2Report", min_value=0, max_value=1, step=1, value=0)  # CO2 levels as integer input
    GruntingReport = st.number_input("GruntingReport", min_value=0, max_value=1, step=1, value=0)  # GruntingReport levels as integer input    
    LVH = st.number_input("LVH", min_value=0, max_value=1, step=1, value=0)  # LVH levels as integer input
    ChestXray_Oligaemic = st.number_input("ChestXray_Oligaemic", min_value=0, max_value=1, step=1, value=0)  # LVH levels as integer input
    XrayReport_Oligaemic = st.number_input("XrayReport_Oligaemic", min_value=0, max_value=1, step=1, value=0)  # LVH levels as integer input
    Disease_TGA = st.number_input("Disease_TGA", min_value=0, max_value=1, step=1, value=0)  # LVH levels as integer input


    result ="" 
      
    # the below line ensures that when the button called 'Predict' is clicked,  
    # the prediction function defined above is called to make the prediction  
    # and store it in the variable result 
    if st.button("Predict"): 
        result = prediction(Age, BirthAsphyxia, HypDistrib, CO2Report,GruntingReport,LVHreport,LVH,ChestXray_Oligaemic,XrayReport_Oligaemic,Disease_TGA) 
    st.success('Is the Infant Sick? : {}'.format(result)) 
     
if __name__=='__main__': 
    main() 