import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st



df=pd.read_csv("Placement_Data_Full_Class.csv")
df.drop(['sl_no','salary','ssc_b','hsc_b'],axis=1,inplace=True)
st.title('Placement Checking')
st.sidebar.header('Placement Data')
st.write(df.head())
st.subheader('Training Data Stats')
st.write(df.describe())

le=LabelEncoder()
lst=['gender','hsc_s','degree_t','workex','specialisation','status']
for i in lst:
  df[i]=le.fit_transform(df[i])
st.subheader('Training Data')
st.write(df.head())

x = df.drop(['status'], axis = 1)
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 1)

# FUNCTION
def user_report():
  gender = st.sidebar.number_input("Gender-M:1,F:0",0,1)
  ssc_p = st.sidebar.number_input('Secondary education percentage', 0.0,100.0, 50.0)
  hsc_p = st.sidebar.number_input('Higher Secondary education percentage', 0.0,100.0, 50.0)
  hsc_s = st.sidebar.number_input('Specialization in Higher Secondary Education-Arts:0,Commerce:1,Science:2',0,2,1)
  degree_p= st.sidebar.number_input('Degree Percentage', 0.0,100.0, 50.0)
  degree_t = st.sidebar.number_input('Degree Type-Comm&Mgmt:0,Sci&Tech:1,others:2',0,2,1)
  workex= st.sidebar.number_input('Work Experience-Yes:1,No:0',0,1)  
  etest_p= st.sidebar.number_input('Employability test percentage', 0.0,100.0, 50.0)
  specialisation = st.sidebar.number_input('Post Graduation(MBA)- Specialisation-Mkt&HR:1,Mkt&Fin:0,',0,2,1)
  mba_p= st.sidebar.number_input('MBA percentage', 0.0,100.0, 50.0)
  user_report_data = {
      'gender':gender,
      'ssc_p':ssc_p,
      'hsc_p':hsc_p,
      'hsc_s':hsc_s,
      'degree_p':degree_p,
      'degree_t':degree_t,
      'workex':workex,
      'etest_p':etest_p,
      'specialisation':specialisation,
      'mba_p':mba_p
     }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data
# PATIENT DATA
user_data = user_report()
st.subheader('Placement Test Data')
st.write(user_data)

#Model
ad = AdaBoostClassifier()
ad.fit(X_train, y_train)
user_result = ad.predict(user_data)



# COLOR FUNCTION
if user_result[0]==0:
  color = 'blue'
else:
  color = 'red'
  

    
st.subheader('Your Report: ')
output=''
if user_result[0]==0:
  output = 'You are not Placed'
else:
  output = 'You are Placed'
st.title(output)
st.subheader('Accuracy: ')
st.write(str(accuracy_score(y_test, ad.predict(X_test))*100)+'%')