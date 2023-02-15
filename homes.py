import streamlit as st
from PIL import Image
st.set_page_config(
	page_title = "Cheating Detection Application")

st.title("Cheating Application ")

st.sidebar.success("Select a page above")



st.image('logo.jpeg', caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
st.title('Introduction')
st.subheader("Purpose")
st.write(""" 
	The main purpose of our project is to safety and better future for our students so that they didn’t do the cheating in exam, if they would not learn since now so their carrier and support for practical life will be disturbed not for their self but they are deceiving their teachers and own self. So, we are going to create our project with exam cheating detection system which is now most useful for now days.""")
st.subheader("Scope")
st.write(""" 
	With the research of exam cheating detection system our team analyzed that when the student's basic starts for learning in his/her life there are two types of students (One who understand with intentions and others don’t learn) when the students don’t learn so in the time of exam, they try to do cheat in exam, so our main scope is to provide knowledge in our students for their better carrier, when we are going to create our web application such as Exam cheating detection by the use of our project when the student will try to cheat in exam with smart watch or try to copy from others exam paper so student will move his/her body our web application will detect him/her and catch them so teacher can easily caught them.""")

st.title("Team Members")

st.write("Imran Ahmed (GL)")
st.write("SE-093-2019")


st.write("Mir Taimoor Iqbal")
st.write("SE-075-2019")


st.write("Muhammad Ali Akbar")
st.write("SE-019-2018")


st.write("Fabeha Qadir")
st.write("SE-076-2019")