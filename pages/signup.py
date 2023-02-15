import pickle
from pathlib import Path
import streamlit as st
import os
import pandas as pd 
import csv 
data = ['Id','Password']

# with open('LoginStatus.csv', 'w') as file:
#     writer = csv.writer(file)
#     writer.writerow(data)
db = {}

l1 = []
l2 = []
ids = st.text_input("Email Address")
password = st.text_input("Password",type="password",key="password")
# l1.append(ids)
# l2.append(password)

# l1.append(ids)
# l2.append(password)
key1 = "Id"
db.setdefault(key1, [])
db[key1].append(ids)

key2 = "password"
db.setdefault(key2, [])
db[key2].append(password)


df = pd.DataFrame(db)
# st.write(db)
# df
if st.button("Register"):
    df.to_csv('LoginStatus.csv', mode='a', header=False, index=False)
    st.success("User Registered Successfully!")

    

# import streamlit as st
# def check_password():
#     """Returns `True` if the user had a correct password."""

#     def password_entered():
#         """Checks whether a password entered by the user is correct."""
#         if (
#             st.session_state["username"] in st.secrets["passwords"]
#             and st.session_state["password"]
#             == st.secrets["passwords"][st.session_state["username"]]
#         ):
#             st.session_state["password_correct"] = True
#             del st.session_state["password"]  # don't store username + password
#             del st.session_state["username"]
#         else:
#             st.session_state["password_correct"] = False

#     if "password_correct" not in st.session_state:
#         # First run, show inputs for username + password.
#         st.text_input("Username", on_change=password_entered, key="username")
#         st.text_input(
#             "Password", type="password", on_change=password_entered, key="password"
#         )
#         return False
#     elif not st.session_state["password_correct"]:
#         # Password not correct, show input + error.
#         st.text_input("Username", on_change=password_entered, key="username")
#         st.text_input(
#             "Password", type="password", on_change=password_entered, key="password"
#         )
#         st.error("😕 User not known or password incorrect")
#         return False
#     else:
#         # Password correct.
#         return True

# if check_password():
#     st.write("Here goes your normal Streamlit app...")
#     st.button("Click me")