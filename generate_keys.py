import pickle
from pathlib import Path
import streamlit_authenticator as stauth
# print("Done !!!")

names = ["dmin", "ser"]

username =["admin", "user"]

password =["admin123", "user123"] 

hashed_passwords =stauth.Hasher(password).generate()

file_path = Path(__file__).parent / "hashed_pw.pkl"

with file_path.open("wb") as file:
    pickle.dump(hashed_passwords, file)
