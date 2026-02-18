import streamlit as st
st.title('IST 488 labs')
lab1 = st.Page('labs/lab1.py', title = 'lab 1')
lab2 = st.Page('labs/lab2.py', title = 'lab 2')
lab3 = st.Page('labs/lab3.py', title ='lab 3')
lab4 = st.Page('labs/lab4.py', title ='lab 4')
lab5 = st.Page('labs/lab5.py', title ='lab 5')
pg = st.navigation([lab1,lab2,lab3,lab4,lab5])

st.set_page_config(page_title = 'IST 488 labs',
                   initial_sidebar_state= 'expanded')
pg.run()
