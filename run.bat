@echo off
:: Activate virtual environment
call venv\Scripts\activate.bat

:: Run the Streamlit app
streamlit run app.py
