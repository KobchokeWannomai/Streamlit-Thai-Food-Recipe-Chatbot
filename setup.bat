@echo off
echo ===== Thai Food Recipe Chatbot Setup =====
echo Setting up environment...

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH. Please install Python and try again.
    exit /b 1
)

:: Check Python version (requires Python 3.8+)
python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"
if %errorlevel% neq 0 (
    echo Error: Python 3.8+ is required.
    echo Please upgrade your Python version.
    exit /b 1
)

echo Python version is compatible.

:: Create virtual environment
echo Creating virtual environment...
python -m venv venv

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Install requirements
echo Installing required packages...
pip install -r requirements.txt

:: Check if the dataset exists
if not exist thai_food_processed.csv (
    echo Error: thai_food_processed.csv not found in the current directory.
    echo Please make sure the data file is in the project directory.
    exit /b 1
)

echo.
echo Setup completed successfully!
echo.
echo To run the chatbot, use the following command:
echo streamlit run app.py
echo.
echo Thank you for using Thai Food Recipe Chatbot!
pause
