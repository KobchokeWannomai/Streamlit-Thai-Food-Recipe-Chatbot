@echo off
chcp 65001 >nul
:: สคริปต์ติดตั้ง Thai Food Recipe Chatbot with Advanced Nutrition สำหรับ Windows

echo ===== 🍲 Thai Food Recipe Chatbot with Advanced Nutrition =====
echo กำลังตั้งค่าสภาพแวดล้อมขั้นสูงบน Windows...
echo.

:: ตรวจสอบว่า Python ถูกติดตั้งหรือไม่
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ ไม่พบ Python กรุณาติดตั้ง Python 3.8 หรือใหม่กว่าแล้วลองใหม่
    echo 📥 ดาวน์โหลดได้จาก: https://www.python.org/downloads/
    pause
    exit /b 1
)

:: แสดงเวอร์ชัน Python
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set python_version=%%i
echo ✅ พบ Python %python_version%

:: ตรวจสอบเวอร์ชัน Python (ต้องการ 3.8 ขึ้นไป)
python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" 2>nul
if %errorlevel% neq 0 (
    echo ❌ ต้องการ Python 3.8 หรือใหม่กว่า (ปัจจุบัน: %python_version%^)
    pause
    exit /b 1
)

:: สร้าง virtual environment
echo 📦 กำลังสร้าง virtual environment...
if exist venv (
    echo ⚠️  พบ virtual environment เดิม กำลังลบและสร้างใหม่...
    rmdir /s /q venv
)

python -m venv venv

:: เปิดใช้งาน virtual environment
echo 🔄 กำลังเปิดใช้งาน virtual environment...
call venv\Scripts\activate.bat

:: อัปเกรด pip
echo ⬆️  กำลังอัปเกรด pip...
python -m pip install --upgrade pip

:: ติดตั้งแพ็คเกจที่จำเป็น
echo 📚 กำลังติดตั้งแพ็คเกจที่จำเป็น...
pip install -r requirements.txt

:: ตรวจสอบการติดตั้ง
echo.
echo 🔍 กำลังตรวจสอบการติดตั้ง...

:: ตรวจสอบแพ็คเกจหลัก
python -c "import streamlit" 2>nul && (echo ✅ streamlit: ติดตั้งสำเร็จ) || (echo ❌ streamlit: ติดตั้งไม่สำเร็จ)
python -c "import pandas" 2>nul && (echo ✅ pandas: ติดตั้งสำเร็จ) || (echo ❌ pandas: ติดตั้งไม่สำเร็จ)
python -c "import sentence_transformers" 2>nul && (echo ✅ sentence-transformers: ติดตั้งสำเร็จ) || (echo ❌ sentence-transformers: ติดตั้งไม่สำเร็จ)
python -c "import plotly" 2>nul && (echo ✅ plotly: ติดตั้งสำเร็จ) || (echo ❌ plotly: ติดตั้งไม่สำเร็จ)
python -c "import requests" 2>nul && (echo ✅ requests: ติดตั้งสำเร็จ) || (echo ❌ requests: ติดตั้งไม่สำเร็จ)

:: ตรวจสอบไฟล์ข้อมูล
echo.
echo 📁 กำลังตรวจสอบไฟล์ข้อมูล...

if exist thai_food_processed.csv (
    echo ✅ พบไฟล์: thai_food_processed.csv
) else (
    echo ⚠️  ไม่พบไฟล์: thai_food_processed.csv
    set missing_main=1
)

if exist thai_ingredients_nutrition.csv (
    echo ✅ พบไฟล์: thai_ingredients_nutrition.csv
) else (
    echo ⚠️  ไม่พบไฟล์: thai_ingredients_nutrition.csv
    set missing_nutrition=1
)

:: สร้างไฟล์ข้อมูลตัวอย่างหากจำเป็น
if defined missing_main (
    echo.
    echo 🎨 กำลังสร้างไฟล์ข้อมูลตัวอย่าง...
    python enhanced_preprocess.py --sample
    
    echo 📊 กำลังประมวลผลข้อมูลตัวอย่างพร้อมโภชนาการ...
    python enhanced_preprocess.py --input thai_food_sample_enhanced.csv --output thai_food_processed.csv --nutrition --enhance
)

:: สร้างไดเรกทอรีที่จำเป็น
echo 📁 กำลังสร้างไดเรกทอรีที่จำเป็น...
if not exist logs mkdir logs
if not exist cache mkdir cache
if not exist data mkdir data
if not exist data\backup mkdir data\backup

:: สร้างไฟล์กำหนดค่า
echo ⚙️  กำลังสร้างไฟล์กำหนดค่า...
(
echo # ตัวอย่างการตั้งค่า Environment Variables
echo # คัดลอกไฟล์นี้เป็น .env และแก้ไขค่าต่างๆ
echo.
echo # USDA FoodData Central API
echo USDA_API_KEY=your_usda_api_key_here
echo.
echo # Nutritionix API
echo NUTRITIONIX_APP_ID=your_nutritionix_app_id_here
echo NUTRITIONIX_APP_KEY=your_nutritionix_app_key_here
echo.
echo # Edamam API
echo EDAMAM_APP_ID=your_edamam_app_id_here
echo EDAMAM_APP_KEY=your_edamam_app_key_here
echo.
echo # การตั้งค่าแอปพลิเคชัน
echo STREAMLIT_SERVER_PORT=8501
echo STREAMLIT_SERVER_ADDRESS=localhost
) > .env.example

:: แสดงผลสรุป
echo.
echo 🎉 การติดตั้งเสร็จสิ้น!
echo.
echo 📋 สรุปการติดตั้ง:
echo    ✅ Virtual environment: พร้อมใช้งาน
echo    ✅ แพ็คเกจ Python: ติดตั้งครบถ้วน
echo    ✅ ข้อมูลตัวอย่าง: พร้อมใช้งาน
echo.
echo 🚀 วิธีการเริ่มใช้งาน:
echo    1. เปิดใช้งาน virtual environment:
echo       call venv\Scripts\activate.bat
echo    2. เริ่มต้นแอปพลิเคชัน:
echo       streamlit run app.py
echo    3. เปิดเว็บเบราว์เซอร์ไปที่: http://localhost:8501
echo.
echo 📖 คำสั่งเพิ่มเติม:
echo    • ประมวลผลข้อมูลใหม่: python enhanced_preprocess.py --help
echo    • เรียกใช้แบบย่อ: run.bat
echo.
echo 💡 เคล็ดลับ:
echo    • ตั้งค่า API keys ในไฟล์ .env เพื่อใช้ข้อมูลโภชนาการจาก API ภายนอก
echo    • ใช้ไฟล์ข้อมูลของคุณเองโดยวางในไดเรกทอรีเดียวกัน
echo.
echo 🔧 หากมีปัญหา:
echo    • ตรวจสอบ logs ในไดเรกทอรี logs\
echo    • อ่านเอกสารใน README.md
echo    • ลองรันคำสั่ง: pip install -r requirements.txt --force-reinstall
echo.
echo 🍲 ขอให้สนุกกับการค้นหาสูตรอาหารไทย!
echo.
pause
