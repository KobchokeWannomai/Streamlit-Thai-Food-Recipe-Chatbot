@echo off
REM run.bat (สำหรับ Windows)

echo 🍲 เริ่มแชทบอทสูตรอาหารไทย...

REM เปิดใช้งาน virtual environment
echo 🔧 เปิดใช้งาน virtual environment...
call venv\Scripts\activate.bat

REM ตรวจสอบว่า virtual environment ทำงานหรือไม่
if "%VIRTUAL_ENV%"=="" (
    echo ⚠️  คำเตือน: virtual environment อาจไม่ได้เปิดใช้งาน
    echo 💡 กรุณารันคำสั่ง setup.bat ก่อนถ้ายังไม่ได้ติดตั้ง
    pause
    exit /b 1
) else (
    echo ✅ virtual environment พร้อมใช้งาน: %VIRTUAL_ENV%
)

echo 🚀 กำลังเริ่มแอปพลิเคชัน Streamlit...
echo 📱 แอปจะเปิดในเบราว์เซอร์ที่ http://localhost:8501
echo 🛑 กด Ctrl+C เพื่อหยุดแอป

REM รันแอป Streamlit
streamlit run streamlit_app.py

if %errorlevel% neq 0 (
    echo ❌ เกิดข้อผิดพลาดในการรันแอป
    echo 💡 กรุณาตรวจสอบ:
    echo    - ติดตั้ง dependencies ครบถ้วนหรือไม่
    echo    - ไฟล์ streamlit_app.py มีอยู่หรือไม่
    echo    - virtual environment เปิดใช้งานถูกต้องหรือไม่
    pause
    exit /b 1
)

echo ✅ แอปทำงานเสร็จสิ้น
pause
