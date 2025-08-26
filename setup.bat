@echo off
chcp 65001 >nul
cls

echo ================================================================
echo 🍲 Thai Food Nutrition Analyzer - การติดตั้งอัตโนมัติ
echo ================================================================
echo.

:: ตรวจสอบสิทธิ์ admin (ถ้าจำเป็น)
net session >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ ตรวจพบสิทธิ์ Administrator
) else (
    echo ⚠️  กำลังรันด้วยสิทธิ์ปกติ
)

echo 🔍 กำลังตรวจสอบความต้องการของระบบ...
echo.

:: ตรวจสอบ Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ ไม่พบ Python! กรุณาติดตั้ง Python 3.8+ จาก https://www.python.org/
    echo.
    echo 📋 วิธีติดตั้ง Python:
    echo    1. ไปที่ https://www.python.org/downloads/
    echo    2. ดาวน์โหลด Python 3.8 หรือใหม่กว่า
    echo    3. ติดตั้งโดยเลือก "Add Python to PATH"
    echo    4. รีสตาร์ท Command Prompt และรันสคริปต์นี้อีกครั้ง
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✅ พบ Python %PYTHON_VERSION%

:: ตรวจสอบ pip
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ ไม่พบ pip! กรุณาติดตั้ง pip
    echo    python -m ensurepip --default-pip
    pause
    exit /b 1
)

echo ✅ พบ pip
echo.

:: ตรวจสอบ Git (ถ้าจำเป็น)
git --version >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ พบ Git
) else (
    echo ⚠️  ไม่พบ Git (จะข้าม git operations)
)

echo.
echo 📁 กำลังสร้างโครงสร้างโฟลเดอร์...
if not exist "data" mkdir data
if not exist "model" mkdir model
if not exist "config" mkdir config
if not exist ".streamlit" mkdir .streamlit
echo ✅ สร้างโฟลเดอร์เรียบร้อย

echo.
echo 🐍 กำลังสร้าง Virtual Environment...
if exist "venv" (
    echo ⚠️  พบ venv เก่า กำลังลบและสร้างใหม่...
    rmdir /s /q venv
)

python -m venv venv
if %errorlevel% neq 0 (
    echo ❌ ไม่สามารถสร้าง virtual environment ได้
    pause
    exit /b 1
)
echo ✅ สร้าง Virtual Environment สำเร็จ

echo.
echo 🔌 กำลังเปิดใช้งาน Virtual Environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ❌ ไม่สามารถเปิดใช้งาน virtual environment ได้
    pause
    exit /b 1
)
echo ✅ เปิดใช้งาน Virtual Environment สำเร็จ

echo.
echo 📦 กำลังอัปเกรด pip...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo ⚠️  ไม่สามารถอัปเกรด pip ได้ แต่จะดำเนินการต่อ
)

echo.
echo 📚 กำลังติดตั้ง Python packages...
echo    (กระบวนการนี้อาจใช้เวลา 5-10 นาที)
echo.

:: ติดตั้ง packages แบบแยก เพื่อให้เห็นความคืบหน้า
echo 🔧 กำลังติดตั้ง Core packages...
pip install streamlit pandas numpy

echo 🤖 กำลังติดตั้ง AI/ML packages...
pip install sentence-transformers scikit-learn transformers

echo 🔥 กำลังติดตั้ง PyTorch...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo 🌐 กำลังติดตั้ง Web packages...
pip install requests urllib3

echo 📊 กำลังติดตั้ง Data packages...
pip install openpyxl plotly matplotlib

echo 🛠️  กำลังติดตั้ง Utility packages...
pip install python-dateutil python-dotenv tqdm fuzzywuzzy python-levenshtein numba

if %errorlevel% neq 0 (
    echo ❌ การติดตั้งล้มเหลว! กำลังลองวิธีอื่น...
    echo 🔄 กำลังติดตั้งจาก requirements.txt...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo ❌ ไม่สามารถติดตั้ง packages ได้
        echo.
        echo 🛠️  การแก้ปัญหา:
        echo    1. ตรวจสอบการเชื่อมต่ออินเทอร์เน็ต
        echo    2. ลองรันคำสั่งนี้ด้วยตนเอง: pip install -r requirements.txt
        echo    3. ตรวจสอบ Python version (ต้องเป็น 3.8+)
        pause
        exit /b 1
    )
)

echo ✅ ติดตั้ง Python packages สำเร็จ!

echo.
echo 🗃️  กำลังตรวจสอบไฟล์ข้อมูล...
if not exist "thai_food_processed_cleaned.csv" (
    if exist "thai_food_processed.csv" (
        echo ⚠️  พบไฟล์ thai_food_processed.csv กำลังประมวลผลใหม่...
        python preprocess.py --input thai_food_processed.csv --output thai_food_processed_cleaned.csv --enhanced
    ) else (
        echo ⚠️  ไม่พบไฟล์ข้อมูลหลัก กรุณาเตรียมไฟล์ thai_food_processed.csv
        echo    หรือ thai_food_processed_cleaned.csv ในโฟลเดอร์หลัก
    )
) else (
    echo ✅ พบไฟล์ข้อมูลหลัก
)

echo.
echo 🔧 กำลังสร้างไฟล์การตั้งค่า...
if not exist ".streamlit\secrets.toml" (
    echo # USDA API Configuration > .streamlit\secrets.toml
    echo USDA_API_KEY = "DEMO_KEY" >> .streamlit\secrets.toml
    echo # Get your free API key from: https://fdc.nal.usda.gov/api-key-signup.html >> .streamlit\secrets.toml
    echo.>> .streamlit\secrets.toml
    echo # Application Settings >> .streamlit\secrets.toml
    echo [app_settings] >> .streamlit\secrets.toml
    echo max_search_results = 10 >> .streamlit\secrets.toml
    echo enable_fuzzy_search = true >> .streamlit\secrets.toml
    echo.
    echo ✅ สร้างไฟล์ .streamlit\secrets.toml
    echo 📝 กรุณาแก้ไข USDA_API_KEY ในไฟล์ .streamlit\secrets.toml
)

echo.
echo 🚀 กำลังทดสอบการติดตั้ง...
python -c "import streamlit, pandas, sentence_transformers; print('✅ All dependencies OK')" 2>nul
if %errorlevel% == 0 (
    echo ✅ การทดสอบสำเร็จ!
) else (
    echo ⚠️  การทดสอบล้มเหลว แต่อาจยังใช้งานได้
)

echo.
echo 📱 กำลังสร้างไฟล์ shortcuts...
echo @echo off > run_app.bat
echo call venv\Scripts\activate.bat >> run_app.bat
echo streamlit run app.py >> run_app.bat
echo pause >> run_app.bat

echo @echo off > fetch_usda_data.bat
echo call venv\Scripts\activate.bat >> fetch_usda_data.bat
echo python usda_nutrition_fetcher.py --recipes thai_food_processed_cleaned.csv --output usda_nutrition_latest.csv >> fetch_usda_data.bat
echo pause >> fetch_usda_data.bat

echo ✅ สร้างไฟล์ shortcuts

echo.
echo ================================================================
echo 🎉 การติดตั้งเสร็จสิ้น!
echo ================================================================
echo.
echo 📋 สรุปการติดตั้ง:
echo    ✅ Virtual Environment: venv\
echo    ✅ Python Packages: ติดตั้งครบถ้วน
echo    ✅ การตั้งค่า: .streamlit\secrets.toml
echo    ✅ Shortcuts: run_app.bat, fetch_usda_data.bat
echo.
echo 🚀 วิธีการรันแอปพลิเคชัน:
echo    1. ดับเบิลคลิก run_app.bat
echo    2. หรือรันในคอมมานด์ไลน์:
echo       - call venv\Scripts\activate.bat
echo       - streamlit run app.py
echo.
echo 🌐 หลังจากรันแล้ว เปิดเว็บเบราว์เซอร์ไปที่:
echo    http://localhost:8501
echo.
echo 📚 ขั้นตอนถัดไป:
echo    1. แก้ไข USDA_API_KEY ในไฟล์ .streamlit\secrets.toml
echo    2. ดาวน์โหลดข้อมูลใหม่ด้วย fetch_usda_data.bat
echo    3. เปิดใช้งานแอปพลิเคชัน
echo.
echo 🆘 หากพบปัญหา:
echo    - ตรวจสอบ README.md สำหรับคู่มือแก้ปัญหา
echo    - หรือเปิด Issue ใน GitHub
echo.
echo กด Enter เพื่อปิดหน้าต่างนี้...
pause >nul
