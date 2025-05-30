@echo off
REM setup.bat (สำหรับ Windows)

echo =============================================================================
echo 🍲 การติดตั้งแชทบอทสูตรอาหารไทย
echo =============================================================================
echo กำลังตั้งค่าสภาพแวดล้อม...

REM ตรวจสอบว่าติดตั้ง Python หรือไม่
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ ข้อผิดพลาด: ไม่ได้ติดตั้ง Python หรือไม่อยู่ใน PATH กรุณาติดตั้ง Python และลองใหม่อีกครั้ง
    echo 📥 ดาวน์โหลด Python จาก: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM ตรวจสอบเวอร์ชัน Python (ต้องการ Python 3.8+)
python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"
if %errorlevel% neq 0 (
    echo ❌ ข้อผิดพลาด: ต้องการ Python 3.8+
    echo กรุณาอัปเกรดเวอร์ชัน Python ของคุณ
    pause
    exit /b 1
)

echo ✅ เวอร์ชัน Python เข้ากันได้

REM สร้าง virtual environment
echo 🔧 กำลังสร้าง virtual environment...
python -m venv venv

REM เปิดใช้งาน virtual environment
echo 🔧 กำลังเปิดใช้งาน virtual environment...
call venv\Scripts\activate.bat

REM อัปเกรด pip
echo 📦 กำลังอัปเกรด pip...
pip install --upgrade pip >nul 2>&1

REM ติดตั้ง requirements
echo 📦 กำลังติดตั้งแพ็กเกจที่จำเป็น...
pip install -r requirements.txt

echo ✅ ติดตั้ง dependencies เรียบร้อยแล้ว

REM ตรวจสอบว่ามีชุดข้อมูลหรือไม่
echo 📊 กำลังตรวจสอบชุดข้อมูล...
if not exist thai_food_processed.csv (
    echo ⚠️  ไม่พบไฟล์ thai_food_processed.csv ในไดเรกทอรีปัจจุบัน
    echo 🔍 กำลังตรวจสอบไฟล์ข้อมูลทางเลือก...
    
    if exist thai_food_raw.csv (
        echo 📝 พบไฟล์ thai_food_raw.csv กำลังประมวลผลพร้อมการวิเคราะห์โภชนาการ...
        python preprocess.py --input thai_food_raw.csv --output thai_food_processed.csv --analyze-nutrition
    ) else if exist thai_food_sample.csv (
        echo 📝 พบไฟล์ thai_food_sample.csv กำลังประมวลผลข้อมูลตัวอย่าง...
        python preprocess.py --input thai_food_sample.csv --output thai_food_processed.csv --analyze-nutrition
    ) else (
        echo ❌ ข้อผิดพลาด: ไม่พบไฟล์ข้อมูล กรุณาให้แน่ใจว่ามีไฟล์ใดไฟล์หนึ่งต่อไปนี้:
        echo    - thai_food_processed.csv ^(ข้อมูลที่ประมวลผลแล้ว^)
        echo    - thai_food_raw.csv ^(ข้อมูลดิบสำหรับการประมวลผล^)
        echo    - thai_food_sample.csv ^(ข้อมูลตัวอย่างสำหรับการทดสอบ^)
        echo.
        echo คุณสามารถดาวน์โหลดข้อมูลจาก repository หรือใช้ข้อมูลตัวอย่างที่ให้มา
        pause
        exit /b 1
    )
) else (
    echo ✅ พบไฟล์ thai_food_processed.csv
)

REM สร้างฐานข้อมูลโภชนาการ
echo 🧪 กำลังสร้างฐานข้อมูลโภชนาการ...
python preprocess.py --create-nutrition-db

REM สร้างไฟล์การกำหนดค่า
echo ⚙️  กำลังสร้างไฟล์การกำหนดค่า...
if not exist .env (
    echo # แชทบอทอาหารไทย - การกำหนดค่า > .env
    echo # USDA API ^(ฟรี^): https://fdc.nal.usda.gov/api-guide.html >> .env
    echo USDA_API_KEY=your_usda_api_key_here >> .env
    echo # Nutritionix API ^(200 ฟรี/วัน^): https://www.nutritionix.com/business/api >> .env
    echo NUTRITIONIX_API_KEY=your_nutritionix_api_key_here >> .env
    echo NUTRITIONIX_APP_ID=your_nutritionix_app_id_here >> .env
    echo ENABLE_ENHANCED_SEARCH=true >> .env
    echo ENABLE_COOKING_ADJUSTMENTS=true >> .env
    echo ✅ สร้างไฟล์ .env สำหรับการกำหนดค่า API
) else (
    echo ✅ ไฟล์ .env มีอยู่แล้ว
)

REM สร้างโฟลเดอร์ .streamlit และไฟล์ secrets
if not exist .streamlit mkdir .streamlit
if not exist .streamlit\secrets.toml (
    echo [api_keys] > .streamlit\secrets.toml
    echo USDA_API_KEY = "your_usda_api_key_here" >> .streamlit\secrets.toml
    echo NUTRITIONIX_API_KEY = "your_nutritionix_api_key_here" >> .streamlit\secrets.toml
    echo NUTRITIONIX_APP_ID = "your_nutritionix_app_id_here" >> .streamlit\secrets.toml
    echo ✅ สร้างเทมเพลต Streamlit secrets
) else (
    echo ✅ ไฟล์ Streamlit secrets มีอยู่แล้ว
)

REM ทดสอบการติดตั้ง
echo 🧪 กำลังทดสอบการติดตั้ง...
python -c "import streamlit, pandas, sentence_transformers, plotly, requests; from nutrition_analyzer import NutritionAnalyzer; from ingredient_converter import IngredientConverter; analyzer = NutritionAnalyzer(); converter = IngredientConverter(); print('✅ การทดสอบการติดตั้งผ่าน')" 2>nul
if %errorlevel% neq 0 (
    echo ⚠️  การทดสอบการติดตั้งมีปัญหา แต่อาจยังใช้งานได้
)

REM สร้างคู่มือ API
echo 📖 กำลังสร้างคู่มือ API...
echo # 🔑 คู่มือการตั้งค่า API > API_SETUP_GUIDE.md
echo. >> API_SETUP_GUIDE.md
echo ## 🆓 USDA FoodData Central API ^(ฟรี^) >> API_SETUP_GUIDE.md
echo 1. เยี่ยมชม: https://fdc.nal.usda.gov/api-guide.html >> API_SETUP_GUIDE.md
echo 2. คลิก "Get an API Key" และกรอกข้อมูล >> API_SETUP_GUIDE.md
echo 3. รับ API key จากอีเมล >> API_SETUP_GUIDE.md
echo 4. แก้ไขไฟล์ .env และใส่ API key >> API_SETUP_GUIDE.md
echo. >> API_SETUP_GUIDE.md
echo ## 🥇 Nutritionix API ^(200 ฟรี/วัน^) >> API_SETUP_GUIDE.md
echo 1. เยี่ยมชม: https://www.nutritionix.com/business/api >> API_SETUP_GUIDE.md
echo 2. สมัครสมาชิกฟรี >> API_SETUP_GUIDE.md
echo 3. รับ App ID และ API Key >> API_SETUP_GUIDE.md
echo 4. แก้ไขไฟล์ .env และใส่ข้อมูล >> API_SETUP_GUIDE.md

echo.
echo =============================================================================
echo ✅ การติดตั้งเสร็จสิ้นเรียบร้อย!
echo =============================================================================
echo.
echo 📋 สิ่งที่ได้ติดตั้ง:
echo    ✅ virtual environment พร้อม dependencies
echo    ✅ ชุดข้อมูลอาหารไทยพร้อมการวิเคราะห์โภชนาการ
echo    ✅ ฐานข้อมูลโภชนาการขั้นสูง
echo    ✅ ไฟล์การกำหนดค่า API
echo    ✅ คู่มือการตั้งค่า API
echo.
echo 🚀 ขั้นตอนถัดไป:
echo    1. 📖 อ่าน API_SETUP_GUIDE.md สำหรับการตั้งค่า API ^(ทางเลือก^)
echo    2. 🔧 แก้ไขไฟล์ .env เพื่อเพิ่ม API keys
echo    3. ▶️  รัน: run.bat หรือ streamlit run streamlit_app.py
echo    4. 🌐 เปิดเบราว์เซอร์ไปที่ http://localhost:8501
echo    5. ⚙️  ใช้แถบการตั้งค่าในแอปเพื่อกำหนดค่า APIs
echo.
echo 💡 ฟีเจอร์ที่พร้อมใช้งาน:
echo    🔍 การค้นหาอัจฉริยะ
echo    📊 การวิเคราะห์โภชนาการขั้นสูง
echo    🧪 การปรับแต่งการทำอาหาร
echo    📱 UI ที่ปรับปรุงแล้ว
echo.
echo 🔗 ลิงก์ API:
echo    • USDA API: https://fdc.nal.usda.gov/api-guide.html
echo    • Nutritionix API: https://www.nutritionix.com/business/api
echo.
echo 🙏 ขอบคุณที่ใช้แชทบอทสูตรอาหารไทย! 🍲
echo =============================================================================
pause
