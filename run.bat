@echo off
chcp 65001 >nul
:: สคริปต์เริ่มต้น Thai Food Recipe Chatbot with Advanced Nutrition สำหรับ Windows

echo 🍲 Thai Food Recipe Chatbot with Advanced Nutrition
echo กำลังเริ่มต้นแอปพลิเคชัน...
echo.

:: ตรวจสอบว่ามี virtual environment หรือไม่
if not exist venv (
    echo ❌ ไม่พบ virtual environment
    echo 💡 กรุณารันคำสั่ง: setup.bat เพื่อติดตั้งก่อน
    pause
    exit /b 1
)

:: เปิดใช้งาน virtual environment
echo 🔄 กำลังเปิดใช้งาน virtual environment...
call venv\Scripts\activate.bat

:: ตรวจสอบว่า streamlit ถูกติดตั้งหรือไม่
streamlit --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ ไม่พบ streamlit
    echo 💡 กรุณารันคำสั่ง: setup.bat เพื่อติดตั้งแพ็คเกจที่จำเป็น
    pause
    exit /b 1
)

:: ตรวจสอบไฟล์ที่จำเป็น
echo 🔍 กำลังตรวจสอบไฟล์ที่จำเป็น...

if not exist app.py (
    echo ❌ ไม่พบไฟล์: app.py
    pause
    exit /b 1
)

if not exist nutrition_api.py (
    echo ❌ ไม่พบไฟล์: nutrition_api.py
    pause
    exit /b 1
)

if not exist recipe_search.py (
    echo ❌ ไม่พบไฟล์: recipe_search.py
    pause
    exit /b 1
)

:: ตรวจสอบไฟล์ข้อมูล
if not exist thai_food_processed.csv (
    echo ⚠️  ไม่พบไฟล์ข้อมูลหลัก: thai_food_processed.csv
    echo 🎨 กำลังสร้างข้อมูลตัวอย่าง...
    
    :: สร้างข้อมูลตัวอย่าง
    python enhanced_preprocess.py --sample
    
    :: ประมวลผลข้อมูลตัวอย่าง
    if exist thai_food_sample_enhanced.csv (
        echo 📊 กำลังประมวลผลข้อมูลตัวอย่าง...
        python enhanced_preprocess.py --input thai_food_sample_enhanced.csv --output thai_food_processed.csv --nutrition
    )
)

:: สร้างไดเรกทอรี logs หากยังไม่มี
if not exist logs mkdir logs

:: กำหนดพอร์ต
set port=8501

:: ตรวจสอบพอร์ตที่ว่าง (พื้นฐาน)
netstat -an | find ":%port%" >nul 2>&1
if not %errorlevel% == 1 (
    echo ⚠️  พอร์ต %port% ถูกใช้งานอยู่
    echo 🔍 จะใช้พอร์ต 8502 แทน
    set port=8502
)

:: แสดงข้อมูลการเริ่มต้น
echo ✅ ทุกอย่างพร้อมแล้ว!
echo.
echo 🌐 เซิร์ฟเวอร์จะเริ่มต้นที่: http://localhost:%port%
echo.
echo 🎯 คุณสมบัติหลัก:
echo    • ค้นหาสูตรอาหารไทยอัจฉริยะ
echo    • คำนวณโภชนาการแบบละเอียด
echo    • รองรับการพิมพ์ผิดและคำไม่ครบ
echo    • แนะนำอาหารตามโภชนาการ
echo    • สนับสนุน API ข้อมูลโภชนาการภายนอก
echo.
echo 💡 เคล็ดลับการใช้งาน:
echo    • ลอง 'แนะนำอาหารแคลอรี่ต่ำ'
echo    • ลอง 'เมนูโปรตีนสูง'
echo    • ลอง 'ไข่เจียว' หรือ 'ผัดกะเพรา'
echo.
echo ⏹️  กด Ctrl+C เพื่อหยุดเซิร์ฟเวอร์
echo.

:: เริ่มต้น Streamlit app
echo 🚀 กำลังเริ่มต้นแอปพลิเคชัน...
streamlit run app.py --server.port %port% --server.address 0.0.0.0
