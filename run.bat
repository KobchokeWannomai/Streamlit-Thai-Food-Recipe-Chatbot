@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

:: Thai Food Nutrition Analyzer - Run Script for Windows
:: สคริปต์รันแอปพลิเคชันแบบครบถ้วนสำหรับ Windows

:: ตั้งค่าตัวแปร
set DEFAULT_PORT=8501
set PORT=%DEFAULT_PORT%
set DEV_MODE=false
set NO_BROWSER=false
set CHECK_ONLY=false

:: ประมวลผลพารามิเตอร์
:parse_args
if "%~1"=="" goto main
if "%~1"=="-p" (
    set PORT=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--port" (
    set PORT=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="-h" goto show_help
if "%~1"=="--help" goto show_help
if "%~1"=="--dev" (
    set DEV_MODE=true
    shift
    goto parse_args
)
if "%~1"=="--no-browser" (
    set NO_BROWSER=true
    shift
    goto parse_args
)
if "%~1"=="--check-only" (
    set CHECK_ONLY=true
    shift
    goto parse_args
)
echo ❌ ตัวเลือกไม่รู้จัก: %~1
goto show_help

:show_help
echo Thai Food Nutrition Analyzer - Run Script
echo.
echo การใช้งาน:
echo   %~nx0 [OPTIONS]
echo.
echo ตัวเลือก:
echo   -p, --port PORT    กำหนดพอร์ต (default: 8501)
echo   -h, --help         แสดงความช่วยเหลือ
echo   --dev              โหมด development (แสดง debug info)
echo   --no-browser       ไม่เปิดเบราว์เซอร์อัตโนมัติ
echo   --check-only       ตรวจสอบระบบเท่านั้น ไม่รันแอป
echo.
echo ตัวอย่าง:
echo   %~nx0                 # รันด้วยการตั้งค่าพื้นฐาน
echo   %~nx0 -p 8080         # รันที่พอร์ต 8080
echo   %~nx0 --dev           # รันในโหมด development
echo.
exit /b 0

:main
cls
echo ================================================================
echo 🍲 Thai Food Nutrition Analyzer
echo ================================================================
echo.

echo 🔍 กำลังตรวจสอบความพร้อม...
call :check_requirements
if errorlevel 1 exit /b 1

echo 🐍 กำลังเปิดใช้งาน Virtual Environment...
call :activate_venv
if errorlevel 1 exit /b 1

echo 🗃️  กำลังตรวจสอบไฟล์ข้อมูล...
call :check_data_files

echo 📚 กำลังตรวจสอบ Python packages...
call :check_dependencies
if errorlevel 1 exit /b 1

if "%CHECK_ONLY%"=="true" (
    echo ✅ ระบบพร้อมใช้งาน!
    pause
    exit /b 0
)

echo 🚀 กำลังเริ่มแอปพลิเคชัน...
call :run_app
goto :eof

:check_requirements
:: ตรวจสอบไฟล์จำเป็น
if not exist "app.py" (
    echo ❌ ไม่พบไฟล์ app.py
    echo 📋 กรุณารัน setup.bat ก่อนใช้งาน
    exit /b 1
)

if not exist "requirements.txt" (
    echo ❌ ไม่พบไฟล์ requirements.txt
    echo 📋 กรุณารัน setup.bat ก่อนใช้งาน
    exit /b 1
)

if not exist "venv" (
    echo ❌ ไม่พบ Virtual Environment
    echo 📋 กรุณารัน setup.bat ก่อนใช้งาน
    exit /b 1
)

echo ✅ ไฟล์จำเป็นครบถ้วน
exit /b 0

:activate_venv
:: เปิดใช้งาน virtual environment
if not exist "venv\Scripts\activate.bat" (
    echo ❌ ไม่พบ Virtual Environment scripts
    exit /b 1
)

call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ ไม่สามารถเปิดใช้งาน Virtual Environment ได้
    exit /b 1
)

echo ✅ Virtual Environment พร้อมใช้งาน
exit /b 0

:check_data_files
:: ตรวจสอบไฟล์ข้อมูล
set MISSING_DATA=
if not exist "thai_food_processed_cleaned.csv" (
    set MISSING_DATA=!MISSING_DATA! thai_food_processed_cleaned.csv
)
if not exist "thai_ingredients_nutrition_data.csv" (
    set MISSING_DATA=!MISSING_DATA! thai_ingredients_nutrition_data.csv
)

if not "!MISSING_DATA!"=="" (
    echo ⚠️  ไม่พบไฟล์ข้อมูล:!MISSING_DATA!
    echo 📋 แอปพลิเคชันจะทำงานได้แต่อาจมีฟังก์ชันบางส่วนไม่ครบถ้วน
    echo.
    set /p response=ต้องการดำเนินการต่อหรือไม่? (y/n): 
    if /i not "!response!"=="y" (
        echo 📋 ยกเลิกการรัน
        pause
        exit /b 1
    )
) else (
    echo ✅ ไฟล์ข้อมูลครบถ้วน
)
exit /b 0

:check_dependencies
:: ตรวจสอบ Python packages
python -c "import streamlit, pandas, numpy, sentence_transformers, sklearn, requests; print('All packages available')" >nul 2>&1
if errorlevel 1 (
    echo ❌ ไม่พบ Python packages ที่จำเป็น
    echo 🔧 กำลังติดตั้ง packages...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ ไม่สามารถติดตั้ง packages ได้
        echo 📋 กรุณาตรวจสอบการเชื่อมต่ออินเทอร์เน็ต
        pause
        exit /b 1
    )
    echo ✅ ติดตั้ง packages สำเร็จ
) else (
    echo ✅ Python packages ครบถ้วน
)
exit /b 0

:check_port
:: ตรวจสอบพอร์ต
netstat -an | findstr ":%1 " >nul 2>&1
if not errorlevel 1 (
    exit /b 1
) else (
    exit /b 0
)

:find_available_port
:: หาพอร์ตที่ว่าง
set TEST_PORT=%PORT%
:port_loop
call :check_port !TEST_PORT!
if errorlevel 1 (
    set /a TEST_PORT+=1
    if !TEST_PORT! gtr 8600 (
        echo ❌ ไม่สามารถหาพอร์ตที่ว่างได้
        pause
        exit /b 1
    )
    goto port_loop
)
set PORT=!TEST_PORT!
exit /b 0

:open_browser
:: เปิดเว็บเบราว์เซอร์
if "%NO_BROWSER%"=="true" exit /b 0

:: รอให้แอปพลิเคชันเริ่มทำงาน
timeout /t 3 /nobreak >nul

:: เปิดเบราว์เซอร์
start http://localhost:%1
exit /b 0

:show_usage_info
echo.
echo 🚀 แอปพลิเคชันกำลังทำงาน!
echo ================================================================
echo.
echo 📋 📱 การเข้าใช้งาน:
echo    🌐 เว็บเบราว์เซอร์: http://localhost:%1
for /f "tokens=2 delims=:" %%i in ('ipconfig ^| findstr "IPv4"') do (
    set IP=%%i
    set IP=!IP: =!
    if not "!IP!"=="" (
        echo    📱 มือถือ (ในเครือข่ายเดียวกัน): http://!IP!:%1
        goto :found_ip
    )
)
:found_ip
echo.
echo 📋 ⌨️  คำสั่งที่มีประโยชน์:
echo    Ctrl + C: หยุดแอปพลิเคชัน
echo    Ctrl + R: รีเฟรชหน้าเว็บ
echo    F11: เต็มหน้าจอ
echo.
echo 📋 🔧 การแก้ปัญหาเบื้องต้น:
echo    - ถ้าเว็บไม่โหลด: รอ 30 วินาที แล้วรีเฟรช
echo    - ถ้าพอร์ตถูกใช้: แอปจะหาพอร์ตอื่นให้อัตโนมัติ
echo    - ถ้าข้อผิดพลาด: ตรวจสอบ logs ด้านล่าง
echo.
echo ================================================================
echo.
exit /b 0

:run_app
:: หาพอร์ตที่ว่าง
call :find_available_port

if not "%PORT%"=="%DEFAULT_PORT%" (
    echo 📋 ใช้พอร์ต %PORT% แทน
)

:: แสดงข้อมูลการใช้งาน
call :show_usage_info %PORT%

:: เปิดเบราว์เซอร์ (รันใน background)
start /b call :open_browser %PORT%

:: ตั้งค่าสำหรับ development mode
if "%DEV_MODE%"=="true" (
    echo 📋 🔧 Development Mode เปิดใช้งาน
    set STREAMLIT_LOGGER_LEVEL=debug
    set STREAMLIT_CLIENT_TOOLBAR_MODE=viewer
)

:: รันแอปพลิเคชัน Streamlit
streamlit run app.py ^
    --server.port %PORT% ^
    --server.address localhost ^
    --server.enableCORS false ^
    --server.enableXsrfProtection false ^
    --browser.gatherUsageStats false ^
    --logger.level warning

echo.
echo 📋 🛑 แอปพลิเคชันหยุดทำงานแล้ว
echo กด Enter เพื่อปิดหน้าต่างนี้...
pause >nul
goto :eof
