#!/bin/bash

echo "🍲 เริ่มแชทบอทสูตรอาหารไทย..."

# เปิดใช้งาน virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    echo "🔧 เปิดใช้งาน virtual environment สำหรับ Windows..."
    source venv/Scripts/activate
else
    # Linux/macOS
    echo "🔧 เปิดใช้งาน virtual environment สำหรับ macOS/Linux..."
    source venv/bin/activate
fi

# ตรวจสอบว่า virtual environment ทำงานหรือไม่
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  คำเตือน: virtual environment อาจไม่ได้เปิดใช้งาน"
    echo "💡 กรุณารันคำสั่ง setup.sh ก่อนถ้ายังไม่ได้ติดตั้ง"
else
    echo "✅ virtual environment พร้อมใช้งาน: $VIRTUAL_ENV"
fi

echo "🚀 กำลังเริ่มแอปพลิเคชัน Streamlit..."
echo "📱 แอปจะเปิดในเบราว์เซอร์ที่ http://localhost:8501"
echo "🛑 กด Ctrl+C เพื่อหยุดแอป"

# รันแอป Streamlit
streamlit run streamlit_app.py

# ตรวจสอบสถานะการทำงาน
if [ $? -ne 0 ]; then
    echo "❌ เกิดข้อผิดพลาดในการรันแอป"
    echo "💡 กรุณาตรวจสอบ:"
    echo "   - ติดตั้ง dependencies ครบถ้วนหรือไม่"
    echo "   - ไฟล์ streamlit_app.py มีอยู่หรือไม่"
    echo "   - virtual environment เปิดใช้งานถูกต้องหรือไม่"
    exit 1
fi

echo "✅ แอปทำงานเสร็จสิ้น"
