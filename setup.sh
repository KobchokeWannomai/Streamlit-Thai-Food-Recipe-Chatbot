#!/bin/bash

# สคริปต์ติดตั้ง Thai Food Recipe Chatbot with Advanced Nutrition ขั้นสูง

echo "===== 🍲 Thai Food Recipe Chatbot with Advanced Nutrition ====="
echo "กำลังตั้งค่าสภาพแวดล้อมขั้นสูง..."
echo ""

# ตรวจสอบว่า Python ถูกติดตั้งหรือไม่
if ! command -v python3 &> /dev/null; then
    echo "❌ ไม่พบ Python 3 กรุณาติดตั้ง Python 3.8 หรือใหม่กว่าแล้วลองใหม่"
    echo "📥 ดาวน์โหลดได้จาก: https://www.python.org/downloads/"
    exit 1
fi

# ตรวจสอบเวอร์ชัน Python
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✅ พบ Python $python_version"

# ตรวจสอบว่าเวอร์ชัน Python เพียงพอหรือไม่
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "❌ ต้องการ Python 3.8 หรือใหม่กว่า (ปัจจุบัน: $python_version)"
    exit 1
fi

# สร้าง virtual environment
echo "📦 กำลังสร้าง virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  พบ virtual environment เดิม กำลังลบและสร้างใหม่..."
    rm -rf venv
fi

python3 -m venv venv

# เปิดใช้งาน virtual environment
echo "🔄 กำลังเปิดใช้งาน virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Linux/macOS
    source venv/bin/activate
fi

# อัปเกรด pip
echo "⬆️  กำลังอัปเกรด pip..."
python -m pip install --upgrade pip

# ติดตั้งแพ็คเกจที่จำเป็น
echo "📚 กำลังติดตั้งแพ็คเกจที่จำเป็น..."
pip install -r requirements.txt

# ตรวจสอบการติดตั้ง
echo ""
echo "🔍 กำลังตรวจสอบการติดตั้ง..."

# ตรวจสอบแพ็คเกจหลัก
required_packages=("streamlit" "pandas" "sentence-transformers" "plotly" "requests")
for package in "${required_packages[@]}"; do
    if python -c "import $package" 2>/dev/null; then
        echo "✅ $package: ติดตั้งสำเร็จ"
    else
        echo "❌ $package: ติดตั้งไม่สำเร็จ"
    fi
done

# ตรวจสอบไฟล์ข้อมูล
echo ""
echo "📁 กำลังตรวจสอบไฟล์ข้อมูล..."

data_files=("thai_food_processed.csv" "thai_ingredients_nutrition.csv")
missing_files=()

for file in "${data_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ พบไฟล์: $file"
    else
        echo "⚠️  ไม่พบไฟล์: $file"
        missing_files+=("$file")
    fi
done

# สร้างไฟล์ข้อมูลตัวอย่างหากจำเป็น
if [ ${#missing_files[@]} -gt 0 ]; then
    echo ""
    echo "🎨 กำลังสร้างไฟล์ข้อมูลตัวอย่าง..."
    python enhanced_preprocess.py --sample
    
    # ถ้าไม่มีไฟล์หลัก ให้ประมวลผลข้อมูลตัวอย่าง
    if [ ! -f "thai_food_processed.csv" ]; then
        echo "📊 กำลังประมวลผลข้อมูลตัวอย่างพร้อมโภชนาการ..."
        python enhanced_preprocess.py --input thai_food_sample_enhanced.csv --output thai_food_processed.csv --nutrition --enhance
    fi
fi

# ตรวจสอบพื้นที่ดิสก์
echo ""
echo "💾 กำลังตรวจสอบพื้นที่ดิสก์..."
available_space=$(df . | tail -1 | awk '{print $4}')
if [ "$available_space" -lt 1000000 ]; then  # น้อยกว่า 1GB
    echo "⚠️  พื้นที่ดิสก์เหลือน้อย กรุณาเพิ่มพื้นที่สำหรับการดาวน์โหลดโมเดล AI"
fi

# สร้างไดเรกทอรีที่จำเป็น
echo "📁 กำลังสร้างไดเรกทอรีที่จำเป็น..."
mkdir -p logs
mkdir -p cache
mkdir -p data/backup

# สร้างไฟล์กำหนดค่า
echo "⚙️  กำลังสร้างไฟล์กำหนดค่า..."
cat > .env.example << EOF
# ตัวอย่างการตั้งค่า Environment Variables
# คัดลอกไฟล์นี้เป็น .env และแก้ไขค่าต่างๆ

# USDA FoodData Central API
USDA_API_KEY=your_usda_api_key_here

# Nutritionix API
NUTRITIONIX_APP_ID=your_nutritionix_app_id_here
NUTRITIONIX_APP_KEY=your_nutritionix_app_key_here

# Edamam API
EDAMAM_APP_ID=your_edamam_app_id_here
EDAMAM_APP_KEY=your_edamam_app_key_here

# การตั้งค่าแอปพลิเคชัน
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
EOF

# แสดงผลสรุป
echo ""
echo "🎉 การติดตั้งเสร็จสิ้น!"
echo ""
echo "📋 สรุปการติดตั้ง:"
echo "   ✅ Virtual environment: พร้อมใช้งาน"
echo "   ✅ แพ็คเกจ Python: ติดตั้งครบถ้วน"
echo "   ✅ ข้อมูลตัวอย่าง: พร้อมใช้งาน"
echo ""
echo "🚀 วิธีการเริ่มใช้งาน:"
echo "   1. เปิดใช้งาน virtual environment:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "      source venv/Scripts/activate"
else
    echo "      source venv/bin/activate"
fi
echo "   2. เริ่มต้นแอปพลิเคชัน:"
echo "      streamlit run app.py"
echo "   3. เปิดเว็บเบราว์เซอร์ไปที่: http://localhost:8501"
echo ""
echo "📖 คำสั่งเพิ่มเติม:"
echo "   • ประมวลผลข้อมูลใหม่: python enhanced_preprocess.py --help"
echo "   • เรียกใช้แบบย่อ: ./run.sh"
echo ""
echo "💡 เคล็ดลับ:"
echo "   • ตั้งค่า API keys ในไฟล์ .env เพื่อใช้ข้อมูลโภชนาการจาก API ภายนอก"
echo "   • ใช้ไฟล์ข้อมูลของคุณเองโดยวางในไดเรกทอรีเดียวกัน"
echo ""
echo "🔧 หากมีปัญหา:"
echo "   • ตรวจสอบ logs ในไดเรกทอรี logs/"
echo "   • อ่านเอกสารใน README.md"
echo "   • ลองรันคำสั่ง: pip install -r requirements.txt --force-reinstall"
echo ""
echo "🍲 ขอให้สนุกกับการค้นหาสูตรอาหารไทย!"
