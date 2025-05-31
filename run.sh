#!/bin/bash

# สคริปต์เริ่มต้น Thai Food Recipe Chatbot with Advanced Nutrition

echo "🍲 Thai Food Recipe Chatbot with Advanced Nutrition"
echo "กำลังเริ่มต้นแอปพลิเคชัน..."
echo ""

# ตรวจสอบว่ามี virtual environment หรือไม่
if [ ! -d "venv" ]; then
    echo "❌ ไม่พบ virtual environment"
    echo "💡 กรุณารันคำสั่ง: ./setup.sh เพื่อติดตั้งก่อน"
    exit 1
fi

# เปิดใช้งาน virtual environment
echo "🔄 กำลังเปิดใช้งาน virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Linux/macOS
    source venv/bin/activate
fi

# ตรวจสอบว่า streamlit ถูกติดตั้งหรือไม่
if ! command -v streamlit &> /dev/null; then
    echo "❌ ไม่พบ streamlit"
    echo "💡 กรุณารันคำสั่ง: ./setup.sh เพื่อติดตั้งแพ็คเกจที่จำเป็น"
    exit 1
fi

# ตรวจสอบไฟล์ข้อมูลที่จำเป็น
required_files=("app.py" "nutrition_api.py" "recipe_search.py")
missing_files=()

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -gt 0 ]; then
    echo "❌ ไม่พบไฟล์ที่จำเป็น: ${missing_files[*]}"
    echo "💡 กรุณาตรวจสอบว่าไฟล์ทั้งหมดอยู่ในไดเรกทอรีเดียวกัน"
    exit 1
fi

# ตรวจสอบไฟล์ข้อมูล
if [ ! -f "thai_food_processed.csv" ]; then
    echo "⚠️  ไม่พบไฟล์ข้อมูลหลัก: thai_food_processed.csv"
    echo "🎨 กำลังสร้างข้อมูลตัวอย่าง..."
    
    # สร้างข้อมูลตัวอย่าง
    python enhanced_preprocess.py --sample
    
    # ประมวลผลข้อมูลตัวอย่าง
    if [ -f "thai_food_sample_enhanced.csv" ]; then
        echo "📊 กำลังประมวลผลข้อมูลตัวอย่าง..."
        python enhanced_preprocess.py --input thai_food_sample_enhanced.csv --output thai_food_processed.csv --nutrition
    fi
fi

# สร้างไดเรกทอรี logs หากยังไม่มี
mkdir -p logs

# ตรวจสอบพอร์ตที่ว่าง
port=8501
if command -v netstat &> /dev/null; then
    if netstat -tuln | grep -q ":$port "; then
        echo "⚠️  พอร์ต $port ถูกใช้งานอยู่"
        echo "🔍 กำลังหาพอร์ตที่ว่าง..."
        
        for p in {8502..8510}; do
            if ! netstat -tuln | grep -q ":$p "; then
                port=$p
                echo "✅ จะใช้พอร์ต $port แทน"
                break
            fi
        done
    fi
fi

# แสดงข้อมูลการเริ่มต้น
echo "✅ ทุกอย่างพร้อมแล้ว!"
echo ""
echo "🌐 เซิร์ฟเวอร์จะเริ่มต้นที่: http://localhost:$port"
echo "📱 สำหรับการเข้าถึงจากอุปกรณ์อื่นในเครือข่าย: http://$(hostname -I | awk '{print $1}'):$port"
echo ""
echo "🎯 คุณสมบัติหลัก:"
echo "   • ค้นหาสูตรอาหารไทยอัจฉริยะ"
echo "   • คำนวณโภชนาการแบบละเอียด"
echo "   • รองรับการพิมพ์ผิดและคำไม่ครบ"
echo "   • แนะนำอาหารตามโภชนาการ"
echo "   • สนับสนุน API ข้อมูลโภชนาการภายนอก"
echo ""
echo "💡 เคล็ดลับการใช้งาน:"
echo "   • ลอง 'แนะนำอาหารแคลอรี่ต่ำ'"
echo "   • ลอง 'เมนูโปรตีนสูง'"
echo "   • ลอง 'ไข่เจียว' หรือ 'ผัดกะเพรา'"
echo ""
echo "⏹️  กด Ctrl+C เพื่อหยุดเซิร์ฟเวอร์"
echo ""

# เริ่มต้น Streamlit app
echo "🚀 กำลังเริ่มต้นแอปพลิเคชัน..."
streamlit run app.py --server.port $port --server.address 0.0.0.0
