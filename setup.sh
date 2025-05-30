#!/bin/bash

# การตั้งค่าแชทบอทสูตรอาหารไทยขั้นสูงพร้อมการวิเคราะห์โภชนาการ

echo "============================================================================="
echo "🍲 การติดตั้งแชทบอทสูตรอาหารไทยพร้อมการวิเคราะห์โภชนาการขั้นสูง"
echo "============================================================================="
echo "กำลังตั้งค่าสภาพแวดล้อมขั้นสูงพร้อมการเชื่อมต่อ API และฟีเจอร์ขั้นสูง..."
echo ""

# สีสำหรับการแสดงผล
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # ไม่มีสี

# ฟังก์ชันสำหรับการแสดงผลแบบมีสี
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_step() {
    echo -e "${BLUE}🔄 $1${NC}"
}

# ตรวจสอบการติดตั้ง Python
print_step "ตรวจสอบการติดตั้ง Python..."
if ! command -v python3 &> /dev/null; then
    print_error "ไม่พบ Python 3 กรุณาติดตั้ง Python 3.8+ และลองใหม่อีกครั้ง"
    echo ""
    echo "📥 ดาวน์โหลด Python จาก: https://www.python.org/downloads/"
    exit 1
fi

# ตรวจสอบเวอร์ชัน Python
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
print_info "พบ Python เวอร์ชัน: $PYTHON_VERSION"

if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
    print_success "เวอร์ชัน Python เข้ากันได้"
else
    print_error "ต้องการ Python 3.8+ เวอร์ชันปัจจุบัน: $PYTHON_VERSION"
    echo "📥 กรุณาอัปเกรด Python และลองใหม่อีกครั้ง"
    exit 1
fi

# สร้าง virtual environment
print_step "สร้าง virtual environment..."
python3 -m venv venv

# เปิดใช้งาน virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Linux/macOS
    source venv/bin/activate
fi

print_success "เปิดใช้งาน virtual environment แล้ว"

# อัปเกรด pip
print_step "อัปเกรด pip..."
pip install --upgrade pip > /dev/null 2>&1

# ติดตั้ง requirements
print_step "ติดตั้งแพ็กเกจที่จำเป็น..."
pip install -r requirements.txt

print_success "ติดตั้ง dependencies เรียบร้อยแล้ว"

# ตรวจสอบว่ามีชุดข้อมูลหรือไม่
print_step "ตรวจสอบความพร้อมของชุดข้อมูล..."
if [ ! -f "thai_food_processed.csv" ]; then
    print_warning "ไม่พบไฟล์ thai_food_processed.csv ในไดเรกทอรีปัจจุบัน"
    print_info "ตรวจสอบไฟล์ข้อมูลทางเลือก..."
    
    if [ -f "thai_food_raw.csv" ]; then
        print_info "พบไฟล์ thai_food_raw.csv กำลังประมวลผลพร้อมการวิเคราะห์โภชนาการขั้นสูง..."
        python preprocess.py --input thai_food_raw.csv --output thai_food_processed.csv --analyze-nutrition
    elif [ -f "thai_food_sample.csv" ]; then
        print_info "พบไฟล์ thai_food_sample.csv กำลังประมวลผลข้อมูลตัวอย่างพร้อมการปรับปรุง..."
        python preprocess.py --input thai_food_sample.csv --output thai_food_processed.csv --analyze-nutrition
    else
        print_error "ไม่พบไฟล์ข้อมูล กรุณาให้แน่ใจว่ามีไฟล์ใดไฟล์หนึ่งต่อไปนี้:"
        echo "   - thai_food_processed.csv (ข้อมูลที่ประมวลผลแล้ว)"
        echo "   - thai_food_raw.csv (ข้อมูลดิบสำหรับการประมวลผล)"
        echo "   - thai_food_sample.csv (ข้อมูลตัวอย่างสำหรับการทดสอบ)"
        echo ""
        print_info "คุณสามารถดาวน์โหลดข้อมูลจาก repository หรือใช้ข้อมูลตัวอย่างที่ให้มา"
        exit 1
    fi
else
    print_success "พบไฟล์ thai_food_processed.csv"
    
    # ตรวจสอบว่ามีข้อมูลโภชนาการในไฟล์ CSV หรือไม่
    if python3 -c "
import pandas as pd
try:
    df = pd.read_csv('thai_food_processed.csv')
    has_nutrition = any(col.startswith('nutrition_') for col in df.columns)
    exit(0 if has_nutrition else 1)
except:
    exit(1)
"; then
        print_success "พบข้อมูลโภชนาการในชุดข้อมูลที่มีอยู่"
    else
        print_warning "ไม่พบข้อมูลโภชนาการ กำลังเพิ่มการวิเคราะห์โภชนาการขั้นสูง..."
        python preprocess.py --input thai_food_processed.csv --output thai_food_processed.csv --analyze-nutrition
    fi
fi

# สร้างฐานข้อมูลโภชนาการขั้นสูง
print_step "สร้างฐานข้อมูลโภชนาการขั้นสูง..."
python preprocess.py --create-nutrition-db

# สร้างไฟล์การกำหนดค่าสำหรับ API keys
print_step "ตั้งค่าไฟล์การกำหนดค่า..."
if [ ! -f ".env" ]; then
    cat > .env << EOL
# แชทบอทอาหารไทยขั้นสูง - การกำหนดค่า API
# รับ API keys เหล่านี้สำหรับการวิเคราะห์โภชนาการขั้นสูง

# USDA FoodData Central API (ฟรี - แนะนำอย่างยิ่ง!)
# สมัครที่: https://fdc.nal.usda.gov/api-guide.html
# ประโยชน์: ฟรี ข้อมูลโภชนาการครบถ้วน รองรับโดยรัฐบาล
USDA_API_KEY=your_usda_api_key_here

# Nutritionix API (ทางเลือก - 200 requests/วัน ฟรี)
# สมัครที่: https://www.nutritionix.com/business/api
# ประโยชน์: ฐานข้อมูลอาหารที่มีการประมวลผลภาษาธรรมชาติ
NUTRITIONIX_API_KEY=your_nutritionix_api_key_here
NUTRITIONIX_APP_ID=your_nutritionix_app_id_here

# การกำหนดค่าฟีเจอร์ขั้นสูง
ENABLE_ENHANCED_SEARCH=true
ENABLE_COOKING_ADJUSTMENTS=true
ENABLE_API_INTEGRATION=true

# การตั้งค่าประสิทธิภาพ
API_TIMEOUT=10
CACHE_DURATION_HOURS=24
MAX_SEARCH_RESULTS=10

# การกำหนดค่า Logging
LOG_LEVEL=INFO
ENABLE_DETAILED_LOGGING=false
EOL
    print_success "สร้างไฟล์ .env สำหรับการกำหนดค่า API ขั้นสูง"
    print_info "📝 แก้ไขไฟล์ .env เพื่อเพิ่ม API keys ของคุณสำหรับฟีเจอร์ขั้นสูง"
else
    print_success "ไฟล์ .env มีอยู่แล้ว"
fi

# สร้างเทมเพลต Streamlit secrets พร้อมการตั้งค่าขั้นสูง
if [ ! -f ".streamlit/secrets.toml" ]; then
    mkdir -p .streamlit
    cat > .streamlit/secrets.toml << EOL
# การกำหนดค่า Streamlit Secrets ขั้นสูง
# คัดลอก API keys ของคุณมาที่นี่สำหรับการ deploy

[api_keys]
# USDA FoodData Central (ฟรี)
USDA_API_KEY = "your_usda_api_key_here"

# Nutritionix (200 requests/วัน ฟรี)
NUTRITIONIX_API_KEY = "your_nutritionix_api_key_here"
NUTRITIONIX_APP_ID = "your_nutritionix_app_id_here"

[features]
# การเปิดใช้ฟีเจอร์ขั้นสูง
enhanced_search = true
cooking_adjustments = true
api_integration = true
auto_scroll = true

[performance]
# การตั้งค่าประสิทธิภาพ
api_timeout = 10
cache_duration = 24
max_search_results = 10
batch_size = 5
EOL
    print_success "สร้างเทมเพลต Streamlit secrets ขั้นสูง"
else
    print_success "ไฟล์ Streamlit secrets มีอยู่แล้ว"
fi

# ทดสอบการติดตั้งขั้นสูง
print_step "ทดสอบการติดตั้งขั้นสูง..."
if python3 -c "
import streamlit
import pandas
import sentence_transformers
import plotly
import requests
from nutrition_analyzer import NutritionAnalyzer
from ingredient_converter import IngredientConverter
print('✅ สามารถ import โมดูลที่จำเป็นทั้งหมดได้')
try:
    # ทดสอบฟีเจอร์ขั้นสูง
    analyzer = NutritionAnalyzer()
    converter = IngredientConverter()
    print('✅ เริ่มต้นตัววิเคราะห์โภชนาการขั้นสูงแล้ว')
    print('✅ เริ่มต้นตัวแปลงส่วนผสมแล้ว')
except Exception as e:
    print(f'⚠️  คำเตือนฟีเจอร์ขั้นสูง: {e}')
"; then
    print_success "การทดสอบการติดตั้งขั้นสูงผ่าน"
else
    print_error "การทดสอบการติดตั้งขั้นสูงล้มเหลว กรุณาตรวจสอบข้อความแสดงข้อผิดพลาดข้างต้น"
    exit 1
fi

# สร้างคู่มือการตั้งค่า API ตัวอย่าง
cat > API_SETUP_GUIDE.md << EOL
# 🔑 คู่มือการตั้งค่า API สำหรับฟีเจอร์ขั้นสูง

## 🆓 USDA FoodData Central API (แนะนำ - ฟรี!)

### ประโยชน์:
- ✅ ฟรีสมบูรณ์ ไม่จำกัดการใช้งาน
- ✅ ฐานข้อมูลโภชนาการครบถ้วน
- ✅ ข้อมูลที่เชื่อถือได้จากรัฐบาล
- ✅ ครอบคลุมอาหารนานาชาติรวมถึงอาหารเอเชีย

### ขั้นตอนการตั้งค่า:
1. เยี่ยมชม: https://fdc.nal.usda.gov/api-guide.html
2. คลิก "Get an API Key"
3. กรอกแบบฟอร์ม:
   - ชื่อ: ชื่อของคุณ
   - อีเมล: ที่อยู่อีเมลของคุณ
   - องค์กร: "การใช้งานส่วนตัว" หรือองค์กรของคุณ
   - วัตถุประสงค์การใช้งาน: "การวิเคราะห์โภชนาการสูตรอาหาร"
4. ตรวจสอบอีเมลเพื่อรับ API key
5. เปิดแอป ไปที่แถบการตั้งค่าด้านข้าง
6. เปิดใช้งาน "USDA API" และใส่ key ของคุณ
7. คลิก "ทดสอบการเชื่อมต่อ"

## 🥇 Nutritionix API (ทางเลือก - 200 requests/วัน ฟรี)

### ประโยชน์:
- ✅ การค้นหาอาหารด้วยภาษาธรรมชาติ
- ✅ ฐานข้อมูลอาหารแบรนด์ที่กว้างขวาง
- ✅ รายการอาหารร้านอาหาร
- ✅ 200 requests ฟรีต่อวัน

### ขั้นตอนการตั้งค่า:
1. เยี่ยมชม: https://www.nutritionix.com/business/api
2. สมัครสมาชิกฟรี
3. เลือก "Free Plan"
4. รับ Application ID และ API Key จากแดชบอร์ด
5. เปิดแอป ไปที่แถบการตั้งค่าด้านข้าง
6. เปิดใช้งาน "Nutritionix API" และใส่ keys ทั้งสอง
7. คลิก "ทดสอบการเชื่อมต่อ"

## 💡 เคล็ดลับ:

- คุณสามารถใช้แอปโดยไม่มี API keys (ใช้ฐานข้อมูลโภชนาการไทยในตัว)
- แนะนำ USDA API สำหรับผลลัพธ์ที่ดีที่สุดและฟรี
- ตรวจสอบแถบการตั้งค่าสำหรับสถานะการเชื่อมต่อ
- จุดเขียว = เชื่อมต่อแล้ว, จุดแดง = ไม่เชื่อมต่อ
EOL

print_success "สร้างคู่มือการตั้งค่า API: API_SETUP_GUIDE.md"

echo ""
echo "============================================================================="
print_success "🎉 การติดตั้งขั้นสูงเสร็จสิ้นเรียบร้อย!"
echo "============================================================================="
echo ""
echo "📋 สิ่งที่ได้ติดตั้ง:"
echo "   ✅ สร้างและเปิดใช้งาน virtual environment"
echo "   ✅ ติดตั้ง dependencies ทั้งหมด (รวมการรองรับ API)"
echo "   ✅ ประมวลผลชุดข้อมูลอาหารไทยพร้อมการวิเคราะห์โภชนาการขั้นสูง"
echo "   ✅ สร้างฐานข้อมูลโภชนาการขั้นสูงพร้อมการปรับแต่งการทำอาหาร"
echo "   ✅ สร้างไฟล์การกำหนดค่า (.env และ Streamlit secrets)"
echo "   ✅ สร้างคู่มือการตั้งค่า API"
echo "   ✅ ทดสอบระบบและทำงานพร้อมฟีเจอร์ขั้นสูง"
echo ""
echo "🚀 ขั้นตอนถัดไป:"
echo "   1. 📖 อ่าน API_SETUP_GUIDE.md สำหรับการกำหนดค่า API (ทางเลือกแต่แนะนำ)"
echo "   2. 🔧 แก้ไขไฟล์ .env เพื่อเพิ่ม API keys ของคุณสำหรับฟีเจอร์ขั้นสูง"
echo "   3. ▶️  รัน: ./run.sh หรือ streamlit run streamlit_app.py"
echo "   4. 🌐 เปิดเบราว์เซอร์ไปที่ URL ที่แสดง (มักจะเป็น http://localhost:8501)"
echo "   5. ⚙️  คลิกปุ่มการตั้งค่าในแถบด้านข้างเพื่อกำหนดค่า APIs"
echo ""
echo "💡 ฟีเจอร์ขั้นสูงที่พร้อมใช้งาน:"
echo "   🔍 การค้นหาอัจฉริยะพร้อมการขยายคำค้นหา"
echo "   📊 การวิเคราะห์โภชนาการขั้นสูงพร้อมการเชื่อมต่อ API"
echo "   🧪 การคำนวณการปรับแต่งการทำอาหาร (การดูดซึมน้ำมัน ฯลฯ)"
echo "   🎯 การตรวจสอบสถานะ API แบบ real-time"
echo "   📱 UI ขั้นสูงพร้อม auto-scroll และการนำทางที่ลื่นไหล"
echo ""
echo "🙏 ขอบคุณที่ใช้แชทบอทสูตรอาหารไทยขั้นสูง! 🍲"
echo "============================================================================="
