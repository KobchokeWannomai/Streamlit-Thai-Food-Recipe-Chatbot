# 🍲 Thai Food Recipe Chatbot with Nutrition Analysis

แชทบอทสูตรอาหารไทยพร้อมระบบวิเคราะห์คุณค่าทางโภชนาการ

## 🌟 คุณสมบัติ

- 🔍 **ค้นหาสูตรอาหารไทย** - ค้นหาด้วยชื่อเมนูหรือวัตถุดิบ
- 📊 **วิเคราะห์โภชนาการ** - ดูข้อมูลแคลอรี่ โปรตีน คาร์โบไฮเดรต ไขมัน วิตามิน และแร่ธาตุ
- 🎯 **ค้นหาตามเกณฑ์โภชนาการ** - หาเมนูแคลอรี่ต่ำ โปรตีนสูง หรือเหมาะสำหรับลดน้ำหนัก
- 💬 **อินเตอร์เฟซแบบแชท** - ใช้งานง่ายผ่านการสนทนา
- 📱 **รองรับทุกอุปกรณ์** - ใช้งานได้ทั้งคอมพิวเตอร์และมือถือ

## 📋 ความต้องการระบบ

- Python 3.8 หรือสูงกว่า
- RAM อย่างน้อย 4GB
- พื้นที่ว่างอย่างน้อย 2GB

## 🚀 การติดตั้ง

### Windows

1. โคลนหรือดาวน์โหลดโปรเจค
```bash
git clone https://github.com/your-repo/thai-food-chatbot.git
cd thai-food-chatbot
```

2. รันสคริปต์ติดตั้ง
```batch
setup.bat
```

3. รันแอปพลิเคชัน
```batch
run.bat
```

### macOS/Linux

1. โคลนหรือดาวน์โหลดโปรเจค
```bash
git clone https://github.com/your-repo/thai-food-chatbot.git
cd thai-food-chatbot
```

2. รันสคริปต์ติดตั้ง
```bash
chmod +x setup.sh
./setup.sh
```

3. รันแอปพลิเคชัน
```bash
./run.sh
```

### การติดตั้งแบบ Manual

1. สร้าง virtual environment
```bash
python -m venv venv
```

2. เปิดใช้งาน virtual environment
- Windows: `venv\Scripts\activate`
- macOS/Linux: `source venv/bin/activate`

3. ติดตั้ง dependencies
```bash
pip install -r requirements.txt
```

4. รันแอปพลิเคชัน
```bash
streamlit run streamlit_app.py
```

## 🔑 การตั้งค่า API Keys (ทางเลือก)

สำหรับข้อมูลโภชนาการที่แม่นยำขึ้น คุณสามารถเพิ่ม API keys:

1. สร้างไฟล์ `.env` ในโฟลเดอร์โปรเจค
2. เพิ่ม API keys:
```env
# USDA FoodData Central API
USDA_API_KEY=your_usda_api_key_here

# Nutritionix API
NUTRITIONIX_API_KEY=your_nutritionix_api_key_here
NUTRITIONIX_APP_ID=your_nutritionix_app_id_here
```

### วิธีขอ API Keys:
- **USDA API**: https://fdc.nal.usda.gov/api-guide.html
- **Nutritionix API**: https://www.nutritionix.com/business/api

## 💻 การใช้งาน

### การค้นหาทั่วไป
- พิมพ์ชื่อเมนู: "ต้มยำกุ้ง", "ผัดไทย", "แกงเขียวหวาน"
- พิมพ์วัตถุดิบ: "เมนูที่มีกุ้ง", "อาหารที่ใส่กะทิ"

### การค้นหาตามโภชนาการ
- "เมนูแคลอรี่ไม่เกิน 300"
- "อาหารโปรตีนสูงมากกว่า 20 กรัม"
- "เมนูลดน้ำหนัก"
- "อาหารเฮลธ์ตี้"

## 📁 โครงสร้างโปรเจค

```
thai-food-chatbot/
├── streamlit_app.py        # แอปพลิเคชันหลัก
├── nutrition_analyzer.py   # ระบบวิเคราะห์โภชนาการ
├── config.py              # การตั้งค่าระบบ
├── preprocess.py          # ประมวลผลข้อมูล
├── batch_nutrition_processor.py  # ประมวลผลแบบ batch
├── requirements.txt       # รายการ dependencies
├── setup.sh              # สคริปต์ติดตั้ง (macOS/Linux)
├── setup.bat             # สคริปต์ติดตั้ง (Windows)
├── run.sh                # สคริปต์รันแอป (macOS/Linux)
├── run.bat               # สคริปต์รันแอป (Windows)
└── thai_food_processed.csv # ข้อมูลสูตรอาหารไทย
```

## 🧪 ตัวอย่างการใช้งาน

รันตัวอย่าง:
```bash
python nutrition_example.py
```

ประมวลผลโภชนาการแบบ batch:
```bash
python batch_nutrition_processor.py --input thai_food_processed.csv
```

## 🔧 การแก้ปัญหา

### ปัญหา: ModuleNotFoundError
- ตรวจสอบว่า activate virtual environment แล้ว
- รัน `pip install -r requirements.txt` อีกครั้ง

### ปัญหา: ข้อมูลโภชนาการไม่แสดง
- ตรวจสอบไฟล์ `thai_food_processed.csv`
- รัน `python preprocess.py --analyze-nutrition`

### ปัญหา: API ไม่ทำงาน
- ตรวจสอบ API keys ในไฟล์ `.env`
- ตรวจสอบการเชื่อมต่ออินเทอร์เน็ต

## 📊 ข้อมูลที่ใช้

- ฐานข้อมูลสูตรอาหารไทยกว่า 100 เมนู
- ข้อมูลโภชนาการวัตถุดิบไทยพื้นฐาน
- API ข้อมูลโภชนาการจาก USDA และ Nutritionix

## 🤝 การมีส่วนร่วม

ยินดีรับ Pull Requests! กรุณา:
1. Fork โปรเจค
2. สร้าง feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit การเปลี่ยนแปลง (`git commit -m 'Add some AmazingFeature'`)
4. Push ไปยัง branch (`git push origin feature/AmazingFeature`)
5. เปิด Pull Request

## 📝 License

โปรเจคนี้เผยแพร่ภายใต้ MIT License

## 👏 กิตติกรรมประกาศ

- ข้อมูลสูตรอาหารไทยจากแหล่งต่างๆ
- USDA FoodData Central สำหรับข้อมูลโภชนาการ
- Streamlit สำหรับ framework การสร้าง web app

## 📧 ติดต่อ

หากมีคำถามหรือข้อเสนอแนะ กรุณาเปิด Issue ใน GitHub

---

Made with ❤️ for Thai food lovers
