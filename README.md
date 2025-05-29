# 🍲 Thai Food Recipe Chatbot with Nutrition Analysis

แชทบอทสูตรอาหารไทยพร้อมระบบวิเคราะห์คุณค่าทางโภชนาการ

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🌟 คุณสมบัติหลัก

- 🔍 **ค้นหาสูตรอาหารไทย** - ค้นหาด้วยชื่อเมนูหรือวัตถุดิบ
- 📊 **วิเคราะห์โภชนาการอัตโนมัติ** - แคลอรี่ โปรตีน คาร์โบไฮเดรต ไขมัน วิตามิน และแร่ธาตุ
- 🎯 **ค้นหาตามเกณฑ์โภชนาการ** - หาเมนูแคลอรี่ต่ำ โปรตีนสูง หรือเหมาะสำหรับลดน้ำหนัก
- 💬 **อินเตอร์เฟซแบบแชท** - ใช้งานง่ายผ่านการสนทนา
- 📱 **รองรับทุกอุปกรณ์** - ใช้งานได้ทั้งคอมพิวเตอร์และมือถือ

## 🚀 Quick Start (เริ่มต้นใน 3 นาที!)

### 1️. ติดตั้งระบบ (ครั้งแรกเท่านั้น)

#### Windows:
```batch
setup.bat
```

#### macOS/Linux:
```bash
chmod +x setup.sh
./setup.sh
```

### 2️. รันแอปพลิเคชัน

#### Windows:
```batch
run.bat
```

#### macOS/Linux:
```bash
./run.sh
```

### 3️. เปิดเบราว์เซอร์
- ไปที่ http://localhost:8501
- เริ่มถามเกี่ยวกับอาหารไทยได้เลย!

## 💡 ตัวอย่างการใช้งาน

### ค้นหาอาหารทั่วไป:
- "ผัดกะเพรา"
- "วิธีทำต้มยำกุ้ง"
- "ส้มตำ"
- "แกงเขียวหวาน"

### ค้นหาตามโภชนาการ:
- "เมนูแคลอรี่ไม่เกิน 300"
- "อาหารโปรตีนสูงมากกว่า 20 กรัม"
- "เมนูลดน้ำหนัก"
- "อาหารเฮลธ์ตี้"

## 📋 ความต้องการระบบ

- Python 3.8 หรือสูงกว่า
- RAM อย่างน้อย 4GB
- พื้นที่ว่างอย่างน้อย 2GB

## 💻 การติดตั้งแบบ Manual

### 1. Clone หรือ Download โปรเจค
```bash
git clone https://github.com/your-repo/thai-food-chatbot.git
cd thai-food-chatbot
```

### 2. สร้าง Virtual Environment
```bash
python -m venv venv
```

### 3. เปิดใช้งาน Virtual Environment
- Windows: `venv\Scripts\activate`
- macOS/Linux: `source venv/bin/activate`

### 4. ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```

### 5. รันแอปพลิเคชัน
```bash
streamlit run streamlit_app.py
```

## 🔑 การตั้งค่า API สำหรับข้อมูลโภชนาการ (ทางเลือก)

ระบบสามารถทำงานได้โดยไม่ต้องมี API keys แต่การมี API keys จะทำให้ข้อมูลโภชนาการแม่นยำและครบถ้วนมากขึ้น

### 🆓 FDC API (แนะนำ - ฟรี)

#### ข้อดี
- ✅ ฟรีไม่มีค่าใช้จ่าย
- ✅ ข้อมูลจากรัฐบาลสหรัฐอเมริกา (เชื่อถือได้)
- ✅ ข้อมูลโภชนาการครบถ้วน
- ✅ ไม่จำกัด Rate limit

#### วิธีการสมัคร

1. **เข้าไปที่เว็บไซต์**
   ```
   https://fdc.nal.usda.gov/api-guide.html
   ```

2. **คลิก "Get an API Key"**

3. **กรอกข้อมูลลงทะเบียน**
   - ชื่อ-นามสกุล
   - อีเมล
   - องค์กร (ใส่ว่า "Personal Use" ได้)
   - วัตถุประสงค์ (ใส่ว่า "Recipe Nutrition Analysis")

4. **รับ API Key ทางอีเมล**

5. **เพิ่มใน .env file**
   ```env
   USDA_API_KEY=your_api_key_here
   ```

### 🥇 Nutritionix API (ทางเลือก)

#### ข้อดี
- ✅ ข้อมูลอาหารหลากหลาย
- ✅ รองรับการค้นหาด้วยประโยค
- ✅ แพ็คเกจฟรีมี 200 requests/วัน

#### วิธีการสมัคร

1. **เข้าไปที่เว็บไซต์**
   ```
   https://www.nutritionix.com/business/api
   ```

2. **Sign Up และเลือก "Free Plan"**

3. **กรอกข้อมูลและรับ**
   - Application ID
   - Application Key

4. **เพิ่มใน .env file**
   ```env
   NUTRITIONIX_APP_ID=your_app_id_here
   NUTRITIONIX_APP_KEY=your_app_key_here
   ```

### 🔧 ตัวอย่างไฟล์ .env

```env
# Copy from .env.example
USDA_API_KEY=your_usda_api_key_here
NUTRITIONIX_API_KEY=your_nutritionix_api_key_here
NUTRITIONIX_APP_ID=your_nutritionix_app_id_here
```

## 📁 โครงสร้างโปรเจค

```
thai-food-chatbot/
├── streamlit_app.py              # แอปพลิเคชันหลัก
├── nutrition_analyzer.py         # ระบบวิเคราะห์โภชนาการ
├── ingredient_converter.py       # ระบบแปลงหน่วยวัตถุดิบ
├── config.py                    # การตั้งค่าระบบ
├── preprocess.py                # ประมวลผลข้อมูล
├── batch_nutrition_processor.py  # ประมวลผลแบบ batch
├── nutrition_example.py         # ตัวอย่างการใช้งาน
├── requirements.txt             # รายการ dependencies
├── setup.sh                     # สคริปต์ติดตั้ง (macOS/Linux)
├── setup.bat                    # สคริปต์ติดตั้ง (Windows)
├── run.sh                       # สคริปต์รันแอป (macOS/Linux)
├── run.bat                      # สคริปต์รันแอป (Windows)
├── thai_food_sample.csv         # ข้อมูลตัวอย่าง
├── .env.example                 # ตัวอย่างการตั้งค่า
├── .gitignore                   # ไฟล์ที่ git ignore
└── LICENSE                      # Apache License 2.0
```

## 🧪 ตัวอย่างการใช้งานขั้นสูง

### วิเคราะห์โภชนาการแบบ Programmatic

```python
from nutrition_analyzer import NutritionAnalyzer

# สร้าง analyzer
analyzer = NutritionAnalyzer()

# วิเคราะห์วัตถุดิบ
ingredients = """
- กุ้ง 200 กรัม
- น้ำมัน 2 ช้อนโต๊ะ
- กระเทียม 3 กลีบ
"""

nutrition_data = analyzer.analyze_ingredients(ingredients)
total = analyzer.calculate_total_nutrition(nutrition_data)

print(f"แคลอรี่รวม: {total.calories:.0f} kcal")
print(f"โปรตีน: {total.protein:.1f} g")
```

### ประมวลผลข้อมูลแบบ Batch

```bash
# วิเคราะห์โภชนาการสำหรับหลายสูตร
python batch_nutrition_processor.py --input recipes.csv --batch-size 10

# สร้างฐานข้อมูลโภชนาการ
python preprocess.py --create-nutrition-db
```

## 🔧 การแก้ปัญหาที่พบบ่อย

### ปัญหา: ModuleNotFoundError
```bash
# ตรวจสอบว่า activate virtual environment แล้ว
# แล้วรัน
pip install -r requirements.txt
```

### ปัญหา: ข้อมูลโภชนาการไม่แสดง
```bash
# ตรวจสอบไฟล์ข้อมูล
python preprocess.py --input thai_food_sample.csv --output thai_food_processed.csv --analyze-nutrition
```

### ปัญหา: Port 8501 ถูกใช้งานแล้ว
```bash
streamlit run streamlit_app.py --server.port 8502
```

### ปัญหา: API ไม่ทำงาน
- ตรวจสอบ API keys ในไฟล์ `.env`
- ตรวจสอบการเชื่อมต่ออินเทอร์เน็ต
- ดู log ที่ `nutrition_processing.log`

## 📊 ข้อมูลที่ใช้

- **ฐานข้อมูลสูตรอาหารไทย** - รองรับมากกว่า 100 เมนู
- **ข้อมูลโภชนาการวัตถุดิบไทย** - มากกว่า 150 ชนิด
- **API ข้อมูลโภชนาการ** - USDA FoodData Central และ Nutritionix

## 🤝 การมีส่วนร่วม

ยินดีรับ Pull Requests! กรุณา:

1. Fork โปรเจค
2. สร้าง feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit การเปลี่ยนแปลง (`git commit -m 'Add AmazingFeature'`)
4. Push ไปยัง branch (`git push origin feature/AmazingFeature`)
5. เปิด Pull Request

## 📝 TODO List

- [ ] เพิ่ม Unit Tests
- [ ] รองรับภาษาอังกฤษ
- [ ] Export เป็น PDF
- [ ] Dark Mode
- [ ] Voice Input
- [ ] Meal Planning Feature
- [ ] Integration กับ Fitness Apps

## 📜 License

โปรเจคนี้เผยแพร่ภายใต้ [Apache License 2.0](LICENSE)

## 👏 กิตติกรรมประกาศ

- ข้อมูลสูตรอาหารไทยจากชุมชน
- [USDA FoodData Central](https://fdc.nal.usda.gov/) สำหรับข้อมูลโภชนาการ
- [Streamlit](https://streamlit.io/) สำหรับ framework
- [Sentence Transformers](https://www.sbert.net/) สำหรับ semantic search

## 📧 ติดต่อ

หากมีคำถามหรือข้อเสนอแนะ:
- เปิด Issue ใน GitHub
- Email: your-email@example.com

---

<p align="center">
  Made with ❤️ for Thai food lovers
  <br>
  <strong>🍲 ขอให้มีความสุขกับการทำอาหารไทย!</strong>
</p>
