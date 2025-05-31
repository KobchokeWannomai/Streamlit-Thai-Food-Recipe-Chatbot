# 🍲 Thai Food Recipe Chatbot with Advanced Nutrition

ระบบแชทบอทสำหรับค้นหาและถามเกี่ยวกับสูตรอาหารไทยพร้อมการคำนวณคุณค่าทางโภชนาการขั้นสูง

## ✨ คุณสมบัติหลัก (Features)

### 🔍 ระบบค้นหาอัจฉริยะ
- **ค้นหาแบบ Fuzzy Matching**: รองรับการพิมพ์ผิดและคำไม่ครบถ้วน
- **ค้นหาตามโภชนาการ**: หาเมนูตามเกณฑ์โภชนาการที่ต้องการ
- **ค้นหาตามประเภทอาหาร**: ทอด, ต้ม, ผัด, ย่าง, ยำ และอื่นๆ
- **ค้นหาตามวัตถุดิบ**: หาเมนูจากวัตถุดิบที่มี

### 🧮 ระบบโภชนาการขั้นสูง
- **คำนวณโภชนาการแม่นยำ**: แคลอรี่, โปรตีน, คาร์โบไฮเดรต, ไขมัน, ใยอาหาร
- **วิตามินและแร่ธาตุ**: วิตามิน A, C, B1, B2, แคลเซียม, เหล็ก, โปแตสเซียม, โซเดียม
- **ปรับสัดส่วนการบริโภค**: คำนวณตามปริมาณที่บริโภคจริง
- **เพิ่มวัตถุดิบที่ขาดหาย**: เสริมข้อมูลวัตถุดิบที่ไม่ได้ระบุแต่ใช้ในการปรุง

### 🌐 รองรับ API ภายนอก
- **USDA FoodData Central**: ข้อมูลโภชนาการมาตรฐานสากล
- **Nutritionix API**: ฐานข้อมูลอาหารครอบคลุม
- **Edamam API**: ข้อมูลโภชนาการและสารก่อภูมิแพ้
- **ฐานข้อมูลท้องถิ่น**: วัตถุดิบไทยที่รวบรวมเฉพาะ

### 📊 การแสดงผลที่สวยงาม
- **กราฟโภชนาการ**: แสดงข้อมูลเป็นกราฟวงกลม แท่ง และเกจ
- **คำแนะนำอัจฉริยะ**: แสดงข้อดีและข้อควรระวังของแต่ละเมนู
- **เปรียบเทียบโภชนาการ**: เปรียบเทียบกับความต้องการประจำวัน
- **รายละเอียดวัตถุดิบ**: แสดงโภชนาการของวัตถุดิบแต่ละชนิด

### 🎯 การแนะนำเฉพาะกลุ่ม
- **ผู้ป่วยเบาหวาน**: เมนูคาร์โบไฮเดรตและโซเดียมต่ำ
- **ผู้ป่วยความดันสูง**: เมนูโซเดียมต่ำและโปแตสเซียมสูง
- **ผู้ลดน้ำหนัก**: เมนูแคลอรี่ต่ำและโปรตีนสูง
- **นักกีฬา**: เมนูโปรตีนสูงและพลังงานเพียงพอ
- **เด็กและผู้สูงอายุ**: เมนูที่เหมาะสมตามวัย

## 🚀 การติดตั้งและใช้งาน

### ข้อกำหนดระบบ
- **Python**: 3.8 หรือใหม่กว่า
- **RAM**: 4GB ขึ้นไป (แนะนำ 8GB)
- **พื้นที่ฮาร์ดดิสก์**: 2GB สำหรับโมเดล AI
- **การเชื่อมต่ออินเทอร์เน็ต**: สำหรับดาวน์โหลดโมเดลและ API

### วิธีติดตั้งแบบง่าย

#### สำหรับ Linux/macOS:
```bash
# โคลนโปรเจค
git clone https://github.com/your-repo/thai-food-chatbot.git
cd thai-food-chatbot

# ติดตั้งอัตโนมัติ
chmod +x setup.sh
./setup.sh

# เริ่มใช้งาน
./run.sh
```

#### สำหรับ Windows:
```cmd
# โคลนโปรเจค
git clone https://github.com/your-repo/thai-food-chatbot.git
cd thai-food-chatbot

# ติดตั้งอัตโนมัติ
setup.bat

# เริ่มใช้งาน
run.bat
```

### วิธีติดตั้งแบบละเอียด

1. **โคลนโปรเจค**:
```bash
git clone https://github.com/your-repo/thai-food-chatbot.git
cd thai-food-chatbot
```

2. **สร้าง Virtual Environment**:
```bash
python -m venv venv

# สำหรับ Linux/macOS
source venv/bin/activate

# สำหรับ Windows
venv\Scripts\activate
```

3. **ติดตั้งแพ็คเกจ**:
```bash
pip install -r requirements.txt
```

4. **เตรียมข้อมูล**:
```bash
# สร้างข้อมูลตัวอย่าง
python enhanced_preprocess.py --sample

# ประมวลผลข้อมูลพร้อมโภชนาการ
python enhanced_preprocess.py --input thai_food_sample_enhanced.csv --output thai_food_processed.csv --nutrition --enhance
```

5. **เริ่มใช้งาน**:
```bash
streamlit run app.py
```

## 📖 วิธีการใช้งาน

### การค้นหาพื้นฐาน
```
ไข่เจียว
ผัดกะเพรา
ต้มยำกุ้ง
```

### การค้นหาตามโภชนาการ
```
แนะนำอาหารแคลอรี่ต่ำ
เมนูโปรตีนสูง
อาหารไขมันต่ำ
เมนูใยอาหารสูง
อาหารแคลเซียมสูง
```

### การค้นหาตามประเภท
```
อาหารทอด
เมนูต้ม
อาหารผัด
เมนูย่าง
อาหารยำ
```

### การค้นหาสำหรับผู้ป่วย
```
อาหารสำหรับเบาหวาน
เมนูความดันสูง
อาหารลดน้ำหนัก
เมนูผู้สูงอายุ
```

## ⚙️ การตั้งค่า API

### USDA FoodData Central API
1. สมัครที่: https://fdc.nal.usda.gov/api-guide.html
2. เพิ่ม API Key ในแถบการตั้งค่าของแอป
3. หรือสร้างไฟล์ `.env`:
```env
USDA_API_KEY=your_api_key_here
```

### Nutritionix API
1. สมัครที่: https://www.nutritionix.com/business/api
2. ตั้งค่า App ID และ App Key ในแอป

### Edamam API
1. สมัครที่: https://developer.edamam.com/
2. เลือก Nutrition Analysis API
3. ตั้งค่า credentials ในแอป

## 🛠️ การประมวลผลข้อมูล

### การประมวลผลพื้นฐาน
```bash
python preprocess.py --input your_data.csv --output processed_data.csv
```

### การประมวลผลขั้นสูง
```bash
# ประมวลผลพร้อมโภชนาการ
python enhanced_preprocess.py --input your_data.csv --output enhanced_data.csv --nutrition

# เพิ่มวัตถุดิบที่ขาดหาย
python enhanced_preprocess.py --input your_data.csv --output enhanced_data.csv --nutrition --enhance

# ใช้ข้อมูลจาก API
python enhanced_preprocess.py --input your_data.csv --output enhanced_data.csv --nutrition --enhance --api
```

### ตัวเลือกการประมวลผล
- `--nutrition`: เพิ่มข้อมูลโภชนาการ
- `--enhance`: เพิ่มวัตถุดิบที่ขาดหาย
- `--api`: ใช้ข้อมูลจาก API ภายนอก
- `--no-adjust`: ไม่ปรับสัดส่วนการบริโภค
- `--sample`: สร้างข้อมูลตัวอย่าง

## 📁 โครงสร้างโปรเจค

```
thai-food-chatbot/
├── app.py                          # แอปพลิเคชัน Streamlit หลัก
├── nutrition_api.py                # ระบบ API โภชนาการขั้นสูง
├── recipe_search.py                # ระบบค้นหาอัจฉริยะ
├── enhanced_preprocess.py          # ประมวลผลข้อมูลขั้นสูง
├── preprocess.py                   # ประมวลผลข้อมูลพื้นฐาน
├── requirements.txt                # รายการแพ็คเกจที่จำเป็น
├── setup.sh / setup.bat           # สคริปต์ติดตั้ง
├── run.sh / run.bat               # สคริปต์เริ่มต้น
├── thai_food_processed.csv        # ข้อมูลสูตรอาหารไทย
├── thai_ingredients_nutrition.csv # ข้อมูลโภชนาการวัตถุดิบ
├── embeddings.pkl                 # (สร้างอัตโนมัติ) Embeddings สำหรับค้นหา
├── model/                         # (สร้างอัตโนมัติ) โมเดล AI
├── logs/                          # ไฟล์ log
├── cache/                         # แคชข้อมูล
├── data/                          # ข้อมูลเสริม
│   └── backup/                    # สำรองข้อมูล
└── .env                           # การตั้งค่า (ไม่อัปโหลด git)
```

## 🧪 การทดสอบและพัฒนา

### การรันการทดสอบ
```bash
# ติดตั้งแพ็คเกจสำหรับทดสอบ
pip install pytest pytest-cov

# รันการทดสอบ
pytest tests/ -v

# ตรวจสอบ code coverage
pytest tests/ --cov=./ --cov-report=html
```

### การแก้ไขปัญหาทั่วไป

#### ปัญหา: โมเดล AI โหลดไม่ได้
```bash
# ลบโฟลเดอร์โมเดลและดาวน์โหลดใหม่
rm -rf model/
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"
```

#### ปัญหา: ข้อมูลโภชนาการไม่ถูกต้อง
```bash
# ลบแคชและสร้างใหม่
rm -f embeddings.pkl
rm -rf cache/
python enhanced_preprocess.py --input thai_food_raw.csv --output thai_food_processed.csv --nutrition --enhance
```

#### ปัญหา: API ไม่ทำงาน
1. ตรวจสอบ API Key ในแถบการตั้งค่า
2. ตรวจสอบการเชื่อมต่ออินเทอร์เน็ต
3. ลองใช้ฐานข้อมูลท้องถิ่นแทน (ปิด API ในการตั้งค่า)

## 📈 การปรับแต่งประสิทธิภาพ

### การเพิ่มความเร็ว
- เปิดใช้งาน GPU สำหรับ sentence transformers
- เพิ่ม RAM สำหรับแคชข้อมูล
- ใช้ SSD สำหรับเก็บโมเดลและข้อมูล

### การประหยัดหน่วยความจำ
- ปิดการแสดงกราฟโภชนาการ
- ลดจำนวนผลลัพธ์การค้นหา
- ใช้โมเดล AI ขนาดเล็กกว่า

## 🤝 การมีส่วนร่วม

เราต้อนรับการมีส่วนร่วมจากทุกคน! วิธีการมีส่วนร่วม:

1. **Fork** โปรเจคนี้
2. สร้าง **feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit** การเปลี่ยนแปลง (`git commit -m 'Add amazing feature'`)
4. **Push** ไปยัง branch (`git push origin feature/amazing-feature`)
5. สร้าง **Pull Request**

### แนวทางการพัฒนา
- ใช้ประโยคคอมเมนต์ภาษาไทย
- เขียนโค้ดที่อ่านง่ายและมี docstring
- เพิ่มการทดสอบสำหรับฟีเจอร์ใหม่
- อัปเดตเอกสารเมื่อมีการเปลี่ยนแปลง

## 📋 TODO List

### ฟีเจอร์ที่วางแผนไว้
- [ ] รองรับภาษาอังกฤษ
- [ ] ระบบแนะนำเมนูประจำวัน
- [ ] การบันทึกเมนูโปรด
- [ ] ระบบแจ้งเตือนสารก่อภูมิแพ้
- [ ] เชื่อมต่อกับ fitness tracker
- [ ] ระบบคำนวณแคลอรี่เผาผลาญ
- [ ] API สำหรับแอปภายนอก
- [ ] รองรับอาหารนานาชาติ

### การปรับปรุง
- [ ] เพิ่มความแม่นยำของ fuzzy search
- [ ] ปรับปรุงการคำนวณโภชนาการ
- [ ] เพิ่มข้อมูลวัตถุดิบไทยเพิ่มเติม
- [ ] ปรับปรุง UI/UX
- [ ] เพิ่มการตรวจสอบข้อมูลอัตโนมัติ

## 📄 ลิขสิทธิ์และใบอนุญาต

โปรเจคนี้อยู่ภายใต้ใบอนุญาต MIT License - ดูรายละเอียดในไฟล์ [LICENSE](LICENSE)

### การใช้ข้อมูล
- ข้อมูลโภชนาการจาก USDA FoodData Central (Public Domain)
- ข้อมูลสูตรอาหารไทยจากแหล่งเปิด
- ข้อมูลวัตถุดิบไทยรวบรวมโดยชุมชน

## 📞 การติดต่อและสนับสนุน

### การรายงานปัญหา
- เปิด **Issue** ใน GitHub
- ระบุรายละเอียดปัญหาและขั้นตอนการทำซ้ำ
- แนบ log files หากมี

### การขอความช่วยเหลือ
- ตรวจสอบ [FAQ](FAQ.md) ก่อน
- ค้นหาใน **Issues** ที่มีอยู่
- สร้าง **Discussion** สำหรับคำถามทั่วไป

### การติดตาม
- ⭐ **Star** โปรเจคหากคุณชอบ
- 🔔 **Watch** เพื่อรับการแจ้งเตือนการอัปเดต
- 🍴 **Fork** เพื่อสร้างเวอร์ชันของคุณเอง

---

## 🙏 กิตติกรรมประกาศ

ขอขอบคุณ:
- **Sentence Transformers** สำหรับโมเดล multilingual
- **Streamlit** สำหรับเฟรมเวิร์กแอปพลิเคชัน
- **USDA FoodData Central** สำหรับข้อมูลโภชนาการ
- **ชุมชนนักพัฒนาไทย** สำหรับข้อมูลและข้อเสนอแนะ

---

**🍲 ขอให้สนุกกับการค้นหาและเรียนรู้เกี่ยวกับอาหารไทย!**

*Made with ❤️ for Thai food lovers*
