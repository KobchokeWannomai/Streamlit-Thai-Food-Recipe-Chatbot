# Thai Food Recipe Chatbot with Nutrition Analysis

ระบบแชทบอทสำหรับค้นหาและถามเกี่ยวกับสูตรอาหารไทย พร้อมข้อมูลคุณค่าทางโภชนาการ

## 🆕 ฟีเจอร์ใหม่ - Nutrition Analysis

### คุณสมบัติที่เพิ่มขึ้น
- 🥗 **วิเคราะห์คุณค่าทางโภชนาการ** - แสดงแคลอรี่, โปรตีน, คาร์โบไฮเดรต, ไขมัน, ใยอาหาร
- 🧪 **วิตามินและแร่ธาตุ** - ข้อมูลรายละเอียดของวิตามินและแร่ธาตุในแต่ละสูตร
- 🔍 **ค้นหาตามโภชนาการ** - ค้นหาสูตรอาหารตามเกณฑ์โภชนาการที่ต้องการ
- 🤖 **การวิเคราะห์อัตโนมัติ** - ระบบวิเคราะห์วัตถุดิบใหม่โดยอัตโนมัติ
- 💾 **ฐานข้อมูลโภชนาการ** - เก็บข้อมูลโภชนาการในฐานข้อมูลเพื่อการค้นหาที่รวดเร็ว

## 📊 แหล่งข้อมูลโภชนาการ

ระบบดึงข้อมูลจากแหล่งที่เชื่อถือได้:
1. **USDA Food Database (FDC API)** - ฐานข้อมูลอาหารของรัฐบาลสหรัฐอเมริกา
2. **Nutritionix API** - ฐานข้อมูลโภชนาการที่ครอบคลุม
3. **Fallback Database** - ข้อมูลพื้นฐานสำหรับวัตถุดิบไทยทั่วไป

## 🚀 การติดตั้งและใช้งาน

### 1. ติดตั้ง Dependencies

```bash
pip install -r requirements_enhanced.txt
```

### 2. ตั้งค่า API Keys (ทำได้ภายหลัง)

สร้างไฟล์ `.env` และเพิ่ม API keys:

```env
# FDC API (ฟรี) - https://fdc.nal.usda.gov/api-guide.html
FDC_API_KEY=your_fdc_api_key_here

# Nutritionix API (ฟรี) - https://www.nutritionix.com/business/api
NUTRITIONIX_APP_ID=your_app_id_here
NUTRITIONIX_APP_KEY=your_app_key_here
```

### 3. ประมวลผลข้อมูลโภชนาการ (ครั้งแรก)

```bash
# ประมวลผลทุกสูตรอาหาร
python batch_nutrition_process.py

# หรือประมวลผลแบบ batch ขนาดเล็ก
python batch_nutrition_process.py --batch-size 3

# บังคับประมวลผลใหม่ทั้งหมด
python batch_nutrition_process.py --force

# สร้างเฉพาะสรุปข้อมูล
python batch_nutrition_process.py --summary-only
```

### 4. เรียกใช้แอป

```bash
# เรียกใช้แอปหลักที่มีฟีเจอร์โภชนาการ
streamlit run app_enhanced.py

# หรือใช้แอปเดิม (ไม่มีฟีเจอร์โภชนาการ)
streamlit run app.py
```

## 📁 โครงสร้างไฟล์ใหม่

```
thai-food-chatbot/
├── app.py                          # แอปเดิม
├── app_enhanced.py                 # แอปใหม่ที่มีฟีเจอร์โภชนาการ
├── nutrition_analyzer.py           # ระบบวิเคราะห์โภชนาการ
├── batch_nutrition_process.py      # ประมวลผลข้อมูลแบบ batch
├── thai_food_processed.csv         # ข้อมูลสูตรอาหารเดิม
├── thai_food_with_nutrition.csv    # ข้อมูลที่เพิ่มโภชนาการแล้ว (สร้างอัตโนมัติ)
├── nutrition.db                    # ฐานข้อมูลโภชนาการ (สร้างอัตโนมัติ)
├── nutrition_results.json          # ผลลัพธ์การวิเคราะห์ (สร้างอัตโนมัติ)
├── nutrition_summary.json          # สรุปข้อมูลโภชนาการ (สร้างอัตโนมัติ)
├── requirements.txt                # dependencies เดิม
├── requirements_enhanced.txt       # dependencies ใหม่
└── .env                           # API keys (สร้างเอง)
```

## 🔧 การใช้งานขั้นสูง

### การเพิ่มวัตถุดิบใหม่

```python
from nutrition_analyzer import NutritionAnalyzer

analyzer = NutritionAnalyzer()

# วิเคราะห์วัตถุดิบใหม่
nutrition = analyzer.analyze_ingredient("มะม่วงดิบ")
print(f"แคลอรี่: {nutrition.calories} kcal")

# วิเคราะห์สูตรอาหารใหม่
result = analyzer.analyze_recipe("ต้มยำกุ้ง", """
- กุ้งนาง 5 ตัว
- น้ำปลา 2 ช้อนโต๊ะ
- พริกขี้หนู 3 เม็ด
""")
```

### การค้นหาตามโภชนาการ

```python
# ค้นหาสูตรที่มีแคลอรี่ต่ำ
criteria = {
    'max_calories': 300,
    'min_protein': 15
}
results = analyzer.search_recipes_by_nutrition(criteria)
```

## 🎯 ตัวอย่างการใช้งาน

### 1. ค้นหาสูตรอาหารทั่วไป
```
ผู้ใช้: "สูตรต้มยำกุ้ง"
ระบบ: [แสดงสูตร + ข้อมูลโภชนาการ]
- แคลอรี่: 245 kcal
- โปรตีน: 28.5 g
- คาร์โบไฮเดรต: 12.3 g
- ไขมัน: 8.7 g
```

### 2. ค้นหาตามเกณฑ์โภชนาการ
- ค้นหาเมนูที่มีแคลอรี่ต่ำกว่า 400
- ค้นหาเมนูที่มีโปรตีนสูงกว่า 20g
- ค้นหาเมนูสำหรับคนลดน้ำหนัก

### 3. วิเคราะห์วัตถุดิบรายตัว
```
ผู้ใช้: "กุ้งมีคุณค่าทางโภชนาการอย่างไร"
ระบบ: [แสดงข้อมูลโภชนาการของกุ้ง]
```

## ⚙️ การปรับแต่ง

### เพิ่มการแปลวัตถุดิบ
แก้ไขใน `nutrition_analyzer.py`:

```python
self.thai_to_english = {
    "มะม่วง": "mango",
    "ส้มตำ": "papaya salad",
    # เพิ่มวัตถุดิบใหม่ที่นี่
}
```

### เพิ่มข้อมูล Fallback
แก้ไข method `get_fallback_nutrition()`:

```python
fallback_data = {
    "วัตถุดิบใหม่": NutritionInfo("วัตถุดิบใหม่", calories, protein, carbs, fat, fiber),
    # เพิ่มข้อมูลใหม่ที่นี่
}
```

## 🐛 การแก้ไขปัญหา

### ปัญหาที่พบบ่อย

1. **API Rate Limit**
   - ลดขนาด batch: `--batch-size 2`
   - เพิ่มระยะเวลาหน่วง: แก้ไข `time.sleep()` ใน batch processor

2. **ข้อมูลโภชนาการไม่ถูกต้อง**
   - ตรวจสอบการแปลวัตถุดิบใน `thai_to_english`
   - เพิ่มข้อมูล fallback ที่แม่นยำขึ้น

3. **ไม่มี API Key**
   - ระบบจะใช้ข้อมูล fallback อัตโนมัติ
   - สมัคร API key ฟรีเพื่อความแม่นยำสูงขึ้น

### Log Files
- `nutrition_processing.log` - บันทึกการประมวลผล
- ตรวจสอบ error ในไฟล์ log

## 📈 สถิติและการติดตาม

ระบบจะสร้างไฟล์สรุป:
- `nutrition_summary.json` - สถิติโภชนาการโดยรวม
- `thai_food_with_nutrition.csv` - ข้อมูลครบถ้วนพร้อมโภชนาการ

## 🤝 การพัฒนาต่อ

### แนวทางการปรับปรุง
1. **เพิ่มภาษาอื่น** - รองรับการค้นหาภาษาอังกฤษ
2. **ระบบแนะนำ** - แนะนำเมนูตามเกณฑ์โภชนาการ
3. **การแชร์** - ส่งออกข้อมูลโภชนาการเป็น PDF
4. **กราฟและชาร์ต** - แสดงข้อมูลในรูปแบบภาพ

### การพัฒนาร่วมกัน
1. Fork repository
2. สร้าง feature branch
3. เพิ่มฟีเจอร์ใหม่
4. ส่ง Pull Request

## 📞 การสนับสนุน

หากพบปัญหาหรือต้องการความช่วยเหลือ:
1. ตรวจสอบ Issues ใน repository
2. สร้าง Issue ใหม่พร้อมรายละเอียด error
3. แนบไฟล์ log ประกอบ

## 📄 License

MIT License - ใช้งานได้อย่างอิสระเพื่อการศึกษาและพัฒนา

---

**หมายเหตุ:** ข้อมูลโภชนาการที่แสดงเป็นการประมาณและควรใช้เป็นข้อมูลอ้างอิงเท่านั้น สำหรับความแม่นยำสูงสุด ควรปรึกษานักโภชนาการหรือแพทย์
