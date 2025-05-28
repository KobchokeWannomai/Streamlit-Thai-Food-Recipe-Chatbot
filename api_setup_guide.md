# 🔑 คู่มือการตั้งค่า API สำหรับข้อมูลโภชนาการ

## 🎯 ภาพรวม

ระบบสามารถทำงานได้โดยไม่ต้องมี API keys แต่การมี API keys จะทำให้ข้อมูลโภชนาการแม่นยำและครบถ้วนมากขึ้น

## 🆓 FDC API (แนะนำ - ฟรี)

### ข้อดี
- ✅ ฟรีไม่มีค่าใช้จ่าย
- ✅ ข้อมูลจากรัฐบาลสหรัฐอเมริกา (เชื่อถือได้)
- ✅ ข้อมูลโภชนาการครบถ้วน
- ✅ Rate limit สูง (ไม่ระบุขีดจำกัด)

### วิธีการสมัคร

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
   FDC_API_KEY=your_fdc_api_key_here
   ```

### ตัวอย่างการใช้งาน
```python
# ระบบจะใช้ API key นี้อัตโนมัติ
analyzer = NutritionAnalyzer()
nutrition = analyzer.analyze_ingredient("กุ้ง")
```

## 🥇 Nutritionix API (ทางเลือก - ฟรี + Premium)

### ข้อดี
- ✅ ข้อมูลอาหารหลากหลาย
- ✅ รองรับการค้นหาด้วยประโยค
- ✅ แพ็คเกจฟรีมี 200 requests/วัน

### ข้อจำกัด
- ❌ Rate limit ต่ำสำหรับแพ็คเกจฟรี
- ❌ ต้องสมัครบัญชี

### วิธีการสมัคร

1. **เข้าไปที่เว็บไซต์**
   ```
   https://www.nutritionix.com/business/api
   ```

2. **คลิก "Sign Up" และเลือก "Free Plan"**

3. **กรอกข้อมูลลงทะเบียน**

4. **ไปที่ Dashboard และรับ**
   - Application ID
   - Application Key

5. **เพิ่มใน .env file**
   ```env
   NUTRITIONIX_APP_ID=your_app_id_here
   NUTRITIONIX_APP_KEY=your_app_key_here
   ```

## 🔧 การตั้งค่าไฟล์ .env

สร้างไฟล์ `.env` ในโฟลเดอร์ root ของโปรเจค:

```env
# FDC API (แนะนำ)
FDC_API_KEY=DEMO_KEY

# Nutritionix API (ทางเลือก)
NUTRITIONIX_APP_ID=your_app_id_here
NUTRITIONIX_APP_KEY=your_app_key_here

# การตั้งค่าอื่นๆ
LOG_LEVEL=INFO
BATCH_SIZE=5
API_TIMEOUT=10
```

## 🔄 ลำดับการใช้ API

ระบบจะพยายามใช้ API ตามลำดับนี้:

1. **FDC API** (ถ้ามี API key)
2. **Nutritionix API** (ถ้า FDC ล้มเหลว)
3. **Fallback Database** (ถ้าทุก API ล้มเหลว)

## 📊 การเปรียบเทียบ API

| คุณสมบัติ | FDC API | Nutritionix API | Fallback |
|-----------|---------|-----------------|----------|
| ราคา | ฟรี | ฟรี 200/วัน | ฟรี |
| ความแม่นยำ | สูงมาก | สูง | ปานกลาง |
| ข้อมูลไทย | น้อย | ปานกลาง | มาก |
| วิตามิน/แร่ธาตุ | ครบ | ครบ | พื้นฐาน |
| Rate Limit | ไม่จำกัด | 200/วัน | ไม่จำกัด |

## 🧪 การทดสอบ API

### ทดสอบการเชื่อมต่อ

```python
from nutrition_analyzer import NutritionAnalyzer

# สร้าง analyzer
analyzer = NutritionAnalyzer()

# ทดสอบ FDC API
try:
    nutrition = analyzer.get_nutrition_from_api("chicken")
    if nutrition:
        print("✅ FDC API ทำงานได้")
        print(f"Calories: {nutrition.calories}")
    else:
        print("❌ FDC API ไม่ทำงาน")
except Exception as e:
    print(f"❌ FDC API Error: {e}")

# ทดสอบ Nutritionix API
try:
    nutrition = analyzer._get_nutrition_from_nutritionix("chicken")
    if nutrition:
        print("✅ Nutritionix API ทำงานได้")
    else:
        print("❌ Nutritionix API ไม่ทำงาน")
except Exception as e:
    print(f"❌ Nutritionix API Error: {e}")
```

### ทดสอบด้วย Command Line

```bash
# ทดสอบการประมวลผลจริง
python -c "
from nutrition_analyzer import NutritionAnalyzer
analyzer = NutritionAnalyzer()
result = analyzer.analyze_ingredient('กุ้ง')
print(f'กุ้ง: {result.calories} kcal, {result.protein}g protein')
"
```

## ⚠️ ข้อควรระวัง

### Rate Limiting
```python
import time

# เพิ่มการหน่วงเวลาระหว่างการเรียก API
def process_with_delay():
    for ingredient in ingredients:
        nutrition = analyzer.analyze_ingredient(ingredient)
        time.sleep(1)  # หน่วง 1 วินาที
```

### Error Handling
```python
def safe_api_call(ingredient):
    try:
        return analyzer.get_nutrition_from_api(ingredient)
    except requests.exceptions.Timeout:
        print(f"Timeout for {ingredient}")
        return analyzer.get_fallback_nutrition(ingredient)
    except requests.exceptions.RequestException as e:
        print(f"API Error for {ingredient}: {e}")
        return analyzer.get_fallback_nutrition(ingredient)
```

## 🔍 การ Debug API

### เปิด Debug Mode

```python
import logging

# เปิด debug logging
logging.basicConfig(level=logging.DEBUG)

# ดู request details
import requests
import http.client
http.client.HTTPConnection.debuglevel = 1
```

### ตรวจสอบ Response

```python
def debug_api_response(ingredient):
    url = f"https://api.nal.usda.gov/fdc/v1/foods/search"
    params = {
        "query": ingredient,
        "api_key": "YOUR_API_KEY",
        "pageSize": 1
    }
    
    response = requests.get(url, params=params)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text[:500]}...")
    
    return response.json()
```

## 🚀 การเพิ่มประสิทธิภาพ

### Caching
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_api_call(ingredient):
    return analyzer.get_nutrition_from_api(ingredient)
```

### Batch Processing
```python
# ประมวลผลทีละ batch เพื่อไม่ให้ถูก rate limit
batch_size = 5
for i in range(0, len(ingredients), batch_size):
    batch = ingredients[i:i+batch_size]
    for ingredient in batch:
        process_ingredient(ingredient)
    time.sleep(2)  # หน่วงระหว่าง batch
```

## 💡 เทคนิคการใช้งาน

### การแปลชื่อวัตถุดิบ
```python
# เพิ่มคำแปลในพจนานุกรม
thai_to_english = {
    "น้ำปลา": "fish sauce",
    "น้ำมันหอย": "oyster sauce",
    "พริกแกง": "curry paste",
    # เพิ่มเติมได้
}
```

### การปรับแต่งความแม่นยำ
```python
def improve_search_accuracy(thai_ingredient):
    # ลบคำที่ไม่จำเป็นออก
    cleaned = thai_ingredient.replace("สด", "").replace("แห้ง", "")
    
    # เพิ่มคำอธิบาย
    if "ปลา" in cleaned:
        return f"fish {cleaned}"
    elif "เนื้อ" in cleaned:
        return f"meat {cleaned}"
    
    return cleaned
```

## ❓ คำถามที่พบบ่อย

### Q: ไม่มี API key ใช้งานได้ไหม?
A: ได้ครับ ระบบจะใช้ข้อมูล fallback แต่อาจไม่แม่นยำเท่า

### Q: API key หมดอายุไหม?
A: FDC API ไม่หมดอายุ, Nutritionix API ขึ้นกับแพ็คเกจ

### Q: ข้อมูลไม่ถูกต้องทำไง?
A: 1) ตรวจสอบการแปลชื่อวัตถุดิบ 2) เพิ่มข้อมูล fallback 3) รายงาน issue

### Q: Rate limit เกินทำไง?
A: 1) ลด batch size 2) เพิ่มเวลาหน่วง 3) ใช้ API อื่น

---

🎉 **เมื่อตั้งค่าเสร็จแล้ว ระบบจะทำงานอัตโนมัติและให้ข้อมูลโภชนาการที่แม่นยำยิ่งขึ้น!**
