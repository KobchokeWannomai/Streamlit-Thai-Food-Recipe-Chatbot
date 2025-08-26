import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import pickle
import re
import requests
import json
import csv
from io import StringIO
import time
from difflib import SequenceMatcher

# การตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="ฐานข้อมูลอาหารไทยและคุณค่าทางโภชนาการ",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ตั้งค่าฟอนต์ภาษาไทย
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@400;700&display=swap');
    html, body, [class*="st-"] {
        font-family: 'Sarabun', sans-serif !important;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .nutrition-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .nutrient-row {
        display: flex;
        justify-content: space-between;
        margin: 5px 0;
        padding: 3px 0;
        border-bottom: 1px solid #f0f0f0;
    }
    .nutrient-name {
        font-weight: bold;
        color: #333;
    }
    .nutrient-value {
        color: #666;
    }
    .ingredient-item {
        background-color: #f8f9fa;
        padding: 8px 12px;
        margin: 5px 0;
        border-radius: 5px;
        border-left: 4px solid #4CAF50;
    }
    .cooking-step {
        background-color: #fff3cd;
        padding: 10px;
        margin: 8px 0;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
    }
    .high-nutrient { color: #28a745; font-weight: bold; }
    .medium-nutrient { color: #ffc107; font-weight: bold; }
    .low-nutrient { color: #6c757d; }
</style>
""", unsafe_allow_html=True)

# เส้นทางไฟล์ต่างๆ
DATA_PATH = "thai_food_processed.csv"
EMBEDDINGS_PATH = "embeddings.pkl"
MODEL_PATH = "model"
NUTRITION_DATA_PATH = "nutrition_data.csv"

# ข้อมูลโภชนาการพื้นฐานของวัตถุดิบทั่วไป (per 100g)
BASIC_NUTRITION_DATA = {
    # เนื้อสัตว์และโปรตีน
    "ไก่": {"calories": 165, "protein": 31, "fat": 3.6, "carbs": 0, "fiber": 0, "calcium": 15, "iron": 1.3, "vitamin_c": 0, "vitamin_a": 10},
    "หมู": {"calories": 242, "protein": 27, "fat": 14, "carbs": 0, "fiber": 0, "calcium": 19, "iron": 1.4, "vitamin_c": 0, "vitamin_a": 2},
    "เนื้อ": {"calories": 250, "protein": 26, "fat": 15, "carbs": 0, "fiber": 0, "calcium": 18, "iron": 2.6, "vitamin_c": 0, "vitamin_a": 0},
    "กุ้ง": {"calories": 106, "protein": 20, "fat": 1.7, "carbs": 1, "fiber": 0, "calcium": 70, "iron": 3.0, "vitamin_c": 0, "vitamin_a": 54},
    "ปลา": {"calories": 150, "protein": 22, "fat": 6, "carbs": 0, "fiber": 0, "calcium": 30, "iron": 1.0, "vitamin_c": 0, "vitamin_a": 15},
    "ไข่": {"calories": 155, "protein": 13, "fat": 11, "carbs": 1.1, "fiber": 0, "calcium": 56, "iron": 1.8, "vitamin_c": 0, "vitamin_a": 160},
    
    # ผักใบเขียว
    "ผักบุ้ง": {"calories": 19, "protein": 2.6, "fat": 0.2, "carbs": 3.1, "fiber": 2.1, "calcium": 77, "iron": 1.7, "vitamin_c": 55, "vitamin_a": 6300},
    "ผักชี": {"calories": 23, "protein": 2.1, "fat": 0.5, "carbs": 3.7, "fiber": 2.8, "calcium": 67, "iron": 1.8, "vitamin_c": 27, "vitamin_a": 3930},
    "คะน้า": {"calories": 22, "protein": 2.2, "fat": 0.3, "carbs": 4.3, "fiber": 1.0, "calcium": 150, "iron": 1.5, "vitamin_c": 120, "vitamin_a": 7000},
    "ผักกาด": {"calories": 16, "protein": 1.5, "fat": 0.2, "carbs": 3.2, "fiber": 1.8, "calcium": 105, "iron": 0.8, "vitamin_c": 45, "vitamin_a": 4500},
    
    # ผักและพืชอื่นๆ
    "หัวหอม": {"calories": 40, "protein": 1.1, "fat": 0.1, "carbs": 9.3, "fiber": 1.7, "calcium": 23, "iron": 0.2, "vitamin_c": 7.4, "vitamin_a": 0},
    "กระเทียม": {"calories": 149, "protein": 6.4, "fat": 0.5, "carbs": 33, "fiber": 2.1, "calcium": 181, "iron": 1.7, "vitamin_c": 31, "vitamin_a": 9},
    "พริก": {"calories": 40, "protein": 1.9, "fat": 0.4, "carbs": 8.8, "fiber": 1.5, "calcium": 14, "iron": 1.0, "vitamin_c": 144, "vitamin_a": 952},
    "มะเขือเทศ": {"calories": 18, "protein": 0.9, "fat": 0.2, "carbs": 3.9, "fiber": 1.2, "calcium": 10, "iron": 0.3, "vitamin_c": 14, "vitamin_a": 833},
    "แตงกวา": {"calories": 16, "protein": 0.7, "fat": 0.1, "carbs": 3.6, "fiber": 0.5, "calcium": 16, "iron": 0.3, "vitamin_c": 2.8, "vitamin_a": 105},
    
    # เครื่องปรุง
    "น้ำมัน": {"calories": 884, "protein": 0, "fat": 100, "carbs": 0, "fiber": 0, "calcium": 0, "iron": 0, "vitamin_c": 0, "vitamin_a": 0},
    "น้ำตาล": {"calories": 387, "protein": 0, "fat": 0, "carbs": 100, "fiber": 0, "calcium": 0, "iron": 0.1, "vitamin_c": 0, "vitamin_a": 0},
    "เกลือ": {"calories": 0, "protein": 0, "fat": 0, "carbs": 0, "fiber": 0, "calcium": 24, "iron": 0.3, "vitamin_c": 0, "vitamin_a": 0},
    
    # แป้งและธัญพืช
    "ข้าว": {"calories": 130, "protein": 2.7, "fat": 0.3, "carbs": 28, "fiber": 0.4, "calcium": 10, "iron": 0.8, "vitamin_c": 0, "vitamin_a": 0},
    "แป้ง": {"calories": 364, "protein": 10, "fat": 1, "carbs": 76, "fiber": 2.7, "calcium": 15, "iron": 1.2, "vitamin_c": 0, "vitamin_a": 0},
    
    # นม และผลิตภัณฑ์จากนม
    "นม": {"calories": 42, "protein": 3.4, "fat": 1, "carbs": 5, "fiber": 0, "calcium": 113, "iron": 0.03, "vitamin_c": 0, "vitamin_a": 28},
    "กะทิ": {"calories": 230, "protein": 2.3, "fat": 24, "carbs": 6, "fiber": 2.2, "calcium": 16, "iron": 1.6, "vitamin_c": 1, "vitamin_a": 0}
}

# หน่วยวัดมาตรฐานและการแปลงหน่วย
UNIT_CONVERSIONS = {
    # น้ำหนัก
    "กิโลกรัม": 1000, "กก.": 1000, "kg": 1000,
    "กรัม": 1, "ก.": 1, "g": 1,
    "ขีด": 15, "บาท": 15,
    "ออนซ์": 28.35, "oz": 28.35,
    "ปอนด์": 453.6, "lb": 453.6,
    
    # ปริมาตร (แปลงเป็น ml แล้วประมาณเป็นกรัม)
    "ลิตร": 1000, "ล.": 1000, "L": 1000,
    "มิลลิลิตร": 1, "มล.": 1, "ml": 1,
    "ถ้วย": 250, "ถ้วยชา": 200, "ถ้วยกาแฟ": 150,
    "ช้อนโต๊ะ": 15, "ช้อนชา": 5,
    "ช้อนกาแฟ": 2.5, "แก้ว": 200,
    
    # หน่วยนับ (ประมาณการ)
    "ตัว": {"default": 50, "ไก่": 1500, "หมู": 2000, "ปลา": 200, "กุ้ง": 20},
    "ผล": {"default": 100, "มะเขือเทศ": 80, "แตงกวา": 150, "มะนาว": 50},
    "หัว": {"default": 50, "หัวหอม": 60, "กระเทียม": 10},
    "กลีบ": {"default": 3, "กระเทียม": 3},
    "เม็ด": {"default": 1, "พริก": 2},
    "ฟอง": {"default": 50, "ไข่": 50},
    "ต้น": {"default": 20, "ผักชี": 15, "ต้นหอม": 10},
    "ราก": {"default": 5, "ผักชี": 3},
    "ใบ": {"default": 1, "มะกรูด": 2},
    "แผ่น": {"default": 5},
    "แว่น": {"default": 3}
}

@st.cache_resource
def load_model():
    """โหลดหรือดาวน์โหลดโมเดล sentence transformer"""
    if os.path.exists(MODEL_PATH):
        return SentenceTransformer(MODEL_PATH)
    else:
        with st.spinner('กำลังดาวน์โหลดโมเดลการค้นหา...'):
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            os.makedirs(MODEL_PATH, exist_ok=True)
            model.save(MODEL_PATH)
            return model

@st.cache_data
def load_data():
    """โหลดข้อมูลสูตรอาหารไทย"""
    try:
        df = pd.read_csv(DATA_PATH)
        # ตรวจสอบชื่อคอลัมน์และปรับแต่ง
        if 'text_ingradiant' in df.columns:
            df = df.rename(columns={'text_ingradiant': 'ingredient'})
        if 'food_method' in df.columns:
            df = df.rename(columns={'food_method': 'method'})
        return df
    except FileNotFoundError:
        st.error(f"ไม่พบไฟล์ {DATA_PATH} กรุณาตรวจสอบให้แน่ใจว่าไฟล์อยู่ในโฟลเดอร์โปรเจค")
        return pd.DataFrame()

@st.cache_data
def get_embeddings(_model, data):
    """สร้างหรือโหลด embeddings สำหรับสูตรอาหารทั้งหมด"""
    if os.path.exists(EMBEDDINGS_PATH) and len(data) > 0:
        try:
            with open(EMBEDDINGS_PATH, 'rb') as f:
                embeddings = pickle.load(f)
                if len(embeddings) == len(data):
                    return embeddings
        except:
            pass
    
    if len(data) == 0:
        return np.array([])
    
    # สร้าง embeddings ใหม่
    with st.spinner('กำลังสร้าง embeddings สำหรับการค้นหา...'):
        texts = []
        for _, row in data.iterrows():
            combined_text = f"{row['name']} {row['ingredient']} {row['method']}"
            texts.append(combined_text)
        
        embeddings = _model.encode(texts)
        
        # บันทึก embeddings
        with open(EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump(embeddings, f)
        
        return embeddings

def fetch_usda_nutrition_data(api_key, max_results=1000):
    """ดาวน์โหลดข้อมูลโภชนาการจาก USDA API"""
    base_url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    
    all_foods = []
    page = 1
    page_size = 50
    
    try:
        while len(all_foods) < max_results:
            params = {
                "api_key": api_key,
                "query": "*",
                "pageSize": min(page_size, max_results - len(all_foods)),
                "pageNumber": page,
                "dataType": ["Foundation", "SR Legacy"]
            }
            
            response = requests.get(base_url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                foods = data.get('foods', [])
                
                if not foods:
                    break
                    
                for food in foods:
                    food_data = {
                        'fdc_id': food.get('fdcId'),
                        'description': food.get('description', ''),
                        'calories': 0,
                        'protein': 0,
                        'fat': 0,
                        'carbs': 0,
                        'fiber': 0,
                        'calcium': 0,
                        'iron': 0,
                        'vitamin_c': 0,
                        'vitamin_a': 0
                    }
                    
                    # แยกข้อมูลโภชนาการ
                    nutrients = food.get('foodNutrients', [])
                    for nutrient in nutrients:
                        nutrient_id = nutrient.get('nutrientId')
                        value = nutrient.get('value', 0)
                        
                        if nutrient_id == 1008:  # พลังงาน (kcal)
                            food_data['calories'] = value
                        elif nutrient_id == 1003:  # โปรตีน
                            food_data['protein'] = value
                        elif nutrient_id == 1004:  # ไขมัน
                            food_data['fat'] = value
                        elif nutrient_id == 1005:  # คาร์โบไฮเดรต
                            food_data['carbs'] = value
                        elif nutrient_id == 1079:  # ใยอาหาร
                            food_data['fiber'] = value
                        elif nutrient_id == 1087:  # แคลเซียม
                            food_data['calcium'] = value
                        elif nutrient_id == 1089:  # เหล็ก
                            food_data['iron'] = value
                        elif nutrient_id == 1162:  # วิตามิน C
                            food_data['vitamin_c'] = value
                        elif nutrient_id == 1106:  # วิตามิน A
                            food_data['vitamin_a'] = value
                    
                    all_foods.append(food_data)
                
                page += 1
                time.sleep(0.1)  # หน่วงเวลาเพื่อไม่ให้เรียก API บ่อยเกินไป
                
            else:
                st.error(f"ข้อผิดพลาด API: {response.status_code}")
                break
    
    except requests.exceptions.RequestException as e:
        st.error(f"เกิดข้อผิดพลาดในการเชื่อมต่อ API: {str(e)}")
        return pd.DataFrame()
    
    return pd.DataFrame(all_foods)

def save_nutrition_data_csv(nutrition_df):
    """บันทึกข้อมูลโภชนาการเป็นไฟล์ CSV"""
    try:
        nutrition_df.to_csv(NUTRITION_DATA_PATH, index=False, encoding='utf-8-sig')
        return True
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการบันทึกไฟล์: {str(e)}")
        return False

def parse_ingredient_amount(ingredient_text):
    """แยกปริมาณและหน่วยจากข้อความวัตถุดิบ"""
    # ลบเครื่องหมาย - ที่ด้านหน้า
    text = ingredient_text.strip().lstrip('-').strip()
    
    # หารูปแบบตัวเลข เศษส่วน และหน่วย
    patterns = [
        r'(\d+(?:[./]\d+)?)\s*([ก-๙a-zA-Z]+)',  # ตัวเลขตามด้วยหน่วย
        r'(\d+(?:[./]\d+)?)',  # เฉพาะตัวเลข
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            amount_str = match.group(1)
            unit = match.group(2) if len(match.groups()) > 1 else ""
            
            # แปลงเศษส่วนเป็นทศนิยม
            if '/' in amount_str:
                parts = amount_str.split('/')
                amount = float(parts[0]) / float(parts[1])
            else:
                amount = float(amount_str)
            
            return amount, unit, text.replace(match.group(0), '').strip()
    
    # ถ้าไม่พบตัวเลข ใช้ค่าเริ่มต้น
    return 1, "", text

def estimate_ingredient_weight(ingredient_text):
    """ประมาณน้ำหนักของวัตถุดิบจากข้อความ"""
    amount, unit, name = parse_ingredient_amount(ingredient_text)
    
    # ค้นหาชื่อวัตถุดิบในฐานข้อมูล
    ingredient_name = None
    for key in BASIC_NUTRITION_DATA.keys():
        if key in name.lower():
            ingredient_name = key
            break
    
    # แปลงหน่วยเป็นกรัม
    weight_grams = 0
    
    if unit in UNIT_CONVERSIONS:
        if isinstance(UNIT_CONVERSIONS[unit], dict):
            # หน่วยนับที่ต้องใช้ชื่อวัตถุดิบ
            conversion_data = UNIT_CONVERSIONS[unit]
            if ingredient_name and ingredient_name in conversion_data:
                weight_grams = amount * conversion_data[ingredient_name]
            else:
                weight_grams = amount * conversion_data["default"]
        else:
            # หน่วยน้ำหนักหรือปริมาตรปกติ
            weight_grams = amount * UNIT_CONVERSIONS[unit]
    else:
        # ถ้าไม่มีหน่วย หรือหน่วยไม่รู้จัก ประมาณจากชื่อวัตถุดิบ
        if ingredient_name:
            if "น้ำมัน" in ingredient_name or "น้ำตาล" in ingredient_name:
                weight_grams = amount * 5  # ประมาณช้อนชา
            else:
                weight_grams = amount * 50  # ประมาณทั่วไป
        else:
            weight_grams = amount * 50  # ค่าเริ่มต้น
    
    # ตรวจสอบว่าได้น้ำหนักที่สมเหตุสมผลหรือไม่
    if weight_grams == 0:
        weight_grams = 50  # ค่าเริ่มต้นถ้าคำนวณไม่ได้
    elif weight_grams > 5000:  # ถ้ามากเกินไป (มากกว่า 5kg)
        weight_grams = 500  # ปรับลงเป็น 500g
    
    return weight_grams, ingredient_name

def calculate_nutrition_for_recipe(ingredients_text, cooking_adjustment=False):
    """คำนวณคุณค่าทางโภชนาการสำหรับสูตรอาหาร"""
    ingredients = [ing.strip() for ing in ingredients_text.split('\n') if ing.strip()]
    
    total_nutrition = {
        'calories': 0, 'protein': 0, 'fat': 0, 'carbs': 0, 'fiber': 0,
        'calcium': 0, 'iron': 0, 'vitamin_c': 0, 'vitamin_a': 0
    }
    
    ingredient_details = []
    
    for ingredient in ingredients:
        if not ingredient or ingredient == '-':
            continue
            
        weight, ingredient_name = estimate_ingredient_weight(ingredient)
        
        if ingredient_name and ingredient_name in BASIC_NUTRITION_DATA:
            nutrition_per_100g = BASIC_NUTRITION_DATA[ingredient_name]
            
            # คำนวณโภชนาการตามน้ำหนักจริง
            actual_nutrition = {}
            for key, value in nutrition_per_100g.items():
                actual_nutrition[key] = (value * weight) / 100
            
            # ปรับค่าสำหรับการบริโภคจริง (ถ้าเปิดใช้งาน)
            if cooking_adjustment:
                # น้ำมันที่ใช้ทอด จะบริโภคเพียง 10-20%
                if "น้ำมัน" in ingredient_name and weight > 50:
                    absorption_rate = 0.15  # ดูดซึม 15%
                    for key in actual_nutrition:
                        actual_nutrition[key] *= absorption_rate
                
                # เครื่องปรุงบางอย่างใช้น้อยมาก
                elif ingredient_name in ["เกลือ", "พริกไทย"] and weight > 10:
                    for key in actual_nutrition:
                        actual_nutrition[key] *= 0.5
            
            # รวมคุณค่าโภชนาการ
            for key in total_nutrition:
                total_nutrition[key] += actual_nutrition.get(key, 0)
            
            ingredient_details.append({
                'name': ingredient,
                'weight': weight,
                'ingredient_type': ingredient_name,
                'nutrition': actual_nutrition
            })
        else:
            # ถ้าไม่พบข้อมูล ใช้ค่าประมาณ
            estimated_nutrition = {
                'calories': weight * 1.5,
                'protein': weight * 0.1,
                'fat': weight * 0.05,
                'carbs': weight * 0.2,
                'fiber': weight * 0.02,
                'calcium': weight * 0.3,
                'iron': weight * 0.01,
                'vitamin_c': weight * 0.1,
                'vitamin_a': weight * 0.5
            }
            
            for key in total_nutrition:
                total_nutrition[key] += estimated_nutrition.get(key, 0)
            
            ingredient_details.append({
                'name': ingredient,
                'weight': weight,
                'ingredient_type': 'unknown',
                'nutrition': estimated_nutrition
            })
    
    return total_nutrition, ingredient_details

def format_nutrition_card(nutrition_data, title="ข้อมูลโภชนาการ"):
    """จัดรูปแบบการแสดงผลข้อมูลโภชนาการ"""
    html = f"""
    <div class="nutrition-card">
        <h4>{title}</h4>
        <div class="nutrient-row">
            <span class="nutrient-name">แคลอรี่:</span>
            <span class="nutrient-value">{nutrition_data['calories']:.1f} kcal</span>
        </div>
        <div class="nutrient-row">
            <span class="nutrient-name">โปรตีน:</span>
            <span class="nutrient-value">{nutrition_data['protein']:.1f} g</span>
        </div>
        <div class="nutrient-row">
            <span class="nutrient-name">ไขมัน:</span>
            <span class="nutrient-value">{nutrition_data['fat']:.1f} g</span>
        </div>
        <div class="nutrient-row">
            <span class="nutrient-name">คาร์โบไฮเดรต:</span>
            <span class="nutrient-value">{nutrition_data['carbs']:.1f} g</span>
        </div>
        <div class="nutrient-row">
            <span class="nutrient-name">ใยอาหาร:</span>
            <span class="nutrient-value">{nutrition_data['fiber']:.1f} g</span>
        </div>
        <div class="nutrient-row">
            <span class="nutrient-name">แคลเซียม:</span>
            <span class="nutrient-value">{nutrition_data['calcium']:.1f} mg</span>
        </div>
        <div class="nutrient-row">
            <span class="nutrient-name">เหล็ก:</span>
            <span class="nutrient-value">{nutrition_data['iron']:.1f} mg</span>
        </div>
        <div class="nutrient-row">
            <span class="nutrient-name">วิตามิน C:</span>
            <span class="nutrient-value">{nutrition_data['vitamin_c']:.1f} mg</span>
        </div>
        <div class="nutrient-row">
            <span class="nutrient-name">วิตามิน A:</span>
            <span class="nutrient-value">{nutrition_data['vitamin_a']:.1f} µg</span>
        </div>
    </div>
    """
    return html

def format_ingredients_list(ingredients_text, ingredient_details=None):
    """จัดรูปแบบรายการวัตถุดิบ"""
    ingredients = [ing.strip() for ing in ingredients_text.split('\n') if ing.strip()]
    
    html = "<div>"
    for i, ingredient in enumerate(ingredients):
        if not ingredient or ingredient == '-':
            continue
        
        detail_info = ""
        if ingredient_details and i < len(ingredient_details):
            detail = ingredient_details[i]
            detail_info = f" <small>(ประมาณ {detail['weight']:.0f}g)</small>"
        
        html += f"""
        <div class="ingredient-item">
            {ingredient}{detail_info}
        </div>
        """
    
    html += "</div>"
    return html

def format_cooking_method(method_text):
    """จัดรูปแบบวิธีการทำอาหาร"""
    # แยกขั้นตอนตามจุด หรือการขึ้นบรรทัดใหม่
    sentences = re.split(r'[.。]\s*|(?<=\s)(?=[ก-ฮ].*(?:จึง|แล้ว|ต่อ|ใส่|นำ|เอา|ทำ))', method_text)
    
    html = "<div>"
    step_num = 1
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 10:  # เฉพาะประโยคที่มีความยาวเพียงพอ
            html += f"""
            <div class="cooking-step">
                <strong>ขั้นตอนที่ {step_num}:</strong> {sentence}
            </div>
            """
            step_num += 1
    
    html += "</div>"
    return html

def similarity(a, b):
    """คำนวณความคล้ายคลึงระหว่างข้อความ"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def find_similar_recipes(query, data, threshold=0.3):
    """ค้นหาสูตรอาหารที่คล้ายคลึงกัน"""
    results = []
    query_lower = query.lower()
    
    for _, row in data.iterrows():
        name_similarity = similarity(query_lower, row['name'].lower())
        ingredient_similarity = similarity(query_lower, row['ingredient'].lower())
        
        max_similarity = max(name_similarity, ingredient_similarity)
        
        if max_similarity >= threshold:
            results.append({
                'name': row['name'],
                'similarity': max_similarity,
                'ingredients': row['ingredient'],
                'method': row['method']
            })
    
    return sorted(results, key=lambda x: x['similarity'], reverse=True)

def search_recipes_by_nutrition(data, nutrition_criteria):
    """ค้นหาอาหารตามเงื่อนไขโภชนาการ"""
    results = []
    
    for _, row in data.iterrows():
        nutrition, _ = calculate_nutrition_for_recipe(row['ingredient'])
        
        match_score = 0
        max_score = len(nutrition_criteria)
        
        for criteria in nutrition_criteria:
            nutrient = criteria['nutrient']
            operator = criteria['operator']  # 'high', 'low', 'medium'
            
            value = nutrition.get(nutrient, 0)
            
            # กำหนดเกณฑ์ตามประเภทสารอาหาร
            thresholds = {
                'calories': {'low': 200, 'high': 400},
                'protein': {'low': 10, 'high': 25},
                'fat': {'low': 5, 'high': 15},
                'carbs': {'low': 20, 'high': 50},
                'fiber': {'low': 3, 'high': 8},
                'calcium': {'low': 50, 'high': 150},
                'iron': {'low': 2, 'high': 5},
                'vitamin_c': {'low': 10, 'high': 50},
                'vitamin_a': {'low': 100, 'high': 500}
            }
            
            threshold = thresholds.get(nutrient, {'low': 5, 'high': 15})
            
            if operator == 'high' and value >= threshold['high']:
                match_score += 1
            elif operator == 'low' and value <= threshold['low']:
                match_score += 1
            elif operator == 'medium' and threshold['low'] < value < threshold['high']:
                match_score += 1
        
        if match_score > 0:
            results.append({
                'name': row['name'],
                'match_score': match_score / max_score,
                'nutrition': nutrition,
                'ingredients': row['ingredient'],
                'method': row['method']
            })
    
    return sorted(results, key=lambda x: x['match_score'], reverse=True)

def search_recipes_semantic(query, model, data, embeddings, top_k=5):
    """ค้นหาสูตรอาหารด้วยความหมาย"""
    if len(embeddings) == 0:
        return []
    
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    top_indices = np.argsort(-similarities)[:top_k]
    results = []
    
    for idx in top_indices:
        if similarities[idx] > 0.1:  # เกณฑ์ความคล้ายคลึงขั้นต่ำ
            results.append({
                'name': data.iloc[idx]['name'],
                'similarity': similarities[idx],
                'ingredients': data.iloc[idx]['ingredient'],
                'method': data.iloc[idx]['method']
            })
    
    return results

def process_nutrition_query(query, data):
    """ประมวลผลคำถามเกี่ยวกับโภชนาการ"""
    query_lower = query.lower()
    
    # รูปแบบคำถามต่างๆ
    nutrition_keywords = {
        'calories': ['แคลอรี่', 'พลังงาน', 'kcal', 'calorie'],
        'protein': ['โปรตีน', 'protein'],
        'fat': ['ไขมัน', 'fat'],
        'carbs': ['คาร์โบไฮเดรต', 'คาร์โบ', 'carb'],
        'fiber': ['ใยอาหาร', 'fiber'],
        'calcium': ['แคลเซียม', 'calcium'],
        'iron': ['เหล็ก', 'iron'],
        'vitamin_c': ['วิตามินซี', 'วิตามิน c', 'vitamin c'],
        'vitamin_a': ['วิตามินเอ', 'วิตามิน a', 'vitamin a']
    }
    
    level_keywords = {
        'high': ['สูง', 'มาก', 'เยอะ', 'high'],
        'low': ['ต่ำ', 'น้อย', 'low'],
        'medium': ['ปานกลาง', 'medium']
    }
    
    criteria = []
    
    # ค้นหาเงื่อนไขในคำถาม
    for nutrient, keywords in nutrition_keywords.items():
        for keyword in keywords:
            if keyword in query_lower:
                for level, level_kw in level_keywords.items():
                    for lkw in level_kw:
                        if lkw in query_lower:
                            criteria.append({
                                'nutrient': nutrient,
                                'operator': level
                            })
                            break
                break
    
    if criteria:
        return search_recipes_by_nutrition(data, criteria)
    else:
        return []

def main():
    # โหลดโมเดลและข้อมูล
    model = load_model()
    data = load_data()
    
    if len(data) == 0:
        st.error("ไม่สามารถโหลดข้อมูลสูตรอาหารได้")
        return
    
    embeddings = get_embeddings(model, data)
    
    # แถบการตั้งค่าด้านซ้าย
    with st.sidebar:
        st.header("⚙️ การตั้งค่า")
        
        # ส่วนการจัดการข้อมูลโภชนาการ
        with st.expander("📊 ข้อมูลโภชนาการจาก USDA"):
            st.write("ดาวน์โหลดข้อมูลโภชนาการจาก USDA API")
            
            usda_api_key = st.text_input("USDA API Key", type="password", 
                                       help="ดาวน์โหลด API Key ฟรีจาก https://fdc.nal.usda.gov/api-guide.html")
            max_records = st.number_input("จำนวนข้อมูลสูงสุด", min_value=100, max_value=5000, value=1000)
            
            if st.button("ดาวน์โหลดข้อมูล USDA"):
                if usda_api_key:
                    with st.spinner("กำลังดาวน์โหลดข้อมูล..."):
                        nutrition_df = fetch_usda_nutrition_data(usda_api_key, max_records)
                        if not nutrition_df.empty:
                            if save_nutrition_data_csv(nutrition_df):
                                st.success(f"ดาวน์โหลดและบันทึกข้อมูลสำเร็จ! จำนวน {len(nutrition_df)} รายการ")
                            else:
                                st.error("เกิดข้อผิดพลาดในการบันทึกไฟล์")
                        else:
                            st.error("ไม่สามารถดาวน์โหลดข้อมูลได้")
                else:
                    st.warning("กรุณาใส่ USDA API Key")
            
            # แสดงสถานะไฟล์ข้อมูลโภชนาการ
            if os.path.exists(NUTRITION_DATA_PATH):
                st.success(f"✅ พบไฟล์ข้อมูลโภชนาการ: {NUTRITION_DATA_PATH}")
            else:
                st.info("ℹ️ ไม่พบไฟล์ข้อมูลโภชนาการ จะใช้ข้อมูลพื้นฐานในระบบ")
        
        # ตัวเลือกการคำนวณ
        with st.expander("🔧 ตัวเลือกการคำนวณ"):
            cooking_adjustment = st.checkbox(
                "ปรับการคำนวณตามการบริโภคจริง",
                value=False,
                help="คำนวณปริมาณสารอาหารที่บริโภคได้จริง (เช่น น้ำมันทอดจะดูดซึมเพียงบางส่วน)"
            )
            
            detailed_nutrition = st.checkbox(
                "แสดงรายละเอียดวิตามินและแร่ธาตุ",
                value=True,
                help="แสดงข้อมูลวิตามินและแร่ธาตุในแต่ละรายการ"
            )
        
        # ข้อมูลสถิติ
        with st.expander("📈 สถิติข้อมูล"):
            st.write(f"จำนวนสูตรอาหารทั้งหมด: {len(data)}")
            st.write(f"จำนวนข้อมูลโภชนาการพื้นฐาน: {len(BASIC_NUTRITION_DATA)}")
            if os.path.exists(NUTRITION_DATA_PATH):
                try:
                    nutrition_df = pd.read_csv(NUTRITION_DATA_PATH)
                    st.write(f"จำนวนข้อมูล USDA: {len(nutrition_df)}")
                except:
                    st.write("ข้อมูล USDA: ไม่สามารถอ่านไฟล์ได้")
    
    # ส่วนหลักของแอปพลิเคชัน
    st.title("🍲 ฐานข้อมูลอาหารไทยและคุณค่าทางโภชนาการ")
    st.write("ค้นหาสูตรอาหารไทย วิเคราะห์คุณค่าทางโภชนาการ และรับคำแนะนำอาหารที่เหมาะสม")
    
    # แท็บต่างๆ
    tab1, tab2, tab3 = st.tabs(["🔍 ค้นหาและถามตอบ", "📊 วิเคราะห์โภชนาการ", "🍽️ แนะนำอาหาร"])
    
    with tab1:
        st.header("ค้นหาสูตรอาหารและถามตอบ")
        
        # พื้นที่สำหรับแชท
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # แสดงประวัติการสนทนา
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant" and "recipe" in message:
                    recipe = message["recipe"]
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"### {recipe['name']}")
                        
                        st.markdown("#### 🥘 วัตถุดิบ")
                        if "ingredient_details" in recipe:
                            st.markdown(format_ingredients_list(recipe["ingredients"], recipe["ingredient_details"]), unsafe_allow_html=True)
                        else:
                            st.markdown(format_ingredients_list(recipe["ingredients"]), unsafe_allow_html=True)
                        
                        st.markdown("#### 👨‍🍳 วิธีทำ")
                        st.markdown(format_cooking_method(recipe["method"]), unsafe_allow_html=True)
                        
                        if "similarity" in recipe:
                            st.markdown(f"*ความเกี่ยวข้อง: {recipe['similarity']:.2f}*")
                    
                    with col2:
                        if "nutrition" in recipe and detailed_nutrition:
                            st.markdown(format_nutrition_card(recipe["nutrition"], "คุณค่าทางโภชนาการ"), unsafe_allow_html=True)
                
                else:
                    st.markdown(message["content"])
        
        # ช่องป้อนคำถาม
        if prompt := st.chat_input("ถามเกี่ยวกับอาหารไทยหรือค้นหาสูตรอาหาร..."):
            # เพิ่มข้อความของผู้ใช้
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("กำลังค้นหาคำตอบ..."):
                    # ประมวลผลคำถาม
                    results = []
                    
                    # ลองค้นหาตามโภชนาการก่อน
                    nutrition_results = process_nutrition_query(prompt, data)
                    if nutrition_results:
                        results = nutrition_results[:3]
                    else:
                        # ค้นหาแบบความหมาย
                        semantic_results = search_recipes_semantic(prompt, model, data, embeddings, top_k=3)
                        if semantic_results:
                            results = semantic_results
                        else:
                            # ค้นหาแบบความคล้ายคลึงของข้อความ
                            similar_results = find_similar_recipes(prompt, data, threshold=0.2)
                            results = similar_results[:3]
                    
                    if results:
                        if len(results) == 1:
                            # แสดงสูตรเดียว
                            recipe = results[0]
                            
                            # คำนวณโภชนาการ
                            nutrition, ingredient_details = calculate_nutrition_for_recipe(
                                recipe['ingredients'], 
                                cooking_adjustment
                            )
                            
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.markdown(f"### {recipe['name']}")
                                
                                st.markdown("#### 🥘 วัตถุดิบ")
                                st.markdown(format_ingredients_list(recipe["ingredients"], ingredient_details), unsafe_allow_html=True)
                                
                                st.markdown("#### 👨‍🍳 วิธีทำ")
                                st.markdown(format_cooking_method(recipe["method"]), unsafe_allow_html=True)
                            
                            with col2:
                                if detailed_nutrition:
                                    st.markdown(format_nutrition_card(nutrition, "คุณค่าทางโภชนาการ"), unsafe_allow_html=True)
                            
                            # บันทึกในประวัติ
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "recipe": {
                                    "name": recipe['name'],
                                    "ingredients": recipe['ingredients'],
                                    "method": recipe['method'],
                                    "nutrition": nutrition,
                                    "ingredient_details": ingredient_details,
                                    "similarity": recipe.get('similarity', recipe.get('match_score', 1.0))
                                }
                            })
                        
                        else:
                            # แสดงหลายสูตร
                            st.markdown(f"พบสูตรอาหารที่เกี่ยวข้อง {len(results)} สูตร:")
                            
                            for i, recipe in enumerate(results, 1):
                                with st.expander(f"{i}. {recipe['name']} (ความเกี่ยวข้อง: {recipe.get('similarity', recipe.get('match_score', 0)):.2f})"):
                                    nutrition, ingredient_details = calculate_nutrition_for_recipe(
                                        recipe['ingredients'], 
                                        cooking_adjustment
                                    )
                                    
                                    col1, col2 = st.columns([2, 1])
                                    
                                    with col1:
                                        st.markdown("**วัตถุดิบ:**")
                                        st.markdown(format_ingredients_list(recipe["ingredients"]), unsafe_allow_html=True)
                                        
                                        st.markdown("**วิธีทำ:**")
                                        st.markdown(format_cooking_method(recipe["method"]), unsafe_allow_html=True)
                                    
                                    with col2:
                                        if detailed_nutrition:
                                            st.markdown(format_nutrition_card(nutrition), unsafe_allow_html=True)
                            
                            # บันทึกในประวัติ
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": f"พบสูตรอาหารที่เกี่ยวข้อง {len(results)} สูตร"
                            })
                    
                    else:
                        response = f"ขออภัย ไม่พบสูตรอาหารที่เกี่ยวข้องกับ '{prompt}' \n\nลองใช้คำค้นหาอื่น เช่น:\n- ชื่ออาหาร (ผัดไทย, ต้มยำกุ้ง)\n- วัตถุดิบ (ไก่, หมู, ปลา)\n- คุณค่าทางโภชนาการ (อาหารโปรตีนสูง, แคลอรี่ต่ำ)"
                        st.markdown(response)
                        
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response
                        })
    
    with tab2:
        st.header("วิเคราะห์คุณค่าทางโภชนาการ")
        
        selected_recipe = st.selectbox(
            "เลือกสูตรอาหารเพื่อวิเคราะห์:",
            options=data['name'].tolist(),
            index=0
        )
        
        if selected_recipe:
            recipe_data = data[data['name'] == selected_recipe].iloc[0]
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown(f"### {selected_recipe}")
                st.markdown("#### วัตถุดิบ")
                
                # คำนวณโภชนาการ
                nutrition, ingredient_details = calculate_nutrition_for_recipe(
                    recipe_data['ingredient'], 
                    cooking_adjustment
                )
                
                st.markdown(format_ingredients_list(recipe_data['ingredient'], ingredient_details), unsafe_allow_html=True)
            
            with col2:
                st.markdown("### การวิเคราะห์ทางโภชนาการ")
                st.markdown(format_nutrition_card(nutrition), unsafe_allow_html=True)
                
                # กราฟแสดงสัดส่วนสารอาหาร
                if nutrition['calories'] > 0:
                    protein_cal = nutrition['protein'] * 4
                    fat_cal = nutrition['fat'] * 9
                    carb_cal = nutrition['carbs'] * 4
                    
                    labels = ['โปรตีน', 'ไขมัน', 'คาร์โบไฮเดรต']
                    values = [protein_cal, fat_cal, carb_cal]
                    
                    fig_data = pd.DataFrame({
                        'สารอาหาร': labels,
                        'แคลอรี่': values
                    })
                    
                    st.bar_chart(fig_data.set_index('สารอาหาร'))
            
            # รายละเอียดแต่ละวัตถุดิบ
            if ingredient_details:
                st.markdown("### รายละเอียดวัตถุดิบแต่ละชนิด")
                
                for detail in ingredient_details:
                    with st.expander(f"{detail['name']} ({detail['weight']:.0f}g)"):
                        if detail['ingredient_type'] != 'unknown':
                            st.markdown(format_nutrition_card(detail['nutrition'], f"โภชนาการใน{detail['name']}"), unsafe_allow_html=True)
                        else:
                            st.write("ไม่พบข้อมูลโภชนาการที่แม่นยำ (ใช้การประมาณ)")
    
    with tab3:
        st.header("แนะนำอาหารตามความต้องการ")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### เลือกเงื่อนไขที่ต้องการ")
            
            # ตัวเลือกการค้นหา
            calorie_level = st.selectbox("ระดับแคลอรี่", ["ไม่ระบุ", "ต่ำ (< 200)", "ปานกลาง (200-400)", "สูง (> 400)"])
            protein_level = st.selectbox("ระดับโปรตีน", ["ไม่ระบุ", "ต่ำ (< 10g)", "ปานกลาง (10-25g)", "สูง (> 25g)"])
            fat_level = st.selectbox("ระดับไขมัน", ["ไม่ระบุ", "ต่ำ (< 5g)", "ปานกลาง (5-15g)", "สูง (> 15g)"])
            fiber_level = st.selectbox("ระดับใยอาหาร", ["ไม่ระบุ", "ต่ำ (< 3g)", "ปานกลาง (3-8g)", "สูง (> 8g)"])
            calcium_level = st.selectbox("ระดับแคลเซียม", ["ไม่ระบุ", "ต่ำ (< 50mg)", "ปานกลาง (50-150mg)", "สูง (> 150mg)"])
            
            if st.button("ค้นหาอาหารแนะนำ"):
                criteria = []
                
                # แปลงตัวเลือกเป็นเงื่อนไข
                level_mapping = {
                    "ต่ำ": "low",
                    "ปานกลาง": "medium", 
                    "สูง": "high"
                }
                
                if "ต่ำ" in calorie_level:
                    criteria.append({"nutrient": "calories", "operator": "low"})
                elif "สูง" in calorie_level:
                    criteria.append({"nutrient": "calories", "operator": "high"})
                elif "ปานกลาง" in calorie_level:
                    criteria.append({"nutrient": "calories", "operator": "medium"})
                
                if "ต่ำ" in protein_level:
                    criteria.append({"nutrient": "protein", "operator": "low"})
                elif "สูง" in protein_level:
                    criteria.append({"nutrient": "protein", "operator": "high"})
                elif "ปานกลาง" in protein_level:
                    criteria.append({"nutrient": "protein", "operator": "medium"})
                
                if "ต่ำ" in fat_level:
                    criteria.append({"nutrient": "fat", "operator": "low"})
                elif "สูง" in fat_level:
                    criteria.append({"nutrient": "fat", "operator": "high"})
                elif "ปานกลาง" in fat_level:
                    criteria.append({"nutrient": "fat", "operator": "medium"})
                
                if "ต่ำ" in fiber_level:
                    criteria.append({"nutrient": "fiber", "operator": "low"})
                elif "สูง" in fiber_level:
                    criteria.append({"nutrient": "fiber", "operator": "high"})
                elif "ปานกลาง" in fiber_level:
                    criteria.append({"nutrient": "fiber", "operator": "medium"})
                
                if "ต่ำ" in calcium_level:
                    criteria.append({"nutrient": "calcium", "operator": "low"})
                elif "สูง" in calcium_level:
                    criteria.append({"nutrient": "calcium", "operator": "high"})
                elif "ปานกลาง" in calcium_level:
                    criteria.append({"nutrient": "calcium", "operator": "medium"})
                
                if criteria:
                    recommendations = search_recipes_by_nutrition(data, criteria)
                    
                    if recommendations:
                        st.session_state.recommendations = recommendations
                    else:
                        st.warning("ไม่พบอาหารที่ตรงตามเงื่อนไข ลองปรับเงื่อนไขใหม่")
                else:
                    st.warning("กรุณาเลือกเงื่อนไขอย่างน้อย 1 ข้อ")
        
        with col2:
            st.markdown("### ผลการแนะนำ")
            
            if 'recommendations' in st.session_state and st.session_state.recommendations:
                for i, recipe in enumerate(st.session_state.recommendations[:5], 1):
                    with st.expander(f"{i}. {recipe['name']} (ตรงเงื่อนไข {recipe['match_score']:.0%})"):
                        col_a, col_b = st.columns([1, 1])
                        
                        with col_a:
                            st.markdown("**วัตถุดิบ:**")
                            st.markdown(format_ingredients_list(recipe['ingredients']), unsafe_allow_html=True)
                        
                        with col_b:
                            st.markdown(format_nutrition_card(recipe['nutrition']), unsafe_allow_html=True)
            else:
                st.info("เลือกเงื่อนไขและคลิก 'ค้นหาอาหารแนะนำ' เพื่อดูผลลัพธ์")
    
    # ส่วนท้าย
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🍲 ฐานข้อมูลอาหารไทยและคุณค่าทางโภชนาการ</p>
        <p>พัฒนาเพื่อส่งเสริมการรับประทานอาหารที่มีคุณค่าทางโภชนาการ</p>
        <small>ข้อมูลโภชนาการเป็นการประมาณการ ควรปรึกษาผู้เชี่ยวชาญด้านโภชนาการสำหรับคำแนะนำที่แม่นยำ</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
