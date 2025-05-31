import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# การตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Thai Food Recipe Chatbot - ปรับปรุงใหม่",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ตั้งค่าฟอนต์ภาษาไทยและ CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@400;500;600;700&display=swap');
    html, body, [class*="st-"] {
        font-family: 'Sarabun', sans-serif !important;
    }
    
    .nutrition-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .vitamin-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .mineral-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .similarity-badge-high {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 0.3rem 0.7rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
        display: inline-block;
        margin-left: 0.5rem;
    }
    
    .similarity-badge-medium {
        background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
        color: white;
        padding: 0.3rem 0.7rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
        display: inline-block;
        margin-left: 0.5rem;
    }
    
    .similarity-badge-low {
        background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
        color: white;
        padding: 0.3rem 0.7rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
        display: inline-block;
        margin-left: 0.5rem;
    }
    
    .search-improvement-note {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: bold;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ข้อมูลโภชนาการพื้นฐาน (รวมในไฟล์เดียว)
NUTRITION_DATABASE = {
    "ข้าว": {"calories": 130, "protein": 2.7, "carbs": 28, "fat": 0.3, "fiber": 0.4, "vitamin_a": 0, "vitamin_c": 0, "vitamin_b1": 0.07, "vitamin_b2": 0.02, "calcium": 10, "iron": 0.8, "potassium": 115, "sodium": 5},
    "ไข่ไก่": {"calories": 155, "protein": 13, "carbs": 1.1, "fat": 11, "fiber": 0, "vitamin_a": 540, "vitamin_c": 0, "vitamin_b1": 0.04, "vitamin_b2": 0.42, "calcium": 56, "iron": 1.75, "potassium": 138, "sodium": 124},
    "กุ้ง": {"calories": 99, "protein": 18, "carbs": 0.2, "fat": 1.4, "fiber": 0, "vitamin_a": 54, "vitamin_c": 2.1, "vitamin_b1": 0.02, "vitamin_b2": 0.04, "calcium": 70, "iron": 0.5, "potassium": 259, "sodium": 111},
    "หมู": {"calories": 242, "protein": 27, "carbs": 0, "fat": 14, "fiber": 0, "vitamin_a": 2, "vitamin_c": 0.7, "vitamin_b1": 0.66, "vitamin_b2": 0.23, "calcium": 19, "iron": 0.87, "potassium": 423, "sodium": 62},
    "ไก่": {"calories": 165, "protein": 31, "carbs": 0, "fat": 3.6, "fiber": 0, "vitamin_a": 21, "vitamin_c": 1.6, "vitamin_b1": 0.07, "vitamin_b2": 0.12, "calcium": 15, "iron": 1.3, "potassium": 256, "sodium": 82},
    "ปลา": {"calories": 112, "protein": 18.7, "carbs": 0, "fat": 3.6, "fiber": 0, "vitamin_a": 45, "vitamin_c": 0.9, "vitamin_b1": 0.02, "vitamin_b2": 0.11, "calcium": 89, "iron": 0.9, "potassium": 358, "sodium": 54},
    "น้ำมันหมู": {"calories": 902, "protein": 0, "carbs": 0, "fat": 100, "fiber": 0, "vitamin_a": 0, "vitamin_c": 0, "vitamin_b1": 0, "vitamin_b2": 0, "calcium": 0, "iron": 0, "potassium": 0, "sodium": 0},
    "น้ำมันพืช": {"calories": 884, "protein": 0, "carbs": 0, "fat": 100, "fiber": 0, "vitamin_a": 0, "vitamin_c": 0, "vitamin_b1": 0, "vitamin_b2": 0, "calcium": 0, "iron": 0, "potassium": 0, "sodium": 0},
    "กระเทียม": {"calories": 149, "protein": 6.4, "carbs": 33, "fat": 0.5, "fiber": 2.1, "vitamin_a": 9, "vitamin_c": 31, "vitamin_b1": 0.2, "vitamin_b2": 0.11, "calcium": 181, "iron": 1.7, "potassium": 401, "sodium": 17},
    "หอมใหญ่": {"calories": 40, "protein": 1.1, "carbs": 9.3, "fat": 0.1, "fiber": 1.7, "vitamin_a": 2, "vitamin_c": 7.4, "vitamin_b1": 0.05, "vitamin_b2": 0.03, "calcium": 23, "iron": 0.21, "potassium": 146, "sodium": 4},
    "ผักชี": {"calories": 23, "protein": 2.1, "carbs": 3.7, "fat": 0.5, "fiber": 2.8, "vitamin_a": 3377, "vitamin_c": 27, "vitamin_b1": 0.07, "vitamin_b2": 0.16, "calcium": 67, "iron": 1.77, "potassium": 521, "sodium": 46},
    "พริกไทย": {"calories": 251, "protein": 10.4, "carbs": 64, "fat": 3.3, "fiber": 25, "vitamin_a": 547, "vitamin_c": 0, "vitamin_b1": 0.11, "vitamin_b2": 0.18, "calcium": 443, "iron": 9.7, "potassium": 1329, "sodium": 20},
    "น้ำปลา": {"calories": 42, "protein": 5.8, "carbs": 1.5, "fat": 0.8, "fiber": 0, "vitamin_a": 0, "vitamin_c": 0, "vitamin_b1": 0.03, "vitamin_b2": 0.22, "calcium": 85, "iron": 2.03, "potassium": 84, "sodium": 6976},
    "น้ำตาล": {"calories": 387, "protein": 0, "carbs": 100, "fat": 0, "fiber": 0, "vitamin_a": 0, "vitamin_c": 0, "vitamin_b1": 0, "vitamin_b2": 0, "calcium": 1, "iron": 0.01, "potassium": 2, "sodium": 1},
    "มะนาว": {"calories": 29, "protein": 0.7, "carbs": 9.3, "fat": 0.2, "fiber": 2.8, "vitamin_a": 22, "vitamin_c": 53, "vitamin_b1": 0.03, "vitamin_b2": 0.02, "calcium": 33, "iron": 0.6, "potassium": 138, "sodium": 2},
    "กะทิ": {"calories": 230, "protein": 2.3, "carbs": 6, "fat": 24, "fiber": 2.2, "vitamin_a": 0, "vitamin_c": 2.8, "vitamin_b1": 0.03, "vitamin_b2": 0, "calcium": 16, "iron": 1.64, "potassium": 263, "sodium": 15},
    "เกลือ": {"calories": 0, "protein": 0, "carbs": 0, "fat": 0, "fiber": 0, "vitamin_a": 0, "vitamin_c": 0, "vitamin_b1": 0, "vitamin_b2": 0, "calcium": 24, "iron": 0.33, "potassium": 8, "sodium": 38758}
}

class SimplifiedNutritionAPI:
    """ระบบคำนวณโภชนาการแบบง่าย"""
    
    def __init__(self):
        self.nutrition_db = NUTRITION_DATABASE
        self.unit_conversion = {
            "ช้อนโต๊ะ": 15, "ช้อนชา": 5, "ถ้วย": 240, "มล": 1, "ลิตร": 1000,
            "กรัม": 1, "กิโลกรัม": 1000, "ตัว": 100, "ฟอง": 50, "หัว": 50,
            "กลีบ": 3, "ต้น": 30, "ใบ": 2, "เม็ด": 0.5, "แว่น": 2, "ผล": 150
        }
    
    def get_nutrition_data(self, ingredient):
        """ดึงข้อมูลโภชนาการ"""
        ingredient_lower = ingredient.lower()
        for key, nutrition in self.nutrition_db.items():
            if key in ingredient_lower or ingredient_lower in key:
                return nutrition
        
        # ค่าเริ่มต้นสำหรับวัตถุดิบที่ไม่รู้จัก
        return {"calories": 50, "protein": 2, "carbs": 10, "fat": 1, "fiber": 1,
                "vitamin_a": 10, "vitamin_c": 5, "vitamin_b1": 0.05, "vitamin_b2": 0.05,
                "calcium": 20, "iron": 0.5, "potassium": 100, "sodium": 10}
    
    def parse_ingredient(self, ingredient_text):
        """แยกปริมาณ หน่วย และชื่อวัตถุดิบ"""
        # ลบเครื่องหมาย - ถ้ามี
        clean_text = re.sub(r'^[-*•]\s*', '', ingredient_text.strip())
        
        # หาตัวเลข
        numbers = re.findall(r'\d+(?:\.\d+)?', clean_text)
        quantity = float(numbers[0]) if numbers else 100
        
        # หาหน่วย
        for unit in self.unit_conversion:
            if unit in clean_text:
                return quantity, unit, clean_text.replace(str(quantity), '').replace(unit, '').strip()
        
        return quantity, "กรัม", clean_text
    
    def convert_to_grams(self, quantity, unit):
        """แปลงเป็นกรัม"""
        return quantity * self.unit_conversion.get(unit, 1)
    
    def calculate_recipe_nutrition(self, ingredients_text):
        """คำนวณโภชนาการของสูตรอาหาร"""
        total_nutrition = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0, "fiber": 0,
                          "vitamin_a": 0, "vitamin_c": 0, "vitamin_b1": 0, "vitamin_b2": 0,
                          "calcium": 0, "iron": 0, "potassium": 0, "sodium": 0}
        
        ingredient_details = []
        
        if not ingredients_text:
            return {"total_nutrition": total_nutrition, "ingredient_details": ingredient_details}
        
        ingredients = [ing.strip() for ing in ingredients_text.split('\n') if ing.strip()]
        
        for ingredient_line in ingredients:
            if not ingredient_line:
                continue
                
            quantity, unit, ingredient_name = self.parse_ingredient(ingredient_line)
            grams = self.convert_to_grams(quantity, unit)
            
            nutrition_per_100g = self.get_nutrition_data(ingredient_name)
            factor = grams / 100
            
            ingredient_nutrition = {}
            for nutrient, value_per_100g in nutrition_per_100g.items():
                nutrient_value = value_per_100g * factor
                ingredient_nutrition[nutrient] = nutrient_value
                total_nutrition[nutrient] += nutrient_value
            
            ingredient_details.append({
                "name": ingredient_name,
                "quantity": quantity,
                "unit": unit,
                "grams": grams,
                "nutrition": ingredient_nutrition
            })
        
        return {"total_nutrition": total_nutrition, "ingredient_details": ingredient_details}

def load_data():
    """โหลดข้อมูลอาหารไทยจากไฟล์หรือสร้างข้อมูลตัวอย่าง"""
    # ลองโหลดจากไฟล์ที่มีอยู่
    possible_files = ["thai_food_processed.csv", "thai_food_sample.csv"]
    
    for file_path in possible_files:
        if os.path.exists(file_path):
            try:
                return pd.read_csv(file_path)
            except:
                continue
    
    # สร้างข้อมูลตัวอย่างหากไม่มีไฟล์
    sample_data = {
        'name': [
            'ผัดกะเพรา', 'ต้มยำกุ้ง', 'ส้มตำ', 'แกงเขียวหวาน', 'ผัดไทย',
            'ไข่เจียว', 'ข้าวผัด', 'ยำวุ้นเส้น', 'ลาบหมู', 'มะม่วงข้าวเหนียว',
            'กุ้งทาพริกไทยกระเทียม', 'ข้าวเม่าทอด', 'เปรี้ยวหวานไข่ม้วน', 'ไข่จ่อม', 'งบปลาทู'
        ],
        'ingredient': [
            '- เนื้อหมูสับ 200 กรัม\n- ใบกะเพรา 1 ถ้วย\n- พริกขี้หนู 5 เม็ด\n- กระเทียม 4 กลีบ\n- น้ำปลา 2 ช้อนโต๊ะ',
            '- กุ้งนาง 300 กรัม\n- เห็ดฟาง 100 กรัม\n- มะนาว 3 ผล\n- ใบมะกรูด 5 ใบ\n- น้ำปลา 3 ช้อนโต๊ะ',
            '- มะละกอดิบ 2 ถ้วย\n- มะเขือเทศ 3 ผล\n- ถั่วฝักยาว 10 เส้น\n- กุ้งแห้ง 2 ช้อนโต๊ะ\n- น้ำปลา 2 ช้อนโต๊ะ',
            '- เนื้อไก่ 400 กรัม\n- กะทิ 2 ถ้วย\n- มะเขือเปราะ 8 ผล\n- ใบโหระพา 1 ถ้วย\n- น้ำปลา 2 ช้อนโต๊ะ',
            '- เส้นจันท์ 200 กรัม\n- กุ้งสด 100 กรัม\n- เต้าหู้ 100 กรัม\n- ไข่ไก่ 2 ฟอง\n- น้ำปลา 2 ช้อนโต๊ะ',
            '- ไข่ไก่ 3 ฟอง\n- น้ำปลา 1 ช้อนชา\n- ต้นหอม 2 ต้น\n- น้ำมันหมู 2 ช้อนโต๊ะ',
            '- ข้าวสวย 3 ถ้วย\n- กุ้งสด 150 กรัม\n- ไข่ไก่ 2 ฟอง\n- หอมใหญ่ 1 หัว\n- น้ำปลา 2 ช้อนโต๊ะ',
            '- วุ้นเส้น 150 กรัม\n- กุ้งสด 200 กรัม\n- หมูสับ 100 กรัม\n- มะนาว 3 ผล\n- น้ำปลา 3 ช้อนโต๊ะ',
            '- เนื้อหมูสับ 300 กรัม\n- ข้าวคั่ว 3 ช้อนโต๊ะ\n- พริกแห้ง 8 เม็ด\n- หอมแดง 5 หัว\n- น้ำปลา 4 ช้อนโต๊ะ',
            '- ข้าวเหนียว 2 ถ้วย\n- มะม่วงสุก 2 ผล\n- กะทิ 1 ถ้วย\n- น้ำตาลปึก 3 ช้อนโต๊ะ\n- เกลือ 1/2 ช้อนชา',
            '- กุ้งนาง 4 ตัว\n- พริกไทย 5 เม็ด\n- กระเทียมกลีบใหญ่ 2 กลีบ\n- รากผักชี 5 ราก\n- น้ำมันหมู 2 ช้อนโต๊ะ',
            '- ข้าวเม่า 2 ถ้วย\n- ไข่ไก่ 3 ฟอง\n- หอมใหญ่ 1 หัว\n- กระเทียม 3 กลีบ\n- น้ำปลา 2 ช้อนโต๊ะ',
            '- ไข่ไก่ 4 ฟอง\n- น้ำตาลทราย 3 ช้อนโต๊ะ\n- น้ำส้มสายชู 2 ช้อนโต๊ะ\n- มะเขือเทศ 2 ผล\n- หอมใหญ่ 1 หัว',
            '- ไข่ไก่ 2 ฟอง\n- กุ้งสด 100 กรัม\n- หมูสับ 50 กรัม\n- น้ำปลา 1 ช้อนโต๊ะ\n- พริกไทย 1/2 ช้อนชา',
            '- ปลาทู 1 ตัว\n- มะเขือเทศ 2 ผล\n- หอมแดง 3 หัว\n- พริกแกง 2 ช้อนโต๊ะ\n- น้ำปลา 2 ช้อนโต๊ะ'
        ],
        'method': [
            'โขลกกระเทียมและพริกให้ละเอียด ผัดหมูสับจนสุก ใส่กะเพราและปรุงรส',
            'ต้มน้ำให้เดือด ใส่เครื่องต้มยำ เมื่อเดือดใส่กุ้งและเห็ด ปรุงรสและใส่มะนาว',
            'โขลกพริก กระเทียม ถั่วลิสง กุ้งแห้งให้หยาบ ใส่มะละกอและผักอื่นๆ ปรุงรสด้วยน้ำปลา น้ำตาล น้ำมะนาว',
            'คั่วน้ำพริกแกงเขียวหวานกับหัวกะทิ ใส่ไก่และกะทิ ปรุงรสและใส่ผัก',
            'แช่เส้นจันท์ให้นุ่ม ผัดกุ้งและเต้าหู้ ใส่ไข่และเส้น ปรุงรสและใส่ผัก',
            'ตอกไข่ใส่ชาม ใส่น้ำปลา ตีให้เข้ากัน ทอดในน้ำมันร้อนจนเหลือง',
            'ตั้งกะทะใส่น้ำมัน ผัดกระเทียมและหอมใหญ่ ใส่กุ้งและไข่ ใส่ข้าวผัดให้เข้ากัน',
            'แช่วุ้นเส้นให้นุ่ม ลวกกุ้งและหมู คลุกทุกอย่างกับน้ำยำ',
            'คั่วข้าวให้เหลืองหอม โขลกให้หยาบ ย่างพริกแห้ง ผสมเนื้อหมูกับเครื่องปรุง',
            'นึ่งข้าวเหนียวให้สุก หั่นมะม่วง ต้มกะทิกับน้ำตาลและเกลือ เสิร์ฟพร้อมกัน',
            'โขลกพริกไทย กระเทียม รากผักชี ใส่กุ้งทาส่วนผสม ทอดในน้ำมันร้อน',
            'ตั้งกะทะใส่น้ำมัน ผัดไข่ให้สุก ใส่ข้าวเม่าผัดให้เข้ากันกับเครื่องปรุง',
            'ตีไข่กับน้ำตาล ทำไข่ม้วน เตรียมน้ำเปรี้ยวหวาน ราดบนไข่ม้วน',
            'ผัดไข่ให้แข็งครึ่งหนึ่ง ใส่กุ้งและหมูสับ ปรุงรสด้วยน้ำปลาและพริกไทย',
            'ย่างปลาทูให้สุก คลุกกับน้ำพริกแกงและผักต่างๆ ปรุงรสให้เข้ากัน'
        ]
    }
    return pd.DataFrame(sample_data)

def improved_fuzzy_search(query, data, threshold=0.4):
    """ระบบค้นหาที่ปรับปรุงแล้ว"""
    query = query.lower().strip()
    matches = []
    
    # ลบคำที่ไม่จำเป็น
    stop_words = ["อาหาร", "เมนู", "สูตร", "วิธีทำ", "ทำ", "ปรุง", "อร่อย", "ง่าย", "เร็ว"]
    query_words = [word for word in query.split() if word not in stop_words and len(word) > 1]
    clean_query = " ".join(query_words) if query_words else query
    
    for idx, recipe_name in enumerate(data['name']):
        recipe_name_lower = recipe_name.lower()
        
        # 1. ความคล้ายคลึงแบบ sequence matching
        sequence_similarity = SequenceMatcher(None, clean_query, recipe_name_lower).ratio()
        
        # 2. การตรวจสอบคำที่ตรงกันทั้งหมด
        exact_match_score = 0
        if clean_query in recipe_name_lower:
            exact_match_score = len(clean_query) / len(recipe_name_lower)
        
        # 3. การตรวจสอบคำต่างๆ แยกกัน
        word_scores = []
        partial_scores = []
        
        for q_word in query_words:
            if len(q_word) <= 1:
                continue
                
            best_match_score = 0
            best_partial_score = 0
            
            recipe_words = recipe_name_lower.split()
            for r_word in recipe_words:
                # การจับคู่แบบเต็ม
                word_similarity = SequenceMatcher(None, q_word, r_word).ratio()
                if word_similarity >= 0.8:
                    best_match_score = max(best_match_score, word_similarity)
                
                # การจับคู่แบบบางส่วน (substring)
                elif len(q_word) >= 3:
                    if q_word in r_word:
                        best_partial_score = max(best_partial_score, 0.7)
                    elif r_word in q_word and len(r_word) >= 3:
                        best_partial_score = max(best_partial_score, 0.6)
            
            if best_match_score > 0:
                word_scores.append(best_match_score)
            elif best_partial_score > 0:
                partial_scores.append(best_partial_score)
        
        # คำนวณคะแนนจากการจับคู่คำ
        word_match_score = 0
        if word_scores:
            word_match_score = sum(word_scores) / len(query_words)
        elif partial_scores:
            word_match_score = sum(partial_scores) / len(query_words) * 0.8
        
        # 4. คำนวณคะแนนรวมแบบถ่วงน้ำหนัก
        final_scores = [
            sequence_similarity * 0.3,
            exact_match_score * 0.9,
            word_match_score * 0.7
        ]
        
        final_score = max(final_scores)
        
        # ปรับคะแนนตามความยาวของชื่อเมนู
        if final_score > 0:
            length_factor = 1.0
            if len(recipe_name_lower) <= 10 and exact_match_score > 0:
                length_factor = 1.2
            elif len(recipe_name_lower) > 20:
                length_factor = 0.9
            
            final_score = min(final_score * length_factor, 1.0)
        
        # เก็บเฉพาะที่มีคะแนนเกินเกณฑ์
        if final_score >= threshold:
            matches.append((recipe_name, final_score, idx))
    
    # เรียงลำดับตามความคล้ายคลึง
    matches.sort(key=lambda x: x[1], reverse=True)
    
    return matches[:5]  # จำกัดผลลัพธ์

def get_similarity_badge_class(similarity):
    """ได้รับ CSS class สำหรับ badge ตามคะแนนความคล้ายคลึง"""
    if similarity >= 0.7:
        return "similarity-badge-high"
    elif similarity >= 0.5:
        return "similarity-badge-medium"
    else:
        return "similarity-badge-low"

def format_ingredients(ingredients_text):
    """จัดรูปแบบรายการวัตถุดิบ"""
    if not ingredients_text:
        return "<p>ไม่มีข้อมูลวัตถุดิบ</p>"
        
    ingredients = ingredients_text.split('\n')
    formatted = "<ul style='margin: 0; padding-left: 1.5rem;'>"
    for item in ingredients:
        if item.strip():
            clean_item = item.strip().lstrip('- ')
            formatted += f"<li style='margin: 0.2rem 0;'>{clean_item}</li>"
    formatted += "</ul>"
    return formatted

def format_cooking_method(method_text):
    """จัดรูปแบบวิธีทำ"""
    if not method_text:
        return "<p>ไม่มีข้อมูลวิธีทำ</p>"
        
    sentences = re.split(r'(?<=[ๆ.।])\s+|(?<=\w)\s{2,}', method_text)
    formatted = "<ol style='margin: 0; padding-left: 1.5rem;'>"
    for sentence in sentences:
        if sentence.strip():
            formatted += f"<li style='margin: 0.3rem 0;'>{sentence.strip()}</li>"
    formatted += "</ol>"
    return formatted

def display_nutrition_chart(nutrition_data, recipe_name):
    """แสดงกราฟโภชนาการ"""
    total_nutrition = nutrition_data['total_nutrition']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('สารอาหารหลัก (กรัม)', 'วิตามิน (mg/IU)', 'แร่ธาตุ (mg)', 'แคลอรี่และใยอาหาร'),
        specs=[[{"type": "pie"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "indicator"}]]
    )
    
    # กราฟ Macronutrients (pie chart)
    macro_labels = ['โปรตีน', 'คาร์โบไฮเดรต', 'ไขมัน']
    macro_values = [total_nutrition['protein'], total_nutrition['carbs'], total_nutrition['fat']]
    macro_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    fig.add_trace(go.Pie(
        labels=macro_labels, 
        values=macro_values,
        marker_colors=macro_colors,
        hole=0.3
    ), row=1, col=1)
    
    # กราฟวิตามิน
    vitamin_labels = ['วิตามิน A', 'วิตามิน C', 'วิตามิน B1', 'วิตามิน B2']
    vitamin_values = [
        total_nutrition['vitamin_a'], total_nutrition['vitamin_c'],
        total_nutrition['vitamin_b1']*1000, total_nutrition['vitamin_b2']*1000
    ]
    
    fig.add_trace(go.Bar(
        x=vitamin_labels,
        y=vitamin_values,
        marker_color=['#FF9F43', '#10AC84', '#5F27CD', '#00D2D3'],
        name='วิตามิน'
    ), row=1, col=2)
    
    # กราฟแร่ธาตุ
    mineral_labels = ['แคลเซียม', 'เหล็ก', 'โปแตสเซียม', 'โซเดียม']
    mineral_values = [
        total_nutrition['calcium'], total_nutrition['iron'],
        total_nutrition['potassium'], total_nutrition['sodium']
    ]
    
    fig.add_trace(go.Bar(
        x=mineral_labels,
        y=mineral_values,
        marker_color=['#2D3436', '#636E72', '#00B894', '#E17055'],
        name='แร่ธาตุ'
    ), row=2, col=1)
    
    # แสดงแคลอรี่และใยอาหาร
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=total_nutrition['calories'],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"แคลอรี่<br><span style='font-size:0.8em;color:gray'>ใยอาหาร: {total_nutrition['fiber']:.1f}g</span>"},
        gauge={
            'axis': {'range': [None, 500]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 200], 'color': "lightgray"},
                {'range': [200, 350], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 400
            }
        }
    ), row=2, col=2)
    
    fig.update_layout(
        title_text=f"ข้อมูลโภชนาการ - {recipe_name}",
        showlegend=False,
        height=800
    )
    
    return fig

def display_nutrition_info(nutrition_data, recipe_name, show_charts=True):
    """แสดงข้อมูลโภชนาการแบบครอบคลุม"""
    total_nutrition = nutrition_data['total_nutrition']
    
    if show_charts:
        fig = display_nutrition_chart(nutrition_data, recipe_name)
        st.plotly_chart(fig, use_container_width=True)
    
    # แสดงข้อมูลโภชนาการในรูปแบบการ์ด
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="nutrition-card">
            <h4>🔥 พลังงานและสารอาหารหลัก</h4>
            <p><strong>แคลอรี่:</strong> {total_nutrition['calories']:.1f} kcal</p>
            <p><strong>โปรตีน:</strong> {total_nutrition['protein']:.1f} g</p>
            <p><strong>คาร์โบไฮเดรต:</strong> {total_nutrition['carbs']:.1f} g</p>
            <p><strong>ไขมัน:</strong> {total_nutrition['fat']:.1f} g</p>
            <p><strong>ใยอาหาร:</strong> {total_nutrition['fiber']:.1f} g</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="vitamin-card">
            <h4>💊 วิตามิน</h4>
            <p><strong>วิตามิน A:</strong> {total_nutrition['vitamin_a']:.1f} IU</p>
            <p><strong>วิตามิน C:</strong> {total_nutrition['vitamin_c']:.1f} mg</p>
            <p><strong>วิตามิน B1:</strong> {total_nutrition['vitamin_b1']:.2f} mg</p>
            <p><strong>วิตามิน B2:</strong> {total_nutrition['vitamin_b2']:.2f} mg</p>
        </div>
        """, unsafe_allow_html=True)

def display_recipe_with_nutrition(recipe, nutrition_data, similarity_score=None):
    """แสดงสูตรอาหารพร้อมข้อมูลโภชนาการ"""
    
    # แสดงชื่อเมนูพร้อมคะแนนความคล้ายคลึง
    title_html = f"### 🍽️ {recipe['name']}"
    if similarity_score is not None:
        similarity_percent = similarity_score * 100
        badge_class = get_similarity_badge_class(similarity_score)
        title_html += f'<span class="{badge_class}">{similarity_percent:.0f}% ตรง</span>'
    
    st.markdown(title_html, unsafe_allow_html=True)
    
    # สร้าง tabs สำหรับแยกข้อมูล
    tab1, tab2, tab3 = st.tabs(["📝 สูตรอาหาร", "📊 โภชนาการ", "🔍 รายละเอียด"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 🥬 วัตถุดิบ")
            st.markdown(format_ingredients(recipe["ingredient"]), unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 👨‍🍳 วิธีทำ")
            st.markdown(format_cooking_method(recipe["method"]), unsafe_allow_html=True)
    
    with tab2:
        if nutrition_data:
            display_nutrition_info(nutrition_data, recipe['name'], show_charts=True)
        else:
            st.info("ไม่มีข้อมูลโภชนาการ")
    
    with tab3:
        if nutrition_data:
            st.markdown("#### 🔬 รายละเอียดวัตถุดิบแต่ละชนิด")
            
            ingredient_df = []
            for ingredient in nutrition_data['ingredient_details']:
                ingredient_df.append({
                    'วัตถุดิบ': ingredient['name'],
                    'ปริมาณ': f"{ingredient['quantity']} {ingredient['unit']}",
                    'น้ำหนัก (กรัม)': f"{ingredient['grams']:.1f}",
                    'แคลอรี่': f"{ingredient['nutrition']['calories']:.1f}",
                    'โปรตีน (g)': f"{ingredient['nutrition']['protein']:.1f}",
                    'ไขมัน (g)': f"{ingredient['nutrition']['fat']:.1f}"
                })
            
            if ingredient_df:
                df = pd.DataFrame(ingredient_df)
                st.dataframe(df, use_container_width=True)
        else:
            st.info("ไม่มีข้อมูลรายละเอียด")

def create_error_message(search_query, available_recipes):
    """สร้างข้อความแสดงข้อผิดพลาดแบบปลอดภัย"""
    message_parts = []
    message_parts.append("❌ ไม่พบสูตรอาหารที่ตรงกับ")
    message_parts.append(f"'{search_query}'")
    message_parts.append("\n\n💡 **คำแนะนำสำหรับการค้นหา:**")
    message_parts.append("- ลองใช้คำค้นหาที่ง่ายกว่า เช่น 'ไข่เจียว' แทน 'วิธีทำไข่เจียว'")
    message_parts.append("- ตรวจสอบการสะกดคำภาษาไทย")
    message_parts.append("- ลองค้นหาด้วยเมนูที่มีชื่อสั้นๆ")
    message_parts.append("- ปรับค่าความเคร่งครัดในการค้นหาในแถบด้านซ้าย (ลดค่าลง)")
    message_parts.append("\n🍽️ **เมนูที่มีในระบบ:**")
    message_parts.append(", ".join(available_recipes))
    
    return " ".join(message_parts)

def main():
    """ฟังก์ชันหลักของแอปพลิเคชัน"""
    
    # แสดงหัวข้อแอป
    st.markdown('<h1 class="main-title">🍲 Thai Food Recipe Chatbot</h1>', unsafe_allow_html=True)
    st.markdown("### 🥘 ระบบค้นหาสูตรอาหารไทย")
    
    # โหลดข้อมูล
    with st.spinner("กำลังเริ่มต้นระบบ..."):
        data = load_data()
        nutrition_api = SimplifiedNutritionAPI()
    
    # แสดงสถิติเบื้องต้น
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📖 จำนวนสูตร", len(data))
    with col2:
        st.metric("🤖 ระบบค้นหา", "✨ ปรับปรุงแล้ว")
    with col3:
        st.metric("📊 โภชนาการ", "✅ พร้อม")
    with col4:
        st.metric("🚀 สถานะ", "🟢 ทำงานได้")
    
    # การตั้งค่าใน sidebar
    st.sidebar.title("🔧 การตั้งค่า")
    fuzzy_threshold = st.sidebar.slider(
        "ความเคร่งครัดในการค้นหา",
        min_value=0.2,
        max_value=0.8,
        value=0.4,
        step=0.1,
        help="ค่าต่ำ = หาได้ง่ายแต่อาจไม่ตรง, ค่าสูง = หาได้ยากแต่ตรงมาก"
    )
    
    # ตัวอย่างคำค้นหา
    st.markdown("#### 💡 ลองค้นหาเมนูเหล่านี้:")
    
    sample_recipes = data['name'].head(5).tolist()
    cols = st.columns(len(sample_recipes))
    for i, recipe_name in enumerate(sample_recipes):
        if cols[i].button(recipe_name, key=f"example_{i}"):
            st.session_state['search_query'] = recipe_name
    
    # เริ่มต้น session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "search_query" not in st.session_state:
        st.session_state.search_query = ""
    
    # แสดงประวัติการสนทนา
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "recipe" in message:
                recipe = message["recipe"]
                nutrition_data = message.get("nutrition_data")
                similarity_score = message.get("similarity_score")
                display_recipe_with_nutrition(recipe, nutrition_data, similarity_score)
            else:
                st.markdown(message["content"])
    
    # ช่องค้นหา
    search_query = st.session_state.get('search_query', '')
    if prompt := st.chat_input("ค้นหาสูตรอาหาร เช่น 'ผัดกะเพรา' หรือ 'ไข่เจียว'... (ระบบค้นหาปรับปรุงใหม่)", key="main_chat"):
        search_query = prompt
        st.session_state.search_query = ""
    
    if search_query:
        # เพิ่มข้อความของผู้ใช้
        st.session_state.messages.append({"role": "user", "content": search_query})
        
        # แสดงข้อความของผู้ใช้
        with st.chat_message("user"):
            st.markdown(search_query)
        
        # ประมวลผลและตอบกลับ
        with st.chat_message("assistant"):
            with st.spinner("กำลังค้นหาด้วยระบบที่ปรับปรุงใหม่..."):
                # ค้นหาสูตรอาหาร
                results = improved_fuzzy_search(search_query, data, threshold=fuzzy_threshold)
                
                if results:
                    best_match = results[0]
                    recipe_name, similarity, recipe_idx = best_match
                    
                    # ดึงข้อมูลสูตร
                    recipe = {
                        'name': recipe_name,
                        'ingredient': data.iloc[recipe_idx]['ingredient'],
                        'method': data.iloc[recipe_idx]['method']
                    }
                    
                    # คำนวณโภชนาการ
                    nutrition_data = nutrition_api.calculate_recipe_nutrition(recipe['ingredient'])
                    
                    response = f"🎯 พบสูตรอาหารที่ตรงกับการค้นหา: **{recipe_name}**"
                    st.markdown(response)
                    
                    # แสดงสูตรและโภชนาการ
                    display_recipe_with_nutrition(recipe, nutrition_data, similarity)
                    
                    # เพิ่มการแนะนำเพิ่มเติม
                    if len(results) > 1:
                        st.markdown("#### 🔍 เมนูอื่นที่น่าสนใจ:")
                        other_results = results[1:min(4, len(results))]
                        
                        cols = st.columns(len(other_results))
                        for i, (other_name, other_sim, other_idx) in enumerate(other_results):
                            with cols[i]:
                                similarity_percent = other_sim * 100
                                if st.button(f"🍽️ {other_name}\n({similarity_percent:.0f}% ตรง)", key=f"other_{i}"):
                                    st.session_state.search_query = other_name
                                    st.rerun()
                    
                    # บันทึกข้อความตอบกลับ
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response, 
                        "recipe": recipe,
                        "nutrition_data": nutrition_data,
                        "similarity_score": similarity
                    })
                else:
                    # สร้างข้อความแสดงข้อผิดพลาดแบบปลอดภัย
                    available_recipes = data['name'].head(10).tolist()
                    response = create_error_message(search_query, available_recipes)
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
