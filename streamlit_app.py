import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import re
import requests
import json
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# การตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Thai Food Recipe Chatbot - Advanced",
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
    
    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: bold;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .nutrition-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 15px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
    }
    
    .vitamin-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 15px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
    }
    
    .mineral-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 15px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
    }
    
    .ingredient-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
        background: white;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .ingredient-table th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 8px;
        text-align: center;
        font-weight: 600;
        font-size: 0.9rem;
        border: none;
    }
    
    .ingredient-table td {
        padding: 10px 8px;
        text-align: center;
        border-bottom: 1px solid #f0f0f0;
        font-size: 0.85rem;
    }
    
    .ingredient-table tr:nth-child(even) {
        background-color: #f8f9fa;
    }
    
    .ingredient-table tr:hover {
        background-color: #e3f2fd;
        transition: background-color 0.3s;
    }
    
    .method-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .method-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e0e0e0;
    }
    
    .method-steps ol {
        line-height: 1.8;
        padding-left: 1.5rem;
    }
    
    .method-steps li {
        margin: 0.8rem 0;
        padding: 0.5rem;
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .method-paragraph {
        line-height: 1.8;
        text-align: justify;
        padding: 1rem;
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .note-section {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .note-title {
        font-weight: 600;
        color: #e65100;
        margin-bottom: 0.5rem;
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
    
    .ingredients-list {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .ingredients-list ul {
        list-style-type: none;
        padding-left: 0;
    }
    
    .ingredients-list li {
        margin: 0.5rem 0;
        padding: 0.5rem;
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        position: relative;
        padding-left: 2rem;
    }
    
    .ingredients-list li:before {
        content: "🥬";
        position: absolute;
        left: 0.5rem;
        top: 0.5rem;
    }
    
    .auto-scroll-button {
        position: fixed !important;
        bottom: 25px !important;
        right: 25px !important;
        z-index: 999999 !important;
        background: linear-gradient(135deg, #4CAF50, #45a049) !important;
        color: white !important;
        border: none !important;
        border-radius: 50% !important;
        width: 60px !important;
        height: 60px !important;
        cursor: pointer !important;
        box-shadow: 0 4px 20px rgba(76, 175, 80, 0.4) !important;
        font-size: 24px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    .auto-scroll-button:hover {
        background: linear-gradient(135deg, #45a049, #3d8b40) !important;
        transform: scale(1.1) translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(76, 175, 80, 0.6) !important;
    }
</style>
""", unsafe_allow_html=True)

class AdvancedNutritionAPI:
    """คลาสสำหรับจัดการข้อมูลโภชนาการขั้นสูง"""
    
    def __init__(self):
        self.nutrition_db = self.load_nutrition_database()
        
        # หน่วยแปลงที่พบบ่อยในอาหารไทย
        self.unit_conversion = {
            "ช้อนโต๊ะ": 15, "ชต": 15, "tbsp": 15,
            "ช้อนชา": 5, "ชช": 5, "tsp": 5,
            "ถ้วย": 240, "cup": 240,
            "ถ้วยชา": 150, "แก้ว": 200,
            "ลิตร": 1000, "l": 1000,
            "มิลลิลิตร": 1, "มล": 1, "ml": 1,
            "กิโลกรัม": 1000, "กก": 1000, "kg": 1000,
            "กรัม": 1, "g": 1, "gram": 1,
            "ตัว": 100, "ฟอง": 50, "หัว": 50, "กลีบ": 3,
            "ต้น": 30, "ใบ": 2, "เม็ด": 0.5, "แว่น": 2,
            "ผล": 150, "ราก": 5, "ท่อน": 20, "ก้อน": 30,
            "หยิบ": 5, "ข้อมือ": 100
        }
        
        # สัดส่วนที่บริโภคจริง
        self.consumption_ratio = {
            "น้ำมันหมู": 0.25, "น้ำมันพืช": 0.25,
            "น้ำมันมะพร้าว": 0.25, "กะทิ": 0.85,
            "น้ำปลา": 1.0, "น้ำตาล": 1.0, "เกลือ": 1.0
        }

    def load_nutrition_database(self) -> Dict:
        """โหลดฐานข้อมูลโภชนาการ"""
        # ลองโหลดจากไฟล์ CSV ก่อน
        if os.path.exists("thai_ingredients_nutrition.csv"):
            try:
                df = pd.read_csv("thai_ingredients_nutrition.csv")
                nutrition_db = {}
                for _, row in df.iterrows():
                    nutrition_db[row['ingredient']] = {
                        "calories": row.get('calories', 0),
                        "protein": row.get('protein', 0),
                        "carbs": row.get('carbs', 0),
                        "fat": row.get('fat', 0),
                        "fiber": row.get('fiber', 0),
                        "vitamin_a": row.get('vitamin_a', 0),
                        "vitamin_c": row.get('vitamin_c', 0),
                        "vitamin_b1": row.get('vitamin_b1', 0),
                        "vitamin_b2": row.get('vitamin_b2', 0),
                        "calcium": row.get('calcium', 0),
                        "iron": row.get('iron', 0),
                        "potassium": row.get('potassium', 0),
                        "sodium": row.get('sodium', 0)
                    }
                return nutrition_db
            except:
                pass
        
        # ใช้ข้อมูลเริ่มต้น
        return self.get_default_nutrition_database()

    def get_default_nutrition_database(self) -> Dict:
        """ฐานข้อมูลโภชนาการเริ่มต้น (ต่อ 100 กรัม)"""
        return {
            "ข้าวสาร": {
                "calories": 130, "protein": 2.7, "carbs": 28, "fat": 0.3, "fiber": 0.4,
                "vitamin_a": 0, "vitamin_c": 0, "vitamin_b1": 0.07, "vitamin_b2": 0.02,
                "calcium": 10, "iron": 0.8, "potassium": 115, "sodium": 5
            },
            "ไข่ไก่": {
                "calories": 155, "protein": 13, "carbs": 1.1, "fat": 11, "fiber": 0,
                "vitamin_a": 540, "vitamin_c": 0, "vitamin_b1": 0.04, "vitamin_b2": 0.42,
                "calcium": 56, "iron": 1.75, "potassium": 138, "sodium": 124
            },
            "ไข่เป็ด": {
                "calories": 185, "protein": 13, "carbs": 1.4, "fat": 14, "fiber": 0,
                "vitamin_a": 674, "vitamin_c": 0, "vitamin_b1": 0.11, "vitamin_b2": 0.44,
                "calcium": 64, "iron": 2.7, "potassium": 222, "sodium": 146
            },
            "กุ้งนาง": {
                "calories": 99, "protein": 18, "carbs": 0.2, "fat": 1.4, "fiber": 0,
                "vitamin_a": 54, "vitamin_c": 2.1, "vitamin_b1": 0.02, "vitamin_b2": 0.04,
                "calcium": 70, "iron": 0.5, "potassium": 259, "sodium": 111
            },
            "เนื้อหมู": {
                "calories": 242, "protein": 27, "carbs": 0, "fat": 14, "fiber": 0,
                "vitamin_a": 2, "vitamin_c": 0.7, "vitamin_b1": 0.66, "vitamin_b2": 0.23,
                "calcium": 19, "iron": 0.87, "potassium": 423, "sodium": 62
            },
            "เนื้อไก่": {
                "calories": 165, "protein": 31, "carbs": 0, "fat": 3.6, "fiber": 0,
                "vitamin_a": 21, "vitamin_c": 1.6, "vitamin_b1": 0.07, "vitamin_b2": 0.12,
                "calcium": 15, "iron": 1.3, "potassium": 256, "sodium": 82
            },
            "ปลาช่อน": {
                "calories": 112, "protein": 18.7, "carbs": 0, "fat": 3.6, "fiber": 0,
                "vitamin_a": 45, "vitamin_c": 0.9, "vitamin_b1": 0.02, "vitamin_b2": 0.11,
                "calcium": 89, "iron": 0.9, "potassium": 358, "sodium": 54
            },
            "น้ำมันพืช": {
                "calories": 884, "protein": 0, "carbs": 0, "fat": 100, "fiber": 0,
                "vitamin_a": 0, "vitamin_c": 0, "vitamin_b1": 0, "vitamin_b2": 0,
                "calcium": 0, "iron": 0, "potassium": 0, "sodium": 0
            },
            "กระเทียม": {
                "calories": 149, "protein": 6.4, "carbs": 33, "fat": 0.5, "fiber": 2.1,
                "vitamin_a": 9, "vitamin_c": 31, "vitamin_b1": 0.2, "vitamin_b2": 0.11,
                "calcium": 181, "iron": 1.7, "potassium": 401, "sodium": 17
            },
            "หอมแดง": {
                "calories": 40, "protein": 1.1, "carbs": 9.3, "fat": 0.1, "fiber": 1.7,
                "vitamin_a": 2, "vitamin_c": 7.4, "vitamin_b1": 0.05, "vitamin_b2": 0.03,
                "calcium": 23, "iron": 0.21, "potassium": 146, "sodium": 4
            },
            "ผักชี": {
                "calories": 23, "protein": 2.1, "carbs": 3.7, "fat": 0.5, "fiber": 2.8,
                "vitamin_a": 3377, "vitamin_c": 27, "vitamin_b1": 0.07, "vitamin_b2": 0.16,
                "calcium": 67, "iron": 1.77, "potassium": 521, "sodium": 46
            },
            "พริกไทย": {
                "calories": 251, "protein": 10.4, "carbs": 64, "fat": 3.3, "fiber": 25,
                "vitamin_a": 547, "vitamin_c": 0, "vitamin_b1": 0.11, "vitamin_b2": 0.18,
                "calcium": 443, "iron": 9.7, "potassium": 1329, "sodium": 20
            },
            "น้ำปลา": {
                "calories": 42, "protein": 5.8, "carbs": 1.5, "fat": 0.8, "fiber": 0,
                "vitamin_a": 0, "vitamin_c": 0, "vitamin_b1": 0.03, "vitamin_b2": 0.22,
                "calcium": 85, "iron": 2.03, "potassium": 84, "sodium": 6976
            },
            "น้ำตาลทราย": {
                "calories": 387, "protein": 0, "carbs": 100, "fat": 0, "fiber": 0,
                "vitamin_a": 0, "vitamin_c": 0, "vitamin_b1": 0, "vitamin_b2": 0,
                "calcium": 1, "iron": 0.01, "potassium": 2, "sodium": 1
            },
            "มะนาว": {
                "calories": 29, "protein": 0.7, "carbs": 9.3, "fat": 0.2, "fiber": 2.8,
                "vitamin_a": 22, "vitamin_c": 53, "vitamin_b1": 0.03, "vitamin_b2": 0.02,
                "calcium": 33, "iron": 0.6, "potassium": 138, "sodium": 2
            },
            "กะทิ": {
                "calories": 230, "protein": 2.3, "carbs": 6, "fat": 24, "fiber": 2.2,
                "vitamin_a": 0, "vitamin_c": 2.8, "vitamin_b1": 0.03, "vitamin_b2": 0,
                "calcium": 16, "iron": 1.64, "potassium": 263, "sodium": 15
            },
            "เกลือ": {
                "calories": 0, "protein": 0, "carbs": 0, "fat": 0, "fiber": 0,
                "vitamin_a": 0, "vitamin_c": 0, "vitamin_b1": 0, "vitamin_b2": 0,
                "calcium": 24, "iron": 0.33, "potassium": 8, "sodium": 38758
            }
        }

    def normalize_ingredient_name(self, ingredient: str) -> str:
        """ปรับแต่งชื่อวัตถุดิบให้เป็นมาตรฐาน"""
        ingredient = re.sub(r'\d+.*', '', ingredient)
        ingredient = re.sub(r'[^\u0E00-\u0E7Fa-zA-Z\s]', '', ingredient)
        ingredient = ingredient.strip()
        
        # แปลงคำพ้องความหมาย
        synonyms = {
            "กุ้งนาง": "กุ้งนาง", "กุ้งตะเข็บ": "กุ้งนาง",
            "เนื้อหมู": "เนื้อหมู", "หมูสับ": "เนื้อหมู",
            "เนื้อไก่": "เนื้อไก่", "ไก่สับ": "เนื้อไก่",
            "หอมหัวใหญ่": "หอมแดง", "น้ำตาลทราย": "น้ำตาลทราย"
        }
        
        for synonym, standard in synonyms.items():
            if synonym in ingredient:
                ingredient = ingredient.replace(synonym, standard)
        
        return ingredient

    def extract_quantity_and_unit(self, ingredient_text: str) -> Tuple[float, str, str]:
        """แยกปริมาณ หน่วย และชื่อวัตถุดิบ"""
        ingredient_text = ingredient_text.strip()
        
        patterns = [
            r'(\d+(?:\.\d+)?(?:/\d+)?)\s*([ก-๙a-zA-Z]+)\s*(.+)',
            r'(.+?)\s+(\d+(?:\.\d+)?(?:/\d+)?)\s*([ก-๙a-zA-Z]+)(?:\s|$)',
            r'(.+)'
        ]
        
        for pattern in patterns:
            match = re.match(pattern, ingredient_text)
            if match:
                groups = match.groups()
                
                if len(groups) == 3 and self.is_number(groups[0]):
                    quantity = self.parse_number(groups[0])
                    unit = groups[1]
                    ingredient = groups[2]
                elif len(groups) == 3 and self.is_number(groups[1]):
                    ingredient = groups[0]
                    quantity = self.parse_number(groups[1])
                    unit = groups[2]
                else:
                    ingredient = groups[0]
                    quantity = self.estimate_default_quantity(ingredient)
                    unit = self.estimate_default_unit(ingredient)
                break
        else:
            ingredient = ingredient_text
            quantity = 100
            unit = "กรัม"
        
        return quantity, unit, self.normalize_ingredient_name(ingredient)

    def is_number(self, text: str) -> bool:
        """ตรวจสอบว่าข้อความเป็นตัวเลขหรือไม่"""
        try:
            self.parse_number(text)
            return True
        except:
            return False

    def parse_number(self, text: str) -> float:
        """แปลงข้อความเป็นตัวเลข"""
        text = text.strip()
        if '/' in text:
            parts = text.split('/')
            if len(parts) == 2:
                return float(parts[0]) / float(parts[1])
        return float(text)

    def estimate_default_quantity(self, ingredient: str) -> float:
        """ประมาณปริมาณเริ่มต้น"""
        ingredient_lower = ingredient.lower()
        defaults = {
            "น้ำมัน": 2, "น้ำปลา": 1.5, "น้ำตาล": 1, "เกลือ": 0.5,
            "กระเทียม": 3, "หอม": 2, "ผักชี": 2, "ไข่": 2,
            "เนื้อ": 200, "หมู": 200, "ไก่": 250, "กุ้ง": 150, "ปลา": 300
        }
        for key, value in defaults.items():
            if key in ingredient_lower:
                return value
        return 100

    def estimate_default_unit(self, ingredient: str) -> str:
        """ประมาณหน่วยเริ่มต้น"""
        ingredient_lower = ingredient.lower()
        unit_map = {
            "น้ำมัน": "ช้อนโต๊ะ", "น้ำปลา": "ช้อนโต๊ะ", "น้ำตาล": "ช้อนชา",
            "กระเทียม": "กลีบ", "หอม": "หัว", "ผักชี": "ต้น", "ไข่": "ฟอง"
        }
        for key, unit in unit_map.items():
            if key in ingredient_lower:
                return unit
        return "กรัม"

    def convert_to_grams(self, quantity: float, unit: str, ingredient: str) -> float:
        """แปลงปริมาณเป็นกรัม"""
        if unit.lower() in ["กรัม", "g", "gram"]:
            return quantity
        
        if unit in self.unit_conversion:
            base_amount = quantity * self.unit_conversion[unit]
            
            if unit in ["ช้อนโต๊ะ", "ช้อนชา", "ถ้วย", "มล", "ลิตร"]:
                density_map = {
                    "น้ำมัน": 0.92, "กะทิ": 0.95, "น้ำปลา": 1.1,
                    "น้ำตาล": 1.6, "เกลือ": 2.16
                }
                density = 1.0
                ingredient_lower = ingredient.lower()
                for key, value in density_map.items():
                    if key in ingredient_lower:
                        density = value
                        break
                return base_amount * density
            else:
                return base_amount
        
        return quantity

    def get_nutrition_data(self, ingredient: str) -> Optional[Dict]:
        """ดึงข้อมูลโภชนาการสำหรับวัตถุดิบ"""
        normalized_ingredient = self.normalize_ingredient_name(ingredient)
        
        # ค้นหาในฐานข้อมูล
        for db_ingredient, nutrition in self.nutrition_db.items():
            if (db_ingredient.lower() in normalized_ingredient.lower() or 
                normalized_ingredient.lower() in db_ingredient.lower()):
                return nutrition.copy()
        
        # ประมาณค่าตามประเภท
        return self.estimate_nutrition_by_type(normalized_ingredient)

    def estimate_nutrition_by_type(self, ingredient: str) -> Dict:
        """ประมาณค่าโภชนาการตามประเภทวัตถุดิบ"""
        ingredient_lower = ingredient.lower()
        
        if any(keyword in ingredient_lower for keyword in ["เนื้อ", "หมู", "ไก่", "ปลา", "กุ้ง"]):
            return {
                "calories": 150, "protein": 20, "carbs": 0, "fat": 6, "fiber": 0,
                "vitamin_a": 20, "vitamin_c": 1, "vitamin_b1": 0.1, "vitamin_b2": 0.15,
                "calcium": 20, "iron": 1.5, "potassium": 250, "sodium": 50
            }
        elif any(keyword in ingredient_lower for keyword in ["ผัก", "ใบ", "ต้น"]):
            return {
                "calories": 25, "protein": 2, "carbs": 5, "fat": 0.2, "fiber": 2,
                "vitamin_a": 1000, "vitamin_c": 30, "vitamin_b1": 0.05, "vitamin_b2": 0.08,
                "calcium": 50, "iron": 1, "potassium": 200, "sodium": 10
            }
        else:
            return {
                "calories": 50, "protein": 2, "carbs": 10, "fat": 1, "fiber": 1,
                "vitamin_a": 10, "vitamin_c": 5, "vitamin_b1": 0.05, "vitamin_b2": 0.05,
                "calcium": 20, "iron": 0.5, "potassium": 100, "sodium": 10
            }

    def calculate_recipe_nutrition(self, ingredients_text: str, adjust_consumption: bool = True) -> Dict:
        """คำนวณค่าโภชนาการของสูตรอาหาร"""
        
        total_nutrition = {
            "calories": 0, "protein": 0, "carbs": 0, "fat": 0, "fiber": 0,
            "vitamin_a": 0, "vitamin_c": 0, "vitamin_b1": 0, "vitamin_b2": 0,
            "calcium": 0, "iron": 0, "potassium": 0, "sodium": 0
        }
        
        ingredient_details = []
        ingredients = [ing.strip() for ing in ingredients_text.split('\n') if ing.strip()]
        
        for ingredient_line in ingredients:
            ingredient_text = re.sub(r'^[-*•]\s*', '', ingredient_line).strip()
            if not ingredient_text:
                continue
            
            quantity, unit, ingredient_name = self.extract_quantity_and_unit(ingredient_text)
            grams = self.convert_to_grams(quantity, unit, ingredient_name)
            
            if adjust_consumption:
                consumption_factor = self.consumption_ratio.get(ingredient_name, 1.0)
                effective_grams = grams * consumption_factor
            else:
                effective_grams = grams
            
            nutrition_per_100g = self.get_nutrition_data(ingredient_name)
            
            if nutrition_per_100g:
                factor = effective_grams / 100
                ingredient_nutrition = {}
                
                for nutrient, value_per_100g in nutrition_per_100g.items():
                    nutrient_value = value_per_100g * factor
                    ingredient_nutrition[nutrient] = nutrient_value
                    total_nutrition[nutrient] += nutrient_value
                
                ingredient_details.append({
                    "name": ingredient_name,
                    "original_text": ingredient_text,
                    "quantity": quantity,
                    "unit": unit,
                    "grams": grams,
                    "effective_grams": effective_grams,
                    "consumption_factor": consumption_factor if adjust_consumption else 1.0,
                    "nutrition": ingredient_nutrition
                })
        
        return {
            "total_nutrition": total_nutrition,
            "ingredient_details": ingredient_details
        }

class AdvancedFuzzyMatcher:
    """คลาสสำหรับการจับคู่ข้อความขั้นสูง"""
    
    def __init__(self):
        # เมนูอาหารไทยยอดนิยม
        self.thai_menu_variations = {
            'ไข่เจียว': ['ไข่เยียว', 'ไข่เจียวฟู', 'ไข่เจียวกรอบ', 'ไขเจียว'],
            'ไข่ดาว': ['ไข่ดาวกรอบ', 'ไข่ทอด', 'ไขดาว'],
            'ผัดกะเพรา': ['กะเพราผัด', 'ผัดใบกะเพรา', 'กะเพราหมูสับ', 'ผัดกระเพรา'],
            'ต้มยำกุ้ง': ['ต้มยำ', 'ต้มยํา', 'ต้มยำใส', 'ต้มยำน้ำใส'],
            'ส้มตำ': ['ส้มตํา', 'ส้มตำไทย', 'ส้มตำอีสาน'],
            'แกงเขียวหวาน': ['แกงเขียวหวานไก่', 'แกงเขียวหวานหมู', 'เขียวหวาน'],
            'ผัดไทย': ['ผัดไท', 'ผัดไทยกุ้ง', 'ผัดไทยหมู'],
            'ยำวุ้นเส้น': ['ยำวุนเส้น', 'วุ้นเส้นยำ', 'ยำวุ้น'],
            'ลาบหมู': ['ลาบ', 'ลาบอีสาน', 'ลาบหมูสับ'],
            'มะม่วงข้าวเหนียว': ['ข้าวเหนียวมะม่วง', 'มะม่วงข้าวเหนียวมูน']
        }
        
        # การแก้ไขการพิมพ์ผิดทั่วไป
        self.common_typos = {
            'ไขเจียว': 'ไข่เจียว', 'ไขดาว': 'ไข่ดาว',
            'กะเพรา': 'กะเพรา', 'กระเพรา': 'กะเพรา',
            'ต้มยำ': 'ต้มยำ', 'ต้มยํา': 'ต้มยำ',
            'ส้มตำ': 'ส้มตำ', 'ส้มตํา': 'ส้มตำ',
            'ผัดไทย': 'ผัดไทย', 'ผัดไท': 'ผัดไทย'
        }

    def calculate_similarity(self, s1: str, s2: str) -> float:
        """คำนวณความคล้ายคลึงระหว่างสองสตริง"""
        s1_lower = s1.lower().strip()
        s2_lower = s2.lower().strip()
        
        # ตรวจสอบการตรงกันแบบตรงตัว
        if s1_lower == s2_lower:
            return 1.0
        
        # ความคล้ายคลึงพื้นฐาน
        basic_similarity = SequenceMatcher(None, s1_lower, s2_lower).ratio()
        
        # ความคล้ายคลึงแบบคำ
        words1 = set(s1_lower.split())
        words2 = set(s2_lower.split())
        if words1 and words2:
            word_similarity = len(words1.intersection(words2)) / len(words1.union(words2))
        else:
            word_similarity = 0
        
        # ความคล้ายคลึงแบบ substring
        substring_similarity = 0
        if len(s1_lower) >= 3 and len(s2_lower) >= 3:
            if s1_lower in s2_lower or s2_lower in s1_lower:
                substring_similarity = 0.8
        
        # คำนวณคะแนนรวม
        final_score = max(basic_similarity * 0.4, word_similarity * 0.4, substring_similarity * 0.2)
        
        return final_score

    def fix_typos(self, text: str) -> str:
        """แก้ไขการพิมพ์ผิด"""
        fixed_text = text.strip()
        for typo, correct in self.common_typos.items():
            if typo in fixed_text.lower():
                fixed_text = re.sub(re.escape(typo), correct, fixed_text, flags=re.IGNORECASE)
        return fixed_text

    def find_best_match(self, query: str, candidates: List[str], threshold: float = 0.4) -> List[Dict]:
        """หาผลลัพธ์ที่ตรงกันมากที่สุด"""
        corrected_query = self.fix_typos(query.lower())
        matches = []
        
        # ตรวจสอบรูปแบบต่างๆ ของเมนู
        for main_menu, variations in self.thai_menu_variations.items():
            if corrected_query == main_menu.lower():
                for i, candidate in enumerate(candidates):
                    if main_menu.lower() in candidate.lower():
                        matches.append({
                            'index': i,
                            'text': candidate,
                            'similarity': 0.98,
                            'match_type': 'exact_variation'
                        })
            
            for variation in variations:
                if corrected_query == variation.lower():
                    for i, candidate in enumerate(candidates):
                        if main_menu.lower() in candidate.lower() or variation.lower() in candidate.lower():
                            matches.append({
                                'index': i,
                                'text': candidate,
                                'similarity': 0.95,
                                'match_type': 'variation_match'
                            })
        
        # การจับคู่แบบทั่วไป
        for i, candidate in enumerate(candidates):
            similarity = self.calculate_similarity(corrected_query, candidate.lower())
            
            if similarity >= threshold:
                # ตรวจสอบว่าไม่ซ้ำกับที่มีแล้ว
                is_duplicate = any(match['index'] == i for match in matches)
                if not is_duplicate:
                    match_type = 'fuzzy'
                    if similarity >= 0.8:
                        match_type = 'high_similarity'
                    elif similarity >= 0.6:
                        match_type = 'medium_similarity'
                    
                    matches.append({
                        'index': i,
                        'text': candidate,
                        'similarity': similarity,
                        'match_type': match_type
                    })
        
        # เรียงลำดับตามความคล้ายคลึง
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        return matches[:10]

# ตัวแปรไฟล์และโฟลเดอร์
DATA_PATH = "thai_food_processed.csv"
SAMPLE_DATA_PATH = "thai_food_sample.csv"
EMBEDDINGS_PATH = "embeddings.pkl"
MODEL_PATH = "model"

@st.cache_resource
def load_model():
    """โหลดหรือดาวน์โหลดโมเดล sentence transformer"""
    try:
        if os.path.exists(MODEL_PATH):
            return SentenceTransformer(MODEL_PATH)
        else:
            with st.spinner("กำลังดาวน์โหลดโมเดล AI... ใช้เวลาประมาณ 1-2 นาที"):
                model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                os.makedirs(MODEL_PATH, exist_ok=True)
                model.save(MODEL_PATH)
                return model
    except Exception as e:
        st.error(f"ไม่สามารถโหลดโมเดลได้: {str(e)}")
        return None

@st.cache_data
def load_data():
    """โหลดข้อมูลอาหารไทย"""
    try:
        if os.path.exists(DATA_PATH):
            return pd.read_csv(DATA_PATH)
        elif os.path.exists(SAMPLE_DATA_PATH):
            return pd.read_csv(SAMPLE_DATA_PATH)
        else:
            return create_sample_data()
    except Exception:
        return create_sample_data()

def create_sample_data():
    """สร้างข้อมูลตัวอย่าง"""
    sample_data = {
        'name': [
            'ไข่เจียว', 'ผัดกะเพรา', 'ต้มยำกุ้ง', 'แกงเขียวหวาน', 'ผัดไทย',
            'ส้มตำ', 'ยำวุ้นเส้น', 'ลาบหมู', 'ข้าวผัด', 'มะม่วงข้าวเหนียว'
        ],
        'ingredient': [
            '- ไข่ไก่ 3 ฟอง\n- น้ำปลา 1 ช้อนชา\n- น้ำมันพืช 2 ช้อนโต๊ะ',
            '- เนื้อหมูสับ 200 กรัม\n- ใบกะเพรา 1 ถ้วย\n- พริกขี้หนู 5 เม็ด\n- กระเทียม 4 กลีบ\n- น้ำปลา 2 ช้อนโต๊ะ',
            '- กุ้งนาง 300 กรัม\n- เห็ดฟาง 100 กรัม\n- มะนาว 3 ผล\n- ใบมะกรูด 5 ใบ\n- น้ำปลา 3 ช้อนโต๊ะ',
            '- เนื้อไก่ 400 กรัม\n- กะทิ 2 ถ้วย\n- มะเขือเปราะ 8 ผล\n- ใบโหระพา 1 ถ้วย\n- น้ำปลา 2 ช้อนโต๊ะ',
            '- เส้นจันท์ 200 กรัม\n- กุ้งสด 100 กรัม\n- เต้าหู้ 100 กรัม\n- ไข่ไก่ 2 ฟอง\n- น้ำปลา 2 ช้อนโต๊ะ',
            '- มะละกอดิบ 2 ถ้วย\n- มะเขือเทศ 3 ผล\n- ถั่วฝักยาว 10 เส้น\n- น้ำปลา 2 ช้อนโต๊ะ',
            '- วุ้นเส้น 150 กรัม\n- กุ้งสด 200 กรัม\n- หมูสับ 100 กรัม\n- มะนาว 3 ผล\n- น้ำปลา 3 ช้อนโต๊ะ',
            '- เนื้อหมูสับ 300 กรัม\n- ข้าวคั่ว 3 ช้อนโต๊ะ\n- หอมแดง 5 หัว\n- น้ำปลา 4 ช้อนโต๊ะ',
            '- ข้าวสวย 3 ถ้วย\n- กุ้งสด 150 กรัม\n- ไข่ไก่ 2 ฟอง\n- หอมใหญ่ 1 หัว\n- น้ำปลา 2 ช้อนโต๊ะ',
            '- ข้าวเหนียว 2 ถ้วย\n- มะม่วงสุก 2 ผล\n- กะทิ 1 ถ้วย\n- น้ำตาลทราย 3 ช้อนโต๊ะ'
        ],
        'method': [
            'ตอกไข่ใส่ชาม ใส่น้ำปลา ตีให้เข้ากัน ตั้งกะทะใส่น้ำมัน ทอดไข่จนเหลือง',
            'โขลกกระเทียมและพริกให้ละเอียด ผัดหมูสับจนสุก ใส่กะเพราและปรุงรส',
            'ต้มน้ำให้เดือด ใส่เครื่องต้มยำ เมื่อเดือดใส่กุ้งและเห็ด ปรุงรสและใส่มะนาว',
            'คั่วน้ำพริกแกงเขียวหวานกับหัวกะทิ ใส่เนื้อไก่และกะทิ ปรุงรสและใส่ผัก',
            'แช่เส้นจันท์ให้นุ่ม ผัดกุ้งและเต้าหู้ ใส่ไข่และเส้น ปรุงรสและใส่ผัก',
            'โขลกพริก กระเทียม ถั่วลิสง ใส่มะละกอและผักอื่นๆ ปรุงรส',
            'แช่วุ้นเส้นให้นุ่ม ลวกกุ้งและหมู คลุกทุกอย่างกับน้ำยำ',
            'คั่วข้าวให้เหลืองหอม โขลกให้หยาบ ผสมเนื้อหมูกับเครื่องปรุง',
            'ตั้งกะทะใส่น้ำมัน ผัดกระเทียมและหอมใหญ่ ใส่กุ้งและไข่ ใส่ข้าวผัดให้เข้ากัน',
            'นึ่งข้าวเหนียวให้สุก หั่นมะม่วง ต้มกะทิกับน้ำตาลและเกลือ เสิร์ฟพร้อมกัน'
        ]
    }
    return pd.DataFrame(sample_data)

@st.cache_data
def get_embeddings(_model, data):
    """สร้างหรือโหลด embeddings สำหรับสูตรอาหาร"""
    if _model is None or data.empty:
        return np.array([])
        
    if os.path.exists(EMBEDDINGS_PATH):
        try:
            with open(EMBEDDINGS_PATH, 'rb') as f:
                embeddings = pickle.load(f)
                if len(embeddings) == len(data):
                    return embeddings
        except:
            pass
    
    texts = []
    for _, row in data.iterrows():
        combined_text = f"{row['name']} {row['ingredient']} {row['method']}"
        texts.append(combined_text)
    
    if texts:
        with st.spinner("กำลังสร้าง embeddings สำหรับการค้นหา..."):
            embeddings = _model.encode(texts)
        
        try:
            with open(EMBEDDINGS_PATH, 'wb') as f:
                pickle.dump(embeddings, f)
        except:
            pass
        
        return embeddings
    return np.array([])

@st.cache_resource
def initialize_nutrition_api():
    """เริ่มต้นระบบข้อมูลโภชนาการ"""
    return AdvancedNutritionAPI()

@st.cache_resource
def initialize_fuzzy_matcher():
    """เริ่มต้นระบบจับคู่ข้อความ"""
    return AdvancedFuzzyMatcher()

def format_ingredients(ingredients_text):
    """จัดรูปแบบรายการวัตถุดิบ"""
    if not ingredients_text:
        return "<p>ไม่มีข้อมูลวัตถุดิบ</p>"
        
    ingredients = ingredients_text.split('\n')
    formatted = """
    <div class="ingredients-list">
        <ul>
    """
    for item in ingredients:
        if item.strip():
            clean_item = item.strip().lstrip('- ')
            # เน้นปริมาณและหน่วย
            highlighted_item = re.sub(
                r'(\d+(?:\.\d+)?)\s*([ก-๙a-zA-Z]+)', 
                r'<strong>\1 \2</strong>', 
                clean_item
            )
            formatted += f"<li>{highlighted_item}</li>"
    formatted += """
        </ul>
    </div>
    """
    return formatted

def format_cooking_method(method_text):
    """จัดรูปแบบวิธีทำใหม่ - รองรับหัวข้อย่อยและย่อหน้า"""
    if not method_text:
        return "<p>ไม่มีข้อมูลวิธีทำ</p>"
    
    # ตรวจสอบว่ามีเลขขั้นตอนอยู่แล้วหรือไม่
    has_numbers = bool(re.search(r'^\s*\d+\.', method_text, re.MULTILINE))
    
    # แบ่งตามบรรทัดใหม่
    lines = method_text.split('\n')
    formatted = '<div class="method-section">'
    
    current_paragraph = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # ตรวจสอบหัวข้อย่อย (ขึ้นต้นด้วย # หรือมีลักษณะเป็นหมายเหตุ)
        if line.startswith('#') or re.match(r'^[หมายเหตุ|สำคัญ|ข้อสังเกต|วิธีทำ|เตรียม]', line):
            # เพิ่มย่อหน้าปัจจุบันก่อน
            if current_paragraph:
                paragraph_text = ' '.join(current_paragraph)
                if has_numbers:
                    formatted += format_method_steps([paragraph_text])
                else:
                    formatted += format_method_paragraph(paragraph_text)
                current_paragraph = []
            
            # เพิ่มหัวข้อย่อย
            clean_header = line.lstrip('#').strip()
            formatted += f'<div class="method-title">{clean_header}</div>'
        else:
            # ถ้ามีเลขขั้นตอนอยู่แล้ว ให้แสดงตามเดิม
            if has_numbers and re.match(r'^\d+\.', line):
                if current_paragraph:
                    paragraph_text = ' '.join(current_paragraph)
                    formatted += format_method_paragraph(paragraph_text)
                    current_paragraph = []
                formatted += format_method_steps([line])
            else:
                # รวมเป็นย่อหน้าเดียว
                current_paragraph.append(line)
    
    # เพิ่มย่อหน้าสุดท้าย
    if current_paragraph:
        paragraph_text = ' '.join(current_paragraph)
        if has_numbers:
            formatted += format_method_steps(current_paragraph)
        else:
            formatted += format_method_paragraph(paragraph_text)
    
    formatted += '</div>'
    return formatted

def format_method_steps(steps):
    """จัดรูปแบบขั้นตอนการทำอาหาร"""
    if not steps:
        return ""
    
    formatted = '<div class="method-steps"><ol>'
    for step in steps:
        if step.strip():
            clean_step = re.sub(r'^\d+\.\s*', '', step.strip())
            formatted += f"<li>{clean_step}</li>"
    formatted += '</ol></div>'
    return formatted

def format_method_paragraph(text):
    """จัดรูปแบบย่อหน้าวิธีทำ"""
    if not text:
        return ""
    
    return f'<div class="method-paragraph">{text}</div>'

def display_nutrition_info(nutrition_data, recipe_name):
    """แสดงข้อมูลโภชนาการแบบบัตร"""
    total_nutrition = nutrition_data['total_nutrition']
    
    col1, col2, col3 = st.columns(3)
    
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
    
    with col3:
        st.markdown(f"""
        <div class="mineral-card">
            <h4>⚡ แร่ธาตุ</h4>
            <p><strong>แคลเซียม:</strong> {total_nutrition['calcium']:.1f} mg</p>
            <p><strong>เหล็ก:</strong> {total_nutrition['iron']:.1f} mg</p>
            <p><strong>โปแตสเซียม:</strong> {total_nutrition['potassium']:.1f} mg</p>
            <p><strong>โซเดียม:</strong> {total_nutrition['sodium']:.1f} mg</p>
        </div>
        """, unsafe_allow_html=True)

def create_ingredient_nutrition_table(ingredient_details):
    """สร้างตารางรายละเอียดวัตถุดิบแต่ละชนิดพร้อมวิตามินและแร่ธาตุ"""
    if not ingredient_details:
        return ""
    
    table_html = """
    <table class="ingredient-table">
        <thead>
            <tr>
                <th>วัตถุดิบ</th>
                <th>ปริมาณ</th>
                <th>แคลอรี่<br>(kcal)</th>
                <th>โปรตีน<br>(g)</th>
                <th>คาร์โบ<br>(g)</th>
                <th>ไขมัน<br>(g)</th>
                <th>ใยอาหาร<br>(g)</th>
                <th>วิต.A<br>(IU)</th>
                <th>วิต.C<br>(mg)</th>
                <th>แคลเซียม<br>(mg)</th>
                <th>เหล็ก<br>(mg)</th>
                <th>โซเดียม<br>(mg)</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for ingredient in ingredient_details:
        nutrition = ingredient['nutrition']
        consumption_note = ""
        if ingredient.get('consumption_factor', 1.0) < 1.0:
            consumption_note = f" *{ingredient['consumption_factor']*100:.0f}%"
        
        # สีพื้นหลังตามระดับโซเดียม
        sodium_color = ""
        if nutrition['sodium'] > 500:
            sodium_color = 'style="background-color: #ffebee; color: #c62828;"'
        elif nutrition['sodium'] > 200:
            sodium_color = 'style="background-color: #fff3e0; color: #ef6c00;"'
        
        # สีพื้นหลังตามระดับวิตามิน
        vit_a_color = ""
        if nutrition['vitamin_a'] > 500:
            vit_a_color = 'style="background-color: #e8f5e8; color: #2e7d32;"'
        
        vit_c_color = ""
        if nutrition['vitamin_c'] > 20:
            vit_c_color = 'style="background-color: #e8f5e8; color: #2e7d32;"'
        
        table_html += f"""
        <tr>
            <td><strong>{ingredient['name']}</strong>{consumption_note}</td>
            <td>{ingredient['quantity']:.1f} {ingredient['unit']}</td>
            <td>{nutrition['calories']:.0f}</td>
            <td>{nutrition['protein']:.1f}</td>
            <td>{nutrition['carbs']:.1f}</td>
            <td>{nutrition['fat']:.1f}</td>
            <td>{nutrition['fiber']:.1f}</td>
            <td {vit_a_color}>{nutrition['vitamin_a']:.0f}</td>
            <td {vit_c_color}>{nutrition['vitamin_c']:.1f}</td>
            <td>{nutrition['calcium']:.0f}</td>
            <td>{nutrition['iron']:.1f}</td>
            <td {sodium_color}>{nutrition['sodium']:.0f}</td>
        </tr>
        """
    
    table_html += """
        </tbody>
    </table>
    """
    
    return table_html

def get_similarity_badge_class(similarity):
    """ได้รับ CSS class สำหรับ badge ตามคะแนนความคล้ายคลึง"""
    if similarity >= 0.8:
        return "similarity-badge-high"
    elif similarity >= 0.6:
        return "similarity-badge-medium"
    else:
        return "similarity-badge-low"

def search_recipes_advanced(query, model, data, embeddings, nutrition_api, fuzzy_matcher, top_k=5):
    """การค้นหาสูตรอาหารขั้นสูง"""
    
    # ใช้ Advanced Fuzzy Matcher
    recipe_names = data['name'].tolist()
    fuzzy_matches = fuzzy_matcher.find_best_match(query, recipe_names, threshold=0.4)
    
    # ค้นหาแบบ semantic search
    if model is not None and len(embeddings) > 0:
        query_embedding = model.encode([query])
        similarities = cosine_similarity(query_embedding, embeddings)[0]
    else:
        similarities = np.zeros(len(data))
    
    # รวมผลลัพธ์
    results = []
    used_indices = set()
    
    # เพิ่ม fuzzy matches ก่อน
    for match in fuzzy_matches[:3]:
        idx = match['index']
        if idx not in used_indices:
            recipe = data.iloc[idx]
            nutrition_data = nutrition_api.calculate_recipe_nutrition(recipe['ingredient'])
            
            results.append({
                'name': recipe['name'],
                'similarity': match['similarity'],
                'match_type': match['match_type'],
                'ingredients': recipe['ingredient'],
                'method': recipe['method'],
                'nutrition': nutrition_data
            })
            used_indices.add(idx)
    
    # เพิ่มผลลัพธ์จาก semantic search
    threshold = 0.3
    top_indices = np.argsort(-similarities)
    
    for idx in top_indices:
        if len(results) >= top_k:
            break
            
        if idx not in used_indices and similarities[idx] > threshold:
            recipe = data.iloc[idx]
            nutrition_data = nutrition_api.calculate_recipe_nutrition(recipe['ingredient'])
            
            results.append({
                'name': recipe['name'],
                'similarity': similarities[idx],
                'match_type': 'semantic',
                'ingredients': recipe['ingredient'],
                'method': recipe['method'],
                'nutrition': nutrition_data
            })
            used_indices.add(idx)
    
    return results

def create_auto_scroll_system():
    """สร้างระบบเลื่อนหน้าอัตโนมัติ"""
    auto_scroll_script = """
    <script>
    let autoScrollActive = false;
    let lastMessageCount = 0;
    
    function createAutoScrollButton() {
        const existingBtn = document.getElementById('auto-scroll-btn');
        if (existingBtn) return;
        
        const scrollBtn = document.createElement('button');
        scrollBtn.id = 'auto-scroll-btn';
        scrollBtn.className = 'auto-scroll-button';
        scrollBtn.innerHTML = '↓';
        scrollBtn.title = 'เลื่อนไปข้อความล่าสุด';
        
        scrollBtn.addEventListener('click', function(e) {
            e.preventDefault();
            scrollToLatestMessage();
        });
        
        document.body.appendChild(scrollBtn);
    }
    
    function scrollToLatestMessage() {
        setTimeout(() => {
            const chatMessages = document.querySelectorAll('[data-testid="stChatMessage"], [data-testid="stExpander"]');
            
            if (chatMessages.length > 0) {
                const lastMessage = chatMessages[chatMessages.length - 1];
                lastMessage.scrollIntoView({ 
                    behavior: 'smooth', 
                    block: 'end',
                    inline: 'nearest'
                });
            } else {
                window.scrollTo({
                    top: document.body.scrollHeight,
                    behavior: 'smooth'
                });
            }
        }, 300);
    }
    
    function checkForNewMessages() {
        const currentMessages = document.querySelectorAll('[data-testid="stChatMessage"]');
        const currentCount = currentMessages.length;
        
        if (currentCount > lastMessageCount && currentCount > 0) {
            lastMessageCount = currentCount;
            
            if (!autoScrollActive) {
                autoScrollActive = true;
                setTimeout(() => {
                    scrollToLatestMessage();
                    setTimeout(() => {
                        autoScrollActive = false;
                    }, 1500);
                }, 600);
            }
        }
    }
    
    function initializeAutoScrollSystem() {
        createAutoScrollButton();
        
        // ตรวจสอบข้อความใหม่ทุก 1 วินาที
        setInterval(checkForNewMessages, 1000);
        
        // ตรวจสอบและสร้างปุ่มซ้ำทุก 5 วินาที
        setInterval(() => {
            if (!document.getElementById('auto-scroll-btn')) {
                createAutoScrollButton();
            }
        }, 5000);
    }
    
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            setTimeout(initializeAutoScrollSystem, 1500);
        });
    } else {
        setTimeout(initializeAutoScrollSystem, 1500);
    }
    </script>
    """
    
    st.components.v1.html(auto_scroll_script, height=0)

def main():
    """ฟังก์ชันหลักของแอปพลิเคชัน"""
    
    st.markdown('<h1 class="main-title">🍲 Thai Food Recipe Chatbot - Advanced</h1>', unsafe_allow_html=True)
    st.markdown("### 🥘 ระบบค้นหาสูตรอาหารไทยขั้นสูงพร้อมข้อมูลโภชนาการครบถ้วน")
    
    # เริ่มต้นระบบ
    with st.spinner("กำลังเริ่มต้นระบบขั้นสูง..."):
        model = load_model()
        data = load_data()
        embeddings = get_embeddings(model, data)
        nutrition_api = initialize_nutrition_api()
        fuzzy_matcher = initialize_fuzzy_matcher()
    
    # สร้างระบบเลื่อนหน้าอัตโนมัติ
    create_auto_scroll_system()
    
    # แถบด้านข้าง
    with st.sidebar:
        st.title("⚙️ การตั้งค่าขั้นสูง")
        
        st.subheader("📊 สถิติระบบ")
        st.metric("สูตรอาหาร", len(data))
        st.metric("ฐานข้อมูลโภชนาการ", len(nutrition_api.nutrition_db))
        
        adjust_consumption = st.checkbox(
            "ปรับสัดส่วนการบริโภคจริง", 
            value=True,
            help="คำนวณปริมาณที่บริโภคจริง เช่น น้ำมันที่ใช้ทอด"
        )
        
        show_detailed_nutrition = st.checkbox(
            "แสดงตารางโภชนาการรายละเอียด", 
            value=True,
            help="แสดงตารางข้อมูลโภชนาการแต่ละวัตถุดิบ"
        )
        
        if "search_count" not in st.session_state:
            st.session_state.search_count = 0
        
        st.metric("การค้นหาในเซสชันนี้", st.session_state.search_count)
        
        # คำแนะนำการใช้งาน
        with st.expander("💡 คำแนะนำการใช้งาน"):
            st.markdown("""
            **ฟีเจอร์ขั้นสูง:**
            - 🔍 รองรับการพิมพ์ผิดเมนูอาหารไทย
            - 📊 ตารางโภชนาการรายละเอียดทุกวัตถุดิบ
            - 🎯 การคำนวณปริมาณการบริโภคจริง
            - 🌈 การแสดงผลสีสันตามระดับโภชนาการ
            - ⬇️ ระบบเลื่อนหน้าอัตโนมัติ
            
            **ตัวอย่างการค้นหา:**
            - ไขเจียว (แก้ไขการพิมพ์ผิดอัตโนมัติ)
            - กะเพรา
            - ต้มยํา
            """)
    
    # ตัวอย่างคำค้นหา
    st.markdown("#### 💡 ลองค้นหาเมนูเหล่านี้:")
    
    sample_recipes = data['name'].head(8).tolist()
    cols = st.columns(4)
    for i, recipe_name in enumerate(sample_recipes):
        with cols[i % 4]:
            if st.button(recipe_name, key=f"example_{i}"):
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
                similarity_score = recipe.get('similarity', 0)
                match_type = recipe.get('match_type', 'semantic')
                
                # แสดงชื่อเมนูพร้อมความคล้ายคลึง
                title_html = f"### 🍽️ {recipe['name']}"
                if similarity_score > 0:
                    badge_class = get_similarity_badge_class(similarity_score)
                    if match_type in ['exact_variation', 'variation_match']:
                        title_html += f'<span class="{badge_class}">แก้ไขการพิมพ์ผิด: {similarity_score:.2f}</span>'
                    elif match_type == 'high_similarity':
                        title_html += f'<span class="{badge_class}">ความคล้ายคลึงสูง: {similarity_score:.2f}</span>'
                    else:
                        title_html += f'<span class="{badge_class}">ความเกี่ยวข้อง: {similarity_score:.2f}</span>'
                
                st.markdown(title_html, unsafe_allow_html=True)
                
                # แสดงข้อมูลโภชนาการ
                display_nutrition_info(recipe['nutrition'], recipe['name'])
                
                # แสดงตารางรายละเอียดวัตถุดิบ (ถ้าเปิดใช้งาน)
                if show_detailed_nutrition and recipe['nutrition'].get('ingredient_details'):
                    st.markdown("#### 📋 ตารางรายละเอียดโภชนาการแต่ละวัตถุดิบ")
                    table_html = create_ingredient_nutrition_table(recipe['nutrition']['ingredient_details'])
                    st.markdown(table_html, unsafe_allow_html=True)
                    
                    # คำอธิบายสัญลักษณ์
                    st.markdown("""
                    <div style="font-size: 0.8rem; color: #666; margin-top: 0.5rem;">
                    💡 <strong>คำอธิบาย:</strong> 
                    *% = ปริมาณที่บริโภคจริง | 
                    <span style="background-color: #ffebee; padding: 2px;">สีแดง</span> = โซเดียมสูง | 
                    <span style="background-color: #e8f5e8; padding: 2px;">สีเขียว</span> = วิตามินสูง
                    </div>
                    """, unsafe_allow_html=True)
                
                # แสดงวัตถุดิบ
                st.markdown("#### 🥬 วัตถุดิบ")
                st.markdown(format_ingredients(recipe["ingredients"]), unsafe_allow_html=True)
                
                # แสดงวิธีทำ
                st.markdown("#### 👨‍🍳 วิธีทำ")
                st.markdown(format_cooking_method(recipe["method"]), unsafe_allow_html=True)
                
            else:
                st.markdown(message["content"])
    
    # ช่องค้นหา
    search_query = st.session_state.get('search_query', '')
    if prompt := st.chat_input("ค้นหาสูตรอาหาร เช่น 'ไข่เจียว' หรือ 'ผัดกะเพรา'... ✨", key="main_chat"):
        search_query = prompt
        st.session_state.search_query = ""
    
    if search_query:
        st.session_state.search_count += 1
        st.session_state.messages.append({"role": "user", "content": search_query})
        
        with st.chat_message("user"):
            st.markdown(search_query)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 กำลังค้นหาด้วยระบบขั้นสูง..."):
                results = search_recipes_advanced(
                    search_query, model, data, embeddings, nutrition_api, fuzzy_matcher
                )
                
                if results:
                    best_match = results[0]
                    
                    response = f"🎯 พบสูตรอาหารที่ตรงกับการค้นหา: **{best_match['name']}**"
                    
                    # ปรับข้อความตามประเภทการจับคู่
                    match_type = best_match.get('match_type', 'semantic')
                    if match_type in ['exact_variation', 'variation_match']:
                        response = f"🔧 แก้ไขการพิมพ์ผิดและพบสูตร: **{best_match['name']}**"
                    elif match_type == 'high_similarity':
                        response = f"✨ พบสูตรอาหารที่คล้ายคลึงมาก: **{best_match['name']}**"
                    
                    st.markdown(response)
                    
                    # คำนวณโภชนาการใหม่ด้วยการตั้งค่า
                    nutrition_data = nutrition_api.calculate_recipe_nutrition(
                        best_match['ingredients'], 
                        adjust_consumption=adjust_consumption
                    )
                    
                    # อัปเดตข้อมูลโภชนาการ
                    best_match['nutrition'] = nutrition_data
                    
                    # แสดงข้อมูลโภชนาการ
                    display_nutrition_info(nutrition_data, best_match['name'])
                    
                    # แสดงตารางรายละเอียดวัตถุดิบ (ถ้าเปิดใช้งาน)
                    if show_detailed_nutrition and nutrition_data.get('ingredient_details'):
                        st.markdown("#### 📋 ตารางรายละเอียดโภชนาการแต่ละวัตถุดิบ")
                        table_html = create_ingredient_nutrition_table(nutrition_data['ingredient_details'])
                        st.markdown(table_html, unsafe_allow_html=True)
                        
                        # คำอธิบายสัญลักษณ์
                        st.markdown("""
                        <div style="font-size: 0.8rem; color: #666; margin-top: 0.5rem;">
                        💡 <strong>คำอธิบาย:</strong> 
                        *% = ปริมาณที่บริโภคจริง | 
                        <span style="background-color: #ffebee; padding: 2px;">สีแดง</span> = โซเดียมสูง | 
                        <span style="background-color: #e8f5e8; padding: 2px;">สีเขียว</span> = วิตามินสูง
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # แสดงวัตถุดิบ
                    st.markdown("#### 🥬 วัตถุดิบ")
                    st.markdown(format_ingredients(best_match["ingredients"]), unsafe_allow_html=True)
                    
                    # แสดงวิธีทำ
                    st.markdown("#### 👨‍🍳 วิธีทำ")
                    st.markdown(format_cooking_method(best_match["method"]), unsafe_allow_html=True)
                    
                    # แนะนำเมนูอื่น
                    if len(results) > 1:
                        st.markdown("#### 🔍 เมนูอื่นที่น่าสนใจ:")
                        other_results = results[1:min(4, len(results))]
                        
                        cols = st.columns(len(other_results))
                        for i, (other_recipe) in enumerate(other_results):
                            with cols[i]:
                                similarity_percent = other_recipe['similarity'] * 100
                                if st.button(f"🍽️ {other_recipe['name']}\n({similarity_percent:.0f}% ตรง)", key=f"other_{i}"):
                                    st.session_state.search_query = other_recipe['name']
                                    st.rerun()
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response, 
                        "recipe": best_match
                    })
                    
                else:
                    response = f"""
                    ❌ ไม่พบสูตรอาหารที่ตรงกับ '{search_query}'
                    
                    💡 **คำแนะนำ:**
                    - ลองใช้คำค้นหาที่ง่ายกว่า เช่น "ไข่เจียว" แทน "วิธีทำไข่เจียว"
                    - ตรวจสอบการสะกดคำภาษาไทย
                    - ลองค้นหาด้วยเมนูที่มีชื่อสั้นๆ
                    
                    🍽️ **เมนูยอดนิยม:** {', '.join(data['name'].head(5).tolist())}
                    """
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
