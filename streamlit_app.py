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
    
    .api-status-connected {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    
    .api-status-disconnected {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
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

# คลาส NutritionAPI
class NutritionAPI:
    """คลาสสำหรับจัดการข้อมูลโภชนาการ"""
    
    def __init__(self):
        self.current_api_source = "local"
        self.local_nutrition_db = self.get_default_nutrition_database()
        
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
            "ผล": 150, "ราก": 5, "ท่อน": 20
        }
        
        # สัดส่วนที่บริโภคจริง
        self.consumption_ratio = {
            "น้ำมันหมู": 0.25, "น้ำมันพืช": 0.25,
            "น้ำมันมะพร้าว": 0.25, "กะทิ": 0.85,
            "น้ำปลา": 1.0, "น้ำตาล": 1.0, "เกลือ": 1.0
        }

    def get_default_nutrition_database(self) -> Dict:
        """ฐานข้อมูลโภชนาการเริ่มต้น (ต่อ 100 กรัม)"""
        return {
            "ข้าว": {"calories": 130, "protein": 2.7, "carbs": 28, "fat": 0.3, "fiber": 0.4,
                    "vitamin_a": 0, "vitamin_c": 0, "vitamin_b1": 0.07, "vitamin_b2": 0.02,
                    "calcium": 10, "iron": 0.8, "potassium": 115, "sodium": 5},
            "ไข่ไก่": {"calories": 155, "protein": 13, "carbs": 1.1, "fat": 11, "fiber": 0,
                     "vitamin_a": 540, "vitamin_c": 0, "vitamin_b1": 0.04, "vitamin_b2": 0.42,
                     "calcium": 56, "iron": 1.75, "potassium": 138, "sodium": 124},
            "กุ้ง": {"calories": 99, "protein": 18, "carbs": 0.2, "fat": 1.4, "fiber": 0,
                    "vitamin_a": 54, "vitamin_c": 2.1, "vitamin_b1": 0.02, "vitamin_b2": 0.04,
                    "calcium": 70, "iron": 0.5, "potassium": 259, "sodium": 111},
            "หมู": {"calories": 242, "protein": 27, "carbs": 0, "fat": 14, "fiber": 0,
                   "vitamin_a": 2, "vitamin_c": 0.7, "vitamin_b1": 0.66, "vitamin_b2": 0.23,
                   "calcium": 19, "iron": 0.87, "potassium": 423, "sodium": 62},
            "ไก่": {"calories": 165, "protein": 31, "carbs": 0, "fat": 3.6, "fiber": 0,
                   "vitamin_a": 21, "vitamin_c": 1.6, "vitamin_b1": 0.07, "vitamin_b2": 0.12,
                   "calcium": 15, "iron": 1.3, "potassium": 256, "sodium": 82},
            "ปลา": {"calories": 112, "protein": 18.7, "carbs": 0, "fat": 3.6, "fiber": 0,
                    "vitamin_a": 45, "vitamin_c": 0.9, "vitamin_b1": 0.02, "vitamin_b2": 0.11,
                    "calcium": 89, "iron": 0.9, "potassium": 358, "sodium": 54},
            "น้ำมันพืช": {"calories": 884, "protein": 0, "carbs": 0, "fat": 100, "fiber": 0,
                         "vitamin_a": 0, "vitamin_c": 0, "vitamin_b1": 0, "vitamin_b2": 0,
                         "calcium": 0, "iron": 0, "potassium": 0, "sodium": 0},
            "กระเทียม": {"calories": 149, "protein": 6.4, "carbs": 33, "fat": 0.5, "fiber": 2.1,
                         "vitamin_a": 9, "vitamin_c": 31, "vitamin_b1": 0.2, "vitamin_b2": 0.11,
                         "calcium": 181, "iron": 1.7, "potassium": 401, "sodium": 17},
            "หอมใหญ่": {"calories": 40, "protein": 1.1, "carbs": 9.3, "fat": 0.1, "fiber": 1.7,
                        "vitamin_a": 2, "vitamin_c": 7.4, "vitamin_b1": 0.05, "vitamin_b2": 0.03,
                        "calcium": 23, "iron": 0.21, "potassium": 146, "sodium": 4},
            "ผักชี": {"calories": 23, "protein": 2.1, "carbs": 3.7, "fat": 0.5, "fiber": 2.8,
                     "vitamin_a": 3377, "vitamin_c": 27, "vitamin_b1": 0.07, "vitamin_b2": 0.16,
                     "calcium": 67, "iron": 1.77, "potassium": 521, "sodium": 46},
            "น้ำปลา": {"calories": 42, "protein": 5.8, "carbs": 1.5, "fat": 0.8, "fiber": 0,
                       "vitamin_a": 0, "vitamin_c": 0, "vitamin_b1": 0.03, "vitamin_b2": 0.22,
                       "calcium": 85, "iron": 2.03, "potassium": 84, "sodium": 6976},
            "น้ำตาล": {"calories": 387, "protein": 0, "carbs": 100, "fat": 0, "fiber": 0,
                       "vitamin_a": 0, "vitamin_c": 0, "vitamin_b1": 0, "vitamin_b2": 0,
                       "calcium": 1, "iron": 0.01, "potassium": 2, "sodium": 1},
            "มะนาว": {"calories": 29, "protein": 0.7, "carbs": 9.3, "fat": 0.2, "fiber": 2.8,
                      "vitamin_a": 22, "vitamin_c": 53, "vitamin_b1": 0.03, "vitamin_b2": 0.02,
                      "calcium": 33, "iron": 0.6, "potassium": 138, "sodium": 2},
            "กะทิ": {"calories": 230, "protein": 2.3, "carbs": 6, "fat": 24, "fiber": 2.2,
                     "vitamin_a": 0, "vitamin_c": 2.8, "vitamin_b1": 0.03, "vitamin_b2": 0,
                     "calcium": 16, "iron": 1.64, "potassium": 263, "sodium": 15}
        }

    def normalize_ingredient_name(self, ingredient: str) -> str:
        """ปรับแต่งชื่อวัตถุดิบให้เป็นมาตรฐาน"""
        ingredient = re.sub(r'\d+.*', '', ingredient)
        ingredient = re.sub(r'[^\u0E00-\u0E7Fa-zA-Z\s]', '', ingredient)
        ingredient = ingredient.strip()
        
        synonyms = {
            "กุ้งนาง": "กุ้ง", "กุ้งตะเข็บ": "กุ้ง",
            "เนื้อหมู": "หมู", "หมูสับ": "หมู",
            "เนื้อไก่": "ไก่", "ไก่สับ": "ไก่",
            "หอมหัวใหญ่": "หอมใหญ่", "น้ำตาลทราย": "น้ำตาล"
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
        
        for local_ingredient, nutrition in self.local_nutrition_db.items():
            if (local_ingredient.lower() in normalized_ingredient.lower() or 
                normalized_ingredient.lower() in local_ingredient.lower()):
                return nutrition.copy()
        
        return self.estimate_nutrition_by_type(normalized_ingredient)

    def estimate_nutrition_by_type(self, ingredient: str) -> Dict:
        """ประมาณค่าโภชนาการตามประเภทวัตถุดิบ"""
        ingredient_lower = ingredient.lower()
        
        if any(keyword in ingredient_lower for keyword in ["เนื้อ", "หมู", "ไก่", "ปลา", "กุ้ง"]):
            return {"calories": 150, "protein": 20, "carbs": 0, "fat": 6, "fiber": 0,
                   "vitamin_a": 20, "vitamin_c": 1, "vitamin_b1": 0.1, "vitamin_b2": 0.15,
                   "calcium": 20, "iron": 1.5, "potassium": 250, "sodium": 50}
        elif any(keyword in ingredient_lower for keyword in ["ผัก", "ใบ", "ต้น"]):
            return {"calories": 25, "protein": 2, "carbs": 5, "fat": 0.2, "fiber": 2,
                   "vitamin_a": 1000, "vitamin_c": 30, "vitamin_b1": 0.05, "vitamin_b2": 0.08,
                   "calcium": 50, "iron": 1, "potassium": 200, "sodium": 10}
        else:
            return {"calories": 50, "protein": 2, "carbs": 10, "fat": 1, "fiber": 1,
                   "vitamin_a": 10, "vitamin_c": 5, "vitamin_b1": 0.05, "vitamin_b2": 0.05,
                   "calcium": 20, "iron": 0.5, "potassium": 100, "sodium": 10}

    def calculate_recipe_nutrition(self, ingredients_text: str, use_api: bool = True, 
                                 adjust_consumption: bool = True, 
                                 enhance_missing: bool = False) -> Dict:
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
                    "quantity": quantity,
                    "unit": unit,
                    "grams": grams,
                    "effective_grams": effective_grams,
                    "nutrition": ingredient_nutrition
                })
        
        return {
            "total_nutrition": total_nutrition,
            "ingredient_details": ingredient_details,
            "settings": {
                "use_api": use_api,
                "adjust_consumption": adjust_consumption,
                "enhance_missing": enhance_missing
            }
        }

# คลาส RecipeSearchEngine
class RecipeSearchEngine:
    """เครื่องมือค้นหาสูตรอาหารขั้นสูง"""
    
    def __init__(self, data: pd.DataFrame, nutrition_api: NutritionAPI):
        self.data = data
        self.nutrition_api = nutrition_api
        self.recipe_nutrition_cache = {}

    def advanced_fuzzy_search(self, query: str, threshold: float = 0.4) -> List[Tuple[str, float, int]]:
        """การค้นหาแบบ fuzzy matching ที่ปรับปรุงแล้ว"""
        query = query.lower().strip()
        matches = []
        
        # ลบคำที่ไม่จำเป็น
        stop_words = ["อาหาร", "เมนู", "สูตร", "วิธีทำ", "ทำ", "ปรุง", "อร่อย", "ง่าย"]
        query_words = [word for word in query.split() if word not in stop_words and len(word) > 1]
        clean_query = " ".join(query_words) if query_words else query
        
        for idx, recipe_name in enumerate(self.data['name']):
            recipe_name_lower = recipe_name.lower()
            
            # 1. ความคล้ายคลึงแบบ sequence matching
            sequence_similarity = SequenceMatcher(None, clean_query, recipe_name_lower).ratio()
            
            # 2. การตรวจสอบคำที่ตรงกันทั้งหมด
            exact_match_score = 0
            if clean_query in recipe_name_lower:
                exact_match_score = min(len(clean_query) / len(recipe_name_lower), 1.0)
            
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
                    word_similarity = SequenceMatcher(None, q_word, r_word).ratio()
                    if word_similarity >= 0.8:
                        best_match_score = max(best_match_score, word_similarity)
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
            
            # 4. ตรวจสอบในส่วนผสมและวิธีทำ
            content_match_score = 0
            if sequence_similarity < threshold and word_match_score < threshold:
                ingredient_text = str(self.data.iloc[idx].get('ingredient', '')).lower()
                method_text = str(self.data.iloc[idx].get('method', '')).lower()
                
                content_matches = 0
                for q_word in query_words:
                    if len(q_word) >= 3:
                        if q_word in ingredient_text:
                            content_matches += 0.3
                        elif q_word in method_text:
                            content_matches += 0.2
                
                if content_matches > 0:
                    content_match_score = min(content_matches / len(query_words), 0.5)
            
            # 5. คำนวณคะแนนรวม
            final_scores = [
                sequence_similarity * 0.3,
                exact_match_score * 0.9,
                word_match_score * 0.7,
                content_match_score * 0.4
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
            
            if final_score >= threshold:
                matches.append((recipe_name, final_score, idx))
        
        # เรียงลำดับและกรองผลลัพธ์
        matches.sort(key=lambda x: x[1], reverse=True)
        
        filtered_matches = []
        seen_scores = set()
        
        for match in matches:
            score_rounded = round(match[1], 2)
            if score_rounded not in seen_scores or len(filtered_matches) < 3:
                filtered_matches.append(match)
                seen_scores.add(score_rounded)
                
                if len(filtered_matches) >= 10:
                    break
        
        return filtered_matches

    def get_recipe_nutrition(self, recipe_index: int, use_api: bool = True, 
                           adjust_consumption: bool = True, 
                           enhance_missing: bool = False) -> Dict:
        """ดึงข้อมูลโภชนาการของสูตรอาหารพร้อมแคช"""
        cache_key = f"{recipe_index}_{use_api}_{adjust_consumption}_{enhance_missing}"
        
        if cache_key in self.recipe_nutrition_cache:
            return self.recipe_nutrition_cache[cache_key]
        
        try:
            recipe = self.data.iloc[recipe_index]
            nutrition_data = self.nutrition_api.calculate_recipe_nutrition(
                recipe['ingredient'], use_api, adjust_consumption, enhance_missing
            )
            
            self.recipe_nutrition_cache[cache_key] = nutrition_data
            return nutrition_data
        except Exception:
            # ส่งคืนข้อมูลโภชนาการเริ่มต้นหากเกิดข้อผิดพลาด
            default_nutrition = {
                "total_nutrition": {
                    "calories": 0, "protein": 0, "carbs": 0, "fat": 0, "fiber": 0,
                    "vitamin_a": 0, "vitamin_c": 0, "vitamin_b1": 0, "vitamin_b2": 0,
                    "calcium": 0, "iron": 0, "potassium": 0, "sodium": 0
                },
                "ingredient_details": [],
                "settings": {"use_api": use_api, "adjust_consumption": adjust_consumption, "enhance_missing": enhance_missing}
            }
            return default_nutrition

    def smart_search(self, query: str, use_api: bool = True, adjust_consumption: bool = True,
                    enhance_missing: bool = False, fuzzy_threshold: float = 0.4,
                    limit: int = 5) -> List[Dict]:
        """ระบบค้นหาอัจฉริยะ"""
        
        fuzzy_results = self.advanced_fuzzy_search(query, threshold=fuzzy_threshold)
        results = []
        
        for recipe_name, similarity, recipe_idx in fuzzy_results[:limit]:
            nutrition_data = self.get_recipe_nutrition(
                recipe_idx, use_api, adjust_consumption, enhance_missing
            )
            
            results.append({
                "name": recipe_name,
                "similarity": similarity,
                "index": recipe_idx,
                "nutrition": nutrition_data,
                "match_reason": f"ความคล้ายคลึงชื่อ: {similarity:.0%}",
                "match_type": "fuzzy"
            })
        
        return results

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
    """สร้างข้อมูลตัวอย่างเมื่อไม่มีไฟล์"""
    sample_data = {
        'name': [
            'ผัดกะเพรา', 'ต้มยำกุ้ง', 'ส้มตำ', 'แกงเขียวหวาน', 'ผัดไทย',
            'ไข่เจียว', 'ข้าวผัด', 'ยำวุ้นเส้น', 'ลาบหมู', 'มะม่วงข้าวเหนียว'
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
            '- ข้าวเหนียว 2 ถ้วย\n- มะม่วงสุก 2 ผล\n- กะทิ 1 ถ้วย\n- น้ำตาลปึก 3 ช้อนโต๊ะ\n- เกลือ 1/2 ช้อนชา'
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
    return NutritionAPI()

@st.cache_resource
def initialize_search_engine(_data, _nutrition_api):
    """เริ่มต้นระบบค้นหา"""
    if _data.empty:
        return None
    return RecipeSearchEngine(_data, _nutrition_api)

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
    """จัดรูปแบบวิธีทำให้เรียบร้อยตามโครงสร้างข้อมูลจริง"""
    if not method_text:
        return "<p>ไม่มีข้อมูลวิธีทำ</p>"
    
    # แปลง **text** เป็น <strong>text</strong>
    method_text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', method_text)
    
    formatted = "<div style='margin: 0; line-height: 1.8; font-size: 1rem;'>"
    
    # แยกบรรทัดเพื่อจัดการทีละบรรทัด
    lines = method_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # ตรวจสอบหมายเหตุ
        if re.match(r'^(หมายเหตุ|สำคัญ|ข้อสังเกต|เคล็ดลับ|วิธีเตรียม|Note|Tip)', line, re.IGNORECASE):
            # แยกหัวข้อและเนื้อหา
            if ':' in line:
                header = line.split(':')[0].strip()
                content = ':'.join(line.split(':')[1:]).strip()
            else:
                # หาส่วนที่เป็นหัวข้อ (คำแรก)
                words = line.split()
                header = words[0] if words else line
                content = ' '.join(words[1:]) if len(words) > 1 else ""
            
            formatted += f"""
            <div style='margin: 1.2rem 0; padding: 1rem; background-color: #fff3cd; border-left: 4px solid #ffc107; border-radius: 4px;'>
                <div style='color: #856404; font-size: 1rem; margin-bottom: 0.3rem; border-bottom: 1px solid #f0c14b; padding-bottom: 0.3rem;'>{header}</div>
                {f"<div style='color: #856404; font-size: 1rem; padding-left: 0.5rem;'>{content}</div>" if content else ""}
            </div>
            """
        
        # ตรวจสอบหัวข้อย่อย (ขึ้นต้นด้วย # หรือเป็นหัวข้อที่มี pattern เฉพาะ)
        elif (line.startswith('#') or 
              re.match(r'^(วิธีทำ|วิธีแต่ง|ส่วนผสม|เครื่องปรุง|การเตรียม)', line, re.IGNORECASE) or
              (line.endswith(':') and len(line) < 50)):
            
            header_text = line.lstrip('#').strip().rstrip(':')
            formatted += f"""
            <div style='margin: 1rem 0; padding: 0.8rem; border-left: 3px solid #28a745; background-color: #f8f9fa;'>
                <div style='color: #28a745; font-size: 1rem; margin-bottom: 0.5rem; border-bottom: 1px solid #dee2e6; padding-bottom: 0.3rem;'>{header_text}</div>
            </div>
            """
        
        # ตรวจสอบเลขข้อ (เริ่มต้นด้วยตัวเลข + จุด)
        elif re.match(r'^\d+\.', line):
            step_number = line.split('.')[0]
            step_content = '.'.join(line.split('.')[1:]).strip()
            
            formatted += f"""
            <div style='margin: 1rem 0; padding: 0.8rem; border-left: 3px solid #667eea; background-color: #f8f9fa;'>
                <div style='color: #667eea; font-size: 1rem; margin-bottom: 0.5rem; border-bottom: 1px solid #dee2e6; padding-bottom: 0.3rem;'>{step_number}</div>
                <div style='font-size: 1rem; padding-left: 0.5rem; color: #333;'>{step_content}</div>
            </div>
            """
        
        # ข้อความธรรมดา
        else:
            # ตรวจสอบว่าเป็นประโยคยาวหรือไม่
            if len(line) > 80:
                # ประโยคยาว แสดงเป็นย่อหน้า
                formatted += f"""
                <div style='margin: 0.8rem 0; padding: 0.6rem; border-left: 3px solid #e3f2fd; background-color: #fafafa;'>
                    <div style='font-size: 1rem; color: #333; text-align: justify;'>{line}</div>
                </div>
                """
            else:
                # ประโยคสั้น แสดงแบบรายการ
                formatted += f"""
                <div style='margin: 0.5rem 0; padding: 0.4rem 0.6rem; background-color: #f8f9fa; border-radius: 3px;'>
                    <div style='font-size: 1rem; color: #333;'>{line}</div>
                </div>
                """
    
    formatted += "</div>"
    return formatted

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

def get_similarity_badge_class(similarity):
    """ได้รับ CSS class สำหรับ badge ตามคะแนนความคล้ายคลึง"""
    if similarity >= 0.7:
        return "similarity-badge-high"
    elif similarity >= 0.5:
        return "similarity-badge-medium"
    else:
        return "similarity-badge-low"

def display_settings_panel(data, model, settings_state):
    """แสดงแถบการตั้งค่าพร้อมสถิติ"""
    st.sidebar.title("🔧 การตั้งค่าขั้นสูง")
    
    # การตั้งค่า API
    st.sidebar.markdown("### 🌐 API ข้อมูลโภชนาการ")
    use_api = st.sidebar.checkbox("เปิดใช้งาน API ภายนอก", value=False, key="use_api")
    
    api_status_detail = "🔴 ไม่ได้เชื่อมต่อ"
    if use_api:
        api_key = st.sidebar.text_input("USDA API Key", type="password", key="usda_api_key")
        if api_key:
            api_status_detail = "🟡 ตั้งค่าแล้ว"
    
    status_class = "api-status-connected" if "🟢" in api_status_detail else "api-status-disconnected"
    st.sidebar.markdown(f'<div class="{status_class}">สถานะ: {api_status_detail}</div>', unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧮 การคำนวณโภชนาการ")
    
    adjust_consumption = st.sidebar.checkbox(
        "ปรับการบริโภคตามความเป็นจริง", value=True, key="adjust_consumption"
    )
    
    enhance_missing = st.sidebar.checkbox(
        "เพิ่มวัตถุดิบที่ขาดหาย", value=False, key="enhance_missing"
    )
    
    st.sidebar.markdown("### 🔍 การค้นหาที่ปรับปรุงแล้ว")
    
    fuzzy_threshold = st.sidebar.slider(
        "ความเคร่งครัดในการค้นหา",
        min_value=0.2, max_value=0.8, value=0.4, step=0.1,
        help="ค่าต่ำ = หาได้ง่าย, ค่าสูง = หาได้ยากแต่ตรงมาก",
        key="fuzzy_threshold"
    )
    
    max_results = st.sidebar.number_input(
        "จำนวนผลลัพธ์สูงสุด", min_value=1, max_value=10, value=3, key="max_results"
    )
    
    st.sidebar.markdown("---")

    # แสดงสถิติโดยรวม
    st.sidebar.markdown("### 📊 สถิติโดยรวม")
    st.sidebar.metric("📖 จำนวนสูตร", len(data))
    st.sidebar.metric("🤖 AI Model", "✅ พร้อม" if model else "❌ ไม่พร้อม")
    
    api_status = "🟢 เชื่อมต่อ" if settings_state.get('use_api', False) else "🔴 ปิดใช้งาน"
    st.sidebar.metric("🌐 API", api_status)
    st.sidebar.metric("🔍 การค้นหา", "✨ ปรับปรุงแล้ว")
    
    return {
        'use_api': use_api, 'adjust_consumption': adjust_consumption,
        'enhance_missing': enhance_missing, 'fuzzy_threshold': fuzzy_threshold,
        'max_results': max_results
    }

def search_recipes_improved(query, model, data, embeddings, search_engine, settings):
    """การค้นหาสูตรอาหารที่ปรับปรุงแล้ว"""
    
    if search_engine:
        try:
            smart_results = search_engine.smart_search(
                query, settings['use_api'], settings['adjust_consumption'],
                settings['enhance_missing'], settings['fuzzy_threshold'],
                limit=settings['max_results']
            )
            
            if smart_results:
                results = []
                for result in smart_results:
                    adjusted_similarity = min(result['similarity'], 1.0)
                    results.append((
                        result['name'], adjusted_similarity,
                        result['index'], result.get('nutrition')
                    ))
                return results
        except Exception as e:
            st.warning(f"เกิดข้อผิดพลาดในระบบค้นหา: {str(e)}")
    
    # ระบบสำรองแบบ fuzzy matching
    matches = []
    query_lower = query.lower().strip()
    
    stop_words = ["อาหาร", "เมนู", "สูตร", "วิธีทำ", "ทำ", "ปรุง"]
    query_words = [word for word in query_lower.split() if word not in stop_words and len(word) > 1]
    clean_query = " ".join(query_words) if query_words else query_lower
    
    for idx, recipe_name in enumerate(data['name']):
        recipe_name_lower = recipe_name.lower()
        
        similarities = []
        
        # 1. ความคล้ายคลึงแบบ sequence
        seq_sim = SequenceMatcher(None, clean_query, recipe_name_lower).ratio()
        similarities.append(seq_sim * 0.4)
        
        # 2. การตรวจสอบคำที่ตรงกัน
        if clean_query in recipe_name_lower:
            exact_sim = len(clean_query) / len(recipe_name_lower)
            similarities.append(exact_sim * 0.8)
        
        # 3. การตรวจสอบคำแยกกัน
        word_matches = 0
        for q_word in query_words:
            if len(q_word) > 2:
                for r_word in recipe_name_lower.split():
                    if SequenceMatcher(None, q_word, r_word).ratio() > 0.8:
                        word_matches += 1
                        break
                    elif q_word in r_word or r_word in q_word:
                        word_matches += 0.5
                        break
        
        if query_words:
            word_sim = word_matches / len(query_words)
            similarities.append(word_sim * 0.6)
        
        final_similarity = max(similarities) if similarities else 0
        
        if final_similarity > 0 and len(recipe_name_lower) <= 10:
            final_similarity = min(final_similarity * 1.1, 1.0)
        
        if final_similarity >= settings['fuzzy_threshold']:
            matches.append((recipe_name, final_similarity, idx))
    
    matches.sort(key=lambda x: x[1], reverse=True)
    
    # คำนวณโภชนาการ
    results = []
    nutrition_api = initialize_nutrition_api()
    
    for recipe_name, similarity, recipe_idx in matches[:settings['max_results']]:
        nutrition_data = None
        try:
            recipe = data.iloc[recipe_idx]
            nutrition_data = nutrition_api.calculate_recipe_nutrition(
                recipe['ingredient'], settings['use_api'],
                settings['adjust_consumption'], settings['enhance_missing']
            )
        except:
            pass
        
        results.append((recipe_name, similarity, recipe_idx, nutrition_data))
    
    return results

def display_recipe_with_nutrition(recipe, nutrition_data, settings, similarity_score=None):
    """แสดงสูตรอาหารพร้อมข้อมูลโภชนาการ"""
    
    title_html = f"### 🍽️ {recipe['name']}"
    if similarity_score is not None:
        similarity_percent = similarity_score * 100
        badge_class = get_similarity_badge_class(similarity_score)
        title_html += f'<span class="{badge_class}">{similarity_percent:.0f}% ตรง</span>'
    
    st.markdown(title_html, unsafe_allow_html=True)
    
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
            display_nutrition_info(nutrition_data, recipe['name'])
        else:
            st.info("ไม่มีข้อมูลโภชนาการ หรือเกิดข้อผิดพลาดในการคำนวณ")
    
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
                    'ไขมัน (g)': f"{ingredient['nutrition']['fat']:.1f}",
                    'วิตามิน A (IU)': f"{ingredient['nutrition']['vitamin_a']:.1f}",
                    'วิตามิน C (mg)': f"{ingredient['nutrition']['vitamin_c']:.1f}",
                    'วิตามิน B1 (mg)': f"{ingredient['nutrition']['vitamin_b1']:.2f}",
                    'วิตามิน B2 (mg)': f"{ingredient['nutrition']['vitamin_b2']:.2f}",
                    'แคลเซียม (mg)': f"{ingredient['nutrition']['calcium']:.1f}",
                    'เหล็ก (mg)': f"{ingredient['nutrition']['iron']:.1f}",
                    'โปแตสเซียม (mg)': f"{ingredient['nutrition']['potassium']:.1f}",
                    'โซเดียม (mg)': f"{ingredient['nutrition']['sodium']:.1f}"
                })
            
            if ingredient_df:
                df = pd.DataFrame(ingredient_df)
                st.dataframe(df, use_container_width=True)
        else:
            st.info("ไม่มีข้อมูลรายละเอียดวัตถุดิบ")

def main():
    """ฟังก์ชันหลักของแอปพลิเคชัน"""
    
    st.markdown('<h1 class="main-title">Thai Food Recipe Chatbot</h1>', unsafe_allow_html=True)
    st.markdown("### 🥘 ระบบค้นหาสูตรอาหารไทย")
    
    # เริ่มต้นระบบ
    with st.spinner("กำลังเริ่มต้นระบบที่ปรับปรุงแล้ว..."):
        model = load_model()
        data = load_data()
        
        if model is None:
            st.error("ไม่สามารถโหลดโมเดล AI ได้")
            return
        
        if data.empty:
            st.error("ไม่มีข้อมูลสูตรอาหาร")
            return
        
        embeddings = get_embeddings(model, data)
        
        # สร้าง session state สำหรับการจัดการ settings
        if 'settings_state' not in st.session_state:
            st.session_state.settings_state = {}
        
        # แถบการตั้งค่าพร้อมสถิติ
        settings = display_settings_panel(data, model, st.session_state.settings_state)
        st.session_state.settings_state = settings
        
        nutrition_api = initialize_nutrition_api()
        search_engine = initialize_search_engine(data, nutrition_api)
    
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
                display_recipe_with_nutrition(recipe, nutrition_data, settings, similarity_score)
            else:
                st.markdown(message["content"])
    
    # ช่องค้นหา
    search_query = st.session_state.get('search_query', '')
    if prompt := st.chat_input("ค้นหาสูตรอาหาร เช่น 'ผัดกะเพรา' หรือ 'ไข่เจียว'... (ระบบค้นหาปรับปรุงใหม่)", key="main_chat"):
        search_query = prompt
        st.session_state.search_query = ""
    
    if search_query:
        st.session_state.messages.append({"role": "user", "content": search_query})
        
        with st.chat_message("user"):
            st.markdown(search_query)
        
        with st.chat_message("assistant"):
            with st.spinner("กำลังค้นหาด้วยระบบที่ปรับปรุงใหม่..."):
                results = search_recipes_improved(
                    search_query, model, data, embeddings, search_engine, settings
                )
                
                if results:
                    best_match = results[0]
                    recipe_name, similarity, recipe_idx, nutrition_data = best_match
                    
                    recipe = {
                        'name': recipe_name,
                        'ingredient': data.iloc[recipe_idx]['ingredient'],
                        'method': data.iloc[recipe_idx]['method']
                    }
                    
                    response = f"🎯 พบสูตรอาหารที่ตรงกับการค้นหา: **{recipe_name}**"
                    st.markdown(response)
                    
                    display_recipe_with_nutrition(recipe, nutrition_data, settings, similarity)
                    
                    # แนะนำเมนูอื่น
                    if len(results) > 1:
                        st.markdown("#### 🔍 เมนูอื่นที่น่าสนใจ:")
                        other_results = results[1:min(4, len(results))]
                        
                        cols = st.columns(len(other_results))
                        for i, (other_name, other_sim, other_idx, _) in enumerate(other_results):
                            with cols[i]:
                                similarity_percent = other_sim * 100
                                if st.button(f"🍽️ {other_name}\n({similarity_percent:.0f}% ตรง)", key=f"other_{i}"):
                                    st.session_state.search_query = other_name
                                    st.rerun()
                    
                    st.session_state.messages.append({
                        "role": "assistant", "content": response, "recipe": recipe,
                        "nutrition_data": nutrition_data, "similarity_score": similarity
                    })
                else:
                    response = f"""
                    ❌ ไม่พบสูตรอาหารที่ตรงกับ '{search_query}'
                    
                    💡 **คำแนะนำสำหรับการค้นหา:**
                    - ลองใช้คำค้นหาที่ง่ายกว่า เช่น "ไข่เจียว" แทน "วิธีทำไข่เจียว"
                    - ตรวจสอบการสะกดคำภาษาไทย
                    - ลองค้นหาด้วยเมนูที่มีชื่อสั้นๆ
                    - ปรับค่าความเคร่งครัดในการค้นหาในแถบด้านซ้าย (ลดค่าลง)
                    
                    🍽️ **เมนูที่มีในระบบ:** {', '.join(data['name'].head(10).tolist())}
                    """
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
