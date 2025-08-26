import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
import pickle
import re
import difflib
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List, Optional, Tuple
import time
import csv

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Thai Food Nutrition Analyzer",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# สไตล์ CSS สำหรับฟอนต์ไทยและ UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600;700&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Sarabun', sans-serif !important;
    }
    
    .main-header {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .nutrition-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .ingredient-item {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 0.5rem;
        margin: 0.25rem 0;
    }
    
    .vitamin-mineral {
        background: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 0.3rem;
        margin: 0.2rem 0;
        font-size: 0.9rem;
    }
    
    .settings-panel {
        background: #f1f3f4;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .metric-box {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# คงตัวแปรพาธ
DATA_PATH = "thai_food_processed_cleaned.csv"
NUTRITION_PATH = "thai_ingredients_nutrition_data.csv"
EMBEDDINGS_PATH = "embeddings.pkl"
MODEL_PATH = "model"
USDA_API_KEY = st.secrets.get("USDA_API_KEY", "DEMO_KEY")

# คลาสหลักสำหรับจัดการข้อมูลโภชนาการ
class NutritionAnalyzer:
    def __init__(self):
        self.nutrition_data = self.load_nutrition_data()
        self.unit_conversions = self.setup_unit_conversions()
        self.cooking_adjustments = self.setup_cooking_adjustments()
        
    def load_nutrition_data(self) -> pd.DataFrame:
        """โหลดข้อมูลโภชนาการจากไฟล์"""
        try:
            if os.path.exists(NUTRITION_PATH):
                return pd.read_csv(NUTRITION_PATH)
            else:
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Error loading nutrition data: {e}")
            return pd.DataFrame()
    
    def setup_unit_conversions(self) -> Dict:
        """ตั้งค่าการแปลงหน่วยมาตรฐาน (แปลงเป็นกรัม)"""
        return {
            'กิโลกรัม': 1000, 'กก.': 1000, 'kg': 1000,
            'กรัม': 1, 'ก.': 1, 'g': 1,
            'ช้อนโต๊ะ': 15, 'ช้อนใหญ่': 15, 'tbsp': 15,
            'ช้อนชา': 5, 'ช้อนเล็ก': 5, 'tsp': 5,
            'ถ้วยตวง': 250, 'ถ้วย': 250, 'cup': 250,
            'ลูก': 50, 'หัว': 100, 'กิ่ง': 2, 'ซีก': 500,
            'แผ่น': 3, 'เส้น': 1, 'ฝัก': 10, 'เม็ด': 5,
            'ใบ': 1, 'ดอก': 0.5, 'แต้ม': 0.1
        }
    
    def setup_cooking_adjustments(self) -> Dict:
        """ตั้งค่าการปรับค่าตามวิธีปรุง"""
        return {
            'น้ำมัน': {
                'ทอด': 0.1,  # ใช้น้ำมันมากแต่บริโภคเพียง 10%
                'ผัด': 0.8,   # บริโภค 80%
                'ต้ม': 0.0    # ไม่บริโภคน้ำมันในการต้ม
            },
            'เกลือ': {
                'ดอง': 0.3,   # ใช้เกลือมากแต่บริโภคเพียง 30%
                'ผัด': 0.9,   # บริโภคเกือบหมด
                'ต้ม': 0.7    # บางส่วนละลายในน้ำ
            },
            'น้ำตาล': {
                'ทอด': 0.9,   # เกือบทั้งหมดติดอาหาร
                'ผัด': 0.9,
                'ต้ม': 0.8
            }
        }
    
    def estimate_ingredient_amount(self, ingredient_text: str) -> Tuple[float, str]:
        """ประมาณปริมาณวัตถุดิบจากข้อความ"""
        # ใช้ regex หาตัวเลขและหน่วย
        pattern = r'(\d+(?:\.\d+)?)\s*([^\d\s]+)'
        matches = re.findall(pattern, ingredient_text)
        
        if matches:
            amount, unit = matches[0]
            amount = float(amount)
            
            # แปลงหน่วยเป็นกรัม
            if unit in self.unit_conversions:
                return amount * self.unit_conversions[unit], 'กรัม'
        
        # ถ้าไม่พบหน่วย ใช้การประมาณจากคำสำคัญ
        if any(word in ingredient_text for word in ['เล็กน้อย', 'เล็ก', 'ปลีก']):
            return 5.0, 'กรัม'
        elif any(word in ingredient_text for word in ['กลาง', 'ปานกลาง']):
            return 50.0, 'กรัม'
        elif any(word in ingredient_text for word in ['ใหญ่', 'มาก']):
            return 100.0, 'กรัม'
        
        return 25.0, 'กรัม'  # ค่าเริ่มต้น
    
    def calculate_nutrition(self, ingredients: List[str], cooking_method: str = "", 
                          use_cooking_adjustments: bool = False) -> Dict:
        """คำนวณคุณค่าทางโภชนาการรวม"""
        total_nutrition = {
            'calories': 0, 'protein': 0, 'fat': 0, 'carbs': 0, 'fiber': 0,
            'vitamin_c': 0, 'calcium': 0, 'iron': 0, 'magnesium': 0,
            'phosphorus': 0, 'potassium': 0, 'zinc': 0, 'sodium': 0,
            'vitamin_b6': 0, 'vitamin_k': 0, 'vitamin_b1': 0, 'vitamin_b2': 0,
            'vitamin_b3': 0, 'folate': 0, 'vitamin_a': 0, 'vitamin_b12': 0,
            'vitamin_e': 0
        }
        
        detailed_breakdown = []
        
        for ingredient in ingredients:
            amount_g, _ = self.estimate_ingredient_amount(ingredient)
            
            # ค้นหาข้อมูลโภชนาการ
            nutrition_info = self.find_nutrition_match(ingredient)
            
            if nutrition_info is not None:
                # คำนวณตามสัดส่วน (ข้อมูลต่อ 100g)
                multiplier = amount_g / 100.0
                
                # ปรับค่าตามวิธีปรุงถ้าเปิดใช้งาน
                if use_cooking_adjustments:
                    multiplier = self.apply_cooking_adjustment(
                        ingredient, cooking_method, multiplier)
                
                item_nutrition = {}
                for col in total_nutrition.keys():
                    col_map = {
                        'calories': 'calories_per_100g',
                        'protein': 'protein_per_100g',
                        'fat': 'fat_per_100g', 
                        'carbs': 'carbohydrates_per_100g',
                        'fiber': 'fiber_per_100g'
                    }
                    
                    db_col = col_map.get(col, f'{col}_per_100g')
                    if db_col in nutrition_info.index:
                        value = pd.to_numeric(nutrition_info[db_col], errors='coerce')
                        if not pd.isna(value):
                            calculated_value = value * multiplier
                            total_nutrition[col] += calculated_value
                            item_nutrition[col] = calculated_value
                
                detailed_breakdown.append({
                    'ingredient': ingredient,
                    'amount_g': amount_g,
                    'nutrition': item_nutrition,
                    'usda_match': nutrition_info.get('english_name', 'Unknown')
                })
        
        return {
            'total': total_nutrition,
            'breakdown': detailed_breakdown
        }
    
    def find_nutrition_match(self, ingredient: str) -> Optional[pd.Series]:
        """ค้นหาข้อมูลโภชนาการที่ตรงกับวัตถุดิบ"""
        if self.nutrition_data.empty:
            return None
            
        ingredient_clean = self.clean_ingredient_name(ingredient)
        
        # ค้นหาแบบตรงก่อน
        exact_match = self.nutrition_data[
            self.nutrition_data['thai_name'].str.contains(ingredient_clean, na=False, case=False)
        ]
        
        if not exact_match.empty:
            return exact_match.iloc[0]
        
        # ค้นหาแบบใกล้เคียง
        thai_names = self.nutrition_data['thai_name'].dropna().tolist()
        closest_matches = difflib.get_close_matches(
            ingredient_clean, thai_names, n=1, cutoff=0.6)
        
        if closest_matches:
            match_row = self.nutrition_data[
                self.nutrition_data['thai_name'] == closest_matches[0]
            ]
            if not match_row.empty:
                return match_row.iloc[0]
        
        return None
    
    def clean_ingredient_name(self, ingredient: str) -> str:
        """ทำความสะอาดชื่อวัตถุดิบ"""
        # ลบตัวเลขและหน่วย
        cleaned = re.sub(r'\d+(?:\.\d+)?\s*[^\s]*', '', ingredient)
        # ลบคำอธิบายพิเศษ
        cleaned = re.sub(r'[()]', '', cleaned)
        return cleaned.strip()
    
    def apply_cooking_adjustment(self, ingredient: str, cooking_method: str, 
                               multiplier: float) -> float:
        """ปรับค่าตามวิธีปรุง"""
        ingredient_lower = ingredient.lower()
        cooking_lower = cooking_method.lower()
        
        for key, adjustments in self.cooking_adjustments.items():
            if key in ingredient_lower:
                for method, factor in adjustments.items():
                    if method in cooking_lower:
                        return multiplier * factor
        
        return multiplier

# คลาสสำหรับดึงข้อมูลจาก USDA API
class USDADataFetcher:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.nal.usda.gov/fdc/v1"
    
    def search_food(self, query: str, limit: int = 5) -> List[Dict]:
        """ค้นหาอาหารจาก USDA API"""
        try:
            url = f"{self.base_url}/foods/search"
            params = {
                'query': query,
                'dataType': ['Foundation', 'SR Legacy'],
                'pageSize': limit,
                'api_key': self.api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get('foods', [])
            else:
                st.error(f"USDA API Error: {response.status_code}")
                return []
                
        except Exception as e:
            st.error(f"Error fetching USDA data: {e}")
            return []
    
    def get_food_details(self, fdc_id: int) -> Optional[Dict]:
        """ดึงรายละเอียดอาหารจาก FDC ID"""
        try:
            url = f"{self.base_url}/food/{fdc_id}"
            params = {'api_key': self.api_key}
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
            return None
            
        except Exception as e:
            st.error(f"Error fetching food details: {e}")
            return None
    
    def export_to_csv(self, data_list: List[Dict], filename: str):
        """ส่งออกข้อมูลเป็นไฟล์ CSV"""
        try:
            df = pd.DataFrame(data_list)
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            return True
        except Exception as e:
            st.error(f"Error exporting to CSV: {e}")
            return False

# ฟังก์ชันโหลดโมเดล
@st.cache_resource
def load_model():
    """โหลดหรือดาวน์โหลดโมเดล sentence transformer"""
    if os.path.exists(MODEL_PATH):
        return SentenceTransformer(MODEL_PATH)
    else:
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        os.makedirs(MODEL_PATH, exist_ok=True)
        model.save(MODEL_PATH)
        return model

# ฟังก์ชันโหลดข้อมูล
@st.cache_data
def load_food_data():
    """โหลดข้อมูลอาหารไทย"""
    try:
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            # ตรวจสอบคอลัมน์ที่จำเป็น
            required_cols = ['name', 'text_ingradiant', 'food_method']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                return pd.DataFrame()
            return df
        else:
            st.error(f"Food data file not found: {DATA_PATH}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading food data: {e}")
        return pd.DataFrame()

# ฟังก์ชันดึง embeddings
@st.cache_data
def get_embeddings(_model, data):
    """ดึงหรือคำนวณ embeddings สำหรับสูตรอาหารทั้งหมด"""
    if os.path.exists(EMBEDDINGS_PATH):
        with open(EMBEDDINGS_PATH, 'rb') as f:
            return pickle.load(f)
    else:
        # รวมข้อความทั้งหมดสำหรับแต่ละสูตร
        texts = []
        for _, row in data.iterrows():
            combined_text = f"{row['name']} {row['text_ingradiant']} {row['food_method']}"
            texts.append(combined_text)
        
        # สร้าง embeddings
        embeddings = _model.encode(texts)
        
        # บันทึก embeddings
        with open(EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump(embeddings, f)
        
        return embeddings

# ฟังก์ชันค้นหาสูตรอาหาร
def search_recipes(query: str, model, data: pd.DataFrame, embeddings, 
                  top_k: int = 5, enable_fuzzy: bool = True) -> List[Dict]:
    """ค้นหาสูตรอาหารตามคำถาม"""
    # ค้นหาแบบ semantic search
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(-similarities)[:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'name': data.iloc[idx]['name'],
            'similarity': similarities[idx],
            'ingredients': data.iloc[idx]['text_ingradiant'],
            'method': data.iloc[idx]['food_method'],
            'index': idx
        })
    
    # ถ้าผลลัพธ์ไม่ดีพอและเปิดการค้นหาแบบ fuzzy
    if enable_fuzzy and (not results or results[0]['similarity'] < 0.3):
        fuzzy_results = fuzzy_search_recipes(query, data, top_k)
        
        # รวมผลลัพธ์และเรียงตามความเกี่ยวข้อง
        all_results = results + fuzzy_results
        all_results = sorted(all_results, key=lambda x: x['similarity'], reverse=True)
        results = all_results[:top_k]
    
    return results

def fuzzy_search_recipes(query: str, data: pd.DataFrame, top_k: int = 5) -> List[Dict]:
    """ค้นหาแบบ fuzzy matching"""
    food_names = data['name'].tolist()
    close_matches = difflib.get_close_matches(query, food_names, n=top_k, cutoff=0.4)
    
    results = []
    for match in close_matches:
        idx = data[data['name'] == match].index[0]
        # คำนวณความคล้ายคลึงจาก difflib
        similarity = difflib.SequenceMatcher(None, query, match).ratio()
        
        results.append({
            'name': match,
            'similarity': similarity,
            'ingredients': data.iloc[idx]['text_ingradiant'],
            'method': data.iloc[idx]['food_method'],
            'index': idx,
            'type': 'fuzzy'
        })
    
    return results

# ฟังก์ชันแนะนำอาหารตามโภชนาการ
def recommend_by_nutrition(data: pd.DataFrame, nutrition_analyzer: NutritionAnalyzer,
                          criteria: Dict, limit: int = 5) -> List[Dict]:
    """แนะนำอาหารตามเกณฑ์โภชนาการ"""
    recommendations = []
    
    for idx, row in data.iterrows():
        ingredients = row['text_ingradiant'].split('\n') if pd.notna(row['text_ingradiant']) else []
        nutrition = nutrition_analyzer.calculate_nutrition(ingredients, row['food_method'])
        
        # ตรวจสอบเกณฑ์
        meets_criteria = True
        total_nutrition = nutrition['total']
        
        for nutrient, (operator, value) in criteria.items():
            nutrient_value = total_nutrition.get(nutrient, 0)
            
            if operator == 'low' and nutrient_value > value:
                meets_criteria = False
                break
            elif operator == 'high' and nutrient_value < value:
                meets_criteria = False
                break
            elif operator == 'range' and not (value[0] <= nutrient_value <= value[1]):
                meets_criteria = False
                break
        
        if meets_criteria:
            recommendations.append({
                'name': row['name'],
                'nutrition': total_nutrition,
                'ingredients': row['text_ingradiant'],
                'method': row['food_method'],
                'index': idx
            })
    
    # เรียงลำดับตามความเหมาะสม
    recommendations = sorted(recommendations, 
                           key=lambda x: sum(x['nutrition'].values()), 
                           reverse=True)[:limit]
    
    return recommendations

# ฟังก์ชันแสดงผลโภชนาการ
def display_nutrition_info(nutrition_data: Dict, show_details: bool = True):
    """แสดงข้อมูลโภชนาการในรูปแบบที่เข้าใจง่าย"""
    total = nutrition_data['total']
    
    # แสดงข้อมูลหลัก
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <h4>🔥 แคลอรี่</h4>
            <h2>{total['calories']:.1f} kcal</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <h4>🥩 โปรตีน</h4>
            <h2>{total['protein']:.1f} g</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <h4>🥑 ไขมัน</h4>
            <h2>{total['fat']:.1f} g</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-box">
            <h4>🍞 คาร์โบไหดเรต</h4>
            <h2>{total['carbs']:.1f} g</h2>
        </div>
        """, unsafe_allow_html=True)
    
    if show_details:
        # แสดงวิตามินและแร่ธาตุ
        st.markdown("### 🌟 วิตามินและแร่ธาตุ")
        
        vitamins_col1, vitamins_col2 = st.columns(2)
        
        with vitamins_col1:
            st.markdown("**วิตามิน**")
            vitamins = {
                'วิตามิน C': total['vitamin_c'],
                'วิตามิน A': total['vitamin_a'], 
                'วิตามิน B1': total['vitamin_b1'],
                'วิตามิน B2': total['vitamin_b2'],
                'วิตามิน B3': total['vitamin_b3'],
                'วิตามิน B6': total['vitamin_b6'],
                'วิตามิน B12': total['vitamin_b12'],
                'วิตามิน E': total['vitamin_e'],
                'วิตามิน K': total['vitamin_k'],
                'โฟเลต': total['folate']
            }
            
            for name, value in vitamins.items():
                if value > 0:
                    unit = 'mg' if 'วิตามิน' in name else 'mcg'
                    st.markdown(f"""
                    <div class="vitamin-mineral">
                        {name}: <strong>{value:.2f} {unit}</strong>
                    </div>
                    """, unsafe_allow_html=True)
        
        with vitamins_col2:
            st.markdown("**แร่ธาตุ**")
            minerals = {
                'แคลเซียม': total['calcium'],
                'เหล็ก': total['iron'],
                'แมกนีเซียม': total['magnesium'],
                'ฟอสฟอรัส': total['phosphorus'],
                'โพแทสเซียม': total['potassium'],
                'สังกะสี': total['zinc'],
                'โซเดียม': total['sodium'],
                'ใยอาหาร': total['fiber']
            }
            
            for name, value in minerals.items():
                if value > 0:
                    st.markdown(f"""
                    <div class="vitamin-mineral">
                        {name}: <strong>{value:.2f} mg</strong>
                    </div>
                    """, unsafe_allow_html=True)

# ฟังก์ชันแสดงรายละเอียดวัตถุดิบ
def display_ingredient_breakdown(nutrition_data: Dict):
    """แสดงรายละเอียดโภชนาการแยกตามวัตถุดิบ"""
    breakdown = nutrition_data['breakdown']
    
    st.markdown("### 📊 รายละเอียดตามวัตถุดิบ")
    
    for item in breakdown:
        with st.expander(f"🥘 {item['ingredient']} ({item['amount_g']:.1f}g)"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**USDA Match:** {item['usda_match']}")
                st.write(f"**แคลอรี่:** {item['nutrition']['calories']:.1f} kcal")
                st.write(f"**โปรตีน:** {item['nutrition']['protein']:.1f} g")
                st.write(f"**ไขมัน:** {item['nutrition']['fat']:.1f} g")
            
            with col2:
                st.write(f"**คาร์โบไหดเรต:** {item['nutrition']['carbs']:.1f} g")
                st.write(f"**ใยอาหาร:** {item['nutrition']['fiber']:.1f} g")
                st.write(f"**แคลเซียม:** {item['nutrition']['calcium']:.1f} mg")
                st.write(f"**เหล็ก:** {item['nutrition']['iron']:.2f} mg")

# ฟังก์ชันหลัก
def main():
    # ส่วนหัว
    st.markdown("""
    <div class="main-header">
        <h1>🍲 ระบบวิเคราะห์คุณค่าทางโภชนาการอาหารไทย</h1>
        <p>Thai Food Nutrition Analyzer with USDA Integration</p>
    </div>
    """, unsafe_allow_html=True)
    
    # โหลดข้อมูลและโมเดล
    with st.spinner("กำลังโหลดระบบ..."):
        model = load_model()
        food_data = load_food_data()
        
        if food_data.empty:
            st.error("ไม่สามารถโหลดข้อมูลอาหารได้ กรุณาตรวจสอบไฟล์ข้อมูล")
            return
            
        embeddings = get_embeddings(model, food_data)
        nutrition_analyzer = NutritionAnalyzer()
        usda_fetcher = USDADataFetcher(USDA_API_KEY)
    
    # แถบเลื่อนการตั้งค่า
    with st.sidebar:
        st.markdown("## ⚙️ การตั้งค่าเพิ่มเติม")
        
        # ส่วนการตั้งค่า USDA API
        st.markdown("### 🌐 USDA API Settings")
        
        if st.button("💾 บันทึกข้อมูลโภชนาการใหม่"):
            with st.spinner("กำลังดึงข้อมูลจาก USDA..."):
                # ดึงรายการวัตถุดิบทั้งหมดที่ไม่ซ้ำ
                all_ingredients = set()
                for ingredients_text in food_data['text_ingradiant'].dropna():
                    for ingredient in ingredients_text.split('\n'):
                        if ingredient.strip():
                            clean_name = nutrition_analyzer.clean_ingredient_name(ingredient)
                            if clean_name:
                                all_ingredients.add(clean_name)
                
                progress_bar = st.progress(0)
                new_data = []
                
                for i, ingredient in enumerate(all_ingredients):
                    progress_bar.progress((i + 1) / len(all_ingredients))
                    
                    # ค้นหาใน USDA
                    usda_results = usda_fetcher.search_food(ingredient, limit=1)
                    
                    if usda_results:
                        food_item = usda_results[0]
                        details = usda_fetcher.get_food_details(food_item['fdcId'])
                        
                        if details:
                            # แปลงข้อมูลเป็นรูปแบบที่ต้องการ
                            nutrition_row = {
                                'thai_name': ingredient,
                                'english_name': food_item.get('description', ''),
                                'usda_description': food_item.get('description', ''),
                                'fdc_id': food_item.get('fdcId', ''),
                                'data_type': food_item.get('dataType', ''),
                                'publication_date': datetime.now().strftime('%Y-%m-%d')
                            }
                            
                            # เพิ่มข้อมูลโภชนาการ
                            for nutrient in details.get('foodNutrients', []):
                                nutrient_name = nutrient.get('nutrient', {}).get('name', '')
                                nutrient_value = nutrient.get('amount', 0)
                                
                                # แมปชื่อสารอาหารกับคอลัมน์
                                if 'Energy' in nutrient_name:
                                    nutrition_row['calories_per_100g'] = nutrient_value
                                elif 'Protein' in nutrient_name:
                                    nutrition_row['protein_per_100g'] = nutrient_value
                                # เพิ่มการแมปอื่นๆ...
                            
                            new_data.append(nutrition_row)
                    
                    time.sleep(0.1)  # หน่วงเวลาเพื่อไม่ให้เกิน API rate limit
                
                # บันทึกข้อมูลใหม่
                if new_data:
                    new_df = pd.DataFrame(new_data)
                    new_df.to_csv(f"usda_nutrition_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", 
                                  index=False, encoding='utf-8-sig')
                    st.success(f"บันทึกข้อมูลโภชนาการใหม่แล้ว {len(new_data)} รายการ")
                else:
                    st.warning("ไม่พบข้อมูลใหม่จาก USDA API")
        
        # การตั้งค่าการคำนวณ
        st.markdown("### 🧮 การตั้งค่าการคำนวณ")
        use_cooking_adjustments = st.checkbox(
            "ปรับค่าตามวิธีปรุงอาหาร",
            value=False,
            help="คำนวณปริมาณที่บริโภคจริงตามวิธีการทำอาหาร เช่น น้ำมันที่ใช้ทอดจะบริโภคเพียงส่วนน้อย"
        )
        
        show_detailed_breakdown = st.checkbox(
            "แสดงรายละเอียดตามวัตถุดิบ",
            value=True,
            help="แสดงการแยกค่าโภชนาการตามวัตถุดิบแต่ละชนิด"
        )
        
        enable_fuzzy_search = st.checkbox(
            "เปิดการค้นหาแบบยืดหยุ่น",
            value=True,
            help="รองรับการพิมพ์ผิดและค้นหาจากความคล้ายคลึงของชื่อ"
        )
    
    # แท็บหลัก
    tab1, tab2, tab3 = st.tabs(["🔍 ค้นหาอาหาร", "🎯 แนะนำตามโภชนาการ", "📚 เรียนรู้เพิ่มเติม"])
    
    with tab1:
        st.markdown("## ค้นหาสูตรอาหารและวิเคราะห์คุณค่าทางโภชนาการ")
        
        # ช่องค้นหา
        search_query = st.text_input(
            "พิมพ์ชื่ออาหารหรือคำถามเกี่ยวกับอาหารไทย:",
            placeholder="เช่น ต้มยำกุ้ง, ผัดไทย, อาหารที่มีโปรตีนสูง"
        )
        
        if search_query:
            with st.spinner("กำลังค้นหา..."):
                results = search_recipes(
                    search_query, model, food_data, embeddings,
                    top_k=5, enable_fuzzy=enable_fuzzy_search
                )
            
            if results and results[0]['similarity'] > 0.2:
                st.markdown("### 🍽️ ผลการค้นหา")
                
                # แสดงผลลัพธ์แรก
                best_result = results[0]
                
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    st.markdown(f"## {best_result['name']}")
                    st.markdown(f"**ความเกี่ยวข้อง:** {best_result['similarity']:.2f}")
                    
                    # แสดงวัตถุดิบ
                    st.markdown("### 🥘 วัตถุดิบ")
                    ingredients_list = best_result['ingredients'].split('\n') if best_result['ingredients'] else []
                    for ingredient in ingredients_list:
                        if ingredient.strip():
                            st.markdown(f"""
                            <div class="ingredient-item">
                                • {ingredient.strip()}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # แสดงวิธีทำ
                    st.markdown("### 👨‍🍳 วิธีทำ")
                    st.write(best_result['method'])
                
                with col2:
                    # คำนวณคุณค่าทางโภชนาการ
                    nutrition_data = nutrition_analyzer.calculate_nutrition(
                        ingredients_list, 
                        best_result['method'],
                        use_cooking_adjustments
                    )
                    
                    # แสดงข้อมูลโภชนาการ
                    display_nutrition_info(nutrition_data, show_details=True)
                
                # แสดงรายละเอียดวัตถุดิบ
                if show_detailed_breakdown:
                    display_ingredient_breakdown(nutrition_data)
                
                # แสดงผลการค้นหาอื่นๆ
                if len(results) > 1:
                    st.markdown("### 🍴 อาหารอื่นที่คล้ายกัน")
                    for i, result in enumerate(results[1:4], 1):
                        with st.expander(f"{i}. {result['name']} (ความเกี่ยวข้อง: {result['similarity']:.2f})"):
                            ingredients = result['ingredients'].split('\n') if result['ingredients'] else []
                            quick_nutrition = nutrition_analyzer.calculate_nutrition(
                                ingredients, result['method'], use_cooking_adjustments)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**วัตถุดิบ:**")
                                for ingredient in ingredients[:5]:
                                    if ingredient.strip():
                                        st.write(f"• {ingredient.strip()}")
                            
                            with col2:
                                total = quick_nutrition['total']
                                st.write(f"**แคลอรี่:** {total['calories']:.0f} kcal")
                                st.write(f"**โปรตีน:** {total['protein']:.1f} g")
                                st.write(f"**ไขมัน:** {total['fat']:.1f} g")
                                st.write(f"**คาร์โบไหดเรต:** {total['carbs']:.1f} g")
            else:
                st.warning("ไม่พบอาหารที่ตรงกับคำค้นหา กรุณาลองใช้คำค้นหาอื่น")
    
    with tab2:
        st.markdown("## 🎯 แนะนำอาหารตามความต้องการทางโภชนาการ")
        
        # เลือกเกณฑ์
        criteria_type = st.selectbox(
            "เลือกประเภทความต้องการ:",
            ["แคลอรี่ต่ำ", "โปรตีนสูง", "ไขมันต่ำ", "คาร์โบไหดเรตต่ำ", 
             "แคลเซียมสูง", "เหล็กสูง", "ใยอาหารสูง", "กำหนดเอง"]
        )
        
        # กำหนดเกณฑ์
        criteria = {}
        if criteria_type == "แคลอรี่ต่ำ":
            criteria = {'calories': ('low', 200)}
        elif criteria_type == "โปรตีนสูง":
            criteria = {'protein': ('high', 15)}
        elif criteria_type == "ไขมันต่ำ":
            criteria = {'fat': ('low', 10)}
        elif criteria_type == "คาร์โบไหดเรตต่ำ":
            criteria = {'carbs': ('low', 20)}
        elif criteria_type == "แคลเซียมสูง":
            criteria = {'calcium': ('high', 100)}
        elif criteria_type == "เหล็กสูง":
            criteria = {'iron': ('high', 3)}
        elif criteria_type == "ใยอาหารสูง":
            criteria = {'fiber': ('high', 5)}
        elif criteria_type == "กำหนดเอง":
            st.markdown("### กำหนดเกณฑ์ด้วยตนเอง")
            col1, col2 = st.columns(2)
            
            with col1:
                max_calories = st.number_input("แคลอรี่สูงสุด (kcal)", min_value=0, value=300)
                min_protein = st.number_input("โปรตีนขั้นต่ำ (g)", min_value=0.0, value=0.0, step=0.5)
                max_fat = st.number_input("ไขมันสูงสุด (g)", min_value=0.0, value=50.0, step=0.5)
            
            with col2:
                max_carbs = st.number_input("คาร์โบไหดเรตสูงสุด (g)", min_value=0.0, value=50.0, step=0.5)
                min_calcium = st.number_input("แคลเซียมขั้นต่ำ (mg)", min_value=0.0, value=0.0, step=1.0)
                min_fiber = st.number_input("ใยอาหารขั้นต่ำ (g)", min_value=0.0, value=0.0, step=0.5)
            
            criteria = {
                'calories': ('low', max_calories),
                'protein': ('high', min_protein),
                'fat': ('low', max_fat),
                'carbs': ('low', max_carbs),
                'calcium': ('high', min_calcium),
                'fiber': ('high', min_fiber)
            }
        
        if st.button("🔍 ค้นหาอาหารที่เหมาะสม"):
            with st.spinner("กำลังวิเคราะห์..."):
                recommendations = recommend_by_nutrition(
                    food_data, nutrition_analyzer, criteria, limit=10)
            
            if recommendations:
                st.markdown(f"### ✨ พบอาหารที่เหมาะสม {len(recommendations)} รายการ")
                
                for i, rec in enumerate(recommendations, 1):
                    with st.expander(f"{i}. {rec['name']}"):
                        col1, col2 = st.columns([2, 3])
                        
                        with col1:
                            total = rec['nutrition']
                            st.metric("แคลอรี่", f"{total['calories']:.0f} kcal")
                            st.metric("โปรตีน", f"{total['protein']:.1f} g")
                            st.metric("ไขมัน", f"{total['fat']:.1f} g")
                            st.metric("คาร์โบไหดเรต", f"{total['carbs']:.1f} g")
                        
                        with col2:
                            st.write("**วัตถุดิบ:**")
                            ingredients = rec['ingredients'].split('\n')[:5]
                            for ingredient in ingredients:
                                if ingredient.strip():
                                    st.write(f"• {ingredient.strip()}")
                            
                            if len(rec['ingredients'].split('\n')) > 5:
                                st.write("...")
            else:
                st.info("ไม่พบอาหารที่ตรงกับเกณฑ์ที่กำหนด")
    
    with tab3:
        st.markdown("## 📚 เรียนรู้เพิ่มเติมเกี่ยวกับระบบ")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🔬 ข้อมูลที่ใช้ในระบบ
            
            - **ฐานข้อมูลสูตรอาหารไทย:** {total_recipes} สูตร
            - **ข้อมูลโภชนาการ USDA:** {nutrition_items} รายการ
            - **โมเดล AI:** Multilingual Sentence Transformer
            - **การค้นหา:** Semantic Search + Fuzzy Matching
            
            ### ⚙️ วิธีการทำงาน
            
            1. **การค้นหา:** ใช้ AI เข้าใจความหมายของคำค้นหา
            2. **การจับคู่:** จับคู่วัตถุดิบไทยกับฐานข้อมูล USDA
            3. **การคำนวณ:** ประมาณปริมาณและคำนวณโภชนาการ
            4. **การปรับค่า:** ปรับตามวิธีการประกอบอาหาร
            """.format(
                total_recipes=len(food_data),
                nutrition_items=len(nutrition_analyzer.nutrition_data)
            ))
        
        with col2:
            st.markdown("""
            ### 💡 เคล็ดลับการใช้งาน
            
            - **คำค้นหา:** สามารถใช้ชื่ออาหาร หรือวัตถุดิบ
            - **ความยืดหยุ่น:** รองรับการพิมพ์ผิด
            - **การปรับแต่ง:** ใช้แถบตั้งค่าด้านซ้าย
            - **ข้อมูลแม่นยำ:** เปิดการปรับค่าตามวิธีปรุง
            
            ### 🚀 ฟีเจอร์พิเศษ
            
            - วิเคราะห์วิตามินและแร่ธาตุครบถ้วน
            - ปรับค่าตามวิธีการทำอาหารจริง
            - แนะนำอาหารตามความต้องการ
            - ดึงข้อมูลใหม่จาก USDA API
            - รองรับการค้นหาแบบไทย-อังกฤษ
            """)
        
        # สถิติการใช้งาน
        st.markdown("### 📊 สถิติระบบ")
        
        if 'usage_stats' not in st.session_state:
            st.session_state.usage_stats = {
                'searches': 0,
                'recommendations': 0,
                'usda_queries': 0
            }
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("การค้นหาทั้งหมด", st.session_state.usage_stats['searches'])
        with col2:
            st.metric("การแนะนำ", st.session_state.usage_stats['recommendations'])
        with col3:
            st.metric("คำขอ USDA API", st.session_state.usage_stats['usda_queries'])

if __name__ == "__main__":
    main()
