import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import re
import difflib
from typing import Dict, List, Optional, Tuple
import requests
import json
from datetime import datetime

# ใช้ try-except สำหรับ optional packages
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    cosine_similarity = None
    TfidfVectorizer = None
    SKLEARN_AVAILABLE = False

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Thai Food Nutrition Analyzer",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# สไตล์ CSS สำหรับฟอนต์ไทยและ UI ที่สวยงาม
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600;700&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Sarabun', sans-serif !important;
    }
    
    .main-header {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .recipe-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .recipe-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .ingredient-list {
        background: #f8f9fa;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    
    .nutrition-metric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .vitamin-mineral-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .vitamin-item {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 0.5rem;
        border-radius: 5px;
        font-size: 0.9rem;
    }
    
    .mineral-item {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 0.5rem;
        border-radius: 5px;
        font-size: 0.9rem;
    }
    
    .search-tips {
        background: #f0f8ff;
        border: 1px solid #b3d9ff;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .similarity-score {
        display: inline-block;
        background: #28a745;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .low-similarity {
        background: #ffc107 !important;
        color: #333 !important;
    }
</style>
""", unsafe_allow_html=True)

# ค่าคงที่สำหรับไฟล์
DATA_PATH = "thai_food_processed_cleaned.csv"  # ลองหาไฟล์ใหม่ก่อน
LEGACY_DATA_PATH = "thai_food_processed.csv"   # ไฟล์เก่า
NUTRITION_PATH = "thai_ingredients_nutrition_data.csv"
EMBEDDINGS_PATH = "embeddings.pkl"
MODEL_PATH = "model"

# คลาส NutritionCalculator แบบง่าย
class SimpleNutritionCalculator:
    """คลาสสำหรับคำนวณโภชนาการอย่างง่าย"""
    
    def __init__(self):
        # ข้อมูลโภชนาการพื้นฐานสำหรับวัตถุดิบทั่วไป (ต่อ 100 กรัม)
        self.basic_nutrition = {
            'ข้าว': {'calories': 130, 'protein': 2.7, 'fat': 0.3, 'carbs': 28},
            'เนื้อหมู': {'calories': 242, 'protein': 27, 'fat': 14, 'carbs': 0},
            'ไก่': {'calories': 165, 'protein': 31, 'fat': 3.6, 'carbs': 0},
            'กุ้ง': {'calories': 106, 'protein': 20, 'fat': 1.7, 'carbs': 0.9},
            'ปลา': {'calories': 206, 'protein': 22, 'fat': 12, 'carbs': 0},
            'ไข่': {'calories': 155, 'protein': 13, 'fat': 11, 'carbs': 1.1},
            'มะเขือเทศ': {'calories': 18, 'protein': 0.9, 'fat': 0.2, 'carbs': 3.9},
            'หอมใหญ่': {'calories': 40, 'protein': 1.1, 'fat': 0.1, 'carbs': 9.3},
            'กระเทียม': {'calories': 149, 'protein': 6.4, 'fat': 0.5, 'carbs': 33},
            'พริก': {'calories': 40, 'protein': 1.9, 'fat': 0.4, 'carbs': 7.3},
            'น้ำมัน': {'calories': 884, 'protein': 0, 'fat': 100, 'carbs': 0},
            'น้ำตาล': {'calories': 387, 'protein': 0, 'fat': 0, 'carbs': 100},
            'เกลือ': {'calories': 0, 'protein': 0, 'fat': 0, 'carbs': 0},
        }
        
    def estimate_ingredient_amount(self, ingredient_text: str) -> float:
        """ประมาณปริมาณวัตถุดิบในหน่วยกรัม"""
        # ลองหาตัวเลขในข้อความ
        numbers = re.findall(r'(\d+(?:\.\d+)?)', ingredient_text)
        
        if numbers:
            amount = float(numbers[0])
            
            # ตรวจสอบหน่วยและแปลงเป็นกรัม
            if any(unit in ingredient_text for unit in ['กิโลกรัม', 'กก.', 'kg']):
                return amount * 1000
            elif any(unit in ingredient_text for unit in ['กรัม', 'ก.', 'g']):
                return amount
            elif any(unit in ingredient_text for unit in ['ช้อนโต๊ะ', 'ช้อนใหญ่']):
                return amount * 15
            elif any(unit in ingredient_text for unit in ['ช้อนชา', 'ช้อนเล็ก']):
                return amount * 5
            elif any(unit in ingredient_text for unit in ['ถ้วย', 'ถ้วยตวง']):
                return amount * 200
        
        # ถ้าไม่มีตัวเลข ใช้การประมาณ
        if any(word in ingredient_text for word in ['เล็กน้อย', 'นิด']):
            return 5
        elif any(word in ingredient_text for word in ['กลาง', 'ปานกลาง']):
            return 30
        elif any(word in ingredient_text for word in ['มาก', 'เยอะ']):
            return 50
        
        return 20  # ค่าเริ่มต้น
    
    def find_nutrition_match(self, ingredient: str) -> Dict:
        """หาข้อมูลโภชนาการที่ตรงกับวัตถุดิบ"""
        ingredient_lower = ingredient.lower()
        
        # ลองจับคู่กับข้อมูลที่มี
        for key, nutrition in self.basic_nutrition.items():
            if key in ingredient or any(word in ingredient_lower for word in key.split()):
                return nutrition
        
        # ถ้าไม่พบ ใช้ค่าเริ่มต้นตามประเภท
        if any(word in ingredient_lower for word in ['เนื้อ', 'หมู', 'วัว']):
            return self.basic_nutrition['เนื้อหมู']
        elif any(word in ingredient_lower for word in ['ไก่', 'เป็ด']):
            return self.basic_nutrition['ไก่']
        elif any(word in ingredient_lower for word in ['ปลา', 'กุ้ง', 'ปู', 'หอย']):
            return self.basic_nutrition['ปลา']
        elif any(word in ingredient_lower for word in ['ผัก', 'ใบ']):
            return {'calories': 25, 'protein': 2, 'fat': 0.3, 'carbs': 4}
        elif any(word in ingredient_lower for word in ['น้ำมัน', 'มัน']):
            return self.basic_nutrition['น้ำมัน']
        else:
            return {'calories': 30, 'protein': 1, 'fat': 0.5, 'carbs': 6}
    
    def calculate_recipe_nutrition(self, ingredients_text: str) -> Dict:
        """คำนวณคุณค่าทางโภชนาการของสูตรอาหาร"""
        total = {'calories': 0, 'protein': 0, 'fat': 0, 'carbs': 0}
        
        if not ingredients_text:
            return total
            
        ingredients = [line.strip() for line in ingredients_text.split('\n') if line.strip()]
        
        for ingredient in ingredients:
            # ลบ prefix
            clean_ingredient = re.sub(r'^[-•*]\s*', '', ingredient)
            
            # ประมาณปริมาณ
            amount_g = self.estimate_ingredient_amount(clean_ingredient)
            
            # หาข้อมูลโภชนาการ
            nutrition = self.find_nutrition_match(clean_ingredient)
            
            # คำนวณตามสัดส่วน (ข้อมูลเป็นต่อ 100 กรัม)
            multiplier = amount_g / 100.0
            
            for nutrient in total.keys():
                total[nutrient] += nutrition.get(nutrient, 0) * multiplier
        
        return total

# ฟังก์ชันโหลดโมเดล
@st.cache_resource
def load_model():
    """โหลดหรือดาวน์โหลดโมเดล sentence transformer"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        st.warning("⚠️ ไม่พบ Sentence Transformers - จะใช้การค้นหาแบพื้นฐาน")
        return None
        
    try:
        if os.path.exists(MODEL_PATH):
            return SentenceTransformer(MODEL_PATH)
        else:
            with st.spinner("กำลังดาวน์โหลดโมเดล AI... (ใช้เวลาประมาณ 2-3 นาที)"):
                model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                os.makedirs(MODEL_PATH, exist_ok=True)
                model.save(MODEL_PATH)
                return model
    except Exception as e:
        st.error(f"ไม่สามารถโหลดโมเดลได้: {e}")
        st.info("💡 จะใช้การค้นหาแบบพื้นฐานแทน")
        return None

# ฟังก์ชันโหลดข้อมูลอาหาร
@st.cache_data
def load_food_data():
    """โหลดข้อมูลอาหารไทย"""
    # ลองหาไฟล์ใหม่ก่อน
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        # ตรวจสอบชื่อคอลัมน์
        if 'text_ingradiant' in df.columns:
            df = df.rename(columns={'text_ingradiant': 'ingredient'})
        if 'food_method' in df.columns:
            df = df.rename(columns={'food_method': 'method'})
        return df
    # ถ้าไม่มี ลองใช้ไฟล์เก่า
    elif os.path.exists(LEGACY_DATA_PATH):
        df = pd.read_csv(LEGACY_DATA_PATH)
        if 'text_ingradiant' in df.columns:
            df = df.rename(columns={'text_ingradiant': 'ingredient'})
        if 'food_method' in df.columns:
            df = df.rename(columns={'food_method': 'method'})
        return df
    else:
        st.error("ไม่พบไฟล์ข้อมูลอาหาร กรุณาตรวจสอบไฟล์ thai_food_processed.csv หรือ thai_food_processed_cleaned.csv")
        return pd.DataFrame()

# ฟังก์ชันดึง embeddings
@st.cache_data
def get_embeddings(_model, data):
    """ดึงหรือคำนวณ embeddings สำหรับสูตรอาหารทั้งหมด"""
    if _model is None or not SENTENCE_TRANSFORMERS_AVAILABLE:
        # ใช้ TF-IDF แทนถ้าไม่มี Sentence Transformers
        return get_tfidf_embeddings(data)
        
    if os.path.exists(EMBEDDINGS_PATH):
        try:
            with open(EMBEDDINGS_PATH, 'rb') as f:
                return pickle.load(f)
        except:
            pass
    
    if data.empty:
        return np.array([])
    
    # รวมข้อความทั้งหมดสำหรับแต่ละสูตร
    texts = []
    for _, row in data.iterrows():
        ingredient_text = str(row.get('ingredient', ''))
        method_text = str(row.get('method', ''))
        combined_text = f"{row['name']} {ingredient_text} {method_text}"
        texts.append(combined_text)
    
    # สร้าง embeddings
    with st.spinner("กำลังสร้างดัชนีการค้นหา... (ใช้เวลาประมาณ 1-2 นาที)"):
        embeddings = _model.encode(texts)
    
    # บันทึก embeddings
    try:
        with open(EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump(embeddings, f)
    except:
        pass
    
    return embeddings

@st.cache_data
def get_tfidf_embeddings(data):
    """สร้าง TF-IDF embeddings แทน Sentence Transformers"""
    if data.empty:
        return np.array([])
    
    # รวมข้อความทั้งหมดสำหรับแต่ละสูตร
    texts = []
    for _, row in data.iterrows():
        ingredient_text = str(row.get('ingredient', ''))
        method_text = str(row.get('method', ''))
        combined_text = f"{row['name']} {ingredient_text} {method_text}"
        texts.append(combined_text)
    
    # ใช้ TF-IDF หากไม่มี scikit-learn ใช้ simple word count
    if SKLEARN_AVAILABLE:
        vectorizer = TfidfVectorizer(max_features=1000)
        embeddings = vectorizer.fit_transform(texts).toarray()
        return embeddings
    else:
def create_simple_embeddings(texts):
    """สร้าง simple embeddings โดยใช้ word count"""
    # สร้าง vocabulary จากคำทั้งหมด
    all_words = set()
    processed_texts = []
    
    for text in texts:
        words = re.findall(r'\w+', text.lower())
        processed_texts.append(words)
        all_words.update(words)
    
    vocab = list(all_words)[:1000]  # จำกัดแค่ 1000 คำ
    
    # สร้าง embeddings จาก word count
    embeddings = []
    for words in processed_texts:
        vector = [words.count(word) for word in vocab]
        embeddings.append(vector)
    
    return np.array(embeddings)

def simple_cosine_similarity(query_vec, embeddings):
    """คำนวณ cosine similarity แบบง่าย"""
    similarities = []
    query_norm = np.linalg.norm(query_vec)
    
    for embedding in embeddings:
        if query_norm == 0 or np.linalg.norm(embedding) == 0:
            similarities.append(0)
        else:
            dot_product = np.dot(query_vec, embedding)
            similarity = dot_product / (query_norm * np.linalg.norm(embedding))
            similarities.append(similarity)
    
    return np.array(similarities)

# ฟังก์ชันค้นหาสูตรอาหาร
def search_recipes(query: str, model, data: pd.DataFrame, embeddings, top_k: int = 5):
    """ค้นหาสูตরอาหารตามคำถาม"""
    if model is None or data.empty or len(embeddings) == 0:
        return []
    
    # ค้นหาแบบ semantic search
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(-similarities)[:top_k]
    
    results = []
    for idx in top_indices:
        if idx < len(data):
            results.append({
                'name': data.iloc[idx]['name'],
                'similarity': similarities[idx],
                'ingredients': data.iloc[idx].get('ingredient', ''),
                'method': data.iloc[idx].get('method', ''),
                'index': idx
            })
    
    # ถ้าผลลัพธ์ไม่ดี ลองค้นหาแบบ fuzzy
    if not results or results[0]['similarity'] < 0.3:
        food_names = data['name'].tolist()
        close_matches = difflib.get_close_matches(query, food_names, n=top_k, cutoff=0.4)
        
        for match in close_matches:
            idx = data[data['name'] == match].index[0]
            similarity = difflib.SequenceMatcher(None, query, match).ratio()
            results.append({
                'name': match,
                'similarity': similarity,
                'ingredients': data.iloc[idx].get('ingredient', ''),
                'method': data.iloc[idx].get('method', ''),
                'index': idx,
                'type': 'fuzzy'
            })
    
    return results

# ฟังก์ชันแสดงผลโภชนาการ
def display_nutrition_card(nutrition_data: Dict, title: str = "ค่าโภชนาการ"):
    """แสดงข้อมูลโภชนาการในรูปแบบการ์ด"""
    st.markdown(f"### 📊 {title}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="nutrition-metric">
            <h4>🔥 แคลอรี่</h4>
            <h2>{nutrition_data.get('calories', 0):.0f}</h2>
            <p>kcal</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="nutrition-metric">
            <h4>🥩 โปรตีน</h4>
            <h2>{nutrition_data.get('protein', 0):.1f}</h2>
            <p>g</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="nutrition-metric">
            <h4>🥑 ไขมัน</h4>
            <h2>{nutrition_data.get('fat', 0):.1f}</h2>
            <p>g</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="nutrition-metric">
            <h4>🍞 คาร์โบไหดเรต</h4>
            <h2>{nutrition_data.get('carbs', 0):.1f}</h2>
            <p>g</p>
        </div>
        """, unsafe_allow_html=True)

# ฟังก์ชันแสดงรายการวัตถุดิบ
def display_ingredients(ingredients_text: str):
    """แสดงรายการวัตถุดิบในรูปแบบที่สวยงาม"""
    if not ingredients_text:
        return
        
    st.markdown("### 🥘 วัตถุดิบ")
    
    ingredients = [line.strip() for line in ingredients_text.split('\n') if line.strip()]
    
    formatted_ingredients = []
    for ingredient in ingredients:
        clean_ingredient = re.sub(r'^[-•*]\s*', '', ingredient)
        if clean_ingredient:
            formatted_ingredients.append(f"• {clean_ingredient}")
    
    ingredients_html = "<br>".join(formatted_ingredients)
    
    st.markdown(f"""
    <div class="ingredient-list">
        {ingredients_html}
    </div>
    """, unsafe_allow_html=True)

# ฟังก์ชันหลัก
def main():
    # ส่วนหัว
    mode_indicator = "🤖 AI Enhanced" if SENTENCE_TRANSFORMERS_AVAILABLE and model else "🔍 Basic Mode"
    
    st.markdown(f"""
    <div class="main-header">
        <h1>🍲 ระบบวิเคราะห์คุณค่าทางโภชนาการอาหารไทย</h1>
        <p>Thai Food Nutrition Analyzer - {mode_indicator}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # โหลดข้อมูลและโมเดล
    with st.spinner("กำลังโหลดระบบ..."):
        model = load_model()
        data = load_food_data()
        
        if data.empty:
            st.error("ไม่สามารถโหลดข้อมูลอาหารได้ กรุณาตรวจสอบไฟล์ข้อมูล")
            st.info("📁 กรุณาเตรียมไฟล์ thai_food_processed.csv หรือ thai_food_processed_cleaned.csv")
            return
        
        embeddings = get_embeddings(model, data)
        nutrition_calculator = SimpleNutritionCalculator()
    
    # แถบด้านข้าง
    with st.sidebar:
        st.markdown("## ⚙️ การตั้งค่า")
        
        # แสดงสถานะระบบ
        st.markdown("### 🖥️ สถานะระบบ")
        if SENTENCE_TRANSFORMERS_AVAILABLE and model:
            st.success("🤖 AI Search: พร้อมใช้งาน")
        else:
            st.warning("🔍 Basic Search: โหมดพื้นฐาน")
        
        if SKLEARN_AVAILABLE:
            st.success("📊 ML Tools: พร้อมใช้งาน")
        else:
            st.info("📊 ML Tools: ใช้ระบบทดแทน")
        
        # ตัวเลือกการค้นหา
        st.markdown("### 🔍 การค้นหา")
        search_mode = st.selectbox(
            "โหมดการค้นหา",
            ["อัตโนมัติ", "Fuzzy Search เท่านั้น"] if not SENTENCE_TRANSFORMERS_AVAILABLE 
            else ["อัตโนมัติ", "AI Search เท่านั้น", "Fuzzy Search เท่านั้น"],
            help="อัตโนมัติ = ใช้วิธีการที่ดีที่สุดที่มี"
        )
        
        max_results = st.slider("จำนวนผลลัพธ์สูงสุด", 1, 10, 5)
        
        # ข้อมูลสถิติ
        st.markdown("### 📊 สถิติข้อมูล")
        st.metric("จำนวนสูตรอาหาร", len(data))
        
        # คำแนะนำสำหรับ AI features
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            st.markdown("### 💡 เพิ่ม AI Features")
            st.info("""
            เพื่อใช้งานฟีเจอร์ AI เต็มรูปแบบ:
            ```
            pip install sentence-transformers
            pip install scikit-learn
            ```
            """)
        
        # ข้อมูลเพิ่มเติม
        with st.expander("🔧 ข้อมูลเทคนิค"):
            st.write(f"**Sentence Transformers:** {'✅' if SENTENCE_TRANSFORMERS_AVAILABLE else '❌'}")
            st.write(f"**Scikit-learn:** {'✅' if SKLEARN_AVAILABLE else '❌'}")
            st.write(f"**โหมดการทำงาน:** {'AI + Fuzzy' if model else 'Fuzzy Only'}")
            st.write(f"**ขนาด Embeddings:** {len(embeddings) if len(embeddings) > 0 else 'N/A'}")
    
    # แท็บหลัก
    tab1, tab2, tab3 = st.tabs(["🔍 ค้นหาอาหาร", "📋 ข้อมูลทั้งหมด", "ℹ️ เกี่ยวกับระบบ"])
    
    with tab1:
        st.markdown("## ค้นหาสูตรอาหารและวิเคราะห์คุณค่าทางโภชนาการ")
        
        st.markdown("""
        <div class="search-tips">
            <h4>💡 เคล็ดลับการค้นหา:</h4>
            <ul>
                <li><strong>ชื่ออาหาร:</strong> ต้มยำกุ้ง, ผัดไทย, แกงเผ็ด</li>
                <li><strong>วัตถุดิบ:</strong> อาหารที่มีกุ้ง, เมนูไก่</li>
                <li><strong>รองรับการพิมพ์ผิด:</strong> ผัดใท → ผัดไทย</li>
                <li><strong>ภาษาอังกฤษ:</strong> tom yum, pad thai</li>
            </ul>
            """ + (f"""
            <div style="background: #e3f2fd; padding: 0.5rem; margin-top: 0.5rem; border-radius: 5px;">
                <strong>ℹ️ โหมดปัจจุบัน:</strong> {'AI + Fuzzy Search' if model and SENTENCE_TRANSFORMERS_AVAILABLE else 'Fuzzy Search (Basic)'}
            </div>
            """ if not SENTENCE_TRANSFORMERS_AVAILABLE else "") + """
        </div>
        """, unsafe_allow_html=True)
        
        # ช่องค้นหา
        query = st.text_input(
            "🔍 ค้นหาอาหารที่ต้องการ:",
            placeholder="เช่น ต้มยำกุ้ง, ผัดไทย, อาหารที่มีโปรตีนสูง...",
            help="พิมพ์ชื่ออาหาร วัตถุดิบ หรือคำอธิบายที่เกี่ยวข้อง"
        )
        
        if query:
            with st.spinner(f"กำลังค้นหา '{query}'..."):
                results = search_recipes(query, model, data, embeddings, max_results)
            
            if results:
                st.markdown(f"### 🍽️ พบ {len(results)} รายการที่เกี่ยวข้อง")
                
                # แสดงผลลัพธ์แรก (รายละเอียดเต็ม)
                best_result = results[0]
                
                # กำหนดสีของ similarity score
                similarity_class = "low-similarity" if best_result['similarity'] < 0.5 else ""
                
                st.markdown(f"""
                <div class="recipe-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                        <h2 style="margin: 0; color: #2c3e50;">🍲 {best_result['name']}</h2>
                        <div class="similarity-score {similarity_class}">
                            ความเกี่ยวข้อง: {best_result['similarity']:.1%}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    # แสดงวัตถุดิบ
                    display_ingredients(best_result['ingredients'])
                    
                    # แสดงวิธีทำ
                    st.markdown("### 👨‍🍳 วิธีทำ")
                    if best_result['method']:
                        # จัดรูปแบบวิธีทำให้อ่านง่าย
                        method_text = best_result['method'].replace('. ', '.\n\n')
                        st.markdown(f"""
                        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #17a2b8;">
                            {method_text}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("ไม่มีข้อมูลวิธีทำ")
                
                with col2:
                    # คำนวณและแสดงคุณค่าทางโภชนาการ
                    nutrition_data = nutrition_calculator.calculate_recipe_nutrition(
                        best_result['ingredients']
                    )
                    display_nutrition_card(nutrition_data)
                    
                    # แสดงข้อมูลเพิ่มเติม
                    st.markdown("### 📋 ข้อมูลเพิ่มเติม")
                    
                    # คำนวณข้อมูลพิเศษ
                    total_cals = nutrition_data.get('calories', 0)
                    if total_cals > 0:
                        protein_pct = (nutrition_data.get('protein', 0) * 4 / total_cals) * 100
                        fat_pct = (nutrition_data.get('fat', 0) * 9 / total_cals) * 100
                        carbs_pct = (nutrition_data.get('carbs', 0) * 4 / total_cals) * 100
                        
                        st.info(f"""
                        **สัดส่วนพลังงาน:**
                        - โปรตีน: {protein_pct:.1f}%
                        - ไขมัน: {fat_pct:.1f}%
                        - คาร์โบไหดเรต: {carbs_pct:.1f}%
                        """)
                        
                        # คำแนะนำ
                        if total_cals < 200:
                            st.success("🥗 เมนูแคลอรี่ต่ำ เหมาะกับผู้ที่ควบคุมน้ำหนัก")
                        elif protein_pct > 30:
                            st.success("💪 เมนูโปรตีนสูง เหมาะกับผู้ที่ออกกำลังกาย")
                        elif fat_pct < 20:
                            st.success("❤️ เมนูไขมันต่ำ เหมาะกับผู้ที่ดูแลสุขภาพหัวใจ")
                
                # แสดงผลการค้นหาอื่นๆ
                if len(results) > 1:
                    st.markdown("### 🍴 อาหารอื่นที่คล้ายกัน")
                    
                    for i, result in enumerate(results[1:4], 1):
                        with st.expander(f"{i}. {result['name']} (ความเกี่ยวข้อง: {result['similarity']:.1%})"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**วัตถุดิบ:**")
                                ingredients = [line.strip() for line in result['ingredients'].split('\n')[:5]]
                                for ingredient in ingredients:
                                    clean_ingredient = re.sub(r'^[-•*]\s*', '', ingredient)
                                    if clean_ingredient:
                                        st.write(f"• {clean_ingredient}")
                                if len(result['ingredients'].split('\n')) > 5:
                                    st.write("...")
                            
                            with col2:
                                quick_nutrition = nutrition_calculator.calculate_recipe_nutrition(
                                    result['ingredients'])
                                st.write(f"**แคลอรี่:** {quick_nutrition.get('calories', 0):.0f} kcal")
                                st.write(f"**โปรตีน:** {quick_nutrition.get('protein', 0):.1f} g")
                                st.write(f"**ไขมัน:** {quick_nutrition.get('fat', 0):.1f} g")
                                st.write(f"**คาร์โบไหดเรต:** {quick_nutrition.get('carbs', 0):.1f} g")
                
            else:
                st.warning(f"ไม่พบอาหารที่ตรงกับคำค้นหา '{query}'")
                st.info("""
                💡 **เคล็ดลับ:**
                - ลองใช้คำค้นหาที่กว้างขึ้น เช่น 'กุ้ง' แทน 'ต้มยำกุ้ง'
                - ตรวจสอบการสะกดคำ
                - ลองค้นหาด้วยภาษาอังกฤษ
                """)
    
    with tab2:
        st.markdown("## 📋 ข้อมูลสูตรอาหารทั้งหมด")
        
        # ตัวกรองข้อมูล
        col1, col2, col3 = st.columns(3)
        
        with col1:
            name_filter = st.text_input("🔍 กรองตามชื่อ:", placeholder="พิมพ์ชื่ออาหาร...")
        
        # กรองข้อมูล
        filtered_data = data.copy()
        
        if name_filter:
            filtered_data = filtered_data[
                filtered_data['name'].str.contains(name_filter, case=False, na=False)
            ]
        
        st.markdown(f"**พบ {len(filtered_data)} รายการ** (จากทั้งหมด {len(data)} รายการ)")
        
        # แสดงข้อมูลในตาราง
        if not filtered_data.empty:
            # เพิ่มคอลัมน์ประมาณคุณค่าทางโภชนาการ
            with st.spinner("กำลังคำนวณคุณค่าทางโภชนาการ..."):
                nutrition_summary = []
                for _, row in filtered_data.head(50).iterrows():  # จำกัดแค่ 50 รายการเพื่อความเร็ว
                    nutrition = nutrition_calculator.calculate_recipe_nutrition(
                        row.get('ingredient', ''))
                    nutrition_summary.append(nutrition)
                
                filtered_data_display = filtered_data.head(50).copy()
                filtered_data_display['แคลอรี่ (kcal)'] = [n.get('calories', 0) for n in nutrition_summary]
                filtered_data_display['โปรตีน (g)'] = [n.get('protein', 0) for n in nutrition_summary]
                filtered_data_display['ไขมัน (g)'] = [n.get('fat', 0) for n in nutrition_summary]
                filtered_data_display['คาร์โบไหดเรต (g)'] = [n.get('carbs', 0) for n in nutrition_summary]
            
            # แสดงตาราง
            st.dataframe(
                filtered_data_display[['name', 'แคลอรี่ (kcal)', 'โปรตีน (g)', 'ไขมัน (g)', 'คาร์โบไหดเรต (g)']].round(1),
                use_container_width=True,
                hide_index=True
            )
            
            # สถิติสรุป
            if len(nutrition_summary) > 0:
                st.markdown("### 📊 สถิติสรุป")
                col1, col2, col3, col4 = st.columns(4)
                
                avg_calories = np.mean([n.get('calories', 0) for n in nutrition_summary])
                avg_protein = np.mean([n.get('protein', 0) for n in nutrition_summary])
                avg_fat = np.mean([n.get('fat', 0) for n in nutrition_summary])
                avg_carbs = np.mean([n.get('carbs', 0) for n in nutrition_summary])
                
                with col1:
                    st.metric("แคลอรี่เฉลี่ย", f"{avg_calories:.0f} kcal")
                with col2:
                    st.metric("โปรตีนเฉลี่ย", f"{avg_protein:.1f} g")
                with col3:
                    st.metric("ไขมันเฉลี่ย", f"{avg_fat:.1f} g")
                with col4:
                    st.metric("คาร์โบไหดเรตเฉลี่ย", f"{avg_carbs:.1f} g")
        else:
            st.info("ไม่พบข้อมูลที่ตรงกับเกณฑ์การกรอง")
    
    with tab3:
        st.markdown("## ℹ️ เกี่ยวกับระบบ")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            ### 🔬 เทคโนโลยีที่ใช้
            
            - **AI Search**: {'✅ Sentence Transformers' if SENTENCE_TRANSFORMERS_AVAILABLE else '❌ ไม่พร้อมใช้งาน'}
            - **โมเดล**: {'paraphrase-multilingual-MiniLM-L12-v2' if model else 'ไม่ได้โหลด'}
            - **การค้นหา**: {'Semantic + Fuzzy Matching' if SENTENCE_TRANSFORMERS_AVAILABLE else 'Fuzzy Matching Only'}
            - **ML Tools**: {'✅ Scikit-learn' if SKLEARN_AVAILABLE else '❌ ใช้ระบบทดแทน'}
            - **UI Framework**: ✅ Streamlit
            - **ข้อมูล**: สูตรอาหารไทยรวบรวม
            
            ### 🎯 คุณสมบัติปัจจุบัน
            
            - {'🤖' if SENTENCE_TRANSFORMERS_AVAILABLE else '🔍'} ค้นหาอาหาร{'ด้วย AI ที่เข้าใจภาษาไทย' if SENTENCE_TRANSFORMERS_AVAILABLE else 'แบบ Fuzzy Matching'}
            - ✅ รองรับการพิมพ์ผิด
            - ✅ คำนวณคุณค่าทางโภชนาการโดยประมาณ
            - ✅ แสดงผลแบบโต้ตอบที่สวยงาม
            - ✅ รองรับทั้งภาษาไทยและอังกฤษ
            """)
            
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                st.warning("""
                **💡 เพิ่มฟีเจอร์ AI:**
                
                เพื่อใช้งาน AI Search ให้ติดตั้ง:
                ```bash
                pip install sentence-transformers
                pip install scikit-learn torch
                ```
                """)
        
        with col2:
            st.markdown(f"""
            ### 📊 ข้อมูลในระบบ
            
            - **จำนวนสูตรอาหาร**: {len(data)} รายการ
            - **ประเภทข้อมูล**: สูตรอาหารไทยแท้
            - **การคำนวณโภชนาการ**: ระบบประมาณการอัตโนมัติ
            - **โหมดการทำงาน**: {'AI + Basic' if SENTENCE_TRANSFORMERS_AVAILABLE else 'Basic Only'}
            
            ### ⚠️ ข้อจำกัด
            
            - ค่าโภชนาการเป็นการประมาณจากข้อมูลพื้นฐาน
            - ความแม่นยำขึ้นอยู่กับคุณภาพข้อมูลต้นทาง
            - {'การค้นหา AI อาจไม่แม่นยำหากไม่มี Sentence Transformers' if not SENTENCE_TRANSFORMERS_AVAILABLE else 'การค้นหา AI ทำงานได้เต็มประสิทธิภาพ'}
            - ควรปรึกษาผู้เชี่ยวชาญด้านโภชนาการสำหรับการใช้งานทางการแพทย์
            
            ### 🔄 เวอร์ชัน
            
            - **เวอร์ชันปัจจุบัน**: {'2.0 Enhanced (Basic Mode)' if not SENTENCE_TRANSFORMERS_AVAILABLE else '2.0 Enhanced (Full AI)'}
            - **อัปเดตล่าสุด**: {datetime.now().strftime("%d/%m/%Y")}
            """)
        
        # ข้อมูลเพิ่มเติม
        st.markdown("### 🚀 การพัฒนาต่อ")
        
        st.info("""
        **ฟีเจอร์ที่กำลังพัฒนา:**
        - รวมข้อมูลจาก USDA FoodData Central
        - การคำนวณโภชนาการที่แม่นยำขึ้น
        - การแนะนำอาหารตามความต้องการเฉพาะ
        - รองรับภาพประกอบอาหาร
        """)

if __name__ == "__main__":
    main()
