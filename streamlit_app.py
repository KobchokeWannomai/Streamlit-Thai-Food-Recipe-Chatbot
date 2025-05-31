import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import re
import hashlib
import time
from difflib import SequenceMatcher
try:
    import Levenshtein
    HAS_LEVENSHTEIN = True
except ImportError:
    HAS_LEVENSHTEIN = False

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_ML = True
except ImportError:
    HAS_ML = False
    st.warning("⚠️ ไม่พบ sentence-transformers, จะใช้การค้นหาแบบธรรมดา")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    st.warning("⚠️ ไม่พบ plotly, จะไม่แสดงกราฟ")

# การตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Thai Food Recipe Chatbot",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ตั้งค่า CSS
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
    
    .warning-card {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 0.8rem;
        border-radius: 8px;
        color: #333;
        margin: 0.3rem 0;
        font-size: 0.9rem;
    }
    
    .status-info {
        background-color: #e7f3ff;
        color: #004085;
        padding: 0.5rem;
        border-radius: 5px;
        border: 1px solid #b8daff;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ข้อมูลโภชนาการเบื้องต้น (ต่อ 100 กรัม)
NUTRITION_DB = {
    "ไข่": {"calories": 155, "protein": 13, "carbs": 1.1, "fat": 11, "fiber": 0, "vitamin_c": 0, "calcium": 56, "iron": 1.75, "sodium": 124},
    "หมู": {"calories": 242, "protein": 27, "carbs": 0, "fat": 14, "fiber": 0, "vitamin_c": 0.7, "calcium": 19, "iron": 0.87, "sodium": 62},
    "ไก่": {"calories": 165, "protein": 31, "carbs": 0, "fat": 3.6, "fiber": 0, "vitamin_c": 1.6, "calcium": 15, "iron": 1.3, "sodium": 82},
    "กุ้ง": {"calories": 99, "protein": 18, "carbs": 0.2, "fat": 1.4, "fiber": 0, "vitamin_c": 2.1, "calcium": 70, "iron": 0.5, "sodium": 111},
    "ปลา": {"calories": 112, "protein": 18.7, "carbs": 0, "fat": 3.6, "fiber": 0, "vitamin_c": 0.9, "calcium": 89, "iron": 0.9, "sodium": 54},
    "น้ำมัน": {"calories": 884, "protein": 0, "carbs": 0, "fat": 100, "fiber": 0, "vitamin_c": 0, "calcium": 0, "iron": 0, "sodium": 0},
    "กระเทียม": {"calories": 149, "protein": 6.4, "carbs": 33, "fat": 0.5, "fiber": 2.1, "vitamin_c": 31, "calcium": 181, "iron": 1.7, "sodium": 17},
    "หอม": {"calories": 40, "protein": 1.1, "carbs": 9.3, "fat": 0.1, "fiber": 1.7, "vitamin_c": 7.4, "calcium": 23, "iron": 0.21, "sodium": 4},
    "ผักชี": {"calories": 23, "protein": 2.1, "carbs": 3.7, "fat": 0.5, "fiber": 2.8, "vitamin_c": 27, "calcium": 67, "iron": 1.77, "sodium": 46},
    "น้ำปลา": {"calories": 42, "protein": 5.8, "carbs": 1.5, "fat": 0.8, "fiber": 0, "vitamin_c": 0, "calcium": 85, "iron": 2.03, "sodium": 6976},
    "น้ำตาล": {"calories": 387, "protein": 0, "carbs": 100, "fat": 0, "fiber": 0, "vitamin_c": 0, "calcium": 1, "iron": 0.01, "sodium": 1},
    "มะนาว": {"calories": 29, "protein": 0.7, "carbs": 9.3, "fat": 0.2, "fiber": 2.8, "vitamin_c": 53, "calcium": 33, "iron": 0.6, "sodium": 2},
    "กะทิ": {"calories": 230, "protein": 2.3, "carbs": 6, "fat": 24, "fiber": 2.2, "vitamin_c": 2.8, "calcium": 16, "iron": 1.64, "sodium": 15},
    "ข้าว": {"calories": 130, "protein": 2.7, "carbs": 28, "fat": 0.3, "fiber": 0.4, "vitamin_c": 0, "calcium": 10, "iron": 0.8, "sodium": 5},
    "ผัก": {"calories": 25, "protein": 2, "carbs": 5, "fat": 0.2, "fiber": 2, "vitamin_c": 30, "calcium": 50, "iron": 1, "sodium": 10}
}

# ข้อมูลตัวอย่างสูตรอาหารไทย
SAMPLE_RECIPES = [
    {
        "name": "ไข่เจียว",
        "ingredient": "- ไข่ไก่ 2 ฟอง\n- น้ำปลา 1 ช้อนชา\n- ต้นหอม 1 ต้น\n- น้ำมันหมู 2 ช้อนโต๊ะ",
        "method": "1. ตีไข่ใส่ชาม ใส่น้ำปลาคนให้เข้ากัน\n2. ตั้งกะทะใส่น้ำมัน รอให้ร้อน\n3. เทไข่ลงทอดจนเหลืองกรอบ\n4. โรยต้นหอมซอย เสิร์ฟร้อนๆ"
    },
    {
        "name": "ผัดกะเพราหมูสับ",
        "ingredient": "- หมูสับ 200 กรัม\n- ใบกะเพรา 1 ถ้วย\n- พริกขี้หนู 5 เม็ด\n- กระเทียม 3 กลีบ\n- น้ำปลา 2 ช้อนโต๊ะ\n- น้ำตาลทราย 1 ช้อนชา\n- น้ำมันพืช 2 ช้อนโต๊ะ",
        "method": "1. โขลกกระเทียมและพริกให้ละเอียด\n2. ตั้งกะทะใส่น้ำมัน ผัดกระเทียมพริกให้หอม\n3. ใส่หมูสับผัดจนสุก\n4. ปรุงรสด้วยน้ำปลา น้ำตาล\n5. ใส่ใบกะเพราผัดให้เข้ากัน"
    },
    {
        "name": "ต้มยำกุ้ง",
        "ingredient": "- กุ้งนาง 300 กรัม\n- เห็ดฟาง 100 กรัม\n- มะนาว 2 ผล\n- ใบมะกรูด 5 ใบ\n- ตะไคร้ 2 ต้น\n- ข่า 3 แว่น\n- พริกขี้หนู 5 เม็ด\n- น้ำปลา 3 ช้อนโต๊ะ\n- น้ำตาลปึก 1 ช้อนโต๊ะ",
        "method": "1. ต้มน้ำให้เดือด ใส่ตะไคร้ ข่า ใบมะกรูด\n2. ใส่เห็ดฟางต้มจนสุก\n3. ใส่กุ้งต้มจนสุก\n4. ปรุงรสด้วยน้ำปลา น้ำตาล\n5. ใส่พริกโขลก คนให้เข้ากัน\n6. ยกลงจากเตา บีบมะนาว"
    },
    {
        "name": "แกงเขียวหวานไก่",
        "ingredient": "- เนื้อไก่ 300 กรัม\n- กะทิ 2 ถ้วย\n- น้ำพริกแกงเขียวหวาน 3 ช้อนโต๊ะ\n- มะเขือเปราะ 5 ผล\n- ใบโหระพา 1 ถ้วย\n- พริกชี้ฟ้า 2 เม็ด\n- น้ำปลา 2 ช้อนโต๊ะ\n- น้ำตาลปึก 1 ช้อนโต๊ะ",
        "method": "1. คั่วน้ำพริกแกงกับหัวกะทิให้หอม\n2. ใส่เนื้อไก่ผัดให้เข้ากัน\n3. เติมกะทิที่เหลือ รอให้เดือด\n4. ใส่มะเขือเปราะ ต้มจนสุก\n5. ปรุงรสด้วยน้ำปลา น้ำตาล\n6. ใส่ใบโหระพา พริกชี้ฟ้า"
    },
    {
        "name": "ส้มตำไทย",
        "ingredient": "- มะละกอดิบขูดฝอย 2 ถ้วย\n- มะเขือเทศ 3 ผล\n- ถั่วฝักยาว 5 เส้น\n- กุ้งแห้ง 2 ช้อนโต๊ะ\n- ถั่วลิสงคั่ว 2 ช้อนโต๊ะ\n- พริกขี้หนู 3 เม็ด\n- กระเทียม 2 กลีบ\n- มะนาว 2 ผล\n- น้ำปลา 2 ช้อนโต๊ะ\n- น้ำตาลปึก 2 ช้อนโต๊ะ",
        "method": "1. ตำพริกกับกระเทียมในครก\n2. ใส่ถั่วฝักยาว ตำพอแตก\n3. ใส่น้ำปลา น้ำตาลปึก คลุกให้ละลาย\n4. ใส่มะเขือเทศ กุ้งแห้ง ถั่วลิสง ตำเบาๆ\n5. ใส่มะละกอ คลุกเคล้าให้เข้ากัน\n6. ชิมรส บีบมะนาว"
    }
]

def generate_unique_key(base_key, extra=""):
    """สร้าง unique key"""
    content = f"{base_key}_{extra}_{time.time()}"
    return hashlib.md5(content.encode()).hexdigest()[:8]

def calculate_similarity(text1, text2):
    """คำนวณความคล้ายคลึงระหว่างข้อความ"""
    if HAS_LEVENSHTEIN:
        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return 1.0
        distance = Levenshtein.distance(text1.lower(), text2.lower())
        return 1 - (distance / max_len)
    else:
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def improved_search(query, recipes, threshold=0.4):
    """การค้นหาที่ปรับปรุงแล้ว"""
    results = []
    query_lower = query.lower().strip()
    
    for idx, recipe in enumerate(recipes):
        name_lower = recipe['name'].lower()
        
        # คำนวณความคล้ายคลึง
        name_similarity = calculate_similarity(query_lower, name_lower)
        
        # ตรวจสอบคำที่มีส่วนตรงกัน
        query_words = query_lower.split()
        name_words = name_lower.split()
        
        word_matches = 0
        for q_word in query_words:
            for n_word in name_words:
                if q_word in n_word or n_word in q_word:
                    word_matches += 1
                    break
        
        word_similarity = word_matches / len(query_words) if query_words else 0
        
        # ตรวจสอบในส่วนผสม
        ingredient_text = recipe.get('ingredient', '').lower()
        ingredient_match = 0
        for q_word in query_words:
            if len(q_word) > 2 and q_word in ingredient_text:
                ingredient_match += 0.3
        
        # คะแนนรวม
        final_score = max(name_similarity * 0.7, word_similarity * 0.6) + min(ingredient_match, 0.3)
        final_score = min(final_score, 1.0)  # จำกัดไม่เกิน 1.0
        
        if final_score >= threshold:
            results.append((recipe['name'], final_score, idx))
    
    return sorted(results, key=lambda x: x[1], reverse=True)

def extract_ingredient_info(ingredient_text):
    """แยกข้อมูลวัตถุดิบ"""
    ingredients = []
    lines = ingredient_text.split('\n')
    
    for line in lines:
        line = line.strip().lstrip('- ')
        if not line:
            continue
            
        # หาตัวเลขและหน่วย
        numbers = re.findall(r'\d+(?:\.\d+)?', line)
        
        # หาชื่อวัตถุดิบ
        ingredient_name = ""
        for name in NUTRITION_DB.keys():
            if name in line:
                ingredient_name = name
                break
        
        if not ingredient_name:
            # พยายามเดาจากคำ
            for word in ['ไข่', 'หมู', 'ไก่', 'กุ้ง', 'ปลา', 'ผัก']:
                if word in line:
                    ingredient_name = word
                    break
        
        # ปริมาณเริ่มต้น
        amount = float(numbers[0]) if numbers else 50
        
        # ปรับปริมาณตามหน่วย
        if 'ช้อนโต๊ะ' in line:
            amount_g = amount * 15
        elif 'ช้อนชา' in line:
            amount_g = amount * 5
        elif 'ถ้วย' in line:
            amount_g = amount * 150
        elif 'กรัม' in line:
            amount_g = amount
        else:
            amount_g = amount * 10  # เดาค่าเริ่มต้น
        
        ingredients.append({
            'name': ingredient_name or 'วัตถุดิบไม่ทราบ',
            'amount': amount_g,
            'original_text': line
        })
    
    return ingredients

def calculate_nutrition(ingredients):
    """คำนวณค่าโภชนาการ"""
    total = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0, "fiber": 0, 
             "vitamin_c": 0, "calcium": 0, "iron": 0, "sodium": 0}
    
    for ing in ingredients:
        ing_name = ing['name']
        amount_g = ing['amount']
        
        # หาข้อมูลโภชนาการ
        nutrition = NUTRITION_DB.get(ing_name)
        if not nutrition:
            # ลองหาแบบ partial match
            for key in NUTRITION_DB.keys():
                if key in ing_name or ing_name in key:
                    nutrition = NUTRITION_DB[key]
                    break
        
        if nutrition:
            factor = amount_g / 100  # แปลงจาก per 100g
            for nutrient in total.keys():
                total[nutrient] += nutrition.get(nutrient, 0) * factor
    
    return total

def format_ingredients(ingredient_text):
    """จัดรูปแบบวัตถุดิบ"""
    if not ingredient_text:
        return "ไม่มีข้อมูลวัตถุดิบ"
    
    lines = ingredient_text.split('\n')
    formatted = "<ul>"
    for line in lines:
        line = line.strip()
        if line:
            clean_line = line.lstrip('- ')
            formatted += f"<li>{clean_line}</li>"
    formatted += "</ul>"
    return formatted

def format_method(method_text):
    """จัดรูปแบบวิธีทำ"""
    if not method_text:
        return "ไม่มีข้อมูลวิธีทำ"
    
    # ตรวจสอบว่ามีเลขขั้นตอนอยู่แล้วหรือไม่
    has_numbers = bool(re.search(r'^\s*\d+\.', method_text, re.MULTILINE))
    
    if has_numbers:
        # มีเลขอยู่แล้ว แยกตามบรรทัด
        lines = method_text.split('\n')
        formatted = ""
        for line in lines:
            line = line.strip()
            if line:
                if line.startswith('#'):
                    formatted += f"<h5>{line.lstrip('#').strip()}</h5>"
                else:
                    formatted += f"<p>{line}</p>"
        return formatted
    else:
        # ไม่มีเลข รวมเป็นย่อหน้า
        paragraphs = method_text.split('\n')
        formatted = ""
        for para in paragraphs:
            para = para.strip()
            if para:
                if para.startswith('#'):
                    formatted += f"<h5>{para.lstrip('#').strip()}</h5>"
                else:
                    formatted += f"<p>{para}</p>"
        return formatted

def display_nutrition_simple(nutrition, recipe_name):
    """แสดงข้อมูลโภชนาการแบบง่าย"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="nutrition-card">
            <h4>🔥 ข้อมูลโภชนาการ - {recipe_name}</h4>
            <p><strong>แคลอรี่:</strong> {nutrition['calories']:.1f} kcal</p>
            <p><strong>โปรตีน:</strong> {nutrition['protein']:.1f} g</p>
            <p><strong>คาร์โบไฮเดรต:</strong> {nutrition['carbs']:.1f} g</p>
            <p><strong>ไขมัน:</strong> {nutrition['fat']:.1f} g</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="nutrition-card">
            <h4>💊 วิตามินและแร่ธาตุ</h4>
            <p><strong>วิตามิน C:</strong> {nutrition['vitamin_c']:.1f} mg</p>
            <p><strong>แคลเซียม:</strong> {nutrition['calcium']:.1f} mg</p>
            <p><strong>เหล็ก:</strong> {nutrition['iron']:.1f} mg</p>
            <p><strong>โซเดียม:</strong> {nutrition['sodium']:.1f} mg</p>
        </div>
        """, unsafe_allow_html=True)
    
    # คำเตือนและคำแนะนำ
    warnings = []
    recommendations = []
    
    if nutrition['calories'] < 200:
        recommendations.append("เมนูแคลอรี่ต่ำ เหมาะสำหรับผู้ควบคุมน้ำหนัก")
    if nutrition['protein'] > 20:
        recommendations.append("โปรตีนสูง เหมาะสำหรับนักกีฬา")
    if nutrition['sodium'] > 1000:
        warnings.append("โซเดียมสูง ผู้ป่วยความดันสูงควรระวัง")
    if nutrition['calories'] > 400:
        warnings.append("แคลอรี่สูง ควรออกกำลังกายเพิ่มเติม")
    
    if recommendations:
        st.success("✅ " + " | ".join(recommendations))
    if warnings:
        st.warning("⚠️ " + " | ".join(warnings))

def display_recipe(recipe, nutrition=None):
    """แสดงสูตรอาหาร"""
    st.markdown(f"### 🍽️ {recipe['name']}")
    
    # สร้าง tabs
    tab1, tab2, tab3 = st.tabs(["📝 สูตรอาหาร", "📊 โภชนาการ", "💡 คำแนะนำ"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 🥬 วัตถุดิบ")
            st.markdown(format_ingredients(recipe['ingredient']), unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 👨‍🍳 วิธีทำ")
            st.markdown(format_method(recipe['method']), unsafe_allow_html=True)
    
    with tab2:
        if nutrition:
            display_nutrition_simple(nutrition, recipe['name'])
        else:
            st.info("กำลังคำนวณข้อมูลโภชนาการ...")
            
            # คำนวณโภชนาการ
            ingredients = extract_ingredient_info(recipe['ingredient'])
            nutrition = calculate_nutrition(ingredients)
            display_nutrition_simple(nutrition, recipe['name'])
    
    with tab3:
        st.markdown("#### 💡 เคล็ดลับและคำแนะนำ")
        
        # คำแนะนำตามชื่อเมนู
        recipe_name = recipe['name'].lower()
        if 'ไข่เจียว' in recipe_name:
            st.write("🥚 **เคล็ดลับ:** ใช้ไฟแรงให้ไข่ฟูกรอบ น้ำมันต้องร้อนจัด")
        elif 'ผัดกะเพรา' in recipe_name:
            st.write("🌿 **เคล็ดลับ:** ใส่ใบกะเพราตอนท้าย ไฟแรงเพื่อให้หอม")
        elif 'ต้มยำ' in recipe_name:
            st.write("🍋 **เคล็ดลับ:** บีบมะนาวตอนท้าย รสจะสดชื่น")
        elif 'แกง' in recipe_name:
            st.write("🥥 **เคล็ดลับ:** คั่วน้ำพริกแกงให้หอมก่อน จะได้รสชาติเข้มข้น")
        elif 'ส้มตำ' in recipe_name:
            st.write("🥒 **เคล็ดลับ:** ตำให้พอแตก ไม่ควรตำแหลก")
        else:
            st.write("👨‍🍳 **คำแนะนำทั่วไป:** ชิมรสระหว่างปรุง ปรับตามความชอบ")
        
        st.markdown("#### 📈 เปรียบเทียบกับความต้องการประจำวัน")
        
        # เปรียบเทียบแบบง่าย
        if nutrition:
            daily_calories = 2000
            daily_protein = 50
            
            cal_percent = (nutrition['calories'] / daily_calories) * 100
            protein_percent = (nutrition['protein'] / daily_protein) * 100
            
            st.write(f"• **แคลอรี่:** {cal_percent:.1f}% ของความต้องการประจำวัน")
            st.write(f"• **โปรตีน:** {protein_percent:.1f}% ของความต้องการประจำวัน")

def load_csv_data():
    """โหลดข้อมูลจาก CSV หากมี"""
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    if csv_files:
        try:
            # ลองโหลดไฟล์ CSV แรกที่พบ
            csv_file = csv_files[0]
            df = pd.read_csv(csv_file)
            
            # ตรวจสอบคอลัมน์
            if 'name' in df.columns and 'ingredient' in df.columns and 'method' in df.columns:
                recipes = []
                for _, row in df.iterrows():
                    recipes.append({
                        'name': str(row['name']),
                        'ingredient': str(row['ingredient']),
                        'method': str(row['method'])
                    })
                return recipes, csv_file
        except Exception as e:
            st.sidebar.error(f"ไม่สามารถอ่านไฟล์ CSV: {e}")
    
    return None, None

def main():
    """ฟังก์ชันหลัก"""
    st.title("🍲 Thai Food Recipe Chatbot")
    st.markdown("### 🥘 ระบบแนะนำสูตรอาหารไทยพร้อมข้อมูลโภชนาการ")
    st.write("ค้นหาสูตรอาหารไทยและดูข้อมูลโภชนาการ รองรับการค้นหาอัจฉริยะ!")
    
    # โหลดข้อมูล
    csv_recipes, csv_filename = load_csv_data()
    
    if csv_recipes:
        st.sidebar.success(f"✅ โหลดข้อมูลจากไฟล์: {csv_filename} ({len(csv_recipes)} เมนู)")
        recipes = csv_recipes
    else:
        st.sidebar.info("📁 ใช้ข้อมูลตัวอย่าง (5 เมนู)")
        recipes = SAMPLE_RECIPES
    
    # แสดงสถานะ
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📖 จำนวนสูตร", len(recipes))
    with col2:
        st.metric("🔍 ระบบค้นหา", "✅ พร้อม")
    with col3:
        st.metric("🧮 โภชนาการ", "✅ พร้อม")
    
    # การตั้งค่า
    st.sidebar.markdown("### ⚙️ การตั้งค่า")
    
    search_threshold = st.sidebar.slider(
        "ความเคร่งครัดในการค้นหา",
        min_value=0.1,
        max_value=0.9,
        value=0.4,
        step=0.1,
        help="ค่าต่ำ = หาได้ง่าย, ค่าสูง = หาได้ยากแต่แม่นยำ"
    )
    
    max_results = st.sidebar.number_input(
        "จำนวนผลลัพธ์สูงสุด",
        min_value=1,
        max_value=10,
        value=3
    )
    
    # ตัวอย่างการค้นหา
    st.markdown("#### 💡 ตัวอย่างการค้นหา:")
    example_queries = ["ไข่เจียว", "ผัดกะเพรา", "ต้มยำกุ้ง", "แกงเขียวหวาน", "ส้มตำ"]
    
    cols = st.columns(len(example_queries))
    for i, query in enumerate(example_queries):
        if cols[i].button(query, key=f"example_{i}"):
            st.session_state['search_query'] = query
    
    # เริ่มต้น session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "search_query" not in st.session_state:
        st.session_state.search_query = ""
    
    # แสดงประวัติการสนทนา
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "recipe" in message:
                display_recipe(message["recipe"], message.get("nutrition"))
            else:
                st.markdown(message["content"])
    
    # ช่องค้นหา
    search_query = st.session_state.get('search_query', '')
    if prompt := st.chat_input("ค้นหาสูตรอาหาร...", key="main_chat"):
        search_query = prompt
        st.session_state.search_query = ""
    
    if search_query:
        # เพิ่มข้อความของผู้ใช้
        st.session_state.messages.append({"role": "user", "content": search_query})
        
        # แสดงข้อความของผู้ใช้
        with st.chat_message("user"):
            st.markdown(search_query)
        
        # ค้นหา
        with st.chat_message("assistant"):
            with st.spinner("กำลังค้นหา..."):
                results = improved_search(search_query, recipes, search_threshold)
                
                if results:
                    best_match = results[0]
                    recipe_name, similarity, recipe_idx = best_match
                    
                    if similarity > 0.2:  # เกณฑ์ขั้นต่ำ
                        recipe = recipes[recipe_idx]
                        
                        # คำนวณโภชนาการ
                        ingredients = extract_ingredient_info(recipe['ingredient'])
                        nutrition = calculate_nutrition(ingredients)
                        
                        similarity_percent = min(similarity * 100, 100)
                        response = f"🎯 พบเมนูที่ตรงกับการค้นหา: **{recipe_name}** (ความคล้ายคลึง: {similarity_percent:.0f}%)"
                        st.markdown(response)
                        
                        # แสดงสูตร
                        display_recipe(recipe, nutrition)
                        
                        # แสดงเมนูอื่นๆ
                        if len(results) > 1:
                            st.markdown("#### 🔍 เมนูอื่นที่น่าสนใจ:")
                            other_results = results[1:min(max_results, len(results))]
                            
                            cols = st.columns(len(other_results))
                            for i, (other_name, other_sim, _) in enumerate(other_results):
                                with cols[i]:
                                    other_sim_percent = min(other_sim * 100, 100)
                                    if st.button(f"🍽️ {other_name}\n({other_sim_percent:.0f}%)", key=f"other_{i}"):
                                        st.session_state.search_query = other_name
                                        st.rerun()
                        
                        # บันทึกการสนทนา
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "recipe": recipe,
                            "nutrition": nutrition
                        })
                    else:
                        response = f"""
                        ❌ ไม่พบเมนูที่ตรงกับ '{search_query}' ในระดับที่เพียงพอ
                        
                        💡 **คำแนะนำ:**
                        - ลองใช้คำที่ง่ายกว่า เช่น "ไข่เจียว", "ผัดกะเพรา"
                        - ค้นหาตามชนิดอาหาร เช่น "แกง", "ยำ", "ทอด"
                        - ลองปรับความเคร่งครัดในการค้นหาในแถบด้านซ้าย
                        """
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    response = """
                    🤔 ไม่พบเมนูที่ตรงกับคำค้นหา
                    
                    ลองค้นหาด้วยคำเหล่านี้:
                    - **ชื่อเมนู:** "ไข่เจียว", "ผัดกะเพรา", "ต้มยำกุ้ง"
                    - **ประเภทอาหาร:** "แกง", "ยำ", "ทอด", "ต้ม"
                    - **วัตถุดิบ:** "ไก่", "หมู", "กุ้ง", "ผัก"
                    """
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
