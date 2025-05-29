import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import pickle
import re
from nutrition_analyzer import NutritionAnalyzer

# การกำหนดค่าหน้าเว็บ
st.set_page_config(
    page_title="Thai Food Recipe Chatbot",
    page_icon="🍲",
    layout="wide"
)

# ตั้งค่าฟอนต์ไทยและ CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@400;700&display=swap');
    html, body, [class*="st-"] {
        font-family: 'Sarabun', sans-serif !important;
    }
    .nutrition-card {
        background-color: #f0f8ff;
        padding: 1rem;  /* แก้จาก 1.5rem เป็น 1rem ให้ตรงกับ stChatMessage */
        border-radius: 12px;
        border-left: 5px solid #4CAF50;
        margin: 15px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        width: 100%;
        box-sizing: border-box;
    }
    .nutrition-item {
        display: inline-block;
        margin: 5px 10px;
        padding: 8px 15px;
        background-color: #e8f5e8;
        border-radius: 20px;
        font-size: 0.9em;
        font-weight: 500;
    }
    .vitamin-mineral {
        display: inline;  /* แก้จาก font-size เป็น display: inline */
        color: #555;
        margin-top: 10px;
        line-height: 1.8;
    }
    .vitamin-mineral-label {
        display: inline;
        font-weight: 600;
        margin-right: 10px;
    }
    .recipe-card {
        background-color: #f9f9f9;
        padding: 1rem;  /* แก้จาก 20px เป็น 1rem ให้ตรงกับ stChatMessage */
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .recipe-title {
        font-size: 1.8em;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 15px;
    }
    .section-title {
        font-size: 1.2em;
        font-weight: 600;
        color: #34495e;
        margin: 15px 0 10px 0;
        padding-bottom: 5px;
        border-bottom: 2px solid #e0e0e0;
    }
    /* ปรับปรุงการแสดงผลรายการ */
    ul, ol {
        margin-left: 20px;
        line-height: 1.8;
    }
    /* ปรับปรุงการแสดงผล metric */
    [data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    /* ซ่อน sidebar */
    section[data-testid="stSidebar"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# เส้นทางของไฟล์
DATA_PATH = "thai_food_processed.csv"
EMBEDDINGS_PATH = "embeddings.pkl"
MODEL_PATH = "model"

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

@st.cache_resource
def load_nutrition_analyzer():
    """โหลดตัววิเคราะห์โภชนาการ"""
    return NutritionAnalyzer()

@st.cache_data
def load_data():
    """โหลดชุดข้อมูลอาหารไทย"""
    return pd.read_csv(DATA_PATH)

@st.cache_data
def get_embeddings(_model, data):
    """รับหือคำนวณ embeddings สำหรับสูตรอาหารทั้งหมด"""
    if os.path.exists(EMBEDDINGS_PATH):
        with open(EMBEDDINGS_PATH, 'rb') as f:
            return pickle.load(f)
    else:
        # รวมข้อความทั้งหมดสำหรับแต่ละสูตร
        texts = []
        for _, row in data.iterrows():
            combined_text = f"{row['name']} {row['ingredient']} {row['method']}"
            texts.append(combined_text)
        
        # สร้าง embeddings
        embeddings = _model.encode(texts)
        
        # บันทึก embeddings
        with open(EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump(embeddings, f)
        
        return embeddings

def format_ingredients(ingredients_text):
    """จัดรูปแบบรายการวัตถุดิบเพื่อการแสดงผลที่ดีขึ้น"""
    ingredients = ingredients_text.split('\n')
    formatted = "<ul style='line-height: 1.8;'>"
    for item in ingredients:
        if item.strip():
            # ลบเครื่องหมาย - ที่อยู่ด้านหน้า
            cleaned_item = item.strip()
            if cleaned_item.startswith('- '):
                cleaned_item = cleaned_item[2:]
            formatted += f"<li>{cleaned_item}</li>"
    formatted += "</ul>"
    return formatted

def format_cooking_method(method_text):
    """จัดรูปแบบวิธีการทำอาหารเพื่อการแสดงผลที่ดีขึ้น"""
    # ตรวจสอบว่ามีเลขขั้นตอนอยู่แล้วหรือไม่
    has_numbered_steps = bool(re.search(r'^\s*\d+\.', method_text, re.MULTILINE))
    
    # แยกบรรทัดและจัดการกับหัวข้อย่อย
    lines = method_text.split('\n')
    formatted = ""
    current_section = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # ตรวจสอบหัวข้อย่อย (ขึ้นต้นด้วย ## หรือ #)
        if line.startswith('##'):
            # ถ้ามีเนื้อหาในส่วนก่อนหน้า ให้จัดรูปแบบก่อน
            if current_section:
                formatted += format_section(current_section, has_numbered_steps)
                current_section = []
            # เพิ่มหัวข้อย่อย (ลบ ## ออกและใช้ขนาดเท่ากับหัวข้อหลัก)
            formatted += f"<div class='section-title' style='font-size: 1.0em; margin-top: 15px;'>{line.replace('##', '').strip()}</div>"
        elif line.startswith('#'):
            # ถ้ามีเนื้อหาในส่วนก่อนหน้า ให้จัดรูปแบบก่อน
            if current_section:
                formatted += format_section(current_section, has_numbered_steps)
                current_section = []
            # เพิ่มหัวข้อย่อย
            formatted += f"<div class='section-title' style='font-size: 1.0em; margin-top: 15px;'>{line.replace('#', '').strip()}</div>"
        elif line.startswith('**หมายเหตุ'):
            # ถ้ามีเนื้อหาในส่วนก่อนหน้า ให้จัดรูปแบบก่อน
            if current_section:
                formatted += format_section(current_section, has_numbered_steps)
                current_section = []
            # เพิ่มหมายเหตุ
            formatted += f"<div style='margin-top: 15px; padding: 10px; background-color: #f9f9f9; border-left: 3px solid #ffa500;'><strong>หมายเหตุ</strong> {line.replace('**หมายเหตุ**', '').replace('**หมายเหตุ', '').strip()}</div>"
        else:
            # เก็บเนื้อหาปกติ
            current_section.append(line)
    
    # จัดรูปแบบส่วนสุดท้าย
    if current_section:
        formatted += format_section(current_section, has_numbered_steps)
    
    return formatted

def format_section(lines, has_numbered_steps):
    """จัดรูปแบบส่วนของวิธีทำ"""
    if not lines:
        return ""
    
    # ถ้ามีเลขขั้นตอนอยู่แล้ว ให้แสดงเป็นลิสต์
    if has_numbered_steps:
        formatted = "<ol style='line-height: 1.8;'>"
        for line in lines:
            # ตรวจสอบว่าบรรทัดขึ้นต้นด้วยตัวเลขหรือไม่
            if re.match(r'^\d+\.', line):
                # ลบตัวเลขออกเพราะ <ol> จะใส่ให้อัตโนมัติ
                clean_line = re.sub(r'^\d+\.\s*', '', line)
                formatted += f"<li>{clean_line}</li>"
            else:
                # ถ้าไม่มีตัวเลข ให้เป็นส่วนต่อของ item ก่อนหน้า
                if formatted.endswith("</li>"):
                    formatted = formatted[:-5] + f" {line}</li>"
                else:
                    formatted += f"<li>{line}</li>"
        formatted += "</ol>"
    else:
        # ถ้าไม่มีเลขขั้นตอน ให้แสดงเป็นย่อหน้า
        # รวมประโยคที่แยกกันด้วยช่องว่างเดียว
        text = ' '.join(lines)
        # แบ่งเป็นประโยคตาม ๆ หรือ .
        sentences = re.split(r'(?<=[ๆ.])\s+', text)
        
        formatted = "<div style='line-height: 1.8; text-align: justify;'>"
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                formatted += sentence.strip()
                # เพิ่มช่องว่างหลังประโยค ยกเว้นประโยคสุดท้าย
                if i < len(sentences) - 1:
                    formatted += " "
        formatted += "</div>"
    
    return formatted

def display_nutrition_info(nutrition_data):
    """แสดงข้อมูลโภชนาการในรูปแบบที่สวยงาม"""
    if not nutrition_data:
        return
    
    total_nutrition = nutrition_data.get('total_nutrition', {})
    
    st.markdown("### 🥗 ข้อมูลโภชนาการ (ต่อหนึ่งที่)")
    
    # สารอาหารหลัก
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("แคลอรี่", f"{total_nutrition.get('calories', 0):.0f} kcal")
    with col2:
        st.metric("โปรตีน", f"{total_nutrition.get('protein', 0):.1f} g")
    with col3:
        st.metric("คาร์โบไฮเดรต", f"{total_nutrition.get('carbs', 0):.1f} g")
    with col4:
        st.metric("ไขมัน", f"{total_nutrition.get('fat', 0):.1f} g")
    with col5:
        st.metric("ใยอาหาร", f"{total_nutrition.get('fiber', 0):.1f} g")
    
    # วิตามินและแร่ธาตุ - แสดงเฉพาะที่มีค่า
    vitamins = total_nutrition.get('vitamins', {})
    minerals = total_nutrition.get('minerals', {})
    
    if vitamins or minerals:
        nutrition_items = []
        
        # แสดงวิตามิน
        vitamin_units = {
            'วิตามิน A': 'mcg',
            'วิตามิน C': 'mg', 
            'วิตามิน D': 'mcg',
            'วิตามิน E': 'mg',
            'วิตามิน K': 'mcg',
            'วิตามิน B1': 'mg',
            'วิตามิน B2': 'mg',
            'วิตามิน B6': 'mg',
            'วิตามิน B12': 'mcg'
        }
        
        # แสดงเฉพาะวิตามินที่มีค่ามากกว่า 0.1
        for vitamin, amount in vitamins.items():
            if amount > 0.1:
                unit = vitamin_units.get(vitamin, 'mg')
                nutrition_items.append(f"<span class='nutrition-item'>{vitamin}: {amount:.1f} {unit}</span>")
        
        # แสดงแร่ธาตุ
        mineral_units = {
            'แคลเซียม': 'mg',
            'เหล็ก': 'mg',
            'แมกนีเซียม': 'mg',
            'ฟอสฟอรัส': 'mg',
            'โพแทสเซียม': 'mg',
            'สังกะสี': 'mg',
            'โซเดียม': 'mg'
        }
        
        # แสดงเฉพาะแร่ธาตุที่มีค่ามากกว่า 0.1
        for mineral, amount in minerals.items():
            if amount > 0.1:
                unit = mineral_units.get(mineral, 'mg')
                # ถ้าเป็นโซเดียม แสดงเป็นจำนวนเต็ม
                if mineral == 'โซเดียม':
                    nutrition_items.append(f"<span class='nutrition-item'>{mineral}: {amount:.0f} {unit}</span>")
                else:
                    nutrition_items.append(f"<span class='nutrition-item'>{mineral}: {amount:.1f} {unit}</span>")
        
        if nutrition_items:
            st.markdown(f"<div><span class='vitamin-mineral-label'>วิตามินและแร่ธาตุ:</span><span class='vitamin-mineral'>{''.join(nutrition_items)}</span></div>", unsafe_allow_html=True)
    
    # รายละเอียดวัตถุดิบ
    with st.expander("📋 รายละเอียดโภชนาการแต่ละวัตถุดิบ"):
        for ingredient_info in nutrition_data.get('ingredients', []):
            # แสดงชื่อวัตถุดิบแบบเต็ม (ใช้ key ที่มีปริมาณ)
            ingredient_full_name = ingredient_info['ingredient']
            nutrition = ingredient_info['nutrition']
            
            # แสดงชื่อและขนาดที่ใช้
            st.write(f"**{ingredient_full_name}**")
            
            # แสดงข้อมูลพื้นฐาน
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.write(f"แคลอรี่: {nutrition.calories:.0f} kcal")
            with col2:
                st.write(f"โปรตีน: {nutrition.protein:.1f} g")
            with col3:
                st.write(f"คาร์โบไฮเดรต: {nutrition.carbs:.1f} g")
            with col4:
                st.write(f"ไขมัน: {nutrition.fat:.1f} g")
            
            # แสดงวิตามินและแร่ธาตุที่มีค่ามากกว่า 0.1
            vitamin_mineral_text = []
            
            # วิตามิน
            if hasattr(nutrition, 'vitamin_a') and nutrition.vitamin_a > 0.1:
                vitamin_mineral_text.append(f"วิตามิน A: {nutrition.vitamin_a:.1f} mcg")
            if hasattr(nutrition, 'vitamin_c') and nutrition.vitamin_c > 0.1:
                vitamin_mineral_text.append(f"วิตามิน C: {nutrition.vitamin_c:.1f} mg")
            if hasattr(nutrition, 'vitamin_d') and nutrition.vitamin_d > 0.1:
                vitamin_mineral_text.append(f"วิตามิน D: {nutrition.vitamin_d:.1f} mcg")
            if hasattr(nutrition, 'vitamin_e') and nutrition.vitamin_e > 0.1:
                vitamin_mineral_text.append(f"วิตามิน E: {nutrition.vitamin_e:.1f} mg")
            if hasattr(nutrition, 'vitamin_k') and nutrition.vitamin_k > 0.1:
                vitamin_mineral_text.append(f"วิตามิน K: {nutrition.vitamin_k:.1f} mcg")
            if hasattr(nutrition, 'vitamin_b1') and nutrition.vitamin_b1 > 0.1:
                vitamin_mineral_text.append(f"วิตามิน B1: {nutrition.vitamin_b1:.1f} mg")
            if hasattr(nutrition, 'vitamin_b6') and nutrition.vitamin_b6 > 0.1:
                vitamin_mineral_text.append(f"วิตามิน B6: {nutrition.vitamin_b6:.1f} mg")
            if hasattr(nutrition, 'vitamin_b12') and nutrition.vitamin_b12 > 0.1:
                vitamin_mineral_text.append(f"วิตามิน B12: {nutrition.vitamin_b12:.1f} mcg")
            
            # แร่ธาตุ
            if hasattr(nutrition, 'calcium') and nutrition.calcium > 0.1:
                vitamin_mineral_text.append(f"แคลเซียม: {nutrition.calcium:.1f} mg")
            if hasattr(nutrition, 'iron') and nutrition.iron > 0.1:
                vitamin_mineral_text.append(f"เหล็ก: {nutrition.iron:.1f} mg")
            if hasattr(nutrition, 'sodium') and nutrition.sodium > 0.1:
                vitamin_mineral_text.append(f"โซเดียม: {nutrition.sodium:.0f} mg")
            if hasattr(nutrition, 'potassium') and nutrition.potassium > 0.1:
                vitamin_mineral_text.append(f"โพแทสเซียม: {nutrition.potassium:.0f} mg")
            if hasattr(nutrition, 'zinc') and nutrition.zinc > 0.1:
                vitamin_mineral_text.append(f"สังกะสี: {nutrition.zinc:.1f} mg")
            if hasattr(nutrition, 'phosphorus') and nutrition.phosphorus > 0.1:
                vitamin_mineral_text.append(f"ฟอสฟอรัส: {nutrition.phosphorus:.0f} mg")
            if hasattr(nutrition, 'magnesium') and nutrition.magnesium > 0.1:
                vitamin_mineral_text.append(f"แมกนีเซียม: {nutrition.magnesium:.0f} mg")
            
            # แสดงใยอาหารถ้ามี
            if hasattr(nutrition, 'fiber') and nutrition.fiber > 0.1:
                vitamin_mineral_text.append(f"ใยอาหาร: {nutrition.fiber:.1f} g")
            
            if vitamin_mineral_text:
                st.write("สารอาหารอื่นๆ: " + ", ".join(vitamin_mineral_text))
            
            # แสดงหมายเหตุสำหรับวัตถุดิบที่มีโซเดียมสูง
            if hasattr(nutrition, 'sodium') and nutrition.sodium > 500:
                st.caption("⚠️ มีโซเดียมสูง")
                
            st.divider()
    
    st.markdown('</div>', unsafe_allow_html=True)

def search_recipes(query, model, data, embeddings, nutrition_analyzer, top_k=5):
    """ค้นหาสูตรอาหารตามคำค้นหา - ปรับปรุงให้แม่นยำขึ้น"""
    query_lower = query.lower()
    
    # ค้นหาแบบตรงตัวก่อน (exact match)
    exact_matches = []
    for idx, row in data.iterrows():
        recipe_name = row['name'].lower()
        if query_lower in recipe_name or recipe_name in query_lower:
            # ให้คะแนนพิเศษสำหรับ exact match
            exact_matches.append({
                'index': idx,
                'score': 1.0 if query_lower == recipe_name else 0.9
            })
    
    # ค้นหาแบบ semantic search
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # รวมผลลัพธ์
    results = []
    used_indices = set()
    
    # เพิ่ม exact matches ก่อน
    for match in sorted(exact_matches, key=lambda x: x['score'], reverse=True):
        idx = match['index']
        if idx not in used_indices:
            recipe_name = data.iloc[idx]['name']
            ingredients = data.iloc[idx]['ingredient']
            
            # รับข้อมูลโภชนาการ
            nutrition_data = nutrition_analyzer.analyze_recipe(recipe_name, ingredients)
            
            results.append({
                'name': recipe_name,
                'similarity': match['score'],
                'ingredients': ingredients,
                'method': data.iloc[idx]['method'],
                'nutrition': nutrition_data
            })
            used_indices.add(idx)
    
    # เพิ่มผลลัพธ์จาก semantic search
    top_indices = np.argsort(-similarities)
    for idx in top_indices:
        if idx not in used_indices and len(results) < top_k:
            if similarities[idx] > 0.3:  # เพิ่ม threshold
                recipe_name = data.iloc[idx]['name']
                ingredients = data.iloc[idx]['ingredient']
                
                # รับข้อมูลโภชนาการ
                nutrition_data = nutrition_analyzer.analyze_recipe(recipe_name, ingredients)
                
                results.append({
                    'name': recipe_name,
                    'similarity': similarities[idx],
                    'ingredients': ingredients,
                    'method': data.iloc[idx]['method'],
                    'nutrition': nutrition_data
                })
                used_indices.add(idx)
    
    return results

def search_by_nutrition_criteria(data, nutrition_analyzer, criteria):
    """ค้นหาสูตรอาหารตามเกณฑ์โภชนาการ"""
    results = []
    
    for _, row in data.iterrows():
        recipe_name = row['name']
        ingredients = row['ingredient']
        
        # วิเคราะห์โภชนาการ
        nutrition_data = nutrition_analyzer.analyze_recipe(recipe_name, ingredients)
        total_nutrition = nutrition_data.get('total_nutrition', {})
        
        # ตรวจสอบเกณฑ์
        match = True
        if 'max_calories' in criteria and total_nutrition.get('calories', 0) > criteria['max_calories']:
            match = False
        if 'min_calories' in criteria and total_nutrition.get('calories', 0) < criteria['min_calories']:
            match = False
        if 'min_protein' in criteria and total_nutrition.get('protein', 0) < criteria['min_protein']:
            match = False
        
        if match:
            results.append({
                'recipe_name': recipe_name,
                'calories': total_nutrition.get('calories', 0),
                'protein': total_nutrition.get('protein', 0),
                'carbs': total_nutrition.get('carbs', 0),
                'fat': total_nutrition.get('fat', 0),
                'fiber': total_nutrition.get('fiber', 0),
                'ingredients': ingredients,
                'method': row['method'],
                'nutrition': nutrition_data
            })
    
    # เรียงลำดับตามแคลอรี่
    results.sort(key=lambda x: x['calories'])
    return results

def detect_nutrition_search(query):
    """ตรวจจับว่าการค้นหาเป็นการค้นหาตามโภชนาการหรือไม่"""
    nutrition_keywords = [
        'แคลอรี่', 'แคลอรี', 'calorie', 'cal', 'kcal',
        'โปรตีน', 'protein',
        'คาร์โบ', 'คาร์โบไฮเดรต', 'carb', 'carbohydrate',
        'ไขมัน', 'fat',
        'ใยอาหาร', 'fiber',
        'ลดน้ำหนัก', 'diet', 'healthy', 'เฮลธ์ตี้',
        'โภชนาการ', 'nutrition',
        'วิตามิน', 'vitamin',
        'แร่ธาตุ', 'mineral',
        'ต่ำ', 'สูง', 'น้อย', 'เยอะ', 'มาก',
        'ไม่เกิน', 'มากกว่า', 'น้อยกว่า'
    ]
    
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in nutrition_keywords)

def extract_nutrition_criteria_from_text(query):
    """แยกเกณฑ์โภชนาการจากข้อความค้นหา"""
    criteria = {}
    query_lower = query.lower()
    
    # ค้นหาแคลอรี่
    calorie_patterns = [
        r'แคลอรี่.*?ไม่เกิน.*?(\d+)',
        r'ไม่เกิน.*?(\d+).*?แคลอรี่',
        r'แคลอรี่.*?น้อยกว่า.*?(\d+)',
        r'น้อยกว่า.*?(\d+).*?แคลอรี่',
        r'แคลอรี.*?ไม่เกิน.*?(\d+)',
        r'ไม่เกิน.*?(\d+).*?แคลอรี'
    ]
    
    for pattern in calorie_patterns:
        match = re.search(pattern, query_lower)
        if match:
            criteria['max_calories'] = int(match.group(1))
            break
    
    # ค้นหาโปรตีน
    protein_patterns = [
        r'โปรตีน.*?มากกว่า.*?(\d+)',
        r'มากกว่า.*?(\d+).*?โปรตีน',
        r'โปรตีน.*?สูง.*?(\d+)',
        r'โปรตีน.*?เยอะ.*?(\d+)',
        r'โปรตีน.*?อย่างน้อย.*?(\d+)'
    ]
    
    for pattern in protein_patterns:
        match = re.search(pattern, query_lower)
        if match:
            criteria['min_protein'] = int(match.group(1))
            break
    
    # เกณฑ์พื้นฐานสำหรับคำค้นหาทั่วไป
    if 'ลดน้ำหนัก' in query_lower or 'diet' in query_lower:
        criteria.update({'max_calories': 400, 'min_protein': 15})
    elif 'แคลอรี่ต่ำ' in query_lower or 'แคลอรีต่ำ' in query_lower:
        criteria['max_calories'] = 300
    elif 'โปรตีนสูง' in query_lower:
        criteria['min_protein'] = 20
    elif 'เฮลธ์ตี้' in query_lower or 'healthy' in query_lower:
        criteria['max_calories'] = 350
    
    return criteria

def main():
    # โหลดโมเดลและข้อมูล
    model = load_model()
    data = load_data()
    embeddings = get_embeddings(model, data)
    nutrition_analyzer = load_nutrition_analyzer()
    
    # แอปหลัก
    st.title("🍲 Thai Food Recipe Chatbot")
    st.markdown("**ค้นหาสูตรอาหารไทยพร้อมข้อมูลโภชนาการ** - ถามเกี่ยวกับวิธีทำอาหารไทยหรือค้นหาตามโภชนาการได้เลย!")
    
    # ตัวอย่างการค้นหา
    st.info("💡 **ตัวอย่างการค้นหา:** ไข่เจียว, ต้มยำกุ้ง, ผัดไทย, เมนูแคลอรี่ไม่เกิน 300, อาหารโปรตีนสูง, เมนูลดน้ำหนัก")
    
    # เริ่มต้นประวัติการสนทนา
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # แสดงประวัติการสนทนา
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "recipe" in message:
                # แสดงสูตรพร้อมโภชนาการ
                recipe = message["recipe"]
                st.markdown(f'<div class="recipe-title">{recipe["name"]}</div>', unsafe_allow_html=True)
                
                # แสดงข้อมูลโภชนาการ
                display_nutrition_info(recipe['nutrition'])
                
                st.markdown('<div class="section-title">📝 วัตถุดิบ</div>', unsafe_allow_html=True)
                st.markdown(format_ingredients(recipe["ingredients"]), unsafe_allow_html=True)
                
                st.markdown('<div class="section-title">👩‍🍳 วิธีทำ</div>', unsafe_allow_html=True)
                st.markdown(format_cooking_method(recipe["method"]), unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            elif message["role"] == "assistant" and "nutrition_results" in message:
                # แสดงผลลัพธ์การค้นหาตามโภชนาการ
                results = message["nutrition_results"]
                st.markdown(f"พบ **{len(results)}** สูตรอาหารที่ตรงเกณฑ์:")
                
                for i, result in enumerate(results[:5], 1):
                    with st.expander(f"{i}. {result['recipe_name']} ({result['calories']:.0f} kcal)"):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.write(f"🔥 แคลอรี่: {result['calories']:.0f} kcal")
                        with col2:
                            st.write(f"🥩 โปรตีน: {result['protein']:.1f} g")
                        with col3:
                            st.write(f"🍞 คาร์โบไฮเดรต: {result['carbs']:.1f} g")
                        with col4:
                            st.write(f"🧈 ไขมัน: {result['fat']:.1f} g")
                        
                        if st.button(f"ดูสูตรอาหาร", key=f"recipe_{i}"):
                            st.write("### วัตถุดิบ")
                            st.write(result['ingredients'])
                            st.write("### วิธีทำ")
                            st.write(result['method'])
                
                if len(results) > 5:
                    st.markdown(f"*และอีก {len(results) - 5} สูตร...*")
            else:
                # แสดงข้อความธรรมดา
                st.markdown(message["content"])
    
    # ช่องใส่ข้อความสำหรับแชท
    if prompt := st.chat_input("ค้นหาสูตรอาหารไทย..."):
        # เพิ่มข้อความของผู้ใช้ลงในประวัติการสนทนา
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # แสดงข้อความของผู้ใช้
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # รับการตอบสนอง
        with st.chat_message("assistant"):
            with st.spinner("กำลังค้นหาและวิเคราะห์..."):
                # ตรวจสอบว่าเป็นการค้นหาตามโภชนาการหรือไม่
                if detect_nutrition_search(prompt):
                    # การค้นหาตามโภชนาการ
                    criteria = extract_nutrition_criteria_from_text(prompt)
                    
                    if criteria:
                        nutrition_results = search_by_nutrition_criteria(data, nutrition_analyzer, criteria)
                        
                        if nutrition_results:
                            response = f"พบ {len(nutrition_results)} สูตรอาหารที่ตรงกับเกณฑ์โภชนาการที่ต้องการ"
                            st.markdown(response)
                            
                            # แสดงผลลัพธ์
                            st.markdown("### 🔍 ผลการค้นหา")
                            for i, result in enumerate(nutrition_results[:5], 1):
                                with st.expander(f"{i}. {result['recipe_name']} ({result['calories']:.0f} kcal)"):
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.write(f"🔥 แคลอรี่: {result['calories']:.0f} kcal")
                                    with col2:
                                        st.write(f"🥩 โปรตีน: {result['protein']:.1f} g")
                                    with col3:
                                        st.write(f"🍞 คาร์โบไฮเดรต: {result['carbs']:.1f} g")
                                    with col4:
                                        st.write(f"🧈 ไขมัน: {result['fat']:.1f} g")
                            
                            if len(nutrition_results) > 5:
                                st.markdown(f"*และอีก {len(nutrition_results) - 5} สูตร...*")
                            
                            # เพิ่มการตอบสนองของแอสซิสแตนต์ลงในประวัติการสนทนา
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response,
                                "nutrition_results": nutrition_results
                            })
                        else:
                            response = "ขออภัย ไม่พบสูตรอาหารที่ตรงกับเกณฑ์โภชนาการที่ต้องการ ลองปรับเกณฑ์ใหม่"
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        # ถ้าตรวจพบคีย์เวิร์ดโภชนาการแต่แยกเกณฑ์ไม่ได้
                        response = "กรุณาระบุเกณฑ์โภชนาการให้ชัดเจนขึ้น เช่น 'เมนูแคลอรี่ไม่เกิน 300' หรือ 'อาหารโปรตีนสูงมากกว่า 20 กรัม'"
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    # การค้นหาทั่วไป
                    results = search_recipes(prompt, model, data, embeddings, nutrition_analyzer)
                    
                    if results:
                        best_match = results[0]
                        
                        # ตรวจสอบว่ามีผลลัพธ์ที่ดี
                        if best_match["similarity"] > 0.3:
                            response = f"พบสูตรอาหารที่คุณค้นหา: **{best_match['name']}**"
                            st.markdown(response)
                            
                            # แสดงสูตรพร้อมโภชนาการ
                            st.markdown(f'<div class="recipe-title">{best_match["name"]}</div>', unsafe_allow_html=True)
                            
                            # แสดงข้อมูลโภชนาการ
                            display_nutrition_info(best_match['nutrition'])
                            
                            st.markdown('<div class="section-title">📝 วัตถุดิบ</div>', unsafe_allow_html=True)
                            st.markdown(format_ingredients(best_match["ingredients"]), unsafe_allow_html=True)
                            
                            st.markdown('<div class="section-title">👩‍🍳 วิธีทำ</div>', unsafe_allow_html=True)
                            st.markdown(format_cooking_method(best_match["method"]), unsafe_allow_html=True)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # แสดงเมนูที่เกี่ยวข้อง
                            if len(results) > 1:
                                st.markdown("### 🍽️ เมนูที่เกี่ยวข้อง")
                                for i, related in enumerate(results[1:4], 1):
                                    st.markdown(f"{i}. **{related['name']}** (ความเกี่ยวข้อง: {related['similarity']:.2f})")
                            
                            # เพิ่มการตอบสนองของแอสซิสแตนต์ลงในประวัติการสนทนาพร้อมข้อมูลสูตร
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response, 
                                "recipe": best_match
                            })
                        else:
                            response = "ขออภัย ฉันไม่พบสูตรอาหารที่ตรงกับคำค้นหาของคุณ กรุณาลองคำค้นหาอื่น"
                            st.markdown(response)
                            
                            # แสดงเมนูแนะนำ
                            st.markdown("### 🍽️ เมนูแนะนำ")
                            random_recipes = data.sample(5)
                            for _, recipe in random_recipes.iterrows():
                                st.markdown(f"- {recipe['name']}")
                            
                            st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        response = "ขออภัย ฉันไม่สามารถค้นหาสูตรอาหารได้ในขณะนี้ กรุณาลองใหม่อีกครั้ง"
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
