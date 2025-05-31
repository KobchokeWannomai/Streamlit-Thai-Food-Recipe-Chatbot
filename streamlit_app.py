import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import pickle
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import hashlib
import time
from difflib import SequenceMatcher

# นำเข้าไฟล์ที่สร้างขึ้นใหม่
from nutrition_api import NutritionAPI
from recipe_search import RecipeSearchEngine

# การตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Thai Food Recipe Chatbot with Advanced Nutrition",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ตั้งค่าฟอนต์ภาษาไทย
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
    
    .recipe-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        background: white;
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 0.8rem;
        border-radius: 8px;
        color: white;
        margin: 0.3rem 0;
        font-size: 0.9rem;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 0.8rem;
        border-radius: 8px;
        color: #333;
        margin: 0.3rem 0;
        font-size: 0.9rem;
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

# ตัวแปรไฟล์และโฟลเดอร์
DATA_PATH = "thai_food_processed.csv"
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
        return pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        st.error(f"ไม่พบไฟล์ข้อมูล: {DATA_PATH}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"ไม่สามารถโหลดข้อมูลได้: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def get_embeddings(_model, data):
    """สร้างหรือโหลด embeddings สำหรับสูตรอาหาร"""
    if _model is None or data.empty:
        return np.array([])
        
    if os.path.exists(EMBEDDINGS_PATH):
        try:
            with open(EMBEDDINGS_PATH, 'rb') as f:
                return pickle.load(f)
        except:
            pass
    
    # รวมข้อความทั้งหมดของแต่ละสูตร
    texts = []
    for _, row in data.iterrows():
        combined_text = f"{row['name']} {row['ingredient']} {row['method']}"
        texts.append(combined_text)
    
    if texts:
        # สร้าง embeddings
        with st.spinner("กำลังสร้าง embeddings สำหรับการค้นหา..."):
            embeddings = _model.encode(texts)
        
        # บันทึก embeddings
        with open(EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump(embeddings, f)
        
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

def generate_unique_key(base_key, recipe_name, chart_type=""):
    """สร้าง unique key สำหรับ elements"""
    content = f"{base_key}_{recipe_name}_{chart_type}_{time.time()}"
    return hashlib.md5(content.encode()).hexdigest()[:8]

def advanced_fuzzy_search(query, data, threshold=0.6):
    """ค้นหาแบบ fuzzy matching ที่ปรับปรุงแล้ว รองรับการพิมพ์ผิดได้ดีขึ้น"""
    query = query.lower().strip()
    matches = []
    
    for idx, recipe_name in enumerate(data['name']):
        recipe_name_lower = recipe_name.lower()
        
        # คำนวณความคล้ายคลึงหลายแบบ
        similarity_scores = []
        
        # 1. ความคล้ายคลึงโดยรวม
        overall_similarity = SequenceMatcher(None, query, recipe_name_lower).ratio()
        similarity_scores.append(overall_similarity)
        
        # 2. ตรวจสอบการมีคำคีย์เวิร์ดบางส่วน
        if query in recipe_name_lower:
            similarity_scores.append(0.95)
        
        # 3. ตรวจสอบคำต่างๆ ในชื่อ (รองรับการค้นหาแบบบางส่วน)
        query_words = query.split()
        recipe_words = recipe_name_lower.split()
        
        word_matches = 0
        partial_matches = 0
        
        for q_word in query_words:
            best_word_match = 0
            for r_word in recipe_words:
                # การจับคู่แบบเต็ม
                word_similarity = SequenceMatcher(None, q_word, r_word).ratio()
                if word_similarity > 0.8:
                    word_matches += word_similarity
                    best_word_match = max(best_word_match, word_similarity)
                # การจับคู่แบบบางส่วน
                elif (len(q_word) > 2 and q_word in r_word) or (len(r_word) > 2 and r_word in q_word):
                    partial_matches += 0.6
                    best_word_match = max(best_word_match, 0.6)
            
            # ค้นหาในส่วนผสมและวิธีทำด้วย
            if best_word_match < 0.5:
                ingredient_text = str(data.iloc[idx].get('ingredient', '')).lower()
                method_text = str(data.iloc[idx].get('method', '')).lower()
                
                if len(q_word) > 2:
                    if q_word in ingredient_text:
                        partial_matches += 0.4
                    elif q_word in method_text:
                        partial_matches += 0.3
        
        # คำนวณคะแนนรวมจากคำ
        if len(query_words) > 0:
            word_score = (word_matches + partial_matches) / len(query_words)
            similarity_scores.append(word_score)
        
        # 4. ตรวจสอบลำดับอักขระ (สำหรับการพิมพ์ผิด)
        # ลบช่องว่างและเปรียบเทียบ
        query_no_space = query.replace(' ', '')
        recipe_no_space = recipe_name_lower.replace(' ', '')
        
        if len(query_no_space) > 2:
            char_similarity = SequenceMatcher(None, query_no_space, recipe_no_space).ratio()
            similarity_scores.append(char_similarity * 0.8)  # ลดน้ำหนักเล็กน้อย
        
        # หาคะแนนสูงสุด
        max_similarity = max(similarity_scores) if similarity_scores else 0
        
        # ตรวจสอบและปรับคะแนนให้ไม่เกิน 1.0
        max_similarity = min(max_similarity, 1.0)
        
        if max_similarity >= threshold:
            matches.append((recipe_name, max_similarity, idx))
    
    # เรียงลำดับตามความคล้ายคลึง
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches

def format_ingredients(ingredients_text):
    """จัดรูปแบบรายการวัตถุดิบให้แสดงผลดี"""
    if not ingredients_text:
        return "<p>ไม่มีข้อมูลวัตถุดิบ</p>"
        
    ingredients = ingredients_text.split('\n')
    formatted = "<ul>"
    for item in ingredients:
        if item.strip():
            clean_item = item.strip().lstrip('- ')
            formatted += f"<li>{clean_item}</li>"
    formatted += "</ul>"
    return formatted

def format_cooking_method(method_text):
    """จัดรูปแบบวิธีทำใหม่ตามข้อกำหนด"""
    if not method_text:
        return "<p>ไม่มีข้อมูลวิธีทำ</p>"
    
    # ตรวจสอบว่ามีเลขขั้นตอนอยู่แล้วหรือไม่
    has_numbers = bool(re.search(r'^\d+\.', method_text.strip(), re.MULTILINE))
    
    # แยกตามการขึ้นบรรทัดใหม่
    lines = method_text.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # ตรวจสอบหัวข้อย่อย (ขึ้นต้นด้วย # หรือมีหมายเหตุ)
        if line.startswith('#'):
            # แปลง # เป็น HTML heading ขนาดเดียวกับหัวข้อหลัก
            heading_level = len(line) - len(line.lstrip('#'))
            heading_text = line.lstrip('#').strip()
            formatted_lines.append(f"<h4>{heading_text}</h4>")
        elif line.startswith('หมายเหตุ') or 'หมายเหตุ:' in line:
            formatted_lines.append(f"<h4>{line}</h4>")
        else:
            formatted_lines.append(f"<p>{line}</p>")
    
    # ถ้าไม่มีเลขขั้นตอนอยู่แล้ว และไม่มีหัวข้อย่อย ให้รวมเป็นย่อหน้าเดียว
    if not has_numbers and not any(line.startswith('<h4>') for line in formatted_lines):
        # รวมทุกย่อหน้าเป็นหนึ่งเดียว
        combined_text = ' '.join([line.replace('<p>', '').replace('</p>', '') for line in formatted_lines if line])
        return f"<p>{combined_text}</p>"
    
    return ''.join(formatted_lines)

def display_nutrition_chart(nutrition_data, recipe_name):
    """แสดงกราฟโภชนาการที่สวยงาม"""
    total_nutrition = nutrition_data['total_nutrition']
    
    # สร้างกราฟแบบ subplot
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

def display_nutrition_info(nutrition_data, recipe_name):
    """แสดงข้อมูลโภชนาการแบบครอบคลุม"""
    total_nutrition = nutrition_data['total_nutrition']
    
    # สร้าง unique key สำหรับ chart
    chart_key = generate_unique_key("nutrition_chart", recipe_name, "main")
    
    # แสดงกราฟโภชนาการ
    fig = display_nutrition_chart(nutrition_data, recipe_name)
    st.plotly_chart(fig, use_container_width=True, key=chart_key)
    
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
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown(f"""
        <div class="mineral-card">
            <h4>⚡ แร่ธาตุหลัก</h4>
            <p><strong>แคลเซียม:</strong> {total_nutrition['calcium']:.1f} mg</p>
            <p><strong>เหล็ก:</strong> {total_nutrition['iron']:.1f} mg</p>
            <p><strong>โปแตสเซียม:</strong> {total_nutrition['potassium']:.1f} mg</p>
            <p><strong>โซเดียม:</strong> {total_nutrition['sodium']:.1f} mg</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # แสดงคำแนะนำและคำเตือน
        recommendations = generate_nutrition_recommendations(total_nutrition)
        warnings = generate_nutrition_warnings(total_nutrition)
        
        if recommendations:
            for rec in recommendations:
                st.markdown(f'<div class="recommendation-card">✅ {rec}</div>', unsafe_allow_html=True)
        
        if warnings:
            for warning in warnings:
                st.markdown(f'<div class="warning-card">⚠️ {warning}</div>', unsafe_allow_html=True)

def generate_nutrition_recommendations(nutrition):
    """สร้างคำแนะนำโภชนาการ"""
    recommendations = []
    
    if nutrition['calories'] < 200:
        recommendations.append("เมนูแคลอรี่ต่ำ เหมาะสำหรับผู้ควบคุมน้ำหนัก")
    
    if nutrition['protein'] > 20:
        recommendations.append("โปรตีนสูง เหมาะสำหรับนักกีฬาและผู้สูงอายุ")
    
    if nutrition['fiber'] > 5:
        recommendations.append("ใยอาหารสูง ช่วยการย่อยและระบบขับถ่าย")
    
    if nutrition['vitamin_c'] > 30:
        recommendations.append("วิตามินซีสูง เสริมสร้างภูมิคุ้มกัน")
    
    if nutrition['calcium'] > 100:
        recommendations.append("แคลเซียมสูง เสริมสร้างกระดูกและฟัน")
    
    if nutrition['iron'] > 3:
        recommendations.append("เหล็กสูง ป้องกันโรคโลหิตจาง")
    
    return recommendations

def generate_nutrition_warnings(nutrition):
    """สร้างคำเตือนโภชนาการ"""
    warnings = []
    
    if nutrition['sodium'] > 1000:
        warnings.append("โซเดียมสูง ผู้ป่วยความดันสูงควรระวัง")
    
    if nutrition['fat'] > 30:
        warnings.append("ไขมันสูง ควรบริโภคในปริมาณจำกัด")
    
    if nutrition['calories'] > 450:
        warnings.append("แคลอรี่สูง ควรออกกำลังกายเพิ่มเติม")
    
    return warnings

def display_settings_panel(data, nutrition_api):
    """แสดงแถบการตั้งค่า พร้อมสถานะระบบ"""
    st.sidebar.title("🔧 การตั้งค่า")
    
    # แสดงสถานะระบบ
    st.sidebar.markdown("### 📊 สถานะระบบ")
    
    # ข้อมูลฐาน
    st.sidebar.markdown(f"""
    <div class="status-info">
        <strong>📖 จำนวนสูตรอาหาร:</strong> {len(data)} รายการ<br>
        <strong>🧮 ฐานข้อมูลโภชนาการ:</strong> {len(nutrition_api.local_nutrition_db)} วัตถุดิบ
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # การตั้งค่า API
    st.sidebar.markdown("### 🌐 API ข้อมูลโภชนาการ")
    
    # เลือกแหล่งข้อมูล API
    api_source = st.sidebar.selectbox(
        "เลือกแหล่งข้อมูลโภชนาการ",
        ["ฐานข้อมูลท้องถิ่น", "USDA FoodData Central", "Nutritionix", "Edamam"],
        key="api_source"
    )
    
    use_api = st.sidebar.checkbox("เปิดใช้งาน API ภายนอก", value=False, key="use_api")
    
    api_status = "🔴 ไม่ได้เชื่อมต่อ"
    if use_api:
        if api_source == "USDA FoodData Central":
            api_key = st.sidebar.text_input(
                "USDA API Key",
                type="password",
                help="ใส่ API Key จาก https://fdc.nal.usda.gov/api-guide.html",
                key="usda_api_key"
            )
            
            if api_key:
                nutrition_api.set_api_key(api_key, "usda")
                
                # ตรวจสอบสถานะ API
                if st.sidebar.button("ตรวจสอบการเชื่อมต่อ", key="check_usda"):
                    with st.spinner("กำลังตรวจสอบ..."):
                        status, message = nutrition_api.check_api_status()
                        if status:
                            api_status = "🟢 เชื่อมต่อแล้ว"
                            st.sidebar.success(f"✅ {message}")
                        else:
                            st.sidebar.error(f"❌ {message}")
        
        elif api_source == "Nutritionix":
            col1, col2 = st.sidebar.columns(2)
            with col1:
                app_id = st.sidebar.text_input("App ID", key="nutritionix_app_id")
            with col2:
                app_key = st.sidebar.text_input("App Key", type="password", key="nutritionix_app_key")
            
            if app_id and app_key:
                nutrition_api.set_nutritionix_credentials(app_id, app_key)
                api_status = "🟡 ตั้งค่าแล้ว"
    
    # แสดงสถานะ API
    status_class = "api-status-connected" if "🟢" in api_status else "api-status-disconnected"
    st.sidebar.markdown(f'<div class="{status_class}">สถานะ: {api_status}</div>', unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # การตั้งค่าการคำนวณโภชนาการ
    st.sidebar.markdown("### 🧮 การคำนวณโภชนาการ")
    
    adjust_consumption = st.sidebar.checkbox(
        "ปรับการบริโภคตามความเป็นจริง",
        value=True,
        help="คำนวณปริมาณที่บริโภคจริง (เช่น น้ำมันทอดจะไม่กินหมด)",
        key="adjust_consumption"
    )
    
    enhance_missing = st.sidebar.checkbox(
        "เพิ่มวัตถุดิบที่ขาดหาย",
        value=False,
        help="เพิ่มวัตถุดิบที่ไม่ได้ระบุแต่ใช้ในการปรุง (เช่น น้ำมันทอด)",
        key="enhance_missing"
    )
    
    # การตั้งค่าการค้นหา
    st.sidebar.markdown("### 🔍 การค้นหา")
    
    fuzzy_threshold = st.sidebar.slider(
        "ความเคร่งครัดในการค้นหา",
        min_value=0.3,
        max_value=0.9,
        value=0.6,
        step=0.1,
        help="ค่าต่ำ = หาได้ง่ายแต่อาจไม่ตรง, ค่าสูง = หาได้ยากแต่ตรงมาก",
        key="fuzzy_threshold"
    )
    
    max_results = st.sidebar.number_input(
        "จำนวนผลลัพธ์สูงสุด",
        min_value=1,
        max_value=10,
        value=3,
        key="max_results"
    )
    
    return {
        'api_source': api_source,
        'use_api': use_api,
        'adjust_consumption': adjust_consumption,
        'enhance_missing': enhance_missing,
        'fuzzy_threshold': fuzzy_threshold,
        'max_results': max_results
    }

def search_recipes(query, model, data, embeddings, search_engine, settings):
    """ค้นหาสูตรอาหารด้วยระบบที่ปรับปรุงแล้ว"""
    # ลองใช้ระบบค้นหาอัจฉริยะก่อน
    if search_engine is not None:
        smart_results = search_engine.smart_search(
            query, 
            settings['use_api'], 
            settings['adjust_consumption'],
            settings['enhance_missing'],
            settings['fuzzy_threshold'],
            limit=settings['max_results']
        )
        
        if smart_results:
            return [(result['name'], min(result['similarity'], 1.0), result['index'], result.get('nutrition')) 
                    for result in smart_results]
    
    # ใช้ระบบ fuzzy search ที่ปรับปรุงแล้ว
    fuzzy_results = advanced_fuzzy_search(query, data, threshold=settings['fuzzy_threshold'])
    
    if fuzzy_results:
        return [(name, similarity, idx, None) for name, similarity, idx in fuzzy_results[:settings['max_results']]]
    
    # หากไม่พบผลลัพธ์ ใช้ embedding search
    if model is not None and embeddings.size > 0:
        query_embedding = model.encode([query])
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        top_indices = np.argsort(-similarities)[:settings['max_results']]
        
        results = []
        for idx in top_indices:
            similarity = min(similarities[idx], 1.0)  # จำกัดให้ไม่เกิน 1.0
            if similarity > settings['fuzzy_threshold'] * 0.5:
                results.append((data.iloc[idx]['name'], similarity, idx, None))
        
        return results
    
    return []

def display_recipe_with_nutrition(recipe, nutrition_data, settings, recipe_key="default"):
    """แสดงสูตรอาหารพร้อมข้อมูลโภชนาการที่ปรับปรุงแล้ว"""
    # แสดงชื่อเมนู
    st.markdown(f"### 🍽️ {recipe['name']}")
    
    # สร้าง unique keys สำหรับ tabs
    tab_key = generate_unique_key("tabs", recipe['name'], recipe_key)
    
    # สร้าง tabs สำหรับแยกข้อมูล
    tab1, tab2, tab3, tab4 = st.tabs(["📝 สูตรอาหาร", "📊 โภชนาการ", "🔍 รายละเอียด", "💡 คำแนะนำ"])
    
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
            st.info("กำลังคำนวณข้อมูลโภชนาการ...")
    
    with tab3:
        if nutrition_data:
            # แสดงรายละเอียดวัตถุดิบพร้อมวิตามินและแร่ธาตุ
            st.markdown("#### 🔬 รายละเอียดวัตถุดิบแต่ละชนิด")
            
            ingredient_df = []
            for ingredient in nutrition_data['ingredient_details']:
                ingredient_df.append({
                    'วัตถุดิบ': ingredient['name'],
                    'ปริมาณ': f"{ingredient['quantity']} {ingredient['unit']}",
                    'น้ำหนัก (กรัม)': f"{ingredient['grams']:.1f}",
                    'น้ำหนักที่บริโภค (กรัม)': f"{ingredient.get('effective_grams', ingredient['grams']):.1f}",
                    'แคลอรี่ (kcal)': f"{ingredient['nutrition']['calories']:.1f}",
                    'โปรตีน (g)': f"{ingredient['nutrition']['protein']:.1f}",
                    'คาร์โบไฮเดรต (g)': f"{ingredient['nutrition']['carbs']:.1f}",
                    'ไขมัน (g)': f"{ingredient['nutrition']['fat']:.1f}",
                    'ใยอาหาร (g)': f"{ingredient['nutrition']['fiber']:.1f}",
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
            st.info("ข้อมูลรายละเอียดวัตถุดิบจะแสดงเมื่อคำนวณโภชนาการเสร็จสิ้น")
    
    with tab4:
        if nutrition_data:
            total_nutrition = nutrition_data['total_nutrition']
            
            # คำแนะนำเกี่ยวกับโภชนาการ
            recommendations = generate_nutrition_recommendations(total_nutrition)
            warnings = generate_nutrition_warnings(total_nutrition)
            
            if recommendations or warnings:
                st.markdown("#### 💡 คำแนะนำและข้อควรระวัง")
                
                if recommendations:
                    st.markdown("**✅ ข้อดี:**")
                    for rec in recommendations:
                        st.markdown(f"- {rec}")
                
                if warnings:
                    st.markdown("**⚠️ ข้อควรระวัง:**")
                    for warning in warnings:
                        st.markdown(f"- {warning}")
            
            # เปรียบเทียบกับความต้องการประจำวัน
            st.markdown("#### 📈 เปรียบเทียบกับความต้องการประจำวัน (ผู้ใหญ่)")
            
            daily_needs = {
                'calories': 2000, 'protein': 50, 'fat': 65, 'carbs': 300,
                'fiber': 25, 'calcium': 1000, 'iron': 18, 'vitamin_c': 90
            }
            
            comparison_data = []
            for nutrient, daily_need in daily_needs.items():
                if nutrient in total_nutrition:
                    percentage = (total_nutrition[nutrient] / daily_need) * 100
                    comparison_data.append({
                        'สารอาหาร': nutrient,
                        'ปริมาณในเมนู': f"{total_nutrition[nutrient]:.1f}",
                        '% ความต้องการประจำวัน': f"{percentage:.1f}%",
                        'สถานะ': '🟢 เพียงพอ' if percentage >= 20 else '🟡 น้อย' if percentage >= 10 else '🔴 น้อยมาก'
                    })
            
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True)

def main():
    """ฟังก์ชันหลักของแอปพลิเคชัน"""
    
    # แสดงหัวข้อแอป
    st.title("🍲 Thai Food Recipe Chatbot")
    st.markdown("### 🥘 ระบบแนะนำสูตรอาหารไทยพร้อมข้อมูลโภชนาการ")
    st.write("ถามเกี่ยวกับสูตรอาหารไทย หรือค้นหาตามคุณค่าทางโภชนาการ รองรับการค้นหาอัจฉริยะและการวิเคราะห์โภชนาการแบบละเอียด!")
    
    # เริ่มต้นระบบ
    with st.spinner("กำลังเริ่มต้นระบบ..."):
        model = load_model()
        data = load_data()
        
        if model is None or data.empty:
            st.error("ไม่สามารถเริ่มต้นระบบได้ กรุณาตรวจสอบไฟล์ข้อมูลและการเชื่อมต่อ")
            return
        
        embeddings = get_embeddings(model, data)
        
        # เริ่มต้น API และระบบค้นหา
        nutrition_api = initialize_nutrition_api()
        search_engine = initialize_search_engine(data, nutrition_api)
        
        # แถบการตั้งค่าพร้อมสถานะ
        settings = display_settings_panel(data, nutrition_api)
    
    # ตัวอย่างคำค้นหา
    st.markdown("#### 💡 ตัวอย่างการค้นหา:")
    example_queries = [
        "แนะนำอาหารแคลอรี่ต่ำ",
        "เมนูโปรตีนสูง", 
        "อาหารทอดกรอบ",
        "ไข่เจียว",
        "แกงเผ็ด",
        "อาหารมีแคลเซียมสูง",
        "เมนูใยอาหารสูง"
    ]
    
    cols = st.columns(len(example_queries))
    for i, query in enumerate(example_queries):
        if cols[i].button(query, key=f"example_{i}"):
            st.session_state['search_query'] = query
    
    # เริ่มต้น session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "search_query" not in st.session_state:
        st.session_state.search_query = ""
    if "message_counter" not in st.session_state:
        st.session_state.message_counter = 0
    
    # แสดงประวัติการสนทนา
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "recipe" in message:
                recipe = message["recipe"]
                nutrition_data = message.get("nutrition_data")
                recipe_key = f"history_{i}_{st.session_state.message_counter}"
                display_recipe_with_nutrition(recipe, nutrition_data, settings, recipe_key)
            else:
                st.markdown(message["content"])
    
    # ช่องค้นหา
    search_query = st.session_state.get('search_query', '')
    if prompt := st.chat_input("ค้นหาสูตรอาหาร หรือถามเกี่ยวกับโภชนาการ...", key="main_chat"):
        search_query = prompt
        st.session_state.search_query = ""
        st.session_state.message_counter += 1
    
    if search_query:
        # เพิ่มข้อความของผู้ใช้
        st.session_state.messages.append({"role": "user", "content": search_query})
        
        # แสดงข้อความของผู้ใช้
        with st.chat_message("user"):
            st.markdown(search_query)
        
        # ประมวลผลและตอบกลับ
        with st.chat_message("assistant"):
            with st.spinner("กำลังค้นหาและคำนวณข้อมูลโภชนาการขั้นสูง..."):
                # ค้นหาสูตรอาหาร
                results = search_recipes(
                    search_query, model, data, embeddings, search_engine, settings
                )
                
                if results:
                    best_match = results[0]
                    recipe_name, similarity, recipe_idx, cached_nutrition = best_match
                    
                    if similarity > 0.2:
                        # ดึงข้อมูลสูตร
                        recipe = {
                            'name': recipe_name,
                            'ingredient': data.iloc[recipe_idx]['ingredient'],
                            'method': data.iloc[recipe_idx]['method']
                        }
                        
                        # คำนวณข้อมูลโภชนาการ
                        if cached_nutrition:
                            nutrition_data = cached_nutrition
                        else:
                            nutrition_data = nutrition_api.calculate_recipe_nutrition(
                                recipe['ingredient'],
                                settings['use_api'],
                                settings['adjust_consumption'],
                                settings['enhance_missing']
                            )
                        
                        response = f"🎯 พบสูตรอาหารที่ตรงกับการค้นหา: **{recipe_name}** (ความคล้ายคลึง: {similarity:.0%})"
                        st.markdown(response)
                        
                        # แสดงสูตรและโภชนาการ
                        main_recipe_key = f"main_{st.session_state.message_counter}"
                        display_recipe_with_nutrition(recipe, nutrition_data, settings, main_recipe_key)
                        
                        # เพิ่มการแนะนำเพิ่มเติม
                        if len(results) > 1:
                            st.markdown("#### 🔍 เมนูอื่นที่น่าสนใจ:")
                            other_results = results[1:min(4, len(results))]
                            
                            cols = st.columns(len(other_results))
                            for i, (other_name, other_sim, other_idx, _) in enumerate(other_results):
                                with cols[i]:
                                    if st.button(f"🍽️ {other_name}\n({other_sim:.0%})", key=f"other_{st.session_state.message_counter}_{i}"):
                                        st.session_state.search_query = other_name
                                        st.rerun()
                        
                        # บันทึกข้อความตอบกลับ
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response, 
                            "recipe": recipe,
                            "nutrition_data": nutrition_data
                        })
                    else:
                        response = f"""
                        ❌ ไม่พบสูตรอาหารที่ตรงกับ '{search_query}' ในระดับที่เพียงพอ
                        
                        💡 **คำแนะนำ:**
                        - ลองใช้คำค้นหาที่ง่ายกว่า เช่น "ไข่เจียว" แทน "วิธีทำไข่เจียว"
                        - ค้นหาตามโภชนาการ เช่น "อาหารแคลอรี่ต่ำ" "เมนูโปรตีนสูง"
                        - ค้นหาตามประเภท เช่น "อาหารทอด" "อาหารต้ม"
                        - ลองปรับการตั้งค่าความเคร่งครัดในการค้นหาในแถบด้านซ้าย
                        """
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    response = """
                    🤔 ไม่พบสูตรอาหารที่ตรงกับคำค้นหา
                    
                    ลองค้นหาด้วยคำเหล่านี้:
                    - **ชื่อเมนูอาหาร:** "ผัดไทย", "ต้มยำกุ้ง", "แกงเขียวหวาน"
                    - **ประเภทอาหาร:** "อาหารทอด", "แกง", "ยำ", "ขนม"
                    - **โภชนาการ:** "แคลอรี่ต่ำ", "โปรตีนสูง", "แคลเซียมสูง"
                    - **วัตถุดิบหลัก:** "ไก่", "หมู", "ปลา", "ผัก"
                    """
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
