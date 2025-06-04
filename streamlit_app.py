import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import re
from typing import Dict, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import จากไฟล์ที่เราสร้าง
from nutrition_api import NutritionAPI
from recipe_search import RecipeSearchEngine
from nutrition_search import NutritionBasedSearchEngine

# การตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Thai Food Recipe Chatbot - Advanced Nutrition",
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
    
    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: bold;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .enhancement-note {
        background: linear-gradient(135deg, #e8f5e8 0%, #f0f8f0 100%);
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

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
            '- ไข่ไก่ 3 ฟอง\n- น้ำปลา 1 ช้อนชา\n- ต้นหอม 2 ต้น',
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
def initialize_search_engines(_data, _nutrition_api):
    """เริ่มต้นระบบค้นหาทั้งหมด"""
    if _data.empty:
        return None, None
    
    recipe_engine = RecipeSearchEngine(_data, _nutrition_api)
    nutrition_engine = NutritionBasedSearchEngine(_data, _nutrition_api)
    
    return recipe_engine, nutrition_engine

def format_ingredients(ingredients_text):
    """จัดรูปแบบรายการวัตถุดิบ"""
    if not ingredients_text:
        return "<p>ไม่มีข้อมูลวัตถุดิบ</p>"
        
    ingredients = ingredients_text.split('\n')
    formatted = "<ul style='margin: 0; padding-left: 1.5rem; line-height: 1.6;'>"
    for item in ingredients:
        if item.strip():
            clean_item = item.strip().lstrip('- ')
            # เน้นปริมาณและหน่วย
            highlighted_item = re.sub(
                r'(\d+(?:\.\d+)?)\s*([ก-๙a-zA-Z]+)', 
                r'<strong>\1 \2</strong>', 
                clean_item
            )
            formatted += f"<li style='margin: 0.3rem 0;'>{highlighted_item}</li>"
    formatted += "</ul>"
    return formatted

def format_cooking_method(method_text):
    """จัดรูปแบบวิธีทำ"""
    if not method_text:
        return "ไม่มีข้อมูลวิธีทำ"
    
    # ตรวจสอบว่ามีเลขขั้นตอนอยู่แล้วหรือไม่
    has_numbers = bool(re.search(r'^\s*\d+\.', method_text, re.MULTILINE))
    
    sentences = re.split(r'[.!?]', method_text)
    formatted = "<div style='margin: 0; line-height: 1.8;'>"
    
    step_num = 1
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        if has_numbers and re.match(r'^\d+', sentence):
            formatted += f"<p style='margin: 0.5rem 0; padding: 0.3rem; background: #f8f9fa; border-left: 3px solid #007bff; border-radius: 3px;'><strong>{sentence}</strong></p>"
        else:
            # เพิ่มหมายเลขขั้นตอนถ้ายังไม่มี
            if not has_numbers and len(sentence) > 10:
                formatted += f"<p style='margin: 0.5rem 0; padding: 0.3rem; background: #f8f9fa; border-left: 3px solid #28a745; border-radius: 3px;'><strong>{step_num}.</strong> {sentence}</p>"
                step_num += 1
            else:
                formatted += f"<p style='margin: 0.5rem 0;'>{sentence}</p>"
    
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
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧮 การคำนวณโภชนาการ")
    
    adjust_consumption = st.sidebar.checkbox(
        "ปรับการบริโภคตามความเป็นจริง", value=True, key="adjust_consumption",
        help="ปรับปริมาณสารอาหารตามที่บริโภคจริง เช่น น้ำมันทอดจะไม่กินหมดทั้งหมด"
    )
    
    enhance_missing = st.sidebar.checkbox(
        "เพิ่มวัตถุดิบที่ขาดหาย", value=False, key="enhance_missing",
        help="เพิ่มวัตถุดิบที่อาจจำเป็นแต่ไม่ได้ระบุไว้ เช่น น้ำมันสำหรับทอด"
    )
    
    # แสดงรายละเอียดการทำงาน
    if adjust_consumption:
        st.sidebar.success("🔧 ระบบจะปรับสัดส่วนการบริโภคจริง")
        with st.sidebar.expander("ดูรายละเอียด"):
            st.write("""
            **ตัวอย่างการปรับสัดส่วน:**
            - น้ำมันทอด: ใช้ 25% (เหลือในกะทะ)
            - กะทิ: ใช้ 85% (ซึมซับส่วนหนึ่ง)
            - น้ำปลา/เกลือ: ใช้ 100% (กินหมด)
            """)
    
    if enhance_missing:
        st.sidebar.success("✨ ระบบจะเพิ่มวัตถุดิบที่ขาดหาย")
        with st.sidebar.expander("ดูรายละเอียด"):
            st.write("""
            **ตัวอย่างการเพิ่มวัตถุดิบ:**
            - อาหารทอด → เพิ่มน้ำมันพืช
            - ไข่เจียว → เพิ่มน้ำมันหมู
            - อาหารผัด → เพิ่มกระเทียม, หอมแดง
            - ยำ → เพิ่มน้ำปลา, มะนาว, น้ำตาล
            """)
    
    st.sidebar.markdown("### 🔍 การค้นหาที่ปรับปรุงแล้ว")
    
    search_method = st.sidebar.selectbox(
        "วิธีการค้นหา",
        options=["hybrid", "fuzzy", "semantic"],
        index=0,
        format_func=lambda x: {
            "hybrid": "🤖 AI + Fuzzy (แนะนำ)",
            "fuzzy": "🔤 Fuzzy Match",
            "semantic": "🧠 AI Semantic"
        }[x],
        key="search_method"
    )
    
    fuzzy_threshold = st.sidebar.slider(
        "ความเคร่งครัดในการค้นหา",
        min_value=0.2, max_value=0.8, value=0.4, step=0.1,
        help="ค่าต่ำ = หาได้ง่าย, ค่าสูง = หาได้ยากแต่ตรงมาก",
        key="fuzzy_threshold"
    )
    
    max_results = st.sidebar.number_input(
        "จำนวนผลลัพธ์สูงสุด", min_value=1, max_value=10, value=5, key="max_results"
    )
    
    st.sidebar.markdown("---")

    # แสดงสถิติโดยรวม
    st.sidebar.markdown("### 📊 สถิติโดยรวม")
    st.sidebar.metric("📖 จำนวนสูตร", len(data))
    st.sidebar.metric("🤖 AI Model", "✅ พร้อม" if model else "❌ ไม่พร้อม")
    
    api_status = "🟢 เชื่อมต่อ" if settings_state.get('use_api', False) else "🔴 ปิดใช้งาน"
    st.sidebar.metric("🌐 API", api_status)
    
    adjust_status = "🟢 เปิด" if settings_state.get('adjust_consumption', True) else "🔴 ปิด"
    st.sidebar.metric("⚖️ ปรับการบริโภค", adjust_status)
    
    enhance_status = "🟢 เปิด" if settings_state.get('enhance_missing', False) else "🔴 ปิด"
    st.sidebar.metric("✨ เพิ่มวัตถุดิบ", enhance_status)
    
    st.sidebar.metric("🔍 การค้นหา", f"✨ {search_method.upper()}")
    
    return {
        'use_api': use_api, 'adjust_consumption': adjust_consumption,
        'enhance_missing': enhance_missing, 'fuzzy_threshold': fuzzy_threshold,
        'max_results': max_results, 'search_method': search_method
    }

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
            # แสดงการตั้งค่าที่ใช้ในการคำนวณ
            settings_info = nutrition_data.get('settings', {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                api_status = "🟢 เปิด" if settings_info.get('use_api', False) else "🔴 ปิด"
                st.metric("API ภายนอก", api_status)
            with col2:
                adjust_status = "🟢 เปิด" if settings_info.get('adjust_consumption', False) else "🔴 ปิด"
                st.metric("ปรับการบริโภค", adjust_status)
            with col3:
                enhance_status = "🟢 เปิด" if settings_info.get('enhance_missing', False) else "🔴 ปิด"
                st.metric("เพิ่มวัตถุดิบ", enhance_status)
            
            # แสดงวัตถุดิบที่ปรับปรุงแล้ว (หากมี)
            enhanced_ingredients = nutrition_data.get('enhanced_ingredients')
            if enhanced_ingredients and enhanced_ingredients != recipe['ingredient']:
                st.markdown("#### 🔧 วัตถุดิบที่ปรับปรุงแล้ว")
                st.markdown("""
                <div class="enhancement-note">
                    ✨ ระบบได้เพิ่มวัตถุดิบที่อาจขาดหายไปตามวิธีการทำ
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**วัตถุดิบเดิม:**")
                    st.markdown(format_ingredients(recipe['ingredient']), unsafe_allow_html=True)
                with col2:
                    st.markdown("**วัตถุดิบที่เพิ่มแล้ว:**")
                    st.markdown(format_ingredients(enhanced_ingredients), unsafe_allow_html=True)
                
                st.markdown("---")
            
            st.markdown("#### 🔬 รายละเอียดวัตถุดิบแต่ละชนิด")
            
            ingredient_df = []
            for ingredient in nutrition_data['ingredient_details']:
                consumption_factor = ingredient.get('consumption_factor', 1.0)
                consumption_note = ""
                if consumption_factor < 1.0:
                    consumption_note = f" (ใช้ {consumption_factor*100:.0f}%)"
                
                ingredient_df.append({
                    'วัตถุดิบ': ingredient['name'] + consumption_note,
                    'ปริมาณ': f"{ingredient['quantity']} {ingredient['unit']}",
                    'น้ำหนัก (กรัม)': f"{ingredient['grams']:.1f}",
                    'น้ำหนักจริง (กรัม)': f"{ingredient['effective_grams']:.1f}",
                    'แคลอรี่': f"{ingredient['nutrition']['calories']:.1f}",
                    'โปรตีน (g)': f"{ingredient['nutrition']['protein']:.1f}",
                    'ไขมัน (g)': f"{ingredient['nutrition']['fat']:.1f}",
                    'วิตามิน C (mg)': f"{ingredient['nutrition']['vitamin_c']:.1f}",
                    'แคลเซียม (mg)': f"{ingredient['nutrition']['calcium']:.1f}",
                    'โซเดียม (mg)': f"{ingredient['nutrition']['sodium']:.1f}"
                })
            
            if ingredient_df:
                df = pd.DataFrame(ingredient_df)
                st.dataframe(df, use_container_width=True)
                
                # คำอธิบายเกี่ยวกับการปรับการบริโภค
                if settings_info.get('adjust_consumption', False):
                    st.markdown("""
                    **💡 คำอธิบาย:** 
                    - **น้ำหนัก (กรัม)**: น้ำหนักของวัตถุดิบตามสูตร
                    - **น้ำหนักจริง (กรัม)**: น้ำหนักที่ปรับตามการบริโภคจริง
                    - วัตถุดิบบางอย่าง เช่น น้ำมันทอด จะไม่กินหมดทั้งหมด
                    """)
        else:
            st.info("ไม่มีข้อมูลรายละเอียดวัตถุดิบ")

def main():
    """ฟังก์ชันหลักของแอปพลิเคชัน"""
    
    st.markdown('<h1 class="main-title">Thai Food Recipe Chatbot</h1>', unsafe_allow_html=True)
    st.markdown("### 🥘 ระบบค้นหาสูตรอาหารไทยขั้นสูงพร้อม AI และโภชนาการ")
    
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
        recipe_engine, nutrition_engine = initialize_search_engines(data, nutrition_api)
        
        # ตั้งค่า model และ embeddings สำหรับ search engines
        if recipe_engine:
            recipe_engine.set_model_and_embeddings(model, embeddings)
    
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
    if prompt := st.chat_input("ค้นหาสูตรอาหาร เช่น 'ผัดกะเพรา' หรือ 'ไข่เจียว' หรือ 'อาหารแคลอรี่ต่ำ'... ", key="main_chat"):
        search_query = prompt
        st.session_state.search_query = ""
    
    if search_query:
        st.session_state.messages.append({"role": "user", "content": search_query})
        
        with st.chat_message("user"):
            st.markdown(search_query)
        
        with st.chat_message("assistant"):
            with st.spinner("กำลังค้นหาด้วยระบบ AI ที่ปรับปรุงแล้ว..."):
                
                # ตรวจสอบว่าเป็นการค้นหาตามโภชนาการหรือไม่
                nutrition_intent = nutrition_engine.detect_enhanced_nutrition_intent(search_query) if nutrition_engine else None
                
                if nutrition_intent and nutrition_intent.get('is_nutrition_query', False):
                    # ใช้ระบบค้นหาตามโภชนาการ
                    nutrition_results = nutrition_engine.search_by_enhanced_nutrition_criteria(
                        nutrition_intent['criteria'], 
                        limit=settings['max_results']
                    )
                    
                    if nutrition_results:
                        response = f"🎯 พบเมนูที่ตรงกับเกณฑ์โภชนาการ: **{search_query}**"
                        
                        # แสดงสถิติโดยรวม
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📋 พบเมนู", f"{len(nutrition_results)} รายการ")
                        with col2:
                            avg_score = sum(r['score'] for r in nutrition_results) / len(nutrition_results)
                            st.metric("⭐ คะแนนเฉลี่ย", f"{avg_score:.1f}")
                        with col3:
                            matched_criteria = nutrition_intent.get('criteria', [])
                            st.metric("🔍 เกณฑ์ตรงกัน", f"{len(matched_criteria)} เกณฑ์")
                        
                        st.markdown(response)
                        
                        # แสดงเมนูแนะนำแบบโภชนาการ
                        for i, result in enumerate(nutrition_results[:settings['max_results']], 1):
                            with st.expander(f"🥘 {i}. {result['name']} (คะแนน: {result['score']:.1f})", expanded=(i == 1)):
                                
                                recipe_data = {
                                    'name': result['name'],
                                    'ingredient': data.iloc[result['index']]['ingredient'],
                                    'method': data.iloc[result['index']]['method']
                                }
                                
                                # แสดงเหตุผลการแนะนำ
                                st.markdown("#### ✨ เหมาะสมเพราะ")
                                for reason in result['reasons']:
                                    st.markdown(f"• {reason}")
                                
                                if result.get('health_benefits'):
                                    st.markdown("#### 💊 ประโยชน์ต่อสุขภาพ")
                                    for benefit in result['health_benefits']:
                                        st.markdown(f"• {benefit}")
                                
                                # แสดงข้อมูลโภชนาการ
                                nutrition_data = {
                                    'total_nutrition': result['nutrition'],
                                    'ingredient_details': [],
                                    'settings': settings
                                }
                                
                                display_nutrition_info(nutrition_data, result['name'])
                        
                        st.session_state.messages.append({
                            "role": "assistant", "content": response
                        })
                    else:
                        response = f"❌ ไม่พบเมนูที่ตรงกับเกณฑ์โภชนาการ '{search_query}'"
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                
                else:
                    # ใช้ระบบค้นหาสูตรอาหารทั่วไป
                    results = recipe_engine.smart_search(
                        search_query, 
                        use_api=settings['use_api'],
                        adjust_consumption=settings['adjust_consumption'],
                        enhance_missing=settings['enhance_missing'],
                        fuzzy_threshold=settings['fuzzy_threshold'],
                        limit=settings['max_results'],
                        search_method=settings['search_method']
                    ) if recipe_engine else []
                    
                    if results:
                        best_match = results[0]
                        recipe_name = best_match['name']
                        similarity = best_match['similarity']
                        recipe_idx = best_match['index']
                        nutrition_data = best_match['nutrition']
                        
                        recipe = {
                            'name': recipe_name,
                            'ingredient': data.iloc[recipe_idx]['ingredient'],
                            'method': data.iloc[recipe_idx]['method']
                        }
                        
                        response = f"🎯 พบสูตรอาหารที่ตรงกับการค้นหา: **{recipe_name}**"
                        
                        # แสดงข้อมูลการปรับปรุงที่ใช้
                        settings_used = []
                        if settings.get('enhance_missing', False):
                            enhanced_ingredients = nutrition_data.get('enhanced_ingredients')
                            if enhanced_ingredients and enhanced_ingredients != recipe['ingredient']:
                                settings_used.append("✨ เพิ่มวัตถุดิบที่ขาดหาย")
                        
                        if settings.get('adjust_consumption', True):
                            # ตรวจสอบว่ามีการปรับสัดส่วนจริงหรือไม่
                            has_adjustment = any(
                                detail.get('consumption_factor', 1.0) < 1.0 
                                for detail in nutrition_data.get('ingredient_details', [])
                            )
                            if has_adjustment:
                                settings_used.append("⚖️ ปรับการบริโภคตามความจริง")
                        
                        if settings_used:
                            response += f"\n\n🔧 **การปรับปรุง:** {' • '.join(settings_used)}"
                        
                        st.markdown(response)
                        
                        display_recipe_with_nutrition(recipe, nutrition_data, settings, similarity)
                        
                        # แนะนำเมนูอื่น
                        if len(results) > 1:
                            st.markdown("#### 🔍 เมนูอื่นที่น่าสนใจ:")
                            other_results = results[1:min(4, len(results))]
                            
                            cols = st.columns(len(other_results))
                            for i, result in enumerate(other_results):
                                with cols[i]:
                                    similarity_percent = result['similarity'] * 100
                                    if st.button(f"🍽️ {result['name']}\n({similarity_percent:.0f}% ตรง)", key=f"other_{i}"):
                                        st.session_state.search_query = result['name']
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
                        - ลองค้นหาตามโภชนาการ เช่น "อาหารแคลอรี่ต่ำ"
                        - ปรับค่าความเคร่งครัดในการค้นหาในแถบด้านซ้าย (ลดค่าลง)
                        
                        🍽️ **เมนูที่มีในระบบ:** {', '.join(data['name'].head(10).tolist())}
                        """
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
