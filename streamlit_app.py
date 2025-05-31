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
    
    .similarity-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin-left: 0.5rem;
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
            # หากไฟล์เสียหาย ให้สร้างใหม่
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

def improved_fuzzy_search(query, data, threshold=0.4):
    """ปรับปรุงการค้นหาแบบ fuzzy matching ให้แม่นยำขึ้น"""
    query = query.lower().strip()
    matches = []
    
    for idx, recipe_name in enumerate(data['name']):
        recipe_name_lower = recipe_name.lower()
        
        # คำนวณความคล้ายคลึงแบบง่าย
        basic_similarity = SequenceMatcher(None, query, recipe_name_lower).ratio()
        
        # ตรวจสอบการมีคำคีย์เวิร์ดที่ตรงกัน
        exact_match_score = 0
        if query in recipe_name_lower:
            exact_match_score = len(query) / len(recipe_name_lower)
        
        # ตรวจสอบคำต่างๆ ในชื่อ
        query_words = query.split()
        recipe_words = recipe_name_lower.split()
        
        word_match_score = 0
        partial_match_score = 0
        
        for q_word in query_words:
            best_word_match = 0
            for r_word in recipe_words:
                # การจับคู่แบบเต็ม
                word_similarity = SequenceMatcher(None, q_word, r_word).ratio()
                if word_similarity > 0.8:
                    best_word_match = max(best_word_match, word_similarity)
                # การจับคู่แบบบางส่วน
                elif len(q_word) > 2 and q_word in r_word:
                    best_word_match = max(best_word_match, 0.7)
                elif len(r_word) > 2 and r_word in q_word:
                    best_word_match = max(best_word_match, 0.6)
            
            if best_word_match > 0:
                word_match_score += best_word_match
                
        # คำนวณคะแนนรวม
        if len(query_words) > 0:
            word_match_score = word_match_score / len(query_words)
        
        # รวมคะแนนทั้งหมด
        final_score = max(
            basic_similarity * 0.4,
            exact_match_score * 0.8,
            word_match_score * 0.7
        )
        
        # ตรวจสอบในส่วนผสมและวิธีทำ (คะแนนน้อยกว่า)
        if final_score < threshold:
            ingredient_text = str(data.iloc[idx].get('ingredient', '')).lower()
            method_text = str(data.iloc[idx].get('method', '')).lower()
            
            ingredient_match = 0
            method_match = 0
            
            for q_word in query_words:
                if len(q_word) > 2:
                    if q_word in ingredient_text:
                        ingredient_match += 0.3
                    if q_word in method_text:
                        method_match += 0.2
            
            final_score = max(final_score, ingredient_match, method_match)
        
        # เก็บเฉพาะที่มีคะแนนเกินเกณฑ์
        if final_score >= threshold:
            matches.append((recipe_name, final_score, idx))
    
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
            # ลบเครื่องหมาย - ถ้ามี
            clean_item = item.strip().lstrip('- ')
            formatted += f"<li>{clean_item}</li>"
    formatted += "</ul>"
    return formatted

def format_cooking_method(method_text):
    """จัดรูปแบบวิธีทำให้แสดงผลดี"""
    if not method_text:
        return "<p>ไม่มีข้อมูลวิธีทำ</p>"
        
    # แบ่งประโยคตามจุด หรือช่องว่างยาว
    sentences = re.split(r'(?<=[ๆ.।])\s+|(?<=\w)\s{2,}', method_text)
    formatted = "<ol>"
    for sentence in sentences:
        if sentence.strip():
            formatted += f"<li>{sentence.strip()}</li>"
    formatted += "</ol>"
    return formatted

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

def display_settings_panel():
    """แสดงแถบการตั้งค่าขั้นสูง"""
    st.sidebar.title("🔧 การตั้งค่าขั้นสูง")
    
    # การตั้งค่า API
    st.sidebar.markdown("### 🌐 API ข้อมูลโภชนาการ")
    
    use_api = st.sidebar.checkbox("เปิดใช้งาน API ภายนอก", value=False, key="use_api")
    
    api_status = "🔴 ไม่ได้เชื่อมต่อ"
    if use_api:
        api_key = st.sidebar.text_input(
            "USDA API Key",
            type="password",
            help="ใส่ API Key จาก https://fdc.nal.usda.gov/api-guide.html",
            key="usda_api_key"
        )
        
        if api_key:
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
        help="คำนวณปริมาณที่บริโภคจริง",
        key="adjust_consumption"
    )
    
    enhance_missing = st.sidebar.checkbox(
        "เพิ่มวัตถุดิบที่ขาดหาย",
        value=False,
        help="เพิ่มวัตถุดิบที่ไม่ได้ระบุแต่ใช้ในการปรุง",
        key="enhance_missing"
    )
    
    # การตั้งค่าการแสดงผล
    st.sidebar.markdown("### 🎨 การแสดงผล")
    
    show_charts = st.sidebar.checkbox("แสดงกราฟโภชนาการ", value=True, key="show_charts")
    show_details = st.sidebar.checkbox("แสดงรายละเอียดวัตถุดิบ", value=True, key="show_details")
    
    # การตั้งค่าการค้นหา
    st.sidebar.markdown("### 🔍 การค้นหา")
    
    fuzzy_threshold = st.sidebar.slider(
        "ความเคร่งครัดในการค้นหา",
        min_value=0.2,
        max_value=0.8,
        value=0.4,
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
        'use_api': use_api,
        'adjust_consumption': adjust_consumption,
        'enhance_missing': enhance_missing,
        'show_charts': show_charts,
        'show_details': show_details,
        'fuzzy_threshold': fuzzy_threshold,
        'max_results': max_results
    }

def search_recipes_improved(query, model, data, embeddings, search_engine, settings):
    """ปรับปรุงการค้นหาสูตรอาหารให้แม่นยำขึ้น"""
    
    # ลองใช้ฟังก์ชันค้นหาที่ปรับปรุงแล้วก่อน
    fuzzy_results = improved_fuzzy_search(query, data, threshold=settings['fuzzy_threshold'])
    
    if fuzzy_results:
        results = []
        for recipe_name, similarity, recipe_idx in fuzzy_results[:settings['max_results']]:
            nutrition_data = None
            try:
                if search_engine:
                    nutrition_data = search_engine.get_recipe_nutrition(
                        recipe_idx, 
                        settings['use_api'], 
                        settings['adjust_consumption'],
                        settings['enhance_missing']
                    )
            except:
                pass
            
            results.append((recipe_name, similarity, recipe_idx, nutrition_data))
        
        return results
    
    # หากไม่พบด้วย fuzzy search ให้ใช้ embedding search
    if embeddings.size > 0 and model is not None:
        try:
            query_embedding = model.encode([query])
            similarities = cosine_similarity(query_embedding, embeddings)[0]
            
            # ปรับเกณฑ์ให้เหมาะสม
            adjusted_threshold = max(0.1, settings['fuzzy_threshold'] * 0.3)
            valid_indices = np.where(similarities >= adjusted_threshold)[0]
            
            if len(valid_indices) > 0:
                # เรียงลำดับและเลือกผลลัพธ์ที่ดีที่สุด
                sorted_indices = valid_indices[np.argsort(-similarities[valid_indices])]
                
                results = []
                for idx in sorted_indices[:settings['max_results']]:
                    nutrition_data = None
                    try:
                        if search_engine:
                            nutrition_data = search_engine.get_recipe_nutrition(
                                idx, 
                                settings['use_api'], 
                                settings['adjust_consumption'],
                                settings['enhance_missing']
                            )
                    except:
                        pass
                    
                    results.append((
                        data.iloc[idx]['name'], 
                        float(similarities[idx]),  # แปลงเป็น Python float
                        idx, 
                        nutrition_data
                    ))
                
                return results
        except Exception as e:
            st.warning(f"เกิดข้อผิดพลาดในการใช้ embedding search: {str(e)}")
    
    return []

def display_recipe_with_nutrition(recipe, nutrition_data, settings, similarity_score=None):
    """แสดงสูตรอาหารพร้อมข้อมูลโภชนาการที่ปรับปรุงแล้ว"""
    
    # แสดงชื่อเมนูพร้อมคะแนนความคล้ายคลึง
    title_html = f"### 🍽️ {recipe['name']}"
    if similarity_score is not None:
        similarity_percent = similarity_score * 100
        badge_color = "#667eea" if similarity_percent >= 70 else "#f093fb" if similarity_percent >= 50 else "#ff9a9e"
        title_html += f'<span style="background: {badge_color}; color: white; padding: 0.2rem 0.5rem; border-radius: 15px; font-size: 0.8rem; font-weight: bold; margin-left: 0.5rem;">{similarity_percent:.0f}% ตรง</span>'
    
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
            display_nutrition_info(nutrition_data, recipe['name'], settings['show_charts'])
        else:
            st.info("ไม่มีข้อมูลโภชนาการ หรือเกิดข้อผิดพลาดในการคำนวณ")
    
    with tab3:
        if nutrition_data and settings['show_details']:
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
            st.info("เปิดใช้งานการแสดงรายละเอียดในการตั้งค่าเพื่อดูข้อมูลเพิ่มเติม")

def main():
    """ฟังก์ชันหลักของแอปพลิเคชัน"""
    
    # แสดงหัวข้อแอป
    st.title("🍲 Thai Food Recipe Chatbot")
    st.markdown("### 🥘 ระบบแนะนำสูตรอาหารไทยพร้อมข้อมูลโภชนาการขั้นสูง")
    st.write("ค้นหาสูตรอาหารไทยด้วยระบบค้นหาที่ปรับปรุงใหม่ รองรับการพิมพ์ผิดและการค้นหาที่แม่นยำยิ่งขึ้น!")
    
    # เริ่มต้นระบบ
    with st.spinner("กำลังเริ่มต้นระบบ..."):
        model = load_model()
        data = load_data()
        
        if model is None or data.empty:
            st.error("ไม่สามารถเริ่มต้นระบบได้ กรุณาตรวจสอบไฟล์ข้อมูลและการเชื่อมต่อ")
            return
        
        embeddings = get_embeddings(model, data)
        
        # แถบการตั้งค่า
        settings = display_settings_panel()
        
        # เริ่มต้น API และระบบค้นหา
        nutrition_api = initialize_nutrition_api()
        search_engine = initialize_search_engine(data, nutrition_api)
    
    # แสดงสถิติเบื้องต้น
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📖 จำนวนสูตร", len(data))
    with col2:
        st.metric("🔥 AI Model", "✅ พร้อม" if model else "❌ ไม่พร้อม")
    with col3:
        api_status = "🟢 เชื่อมต่อ" if settings['use_api'] else "🔴 ปิดใช้งาน"
        st.metric("🌐 API", api_status)
    with col4:
        search_status = "🟢 ปรับปรุงแล้ว"
        st.metric("🔍 การค้นหา", search_status)
    
    # ตัวอย่างคำค้นหา
    st.markdown("#### 💡 ตัวอย่างการค้นหา:")
    example_queries = [
        "ผัดกะเพรา", "ต้มยำกุ้ง", "ไข่เจียว", "ส้มตำ", "แกงเขียวหวาน"
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
    if prompt := st.chat_input("ค้นหาสูตรอาหาร เช่น 'ผัดกะเพรา' หรือ 'ไข่เจียว'...", key="main_chat"):
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
                results = search_recipes_improved(
                    search_query, model, data, embeddings, search_engine, settings
                )
                
                if results:
                    best_match = results[0]
                    recipe_name, similarity, recipe_idx, nutrition_data = best_match
                    
                    # ดึงข้อมูลสูตร
                    recipe = {
                        'name': recipe_name,
                        'ingredient': data.iloc[recipe_idx]['ingredient'],
                        'method': data.iloc[recipe_idx]['method']
                    }
                    
                    response = f"🎯 พบสูตรอาหารที่ตรงกับการค้นหา: **{recipe_name}**"
                    st.markdown(response)
                    
                    # แสดงสูตรและโภชนาการ
                    display_recipe_with_nutrition(recipe, nutrition_data, settings, similarity)
                    
                    # เพิ่มการแนะนำเพิ่มเติม
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
                    
                    # บันทึกข้อความตอบกลับ
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response, 
                        "recipe": recipe,
                        "nutrition_data": nutrition_data,
                        "similarity_score": similarity
                    })
                else:
                    response = f"""
                    ❌ ไม่พบสูตรอาหารที่ตรงกับ '{search_query}'
                    
                    💡 **คำแนะนำ:**
                    - ลองใช้คำค้นหาที่ง่ายกว่า เช่น "ไข่เจียว" แทน "วิธีทำไข่เจียว"
                    - ตรวจสอบการสะกดคำ
                    - ลองค้นหาด้วยเมนูยอดนิยม เช่น "ผัดกะเพรา", "ต้มยำกุ้ง"
                    - ปรับการตั้งค่าความเคร่งครัดในการค้นหาในแถบด้านซ้าย
                    
                    🍽️ **เมนูยอดนิยม:** ผัดกะเพรา, ต้มยำกุ้ง, ไข่เจียว, ส้มตำ, แกงเขียวหวาน
                    """
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
