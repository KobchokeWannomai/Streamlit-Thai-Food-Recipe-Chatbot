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

# นำเข้าไฟล์ที่สร้างขึ้นใหม่
from nutrition_api import NutritionAPI
from recipe_search import RecipeSearchEngine

# การตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Thai Food Recipe Chatbot with Nutrition",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="collapsed"
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
    }
    
    .vitamin-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .mineral-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .recipe-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .settings-panel {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
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
    if os.path.exists(MODEL_PATH):
        return SentenceTransformer(MODEL_PATH)
    else:
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        os.makedirs(MODEL_PATH, exist_ok=True)
        model.save(MODEL_PATH)
        return model

@st.cache_data
def load_data():
    """โหลดข้อมูลอาหารไทย"""
    return pd.read_csv(DATA_PATH)

@st.cache_data
def get_embeddings(_model, data):
    """สร้างหรือโหลด embeddings สำหรับสูตรอาหาร"""
    if os.path.exists(EMBEDDINGS_PATH):
        with open(EMBEDDINGS_PATH, 'rb') as f:
            return pickle.load(f)
    else:
        # รวมข้อความทั้งหมดของแต่ละสูตร
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

@st.cache_resource
def initialize_nutrition_api():
    """เริ่มต้นระบบข้อมูลโภชนาการ"""
    return NutritionAPI()

@st.cache_resource
def initialize_search_engine(_data, _nutrition_api):
    """เริ่มต้นระบบค้นหา"""
    return RecipeSearchEngine(_data, _nutrition_api)

def format_ingredients(ingredients_text):
    """จัดรูปแบบรายการวัตถุดิบให้แสดงผลดี"""
    ingredients = ingredients_text.split('\n')
    formatted = "<ul>"
    for item in ingredients:
        if item.strip():
            formatted += f"<li>{item.strip()}</li>"
    formatted += "</ul>"
    return formatted

def format_cooking_method(method_text):
    """จัดรูปแบบวิธีทำให้แสดงผลดี"""
    sentences = re.split(r'(?<=[ๆ.]) ', method_text)
    formatted = "<ol>"
    for sentence in sentences:
        if sentence.strip():
            formatted += f"<li>{sentence.strip()}</li>"
    formatted += "</ol>"
    return formatted

def display_nutrition_info(nutrition_data, title="ข้อมูลโภชนาการ"):
    """แสดงข้อมูลโภชนาการในรูปแบบกราฟและตาราง"""
    total_nutrition = nutrition_data['total_nutrition']
    
    # สร้างกราฟโภชนาการหลัก
    col1, col2 = st.columns(2)
    
    with col1:
        # กราฟ Macronutrients
        macro_labels = ['โปรตีน', 'คาร์โบไฮเดรต', 'ไขมัน']
        macro_values = [total_nutrition['protein'], total_nutrition['carbs'], total_nutrition['fat']]
        macro_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        fig_macro = px.pie(values=macro_values, names=macro_labels, 
                          title="สัดส่วนสารอาหารหลัก (กรัม)",
                          color_discrete_sequence=macro_colors)
        fig_macro.update_layout(height=300)
        st.plotly_chart(fig_macro, use_container_width=True)
    
    with col2:
        # แสดงข้อมูลสารอาหารหลักเป็นตัวเลข
        st.markdown(f"""
        <div class="nutrition-card">
            <h4>🔥 แคลอรี่: {total_nutrition['calories']:.1f} kcal</h4>
            <p>🥩 โปรตีน: {total_nutrition['protein']:.1f} g</p>
            <p>🍚 คาร์โบไฮเดรต: {total_nutrition['carbs']:.1f} g</p>
            <p>🧈 ไขมัน: {total_nutrition['fat']:.1f} g</p>
            <p>🌾 ใยอาหาร: {total_nutrition['fiber']:.1f} g</p>
        </div>
        """, unsafe_allow_html=True)
    
    # แสดงวิตามินและแร่ธาตุ
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown(f"""
        <div class="vitamin-card">
            <h4>💊 วิตามิน</h4>
            <p>🥕 วิตามิน A: {total_nutrition['vitamin_a']:.1f} IU</p>
            <p>🍋 วิตามิน C: {total_nutrition['vitamin_c']:.1f} mg</p>
            <p>🌾 วิตามิน B1: {total_nutrition['vitamin_b1']:.2f} mg</p>
            <p>🥛 วิตามิน B2: {total_nutrition['vitamin_b2']:.2f} mg</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="mineral-card">
            <h4>⚡ แร่ธาตุ</h4>
            <p>🦴 แคลเซียม: {total_nutrition['calcium']:.1f} mg</p>
            <p>🩸 เหล็ก: {total_nutrition['iron']:.1f} mg</p>
            <p>🍌 โปแตสเซียม: {total_nutrition['potassium']:.1f} mg</p>
            <p>🧂 โซเดียม: {total_nutrition['sodium']:.1f} mg</p>
        </div>
        """, unsafe_allow_html=True)
    
    # แสดงรายละเอียดวัตถุดิบ
    if st.expander("🔍 รายละเอียดวัตถุดิบแต่ละชนิด"):
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

def display_settings_panel():
    """แสดงแถบการตั้งค่า"""
    st.sidebar.markdown("## ⚙️ การตั้งค่า")
    
    # การตั้งค่า API
    st.sidebar.markdown("### 🌐 API ข้อมูลโภชนาการ")
    
    use_api = st.sidebar.checkbox("เปิดใช้งาน API", value=False, key="use_api")
    
    if use_api:
        api_key = st.sidebar.text_input(
            "USDA FoodData Central API Key",
            type="password",
            help="ใส่ API Key จาก https://fdc.nal.usda.gov/api-guide.html"
        )
        
        if api_key:
            nutrition_api.set_api_key(api_key)
            
            # ตรวจสอบสถานะ API
            if st.sidebar.button("ตรวจสอบการเชื่อมต่อ"):
                status, message = nutrition_api.check_api_status()
                if status:
                    st.sidebar.success(f"✅ {message}")
                else:
                    st.sidebar.error(f"❌ {message}")
    
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
    
    # การตั้งค่าการแสดงผล
    st.sidebar.markdown("### 🎨 การแสดงผล")
    
    show_charts = st.sidebar.checkbox("แสดงกราฟโภชนาการ", value=True, key="show_charts")
    show_details = st.sidebar.checkbox("แสดงรายละเอียดวัตถุดิบ", value=True, key="show_details")
    
    return {
        'use_api': use_api,
        'adjust_consumption': adjust_consumption,
        'enhance_missing': enhance_missing,
        'show_charts': show_charts,
        'show_details': show_details
    }

def search_recipes(query, model, data, embeddings, search_engine, settings, top_k=3):
    """ค้นหาสูตรอาหารด้วยระบบอัจฉริยะ"""
    # ใช้ระบบค้นหาอัจฉริยะก่อน
    smart_results = search_engine.smart_search(
        query, 
        settings['use_api'], 
        settings['adjust_consumption'],
        limit=top_k
    )
    
    if smart_results:
        return [(result['name'], result['similarity'], result['index'], result.get('nutrition')) 
                for result in smart_results]
    
    # หากไม่พบผลลัพธ์จากระบบอัจฉริยะ ใช้ embedding search แบบเดิม
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(-similarities)[:top_k]
    
    results = []
    for idx in top_indices:
        if similarities[idx] > 0.1:  # ลดเกณฑ์เพื่อให้หาได้มากขึ้น
            results.append({
                'name': data.iloc[idx]['name'],
                'similarity': similarities[idx],
                'index': idx,
                'nutrition': None
            })
    
    return [(result['name'], result['similarity'], result['index'], result['nutrition']) 
            for result in results]

def display_recipe_with_nutrition(recipe, nutrition_data, settings):
    """แสดงสูตรอาหารพร้อมข้อมูลโภชนาการ"""
    # แสดงชื่อเมนู
    st.markdown(f"### 🍽️ {recipe['name']}")
    
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
        if nutrition_data and settings['show_charts']:
            display_nutrition_info(nutrition_data, f"ข้อมูลโภชนาการ - {recipe['name']}")
        else:
            st.info("เปิดใช้งานการแสดงกราฟในการตั้งค่าเพื่อดูข้อมูลโภชนาการ")
    
    with tab3:
        if nutrition_data and settings['show_details']:
            # แสดงรายละเอียดเพิ่มเติม
            total_nutrition = nutrition_data['total_nutrition']
            
            # คำแนะนำเกี่ยวกับโภชนาการ
            recommendations = []
            
            if total_nutrition['calories'] < 200:
                recommendations.append("🟢 เมนูแคลอรี่ต่ำ เหมาะสำหรับผู้ควบคุมน้ำหนัก")
            elif total_nutrition['calories'] > 400:
                recommendations.append("🔴 เมนูแคลอรี่สูง เหมาะสำหรับผู้ต้องการพลังงาน")
            
            if total_nutrition['protein'] > 20:
                recommendations.append("💪 โปรตีนสูง เหมาะสำหรับนักกีฬาและผู้สูงอายุ")
            
            if total_nutrition['fiber'] > 5:
                recommendations.append("🌾 ใยอาหารสูง ดีต่อระบบทางเดิน")
            
            if total_nutrition['sodium'] > 1000:
                recommendations.append("⚠️ โซเดียมสูง ผู้ป่วยความดันสูงควรระวัง")
            
            if recommendations:
                st.markdown("#### 💡 คำแนะนำ")
                for rec in recommendations:
                    st.markdown(f"- {rec}")
            
            # แสดงเปรียบเทียบกับความต้องการประจำวัน (สำหรับผู้ใหญ่)
            daily_needs = {
                'calories': 2000, 'protein': 50, 'fat': 65, 'carbs': 300,
                'fiber': 25, 'calcium': 1000, 'iron': 18, 'vitamin_c': 90
            }
            
            st.markdown("#### 📈 เปรียบเทียบกับความต้องการประจำวัน")
            comparison_data = []
            
            for nutrient, daily_need in daily_needs.items():
                if nutrient in total_nutrition:
                    percentage = (total_nutrition[nutrient] / daily_need) * 100
                    comparison_data.append({
                        'สารอาหาร': nutrient,
                        'ปริมาณในเมนู': f"{total_nutrition[nutrient]:.1f}",
                        '% ความต้องการประจำวัน': f"{percentage:.1f}%"
                    })
            
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True)

def main():
    # เริ่มต้นระบบ
    model = load_model()
    data = load_data()
    embeddings = get_embeddings(model, data)
    
    # แถบการตั้งค่า
    settings = display_settings_panel()
    
    # เริ่มต้น API และระบบค้นหา
    global nutrition_api
    nutrition_api = initialize_nutrition_api()
    search_engine = initialize_search_engine(data, nutrition_api)
    
    # หัวข้อแอป
    st.title("🍲 Thai Food Recipe Chatbot")
    st.markdown("### 🥘 ระบบแนะนำสูตรอาหารไทยพร้อมข้อมูลโภชนาการ")
    st.write("ถามเกี่ยวกับสูตรอาหารไทย หรือค้นหาตามคุณค่าทางโภชนาการ!")
    
    # แสดงสถิติเบื้องต้น
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📖 จำนวนสูตร", len(data))
    with col2:
        st.metric("🔥 ใช้ AI", "✅")
    with col3:
        api_status = "🟢 เชื่อมต่อ" if settings['use_api'] else "🔴 ปิดใช้งาน"
        st.metric("🌐 API", api_status)
    with col4:
        nutrition_status = "🟢 เปิด" if settings['adjust_consumption'] else "🟡 ปกติ"
        st.metric("🧮 โภชนาการ", nutrition_status)
    
    # ตัวอย่างคำค้นหา
    st.markdown("#### 💡 ตัวอย่างการค้นหา:")
    example_queries = [
        "แนะนำอาหารแคลอรี่ต่ำ",
        "เมนูที่มีโปรตีนสูง", 
        "อาหารทอดกรอบ",
        "ไข่เจียว",
        "แกงเผ็ด"
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
                display_recipe_with_nutrition(recipe, nutrition_data, settings)
            else:
                st.markdown(message["content"])
    
    # ช่องค้นหา
    search_query = st.session_state.get('search_query', '')
    if prompt := st.chat_input("ค้นหาสูตรอาหาร หรือถามเกี่ยวกับโภชนาการ...", key="main_chat"):
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
            with st.spinner("กำลังค้นหาและคำนวณข้อมูลโภชนาการ..."):
                # ค้นหาสูตรอาหาร
                results = search_recipes(
                    search_query, model, data, embeddings, search_engine, settings
                )
                
                if results:
                    best_match = results[0]
                    recipe_name, similarity, recipe_idx, cached_nutrition = best_match
                    
                    if similarity > 0.3:  # ลดเกณฑ์ให้หาได้ง่ายขึ้น
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
                                settings['adjust_consumption']
                            )
                        
                        response = f"🎯 พบสูตรอาหารที่ตรงกับการค้นหา: **{recipe_name}**"
                        st.markdown(response)
                        
                        # แสดงสูตรและโภชนาการ
                        display_recipe_with_nutrition(recipe, nutrition_data, settings)
                        
                        # เพิ่มการแนะนำเพิ่มเติม
                        if len(results) > 1:
                            st.markdown("#### 🔍 เมนูอื่นที่น่าสนใจ:")
                            other_results = results[1:3]  # แสดง 2 เมนูถัดมา
                            
                            cols = st.columns(len(other_results))
                            for i, (other_name, other_sim, other_idx, _) in enumerate(other_results):
                                with cols[i]:
                                    if st.button(f"🍽️ {other_name}", key=f"other_{i}"):
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
                        ❌ ไม่พบสูตรอาหารที่ตรงกับ '{search_query}' 
                        
                        💡 **คำแนะนำ:**
                        - ลองใช้คำค้นหาที่ง่ายกว่า เช่น "ไข่เจียว" แทน "วิธีทำไข่เจียว"
                        - ค้นหาตามโภชนาการ เช่น "อาหารแคลอรี่ต่ำ" 
                        - ค้นหาตามประเภท เช่น "อาหารทอด" "อาหารต้ม"
                        """
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    response = """
                    🤔 ไม่พบสูตรอาหารที่ตรงกับคำค้นหา
                    
                    ลองค้นหาด้วยคำเหล่านี้:
                    - ชื่อเมนูอาหาร: "ผัดไทย", "ต้มยำกุ้ง"
                    - ประเภทอาหาร: "อาหารทอด", "แกง", "ยำ"
                    - โภชนาการ: "แคลอรี่ต่ำ", "โปรตีนสูง"
                    """
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
