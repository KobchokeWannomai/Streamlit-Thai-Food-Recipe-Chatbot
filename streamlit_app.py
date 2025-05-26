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
from nutrition_analyzer import NutritionAnalyzer

# การตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="แชทบอทสูตรอาหารไทย",
    page_icon="🍲",
    layout="wide"
)

# ตั้งค่าฟอนต์ไทย
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@400;700&display=swap');
    html, body, [class*="st-"] {
        font-family: 'Sarabun', sans-serif !important;
    }
    .nutrition-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .ingredient-highlight {
        background-color: #e8f4fd;
        padding: 0.2rem;
        border-radius: 0.3rem;
        margin: 0.1rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# เส้นทางไฟล์
DATA_PATH = "thai_food_processed.csv"
NUTRITION_DATA_PATH = "thai_food_with_nutrition.csv"
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
    """โหลดข้อมูลสูตรอาหารไทย"""
    # ลองโหลดไฟล์ที่มีข้อมูลคุณค่าทางโภชนาการก่อน
    if os.path.exists(NUTRITION_DATA_PATH):
        return pd.read_csv(NUTRITION_DATA_PATH)
    else:
        return pd.read_csv(DATA_PATH)

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
            combined_text = f"{row['name']} {row['ingredient']} {row['method']}"
            texts.append(combined_text)
        
        # สร้าง embeddings
        embeddings = _model.encode(texts)
        
        # บันทึก embeddings
        with open(EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump(embeddings, f)
        
        return embeddings

def format_ingredients(ingredients_text):
    """จัดรูปแบบรายการวัตถุดิบให้แสดงผลดีขึ้น"""
    if pd.isna(ingredients_text):
        return "ไม่มีข้อมูลวัตถุดิบ"
    
    ingredients = ingredients_text.split('\n')
    formatted = "<ul>"
    for item in ingredients:
        if item.strip():
            # เน้นชื่อวัตถุดิบด้วย highlight
            highlighted_item = f'<span class="ingredient-highlight">{item.strip()}</span>'
            formatted += f"<li>{highlighted_item}</li>"
    formatted += "</ul>"
    return formatted

def format_cooking_method(method_text):
    """จัดรูปแบบวิธีทำอาหารให้แสดงผลดีขึ้น"""
    if pd.isna(method_text):
        return "ไม่มีข้อมูลวิธีทำ"
    
    # แยกประโยคตามจุด หรือ space ใน cooking instructions ภาษาไทย
    sentences = re.split(r'(?<=[ๆ.]) ', method_text)
    formatted = "<ol>"
    for sentence in sentences:
        if sentence.strip():
            formatted += f"<li>{sentence.strip()}</li>"
    formatted += "</ol>"
    return formatted

def display_nutrition_info(row):
    """แสดงข้อมูลคุณค่าทางโภชนาการ"""
    nutrition_cols = ['calories', 'protein', 'carbohydrates', 'fat', 'fiber', 
                     'calcium', 'iron', 'vitamin_a', 'vitamin_c', 'vitamin_d']
    
    # ตรวจสอบว่ามีข้อมูลคุณค่าทางโภชนาการหรือไม่
    has_nutrition = any(col in row.index and pd.notna(row.get(col, 0)) and row.get(col, 0) > 0 
                       for col in nutrition_cols)
    
    if not has_nutrition:
        st.info("💡 ข้อมูลคุณค่าทางโภชนาการยังไม่พร้อมใช้งาน กรุณารันระบบวิเคราะห์คุณค่าทางโภชนาการ")
        return
    
    st.markdown("#### 🍎 ข้อมูลคุณค่าทางโภชนาการ (ต่อ 100 กรัม)")
    
    # แสดงข้อมูลในรูปแบบ columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="nutrition-card">
        <strong>🔥 พลังงาน:</strong> {row.get('calories', 0):.1f} แคลอรี่<br>
        <strong>🥩 โปรตีน:</strong> {row.get('protein', 0):.1f} กรัม<br>
        <strong>🍞 คาร์โบไฮเดรต:</strong> {row.get('carbohydrates', 0):.1f} กรัม<br>
        <strong>🧈 ไขมัน:</strong> {row.get('fat', 0):.1f} กรัม
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="nutrition-card">
        <strong>🦴 แคลเซียม:</strong> {row.get('calcium', 0):.1f} มิลลิกรัม<br>
        <strong>🩸 เหล็ก:</strong> {row.get('iron', 0):.1f} มิลลิกรัม<br>
        <strong>🥕 วิตามิน A:</strong> {row.get('vitamin_a', 0):.1f} ไมโครกรัม<br>
        <strong>🍊 วิตามิน C:</strong> {row.get('vitamin_c', 0):.1f} มิลลิกรัม
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # สร้างกราฟแสดงสัดส่วนสารอาหารหลัก
        if row.get('protein', 0) > 0 or row.get('carbohydrates', 0) > 0 or row.get('fat', 0) > 0:
            fig = go.Figure(data=[go.Pie(
                labels=['โปรตีน', 'คาร์โบไฮเดรท', 'ไขมัน'],
                values=[row.get('protein', 0), row.get('carbohydrates', 0), row.get('fat', 0)],
                hole=0.3
            )])
            fig.update_layout(
                title="สัดส่วนสารอาหารหลัก",
                height=250,
                margin=dict(t=50, b=0, l=0, r=0)
            )
            st.plotly_chart(fig, use_container_width=True)

def search_recipes(query, model, data, embeddings, top_k=3, filters=None):
    """ค้นหาสูตรอาหารตามคำค้นหาและตัวกรอง"""
    # เข้ารหัสคำค้นหา
    query_embedding = model.encode([query])
    
    # คำนวณความคล้ายคลึง
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # ใช้ตัวกรองถ้ามี
    filtered_indices = list(range(len(data)))
    
    if filters:
        # กรองตามคุณค่าทางโภชนาการ
        if filters.get('max_calories'):
            filtered_indices = [i for i in filtered_indices 
                             if data.iloc[i].get('calories', 0) <= filters['max_calories']]
        
        if filters.get('min_protein'):
            filtered_indices = [i for i in filtered_indices 
                             if data.iloc[i].get('protein', 0) >= filters['min_protein']]
        
        if filters.get('max_fat'):
            filtered_indices = [i for i in filtered_indices 
                             if data.iloc[i].get('fat', 0) <= filters['max_fat']]
        
        if filters.get('high_vitamin_c'):
            filtered_indices = [i for i in filtered_indices 
                             if data.iloc[i].get('vitamin_c', 0) >= 10]
    
    # เลือกผลลัพธ์ที่ดีที่สุดจากรายการที่ผ่านการกรอง
    if filtered_indices:
        filtered_similarities = [(i, similarities[i]) for i in filtered_indices]
        filtered_similarities.sort(key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in filtered_similarities[:top_k]]
    else:
        top_indices = np.argsort(-similarities)[:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'name': data.iloc[idx]['name'],
            'similarity': similarities[idx],
            'ingredients': data.iloc[idx]['ingredient'],
            'method': data.iloc[idx]['method'],
            'row_data': data.iloc[idx]
        })
    
    return results

def create_nutrition_analyzer():
    """สร้างตัววิเคราะห์คุณค่าทางโภชนาการ"""
    # ใช้ API key จาก environment variable หรือให้ผู้ใช้ใส่
    api_key = os.getenv('USDA_API_KEY')
    return NutritionAnalyzer(api_key=api_key)

def main():
    # โหลดโมเดลและข้อมูล
    model = load_model()
    data = load_data()
    embeddings = get_embeddings(model, data)
    
    # หัวข้อหลักของแอป
    st.title("🍲 แชทบอทสูตรอาหารไทย")
    st.write("ถามเกี่ยวกับวิธีทำอาหารไทยได้เลย! รองรับการค้นหาตามคุณค่าทางโภชนาการ")
    
    # แถบข้างสำหรับตัวกรองและตั้งค่า
    with st.sidebar:
        st.header("🎛️ ตัวกรองการค้นหา")
        
        # ตัวกรองคุณค่าทางโภชนาการ
        st.subheader("คุณค่าทางโภชนาการ")
        max_calories = st.slider("พลังงานสูงสุด (แคลอรี่)", 0, 1000, 1000)
        min_protein = st.slider("โปรตีนต่ำสุด (กรัม)", 0, 50, 0)
        max_fat = st.slider("ไขมันสูงสุด (กรัม)", 0, 100, 100)
        high_vitamin_c = st.checkbox("อุดมด้วยวิตามิน C")
        
        # ปุ่มจัดการข้อมูลคุณค่าทางโภชนาการ
        st.subheader("⚙️ จัดการข้อมูล")
        if st.button("🔍 วิเคราะห์คุณค่าทางโภชนาการ"):
            with st.spinner("กำลังวิเคราะห์คุณค่าทางโภชนาการ..."):
                analyzer = create_nutrition_analyzer()
                if analyzer.api_key:
                    analyzer.process_csv_file(DATA_PATH, NUTRITION_DATA_PATH)
                    st.success("วิเคราะห์เสร็จสิ้น! กรุณาโหลดหน้าใหม่เพื่อดูผลลัพธ์")
                    st.experimental_rerun()
                else:
                    st.error("กรุณาตั้งค่า USDA_API_KEY ใน environment variables")
        
        # สถิติข้อมูล
        st.subheader("📊 สถิติข้อมูล")
        st.metric("จำนวนสูตรอาหาร", len(data))
        
        if 'calories' in data.columns:
            avg_calories = data['calories'].mean()
            st.metric("พลังงานเฉลี่ย", f"{avg_calories:.1f} แคลอรี่")
    
    # สร้างตัวกรอง
    filters = {
        'max_calories': max_calories if max_calories < 1000 else None,
        'min_protein': min_protein if min_protein > 0 else None,
        'max_fat': max_fat if max_fat < 100 else None,
        'high_vitamin_c': high_vitamin_c
    }
    
    # เริ่มต้นประวัติการสนทนา
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # แสดงประวัติการสนทนา
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "recipe" in message:
                # แสดงสูตรอาหาร
                recipe = message["recipe"]
                st.markdown(f"### {recipe['name']}")
                
                # แสดงวัตถุดิบ
                st.markdown("#### 🥘 วัตถุดิบ")
                st.markdown(format_ingredients(recipe["ingredients"]), unsafe_allow_html=True)
                
                # แสดงวิธีทำ
                st.markdown("#### 👩‍🍳 วิธีทำ")
                st.markdown(format_cooking_method(recipe["method"]), unsafe_allow_html=True)
                
                # แสดงข้อมูลคุณค่าทางโภชนาการ
                if 'row_data' in recipe:
                    display_nutrition_info(recipe['row_data'])
                
                st.markdown(f"*ความเกี่ยวข้อง: {recipe['similarity']:.2f}*")
            else:
                # แสดงข้อความทั่วไป
                st.markdown(message["content"])
    
    # ช่องกรอกข้อความสำหรับสนทนา
    if prompt := st.chat_input("ถามเกี่ยวกับอาหารไทย... (เช่น อาหารที่มีโปรตีนสูง, อาหารแคลอรี่ต่ำ)"):
        # เพิ่มข้อความของผู้ใช้ในประวัติ
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # แสดงข้อความของผู้ใช้
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # ให้คำตอบ
        with st.chat_message("assistant"):
            with st.spinner("กำลังค้นหา..."):
                results = search_recipes(prompt, model, data, embeddings, filters=filters)
                
                if results:
                    best_match = results[0]
                    
                    # ตรวจสอบว่าผลลัพธ์ดีพอหรือไม่
                    if best_match["similarity"] > 0.3:
                        response = f"ฉันพบสูตรอาหารที่คุณต้องการ: {best_match['name']}"
                        st.markdown(response)
                        
                        # แสดงสูตรอาหาร
                        st.markdown(f"### {best_match['name']}")
                        
                        # แสดงวัตถุดิบ
                        st.markdown("#### 🥘 วัตถุดิบ")
                        st.markdown(format_ingredients(best_match["ingredients"]), unsafe_allow_html=True)
                        
                        # แสดงวิธีทำ
                        st.markdown("#### 👩‍🍳 วิธีทำ")
                        st.markdown(format_cooking_method(best_match["method"]), unsafe_allow_html=True)
                        
                        # แสดงข้อมูลคุณค่าทางโภชนาการ
                        display_nutrition_info(best_match['row_data'])
                        
                        st.markdown(f"*ความเกี่ยวข้อง: {best_match['similarity']:.2f}*")
                        
                        # เพิ่มคำตอบในประวัติ
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response, 
                            "recipe": best_match
                        })
                        
                        # แสดงสูตรอื่นๆ ที่เกี่ยวข้อง
                        if len(results) > 1:
                            with st.expander("🔍 สูตรอื่นๆ ที่เกี่ยวข้อง"):
                                for i, result in enumerate(results[1:], 1):
                                    st.write(f"**{i+1}. {result['name']}** (ความคล้ายคลึง: {result['similarity']:.2f})")
                    
                    else:
                        response = "ขออภัย ฉันไม่พบสูตรอาหารที่ตรงกับคำค้นหาของคุณ กรุณาลองใช้คำค้นหาอื่น หรือปรับตัวกรองการค้นหา"
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        # แสดงคำแนะนำ
                        st.info("💡 คำแนะนำ: ลองค้นหาด้วยคำเช่น 'ต้มยำ', 'แกงเขียวหวาน', 'ผัดไทย', 'อาหารโปรตีนสูง', 'อาหารแคลอรี่ต่ำ'")
                
                else:
                    response = "ขออภัย ฉันไม่สามารถค้นหาสูตรอาหารได้ในขณะนี้ กรุณาลองใหม่อีกครั้ง"
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # แสดงตัวอย่างคำค้นหา
    st.markdown("---")
    st.markdown("### 💭 ตัวอย่างคำถามที่น่าสนใจ")
    
    example_queries = [
        "อาหารที่มีโปรตีนสูง",
        "เมนูแคลอรี่ต่ำ",
        "อาหารอุดมไปด้วยวิตามิน C",
        "แกงที่ทำง่าย",
        "อาหารจานเดียว",
        "ขนมไทยโบราณ"
    ]
    
    cols = st.columns(3)
    for i, query in enumerate(example_queries):
        with cols[i % 3]:
            if st.button(query, key=f"example_{i}"):
                # เพิ่มคำค้นหาตัวอย่างในช่องแชท
                st.rerun()

if __name__ == "__main__":
    main()
