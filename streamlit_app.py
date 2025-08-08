import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import pickle
import re
from nutrition_analyzer import ThaiNutritionAnalyzer

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Thai Food Recipe Chatbot",
    page_icon="🍲",
    layout="wide"
)

# ตั้งค่าฟอนต์ภาษาไทย
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@400;700&display=swap');
    html, body, [class*="st-"] {
        font-family: 'Sarabun', sans-serif !important;
    }
</style>
""", unsafe_allow_html=True)

# เส้นทางไฟล์
DATA_PATH = "thai_food_processed.csv"
EMBEDDINGS_PATH = "embeddings.pkl"
MODEL_PATH = "model"
NUTRITION_EMBEDDINGS_PATH = "nutrition_embeddings.pkl"

# สร้างอินสแตนซ์ของ NutritionAnalyzer
nutrition_analyzer = ThaiNutritionAnalyzer()

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
    """โหลดชุดข้อมูลอาหารไทย"""
    df = pd.read_csv(DATA_PATH)
    # เพิ่มคอลัมน์คุณค่าทางโภชนาการถ้ายังไม่มี
    if 'nutrition_info' not in df.columns:
        df['nutrition_info'] = ''
        # วิเคราะห์คุณค่าทางโภชนาการสำหรับแต่ละเมนู
        for idx, row in df.iterrows():
            nutrition_data = nutrition_analyzer.analyze_recipe_nutrition(row['ingredient'])
            df.at[idx, 'nutrition_info'] = nutrition_analyzer.get_nutrition_summary(nutrition_data)
        # บันทึกกลับไปยังไฟล์ CSV
        df.to_csv(DATA_PATH, index=False)
    return df

@st.cache_data
def get_embeddings(_model, data):
    """สร้างหรือโหลด embeddings สำหรับสูตรอาหารทั้งหมด"""
    # ตรวจสอบ embeddings ที่มีคุณค่าทางโภชนาการ
    if os.path.exists(NUTRITION_EMBEDDINGS_PATH):
        with open(NUTRITION_EMBEDDINGS_PATH, 'rb') as f:
            return pickle.load(f)
    else:
        # รวมข้อความทั้งหมดสำหรับแต่ละสูตรอาหาร (รวมคุณค่าทางโภชนาการ)
        texts = []
        for _, row in data.iterrows():
            combined_text = f"{row['name']} {row['ingredient']} {row['method']} {row.get('nutrition_info', '')}"
            texts.append(combined_text)
        
        # สร้าง embeddings
        embeddings = _model.encode(texts)
        
        # บันทึก embeddings
        with open(NUTRITION_EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump(embeddings, f)
        
        return embeddings

def format_ingredients(ingredients_text):
    """จัดรูปแบบรายการวัตถุดิบให้แสดงผลสวยงาม"""
    ingredients = ingredients_text.split('\n')
    formatted = "<ul>"
    for item in ingredients:
        if item.strip():
            formatted += f"<li>{item.strip()}</li>"
    formatted += "</ul>"
    return formatted

def format_cooking_method(method_text):
    """จัดรูปแบบวิธีทำอาหารให้แสดงผลสวยงาม"""
    # แบ่งตามประโยค (ภาษาไทยมักใช้ช่องว่างเป็นตัวแบ่งประโยคในวิธีทำอาหาร)
    sentences = re.split(r'(?<=[ๆ.]) ', method_text)
    formatted = "<ol>"
    for sentence in sentences:
        if sentence.strip():
            formatted += f"<li>{sentence.strip()}</li>"
    formatted += "</ol>"
    return formatted

def search_recipes(query, model, data, embeddings, top_k=3, nutrition_filter=None):
    """ค้นหาสูตรอาหารตามคำค้นหา พร้อมกรองตามคุณค่าทางโภชนาการ"""
    # เข้ารหัสคำค้นหา
    query_embedding = model.encode([query])
    
    # คำนวณความคล้ายคลึง
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # กรองตามคุณค่าทางโภชนาการถ้ามี
    if nutrition_filter:
        filtered_indices = []
        for idx, row in data.iterrows():
            nutrition_data = nutrition_analyzer.analyze_recipe_nutrition(row['ingredient'])
            total_nutrition = nutrition_data.get('total_nutrition', {})
            
            # ตรวจสอบเงื่อนไขการกรอง
            match = True
            if 'max_calories' in nutrition_filter and 'พลังงาน' in total_nutrition:
                if total_nutrition['พลังงาน'] > nutrition_filter['max_calories']:
                    match = False
            if 'min_protein' in nutrition_filter and 'โปรตีน' in total_nutrition:
                if total_nutrition['โปรตีน'] < nutrition_filter['min_protein']:
                    match = False
            if 'max_fat' in nutrition_filter and 'ไขมัน' in total_nutrition:
                if total_nutrition['ไขมัน'] > nutrition_filter['max_fat']:
                    match = False
                    
            if match:
                filtered_indices.append(idx)
        
        # กรองผลลัพธ์
        if filtered_indices:
            filtered_similarities = np.zeros_like(similarities)
            for idx in filtered_indices:
                filtered_similarities[idx] = similarities[idx]
            similarities = filtered_similarities
    
    # หาผลลัพธ์ที่ดีที่สุด
    top_indices = np.argsort(-similarities)[:top_k]
    results = []
    
    for idx in top_indices:
        if similarities[idx] > 0:
            results.append({
                'name': data.iloc[idx]['name'],
                'similarity': similarities[idx],
                'ingredients': data.iloc[idx]['ingredient'],
                'method': data.iloc[idx]['method'],
                'nutrition_info': data.iloc[idx].get('nutrition_info', '')
            })
    
    return results

def main():
    # โหลดโมเดลและข้อมูล
    model = load_model()
    data = load_data()
    embeddings = get_embeddings(model, data)
    
    # หัวข้อแอป
    st.title("🍲 Thai Food Recipe Chatbot")
    st.write("ถามเกี่ยวกับวิธีทำอาหารไทยได้เลย! Ask about Thai food recipes!")
    
    # แท็บสำหรับฟีเจอร์ต่างๆ
    tab1, tab2, tab3 = st.tabs(["🔍 ค้นหาสูตรอาหาร", "🥗 ค้นหาตามคุณค่าทางโภชนาการ", "📊 วิเคราะห์วัตถุดิบ"])
    
    with tab1:
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
                    st.markdown("#### วัตถุดิบ (Ingredients)")
                    st.markdown(format_ingredients(recipe["ingredients"]), unsafe_allow_html=True)
                    st.markdown("#### วิธีทำ (Method)")
                    st.markdown(format_cooking_method(recipe["method"]), unsafe_allow_html=True)
                    if recipe.get('nutrition_info'):
                        st.markdown("#### คุณค่าทางโภชนาการ (Nutritional Information)")
                        st.markdown(recipe['nutrition_info'])
                    st.markdown(f"*ความเกี่ยวข้อง (Relevance): {recipe['similarity']:.2f}*")
                else:
                    # แสดงข้อความปกติ
                    st.markdown(message["content"])
        
        # ช่องพิมพ์ข้อความ
        if prompt := st.chat_input("ถามเกี่ยวกับอาหารไทย..."):
            # เพิ่มข้อความผู้ใช้ในประวัติการสนทนา
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # แสดงข้อความผู้ใช้
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # รับคำตอบ
            with st.chat_message("assistant"):
                with st.spinner("กำลังค้นหา..."):
                    results = search_recipes(prompt, model, data, embeddings)
                    
                    if results:
                        best_match = results[0]
                        
                        # ตรวจสอบว่ามีผลลัพธ์ที่ดี
                        if best_match["similarity"] > 0.3:
                            response = f"ฉันพบสูตรอาหารที่คุณต้องการ: {best_match['name']}"
                            st.markdown(response)
                            
                            # แสดงสูตรอาหาร
                            st.markdown(f"### {best_match['name']}")
                            st.markdown("#### วัตถุดิบ (Ingredients)")
                            st.markdown(format_ingredients(best_match["ingredients"]), unsafe_allow_html=True)
                            st.markdown("#### วิธีทำ (Method)")
                            st.markdown(format_cooking_method(best_match["method"]), unsafe_allow_html=True)
                            if best_match.get('nutrition_info'):
                                st.markdown("#### คุณค่าทางโภชนาการ (Nutritional Information)")
                                st.markdown(best_match['nutrition_info'])
                            st.markdown(f"*ความเกี่ยวข้อง (Relevance): {best_match['similarity']:.2f}*")
                            
                            # เพิ่มคำตอบของผู้ช่วยในประวัติการสนทนาพร้อมข้อมูลสูตรอาหาร
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response, 
                                "recipe": best_match
                            })
                        else:
                            response = "ขออภัย ฉันไม่พบสูตรอาหารที่ตรงกับคำถามของคุณ กรุณาลองถามใหม่อีกครั้ง"
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        response = "ขออภัย ฉันไม่สามารถค้นหาสูตรอาหารได้ในขณะนี้ กรุณาลองใหม่อีกครั้ง"
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
    
    with tab2:
        st.header("🥗 ค้นหาอาหารตามคุณค่าทางโภชนาการ")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("ตั้งค่าเงื่อนไขการค้นหา")
            max_calories = st.number_input("พลังงานสูงสุด (แคลอรี)", min_value=0, value=500, step=50)
            min_protein = st.number_input("โปรตีนขั้นต่ำ (กรัม)", min_value=0, value=10, step=5)
            max_fat = st.number_input("ไขมันสูงสุด (กรัม)", min_value=0, value=30, step=5)
            
            nutrition_filter = {
                'max_calories': max_calories,
                'min_protein': min_protein,
                'max_fat': max_fat
            }
        
        with col2:
            st.subheader("คำค้นหา")
            nutrition_query = st.text_input("ค้นหาอาหาร (เช่น ต้มยำ, ผัดไทย)")
            
            if st.button("🔍 ค้นหา"):
                if nutrition_query:
                    with st.spinner("กำลังค้นหา..."):
                        results = search_recipes(nutrition_query, model, data, embeddings, 
                                               top_k=5, nutrition_filter=nutrition_filter)
                        
                        if results:
                            st.success(f"พบ {len(results)} เมนูที่ตรงกับเงื่อนไข")
                            
                            for result in results:
                                with st.expander(f"📍 {result['name']} (ความเกี่ยวข้อง: {result['similarity']:.2f})"):
                                    col1, col2 = st.columns([2, 1])
                                    
                                    with col1:
                                        st.markdown("**วัตถุดิบ:**")
                                        st.markdown(format_ingredients(result["ingredients"]), unsafe_allow_html=True)
                                        
                                    with col2:
                                        if result.get('nutrition_info'):
                                            st.markdown("**คุณค่าทางโภชนาการ:**")
                                            st.markdown(result['nutrition_info'])
                        else:
                            st.warning("ไม่พบเมนูที่ตรงกับเงื่อนไขที่กำหนด")
    
    with tab3:
        st.header("📊 วิเคราะห์คุณค่าทางโภชนาการของวัตถุดิบ")
        
        # ให้ผู้ใช้ป้อนรายการวัตถุดิบ
        st.subheader("ป้อนรายการวัตถุดิบ")
        ingredients_input = st.text_area(
            "ใส่รายการวัตถุดิบ (บรรทัดละ 1 รายการ)",
            height=200,
            placeholder="ตัวอย่าง:\n- กุ้งนาง 500 กรัม\n- หมูสับ 200 กรัม\n- ไข่ไก่ 2 ฟอง"
        )
        
        if st.button("วิเคราะห์คุณค่าทางโภชนาการ"):
            if ingredients_input:
                with st.spinner("กำลังวิเคราะห์..."):
                    # วิเคราะห์คุณค่าทางโภชนาการ
                    nutrition_data = nutrition_analyzer.analyze_recipe_nutrition(ingredients_input)
                    
                    # แสดงผลลัพธ์
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.subheader("📋 สรุปคุณค่าทางโภชนาการรวม")
                        summary = nutrition_analyzer.get_nutrition_summary(nutrition_data)
                        st.markdown(summary)
                    
                    with col2:
                        st.subheader("🥘 รายละเอียดแต่ละวัตถุดิบ")
                        for ingredient in nutrition_data.get('ingredients', []):
                            with st.expander(f"{ingredient['name']} ({ingredient['quantity']} {ingredient['unit']})"):
                                nutrition = ingredient['nutrition']
                                if nutrition:
                                    for nutrient, value in nutrition.items():
                                        st.write(f"- {nutrient}: {value}")
                                else:
                                    st.write("ไม่มีข้อมูลคุณค่าทางโภชนาการ")
            else:
                st.warning("กรุณาใส่รายการวัตถุดิบ")
        
        # ส่วนสำหรับเพิ่มข้อมูลวัตถุดิบใหม่
        st.divider()
        st.subheader("➕ เพิ่มข้อมูลคุณค่าทางโภชนาการของวัตถุดิบใหม่")
        
        new_ingredient_name = st.text_input("ชื่อวัตถุดิบ")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**สารอาหารหลัก (ต่อ 100 กรัม)**")
            energy = st.number_input("พลังงาน (แคลอรี)", min_value=0.0, step=0.1)
            protein = st.number_input("โปรตีน (กรัม)", min_value=0.0, step=0.1)
            fat = st.number_input("ไขมัน (กรัม)", min_value=0.0, step=0.1)
            carb = st.number_input("คาร์โบไฮเดรต (กรัม)", min_value=0.0, step=0.1)
        
        with col2:
            st.write("**แร่ธาตุ (มิลลิกรัม)**")
            calcium = st.number_input("แคลเซียม", min_value=0.0, step=0.1)
            phosphorus = st.number_input("ฟอสฟอรัส", min_value=0.0, step=0.1)
            iron = st.number_input("เหล็ก", min_value=0.0, step=0.1)
            sodium = st.number_input("โซเดียม", min_value=0.0, step=0.1)
        
        with col3:
            st.write("**วิตามิน**")
            vit_a = st.number_input("วิตามินเอ (IU)", min_value=0.0, step=0.1)
            vit_b1 = st.number_input("วิตามินบี1 (มก.)", min_value=0.0, step=0.1)
            vit_b2 = st.number_input("วิตามินบี2 (มก.)", min_value=0.0, step=0.1)
            vit_c = st.number_input("วิตามินซี (มก.)", min_value=0.0, step=0.1)
        
        if st.button("บันทึกข้อมูลวัตถุดิบใหม่"):
            if new_ingredient_name:
                new_nutrition = {
                    'พลังงาน': energy,
                    'โปรตีน': protein,
                    'ไขมัน': fat,
                    'คาร์โบไฮเดรต': carb,
                    'แคลเซียม': calcium,
                    'ฟอสฟอรัส': phosphorus,
                    'เหล็ก': iron,
                    'โซเดียม': sodium,
                    'วิตามินเอ': vit_a,
                    'วิตามินบี1': vit_b1,
                    'วิตามินบี2': vit_b2,
                    'วิตามินซี': vit_c
                }
                
                nutrition_analyzer.add_new_ingredient_nutrition(new_ingredient_name, new_nutrition)
                st.success(f"✅ บันทึกข้อมูลคุณค่าทางโภชนาการของ '{new_ingredient_name}' เรียบร้อยแล้ว")
                
                # ลบ embeddings เก่าเพื่อให้สร้างใหม่
                if os.path.exists(NUTRITION_EMBEDDINGS_PATH):
                    os.remove(NUTRITION_EMBEDDINGS_PATH)
                st.info("ระบบจะอัพเดท embeddings ใหม่ในครั้งถัดไปที่รีสตาร์ทแอป")
            else:
                st.error("กรุณาใส่ชื่อวัตถุดิบ")

if __name__ == "__main__":
    main()
