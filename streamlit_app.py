import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import pickle
import re
import json
from nutrition_analyzer import NutritionAnalyzer

# การกำหนดค่าหน้าเว็บ
st.set_page_config(
    page_title="Thai Food Recipe Chatbot with Nutrition",
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
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin: 10px 0;
    }
    .nutrition-item {
        display: inline-block;
        margin: 5px 10px;
        padding: 5px 10px;
        background-color: #e8f5e8;
        border-radius: 15px;
        font-size: 0.9em;
    }
    .vitamin-mineral {
        font-size: 0.8em;
        color: #666;
        margin-top: 5px;
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
    formatted = "<ul>"
    for item in ingredients:
        if item.strip():
            formatted += f"<li>{item.strip()}</li>"
    formatted += "</ul>"
    return formatted

def format_cooking_method(method_text):
    """จัดรูปแบบวิธีการทำอาหารเพื่อการแสดงผลที่ดีขึ้น"""
    sentences = re.split(r'(?<=[ๆ.]) ', method_text)
    formatted = "<ol>"
    for sentence in sentences:
        if sentence.strip():
            formatted += f"<li>{sentence.strip()}</li>"
    formatted += "</ol>"
    return formatted

def display_nutrition_info(nutrition_data):
    """แสดงข้อมูลโภชนาการในรูปแบบที่สวยงาม"""
    if not nutrition_data:
        return
    
    total_nutrition = nutrition_data.get('total_nutrition', {})
    
    st.markdown('<div class="nutrition-card">', unsafe_allow_html=True)
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
    
    # วิตามินและแร่ธาตุ
    vitamins = total_nutrition.get('vitamins', {})
    minerals = total_nutrition.get('minerals', {})
    
    if vitamins or minerals:
        st.markdown("**วิตามินและแร่ธาตุ:**")
        
        nutrition_items = []
        for vitamin, amount in vitamins.items():
            if amount > 0:
                nutrition_items.append(f"<span class='nutrition-item'>{vitamin}: {amount:.1f}</span>")
        
        for mineral, amount in minerals.items():
            if amount > 0:
                nutrition_items.append(f"<span class='nutrition-item'>{mineral}: {amount:.1f}</span>")
        
        if nutrition_items:
            st.markdown(f"<div class='vitamin-mineral'>{''.join(nutrition_items)}</div>", unsafe_allow_html=True)
    
    # รายละเอียดวัตถุดิบ
    with st.expander("รายละเอียดโภชนาการแต่ละวัตถุดิบ"):
        for ingredient_info in nutrition_data.get('ingredients', []):
            ingredient = ingredient_info['ingredient']
            nutrition = ingredient_info['nutrition']
            
            st.write(f"**{ingredient}** (ต่อ 100g)")
            st.write(f"- แคลอรี่: {nutrition.calories:.0f} kcal")
            st.write(f"- โปรตีน: {nutrition.protein:.1f} g")
            st.write(f"- คาร์โบไฮเดรต: {nutrition.carbs:.1f} g")
            st.write(f"- ไขมัน: {nutrition.fat:.1f} g")
            st.divider()
    
    st.markdown('</div>', unsafe_allow_html=True)

def search_recipes(query, model, data, embeddings, nutrition_analyzer, top_k=3):
    """ค้นหาสูตรอาหารตามคำค้นหา"""
    # เข้ารหัสคำค้นหา
    query_embedding = model.encode([query])
    
    # คำนวณความคล้ายคลึง
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # รับผลลัพธ์ที่ตรงที่สุด
    top_indices = np.argsort(-similarities)[:top_k]
    results = []
    
    for idx in top_indices:
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
    
    return results

def search_by_nutrition_criteria(nutrition_analyzer, criteria):
    """ค้นหาสูตรอาหารตามเกณฑ์โภชนาการ"""
    return nutrition_analyzer.search_recipes_by_nutrition(criteria)

def detect_nutrition_search(query):
    """ตรวจจับว่าการค้นหาเป็นการค้นหาตามโภชนาการหรือไม่"""
    nutrition_keywords = [
        'แคลอรี่', 'แคลอรี', 'calorie', 'cal', 'kcal',
        'โปรตีน', 'protein',
        'คาร์โบ', 'คาร์โบไฮเดรต', 'carb', 'carbohydrate',
        'ไขมัน', 'fat',
        'ใยอาหาร', 'fiber',
        'ลดน้ำหนัก', 'diet', 'healthy',
        'โภชนาการ', 'nutrition',
        'วิตามิน', 'vitamin',
        'แร่ธาตุ', 'mineral',
        'ต่ำ', 'สูง', 'น้อย', 'เยอะ',
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
        r'น้อยกว่า.*?(\d+).*?แคลอรี่'
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
        r'โปรตีน.*?เยอะ.*?(\d+)'
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
    
    return criteria

def main():
    # โหลดโมเดลและข้อมูล
    model = load_model()
    data = load_data()
    embeddings = get_embeddings(model, data)
    nutrition_analyzer = load_nutrition_analyzer()
    
    # แถบด้านข้างสำหรับค้นหาตามโภชนาการ
    with st.sidebar:
        st.header("🔍 ค้นหาตามเกณฑ์โภชนาการ")
        st.markdown("*ใช้ฟอร์มนี้สำหรับการค้นหาแบบละเอียด*")
        
        st.subheader("เกณฑ์การค้นหา")
        
        # เกณฑ์แคลอรี่
        calorie_range = st.slider(
            "แคลอรี่ (kcal)",
            min_value=0,
            max_value=1000,
            value=(0, 500),
            step=10
        )
        
        # เกณฑ์โปรตีน
        min_protein = st.number_input(
            "โปรตีนขั้นต่ำ (g)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.5
        )
        
        if st.button("ค้นหาสูตรอาหาร"):
            criteria = {
                'min_calories': calorie_range[0],
                'max_calories': calorie_range[1],
                'min_protein': min_protein
            }
            
            nutrition_results = search_by_nutrition_criteria(nutrition_analyzer, criteria)
            
            if nutrition_results:
                st.success(f"พบ {len(nutrition_results)} สูตรอาหารที่ตรงเกณฑ์")
                
                for result in nutrition_results[:5]:  # แสดง 5 อันดับแรก
                    with st.expander(f"{result['recipe_name']}"):
                        st.write(f"**แคลอรี่:** {result['calories']:.0f} kcal")
                        st.write(f"**โปรตีน:** {result['protein']:.1f} g")
                        st.write(f"**คาร์โบไฮเดรต:** {result['carbs']:.1f} g")
                        st.write(f"**ไขมัน:** {result['fat']:.1f} g")
            else:
                st.warning("ไม่พบสูตรอาหารที่ตรงเกณฑ์")
        
        st.markdown("---")
        st.markdown("### 💡 เทคนิคการค้นหา")
        st.markdown("""
        **ค้นหาทั่วไป:**
        - "ต้มยำกุ้ง"
        - "เมนูไก่"
        - "อาหารจานเดียว"
        
        **ค้นหาตามโภชนาการ:**
        - "เมนูแคลอรี่ไม่เกิน 300"
        - "อาหารโปรตีนสูงมากกว่า 20 กรัม"
        - "เมนูลดน้ำหนัก"
        - "อาหารแคลอรี่ต่ำ"
        """)
    
    # แอปหลัก
    st.title("🍲 Thai Food Recipe Chatbot")
    st.write("ถามเกี่ยวกับวิธีทำอาหารไทยได้เลย! พร้อมข้อมูลโภชนาการ")
    st.write("💡 **ใหม่!** สามารถค้นหาตามเกณฑ์โภชนาการได้ เช่น 'เมนูแคลอรี่ไม่เกิน 300' หรือ 'อาหารโปรตีนสูง'")
    
    # เริ่มต้นประวัติการสนทนา
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # แสดงประวัติการสนทนา
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "recipe" in message:
                # แสดงสูตรพร้อมโภชนาการ
                recipe = message["recipe"]
                st.markdown(f"### {recipe['name']}")
                
                # แสดงข้อมูลโภชนาการ
                display_nutrition_info(recipe['nutrition'])
                
                st.markdown("#### วัตถุดิบ (Ingredients)")
                st.markdown(format_ingredients(recipe["ingredients"]), unsafe_allow_html=True)
                st.markdown("#### วิธีทำ (Method)")
                st.markdown(format_cooking_method(recipe["method"]), unsafe_allow_html=True)
                st.markdown(f"*ความเกี่ยวข้อง (Relevance): {recipe['similarity']:.2f}*")
            elif message["role"] == "assistant" and "nutrition_results" in message:
                # แสดงผลลัพธ์การค้นหาตามโภชนาการ
                results = message["nutrition_results"]
                st.markdown(f"พบ **{len(results)}** สูตรอาหารที่ตรงเกณฑ์:")
                
                for i, result in enumerate(results[:5], 1):
                    st.markdown(f"""
                    **{i}. {result['recipe_name']}**
                    - แคลอรี่: {result['calories']:.0f} kcal
                    - โปรตีน: {result['protein']:.1f} g
                    - คาร์โบไฮเดรต: {result['carbs']:.1f} g
                    - ไขมัน: {result['fat']:.1f} g
                    """)
                
                if len(results) > 5:
                    st.markdown(f"*และอีก {len(results) - 5} สูตร...*")
            else:
                # แสดงข้อความธรรมดา
                st.markdown(message["content"])
    
    # ช่องใส่ข้อความสำหรับแชท
    if prompt := st.chat_input("ถามเกี่ยวกับอาหารไทยหรือค้นหาตามโภชนาการ..."):
        # เพิ่มข้อความของผู้ใช้ลงในประวัติการสนทนา
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # แสดงข้อความของผู้ใช้
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # รับการตอบสนอง
        with st.chat_message("assistant"):
            with st.spinner("กำลังค้นหาและวิเคราะห์โภชนาการ..."):
                # ตรวจสอบว่าเป็นการค้นหาตามโภชนาการหรือไม่
                if detect_nutrition_search(prompt):
                    # การค้นหาตามโภชนาการ
                    criteria = extract_nutrition_criteria_from_text(prompt)
                    
                    if criteria:
                        nutrition_results = search_by_nutrition_criteria(nutrition_analyzer, criteria)
                        
                        if nutrition_results:
                            response = f"พบ {len(nutrition_results)} สูตรอาหารที่ตรงกับเกณฑ์โภชนาการที่ต้องการ"
                            st.markdown(response)
                            
                            # แสดงผลลัพธ์
                            st.markdown("### 🔍 ผลการค้นหา")
                            for i, result in enumerate(nutrition_results[:5], 1):
                                st.markdown(f"""
                                **{i}. {result['recipe_name']}**
                                - แคลอรี่: {result['calories']:.0f} kcal
                                - โปรตีน: {result['protein']:.1f} g
                                - คาร์โบไฮเดรต: {result['carbs']:.1f} g
                                - ไขมัน: {result['fat']:.1f} g
                                """)
                            
                            if len(nutrition_results) > 5:
                                st.markdown(f"*และอีก {len(nutrition_results) - 5} สูตร...*")
                            
                            # เพิ่มการตอบสนองของแอสซิสแตนต์ลงในประวัติการสนทนา
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response,
                                "nutrition_results": nutrition_results
                            })
                        else:
                            response = "ขออภัย ไม่พบสูตรอาหารที่ตรงกับเกณฑ์โภชนาการที่ต้องการ ลองปรับเกณฑ์ใหม่หรือใช้ฟอร์มในแถบด้านข้าง"
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        # ถ้าตรวจพบคีย์เวิร์ดโภชนาการแต่แยกเกณฑ์ไม่ได้
                        response = "กรุณาระบุเกณฑ์โภชนาการให้ชัดเจนขึ้น เช่น 'เมนูแคลอรี่ไม่เกิน 300' หรือ 'อาหารโปรตีนสูงมากกว่า 20 กรัม' หรือใช้ฟอร์มในแถบด้านข้าง"
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    # การค้นหาทั่วไป
                    results = search_recipes(prompt, model, data, embeddings, nutrition_analyzer)
                    
                    if results:
                        best_match = results[0]
                        
                        # ตรวจสอบว่ามีผลลัพธ์ที่ดี
                        if best_match["similarity"] > 0.3:
                            response = f"ฉันพบสูตรอาหารที่คุณต้องการ: {best_match['name']}"
                            st.markdown(response)
                            
                            # แสดงสูตรพร้อมโภชนาการ
                            st.markdown(f"### {best_match['name']}")
                            
                            # แสดงข้อมูลโภชนาการ
                            display_nutrition_info(best_match['nutrition'])
                            
                            st.markdown("#### วัตถุดิบ (Ingredients)")
                            st.markdown(format_ingredients(best_match["ingredients"]), unsafe_allow_html=True)
                            st.markdown("#### วิธีทำ (Method)")
                            st.markdown(format_cooking_method(best_match["method"]), unsafe_allow_html=True)
                            st.markdown(f"*ความเกี่ยวข้อง (Relevance): {best_match['similarity']:.2f}*")
                            
                            # เพิ่มการตอบสนองของแอสซิสแตนต์ลงในประวัติการสนทนาพร้อมข้อมูลสูตร
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response, 
                                "recipe": best_match
                            })
                        else:
                            response = "ขออภัย ฉันไม่พบสูตรอาหารที่ตรงกับคำถามของคุณ กรุณาลองถามใหม่อีกครั้ง หรือลองค้นหาตามโภชนาการ"
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        response = "ขออภัย ฉันไม่สามารถค้นหาสูตรอาหารได้ในขณะนี้ กรุณาลองใหม่อีกครั้ง"
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

    # ส่วนท้ายพร้อมข้อมูลโภชนาการ
    st.markdown("---")
    st.markdown("""
    ### 📊 เกี่ยวกับข้อมูลโภชนาการ
    - ข้อมูลโภชนาการคำนวณจากวัตถุดิบในสูตรอาหาร
    - ค่าที่แสดงเป็นการประมาณต่อหนึ่งที่ (1 serving)
    - แหล่งข้อมูล: USDA Food Database และข้อมูลอ้างอิงจากแหล่งที่เชื่อถือได้
    - สำหรับข้อมูลโภชนาการที่แม่นยำ ควรปรึกษานักโภชนาการ
    
    ### 🔍 วิธีการค้นหา
    - **ค้นหาทั่วไป**: ใส่ชื่ออาหารหรือวัตถุดิบ เช่น "ต้มยำกุ้ง", "เมนูไก่"
    - **ค้นหาตามโภชนาการ**: ใส่เกณฑ์โภชนาการ เช่น "เมนูแคลอรี่ไม่เกิน 300", "อาหารโปรตีนสูง"
    - **ค้นหาละเอียด**: ใช้ฟอร์มในแถบด้านข้างสำหรับการตั้งค่าที่แม่นยำ
    """)

if __name__ == "__main__":
    main()
