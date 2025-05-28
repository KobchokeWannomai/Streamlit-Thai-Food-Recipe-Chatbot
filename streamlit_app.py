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

# Page config
st.set_page_config(
    page_title="Thai Food Recipe Chatbot with Nutrition",
    page_icon="🍲",
    layout="wide"
)

# Set Thai font
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

# Paths
DATA_PATH = "thai_food_processed.csv"
EMBEDDINGS_PATH = "embeddings.pkl"
MODEL_PATH = "model"

@st.cache_resource
def load_model():
    """Load or download the sentence transformer model"""
    if os.path.exists(MODEL_PATH):
        return SentenceTransformer(MODEL_PATH)
    else:
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        os.makedirs(MODEL_PATH, exist_ok=True)
        model.save(MODEL_PATH)
        return model

@st.cache_resource
def load_nutrition_analyzer():
    """Load nutrition analyzer"""
    return NutritionAnalyzer()

@st.cache_data
def load_data():
    """Load the Thai food dataset"""
    return pd.read_csv(DATA_PATH)

@st.cache_data
def get_embeddings(_model, data):
    """Get or compute embeddings for all recipes"""
    if os.path.exists(EMBEDDINGS_PATH):
        with open(EMBEDDINGS_PATH, 'rb') as f:
            return pickle.load(f)
    else:
        # Combine all text for each recipe
        texts = []
        for _, row in data.iterrows():
            combined_text = f"{row['name']} {row['ingredient']} {row['method']}"
            texts.append(combined_text)
        
        # Generate embeddings
        embeddings = _model.encode(texts)
        
        # Save embeddings
        with open(EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump(embeddings, f)
        
        return embeddings

def format_ingredients(ingredients_text):
    """Format the ingredients list for better display"""
    ingredients = ingredients_text.split('\n')
    formatted = "<ul>"
    for item in ingredients:
        if item.strip():
            formatted += f"<li>{item.strip()}</li>"
    formatted += "</ul>"
    return formatted

def format_cooking_method(method_text):
    """Format the cooking method for better display"""
    sentences = re.split(r'(?<=[ๆ.]) ', method_text)
    formatted = "<ol>"
    for sentence in sentences:
        if sentence.strip():
            formatted += f"<li>{sentence.strip()}</li>"
    formatted += "</ol>"
    return formatted

def display_nutrition_info(nutrition_data):
    """Display nutrition information in a nice format"""
    if not nutrition_data:
        return
    
    total_nutrition = nutrition_data.get('total_nutrition', {})
    
    st.markdown('<div class="nutrition-card">', unsafe_allow_html=True)
    st.markdown("### 🥗 ข้อมูลโภชนาการ (ต่อหนึ่งที่)")
    
    # Main nutrients
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
    
    # Vitamins and minerals
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
    
    # Ingredient breakdown
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
    """Search for recipes based on the query"""
    # Encode the query
    query_embedding = model.encode([query])
    
    # Calculate similarity
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # Get top matches
    top_indices = np.argsort(-similarities)[:top_k]
    results = []
    
    for idx in top_indices:
        recipe_name = data.iloc[idx]['name']
        ingredients = data.iloc[idx]['ingredient']
        
        # Get nutrition data
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
    """Search recipes by nutrition criteria"""
    return nutrition_analyzer.search_recipes_by_nutrition(criteria)

def main():
    # Load model and data
    model = load_model()
    data = load_data()
    embeddings = get_embeddings(model, data)
    nutrition_analyzer = load_nutrition_analyzer()
    
    # Sidebar for nutrition search
    with st.sidebar:
        st.header("🔍 ค้นหาตามโภชนาการ")
        
        search_mode = st.radio(
            "เลือกวิธีการค้นหา",
            ["ค้นหาทั่วไป", "ค้นหาตามโภชนาการ"]
        )
        
        if search_mode == "ค้นหาตามโภชนาการ":
            st.subheader("เกณฑ์การค้นหา")
            
            # Calorie criteria
            calorie_range = st.slider(
                "แคลอรี่ (kcal)",
                min_value=0,
                max_value=1000,
                value=(0, 500),
                step=10
            )
            
            # Protein criteria
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
                    
                    for result in nutrition_results[:5]:  # Show top 5
                        with st.expander(f"{result['recipe_name']}"):
                            st.write(f"**แคลอรี่:** {result['calories']:.0f} kcal")
                            st.write(f"**โปรตีน:** {result['protein']:.1f} g")
                            st.write(f"**คาร์โบไฮเดรต:** {result['carbs']:.1f} g")
                            st.write(f"**ไขมัน:** {result['fat']:.1f} g")
                else:
                    st.warning("ไม่พบสูตรอาหารที่ตรงเกณฑ์")
    
    # Main app
    st.title("🍲 Thai Food Recipe Chatbot")
    st.write("ถามเกี่ยวกับวิธีทำอาหารไทยได้เลย! พร้อมข้อมูลโภชนาการ")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "recipe" in message:
                # Display recipe with nutrition
                recipe = message["recipe"]
                st.markdown(f"### {recipe['name']}")
                
                # Display nutrition info
                display_nutrition_info(recipe['nutrition'])
                
                st.markdown("#### วัตถุดิบ (Ingredients)")
                st.markdown(format_ingredients(recipe["ingredients"]), unsafe_allow_html=True)
                st.markdown("#### วิธีทำ (Method)")
                st.markdown(format_cooking_method(recipe["method"]), unsafe_allow_html=True)
                st.markdown(f"*ความเกี่ยวข้อง (Relevance): {recipe['similarity']:.2f}*")
            else:
                # Display regular message
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("ถามเกี่ยวกับอาหารไทย..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("กำลังค้นหาและวิเคราะห์โภชนาการ..."):
                results = search_recipes(prompt, model, data, embeddings, nutrition_analyzer)
                
                if results:
                    best_match = results[0]
                    
                    # Check if there's a good match
                    if best_match["similarity"] > 0.3:
                        response = f"ฉันพบสูตรอาหารที่คุณต้องการ: {best_match['name']}"
                        st.markdown(response)
                        
                        # Display recipe with nutrition
                        st.markdown(f"### {best_match['name']}")
                        
                        # Display nutrition info
                        display_nutrition_info(best_match['nutrition'])
                        
                        st.markdown("#### วัตถุดิบ (Ingredients)")
                        st.markdown(format_ingredients(best_match["ingredients"]), unsafe_allow_html=True)
                        st.markdown("#### วิธีทำ (Method)")
                        st.markdown(format_cooking_method(best_match["method"]), unsafe_allow_html=True)
                        st.markdown(f"*ความเกี่ยวข้อง (Relevance): {best_match['similarity']:.2f}*")
                        
                        # Add assistant response to chat history with recipe data
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

    # Footer with nutrition info
    st.markdown("---")
    st.markdown("""
    ### 📊 เกี่ยวกับข้อมูลโภชนาการ
    - ข้อมูลโภชนาการคำนวณจากวัตถุดิบในสูตรอาหาร
    - ค่าที่แสดงเป็นการประมาณต่อหนึ่งที่ (1 serving)
    - แหล่งข้อมูล: USDA Food Database และข้อมูลอ้างอิงจากแหล่งที่เชื่อถือได้
    - สำหรับข้อมูลโภชนาการที่แม่นยำ ควรปรึกษานักโภชนาการ
    """)

if __name__ == "__main__":
    main()
