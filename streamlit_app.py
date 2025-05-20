import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import pickle
import re
import requests
import json
from typing import Dict, List, Tuple, Optional, Union

# Page config
st.set_page_config(
    page_title="Thai Food Recipe Chatbot - โภชนาการ",
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
</style>
""", unsafe_allow_html=True)

# Paths
DATA_PATH = "thai_food_processed.csv"
EMBEDDINGS_PATH = "embeddings.pkl"
MODEL_PATH = "model"
NUTRITION_DB_PATH = "nutrition_db.csv"
RECIPE_NUTRITION_PATH = "recipe_nutrition.pkl"

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

@st.cache_data
def load_data():
    """Load the Thai food dataset"""
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_nutrition_database():
    """Load or create the nutrition database"""
    if os.path.exists(NUTRITION_DB_PATH):
        return pd.read_csv(NUTRITION_DB_PATH)
    else:
        # Create sample nutrition database with common Thai ingredients
        data = {
            'ingredient': [
                'กุ้ง', 'ปลา', 'ไข่', 'ไก่', 'หมู', 'ข้าว', 'มะเขือเทศ', 'พริก', 
                'กระเทียม', 'ขิง', 'ตะไคร้', 'ใบมะกรูด', 'น้ำตาล', 'น้ำปลา', 
                'มะพร้าว', 'กะทิ', 'ถั่ว', 'ผักกาด', 'คะน้า', 'ผักบุ้ง'
            ],
            'calories': [
                100, 120, 70, 165, 242, 130, 18, 40, 
                42, 80, 60, 45, 387, 10, 354, 230, 
                347, 14, 28, 19
            ],
            'protein': [
                20.1, 22.3, 6.3, 31, 25.7, 2.7, 0.9, 1.9, 
                1.8, 1.8, 1.8, 1.2, 0, 0.4, 3.3, 2.2, 
                38.3, 1.2, 3.0, 2.6
            ],
            'fat': [
                1.7, 2.6, 5.0, 3.6, 14.0, 0.3, 0.2, 0.4, 
                0.1, 0.8, 1.2, 0.7, 0, 0, 33.5, 23.8, 
                18.0, 0.2, 0.4, 0.3
            ],
            'carbs': [
                0.9, 0, 0.6, 0, 0, 28.2, 3.9, 8.8, 
                9.4, 17.8, 12.5, 11.0, 100, 2.0, 15.2, 5.5, 
                29.0, 2.4, 4.4, 3.0
            ],
            'serving_size': [
                100, 100, 50, 100, 100, 100, 100, 20, 
                10, 10, 10, 5, 10, 15, 100, 100, 
                100, 100, 100, 100
            ],
            'unit': [
                'กรัม', 'กรัม', 'กรัม', 'กรัม', 'กรัม', 'กรัม', 'กรัม', 'กรัม', 
                'กรัม', 'กรัม', 'กรัม', 'ใบ', 'กรัม', 'มิลลิลิตร', 'กรัม', 'มิลลิลิตร', 
                'กรัม', 'กรัม', 'กรัม', 'กรัม'
            ]
        }
        df = pd.DataFrame(data)
        df.to_csv(NUTRITION_DB_PATH, index=False)
        return df

@st.cache_data
def get_recipe_nutrition():
    """Get or calculate nutrition information for all recipes"""
    if os.path.exists(RECIPE_NUTRITION_PATH):
        with open(RECIPE_NUTRITION_PATH, 'rb') as f:
            return pickle.load(f)
    else:
        nutrition_db = load_nutrition_database()
        data = load_data()
        
        all_nutrition = {}
        
        for idx, row in data.iterrows():
            recipe_name = row['name']
            ingredients_text = row['ingredient']
            nutrition = calculate_recipe_nutrition(ingredients_text, nutrition_db)
            all_nutrition[recipe_name] = nutrition
        
        with open(RECIPE_NUTRITION_PATH, 'wb') as f:
            pickle.dump(all_nutrition, f)
        
        return all_nutrition

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

def parse_ingredients(ingredients_text: str) -> List[Dict[str, str]]:
    """Parse ingredients from text into a structured format"""
    ingredients_list = []
    
    # Split by line breaks (usually each ingredient is on a new line)
    lines = ingredients_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Remove leading dash or bullet if present
        if line.startswith('-'):
            line = line[1:].strip()
        
        # Try to extract amount and unit
        # Pattern: quantity (number or fraction) followed by unit
        match = re.search(r'(\d+(?:\.\d+)?(?:/\d+)?)\s*([ก-๙a-zA-Z]+)?', line)
        
        if match:
            amount_str = match.group(1)
            unit = match.group(2) if match.group(2) else ""
            
            # Convert fraction to decimal if needed
            if '/' in amount_str:
                num, denom = amount_str.split('/')
                amount = float(num) / float(denom)
            else:
                amount = float(amount_str)
            
            # Get the ingredient name (everything after the amount and unit)
            ingredient_name = line[match.end():].strip()
            
            # Clean up the ingredient name
            if ingredient_name.startswith('ของ'):
                ingredient_name = ingredient_name[3:].strip()
            
            # Remove additional specifications in parentheses for matching
            clean_name = re.sub(r'\(.*?\)', '', ingredient_name).strip()
            
            ingredients_list.append({
                'raw_text': line,
                'name': clean_name,
                'amount': amount,
                'unit': unit,
                'full_name': ingredient_name
            })
        else:
            # If we couldn't parse amount/unit, just add the whole line as the ingredient name
            ingredients_list.append({
                'raw_text': line,
                'name': line,
                'amount': 1.0,
                'unit': '',
                'full_name': line
            })
    
    return ingredients_list

def match_ingredient_to_nutrition_db(ingredient: Dict[str, str], nutrition_db: pd.DataFrame) -> Tuple[bool, str, float]:
    """Match an ingredient to the nutrition database with some fuzzy matching"""
    # Get the clean ingredient name for matching
    ingredient_name = ingredient['name'].lower()
    
    # Try direct match first
    direct_match = nutrition_db[nutrition_db['ingredient'].str.lower() == ingredient_name]
    if not direct_match.empty:
        return True, direct_match.iloc[0]['ingredient'], 1.0
    
    # Try partial matching
    for idx, db_entry in nutrition_db.iterrows():
        db_ingredient = db_entry['ingredient'].lower()
        
        # Check if the database ingredient appears in the ingredient name
        if db_ingredient in ingredient_name:
            return True, db_entry['ingredient'], 0.8
        
        # Check if the ingredient name appears in the database ingredient
        if ingredient_name in db_ingredient:
            return True, db_entry['ingredient'], 0.6
    
    # No match found
    return False, "", 0.0

def calculate_recipe_nutrition(ingredients_text: str, nutrition_db: pd.DataFrame) -> Dict[str, float]:
    """Calculate nutrition information for a recipe based on its ingredients"""
    ingredients = parse_ingredients(ingredients_text)
    
    # Initialize nutrition values
    nutrition = {
        'calories': 0.0,
        'protein': 0.0,
        'fat': 0.0,
        'carbs': 0.0,
        'matched_ingredients': 0,
        'total_ingredients': len(ingredients)
    }
    
    # Calculate nutrition for each ingredient
    for ingredient in ingredients:
        matched, db_ingredient, confidence = match_ingredient_to_nutrition_db(ingredient, nutrition_db)
        
        if matched:
            nutrition['matched_ingredients'] += 1
            
            # Get nutrition values from the database
            db_entry = nutrition_db[nutrition_db['ingredient'] == db_ingredient].iloc[0]
            
            # Default to 1.0 if amount cannot be determined
            amount = ingredient.get('amount', 1.0)
            
            # Get the standard serving size for this ingredient
            serving_size = db_entry['serving_size']
            
            # Calculate the proportion of the standard serving
            proportion = amount / serving_size if serving_size > 0 else 1.0
            
            # Scale nutrition values by the proportion and confidence
            scale_factor = proportion * confidence
            nutrition['calories'] += db_entry['calories'] * scale_factor
            nutrition['protein'] += db_entry['protein'] * scale_factor
            nutrition['fat'] += db_entry['fat'] * scale_factor
            nutrition['carbs'] += db_entry['carbs'] * scale_factor
    
    # Add coverage percentage
    if nutrition['total_ingredients'] > 0:
        nutrition['coverage'] = nutrition['matched_ingredients'] / nutrition['total_ingredients']
    else:
        nutrition['coverage'] = 0.0
    
    return nutrition

def search_recipes(query, model, data, embeddings, nutrition_filter=None, top_k=3):
    """Search for recipes based on the query and optional nutrition filters"""
    # Encode the query
    query_embedding = model.encode([query])
    
    # Calculate similarity
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # Get all recipes with their scores
    recipes_with_scores = [(data.iloc[idx], similarities[idx]) for idx in range(len(similarities))]
    
    # Apply nutrition filter if provided
    if nutrition_filter:
        recipe_nutrition = get_recipe_nutrition()
        filtered_recipes = []
        
        for recipe, score in recipes_with_scores:
            recipe_name = recipe['name']
            if recipe_name in recipe_nutrition:
                nutrition = recipe_nutrition[recipe_name]
                
                # Check each filter criterion
                meets_criteria = True
                
                if 'min_calories' in nutrition_filter and nutrition['calories'] < nutrition_filter['min_calories']:
                    meets_criteria = False
                
                if 'max_calories' in nutrition_filter and nutrition['calories'] > nutrition_filter['max_calories']:
                    meets_criteria = False
                    
                if 'min_protein' in nutrition_filter and nutrition['protein'] < nutrition_filter['min_protein']:
                    meets_criteria = False
                
                if 'max_protein' in nutrition_filter and nutrition['protein'] > nutrition_filter['max_protein']:
                    meets_criteria = False
                    
                if 'min_fat' in nutrition_filter and nutrition['fat'] < nutrition_filter['min_fat']:
                    meets_criteria = False
                
                if 'max_fat' in nutrition_filter and nutrition['fat'] > nutrition_filter['max_fat']:
                    meets_criteria = False
                    
                if 'min_carbs' in nutrition_filter and nutrition['carbs'] < nutrition_filter['min_carbs']:
                    meets_criteria = False
                
                if 'max_carbs' in nutrition_filter and nutrition['carbs'] > nutrition_filter['max_carbs']:
                    meets_criteria = False
                
                if meets_criteria:
                    filtered_recipes.append((recipe, score))
            else:
                # If we don't have nutrition data, include it only if we're not strict
                if not nutrition_filter.get('strict', False):
                    filtered_recipes.append((recipe, score))
        
        recipes_with_scores = filtered_recipes
    
    # Sort by similarity score
    recipes_with_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Get top results
    top_recipes = recipes_with_scores[:top_k]
    
    results = []
    for recipe, similarity in top_recipes:
        results.append({
            'name': recipe['name'],
            'similarity': similarity,
            'ingredients': recipe['ingredient'],
            'method': recipe['method']
        })
    
    return results

def extract_nutrition_criteria_from_query(query: str) -> Tuple[str, Optional[Dict[str, float]]]:
    """Extract nutrition criteria from the query and return the cleaned query and criteria"""
    # Define patterns for nutrition criteria
    patterns = {
        'min_calories': r'อย่างน้อย\s*(\d+)\s*แคลอรี่|มากกว่า\s*(\d+)\s*แคลอรี่|calories\s*(>|>=|at least|more than)\s*(\d+)',
        'max_calories': r'ไม่เกิน\s*(\d+)\s*แคลอรี่|น้อยกว่า\s*(\d+)\s*แคลอรี่|calories\s*(<|<=|less than|at most)\s*(\d+)',
        'min_protein': r'โปรตีนอย่างน้อย\s*(\d+)|โปรตีนมากกว่า\s*(\d+)|protein\s*(>|>=|at least|more than)\s*(\d+)',
        'max_protein': r'โปรตีนไม่เกิน\s*(\d+)|โปรตีนน้อยกว่า\s*(\d+)|protein\s*(<|<=|less than|at most)\s*(\d+)',
        'min_fat': r'ไขมันอย่างน้อย\s*(\d+)|ไขมันมากกว่า\s*(\d+)|fat\s*(>|>=|at least|more than)\s*(\d+)',
        'max_fat': r'ไขมันไม่เกิน\s*(\d+)|ไขมันน้อยกว่า\s*(\d+)|fat\s*(<|<=|less than|at most)\s*(\d+)',
        'min_carbs': r'คาร์บอย่างน้อย\s*(\d+)|คาร์บมากกว่า\s*(\d+)|carbs\s*(>|>=|at least|more than)\s*(\d+)',
        'max_carbs': r'คาร์บไม่เกิน\s*(\d+)|คาร์บน้อยกว่า\s*(\d+)|carbs\s*(<|<=|less than|at most)\s*(\d+)'
    }
    
    # Initialize nutrition criteria
    nutrition_criteria = {}
    
    # Check for each pattern
    for criterion, pattern in patterns.items():
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            # Extract the value from the appropriate group
            value = next((g for g in match.groups() if g is not None), None)
            if value:
                nutrition_criteria[criterion] = float(value)
                # Remove the matched part from the query
                query = re.sub(pattern, '', query, flags=re.IGNORECASE)
    
    # Clean up the query
    query = query.strip()
    
    # If the query is now empty but we have nutrition criteria, set a default query
    if not query and nutrition_criteria:
        query = "อาหาร"  # Default to searching for "food" in Thai
    
    return query, nutrition_criteria or None

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
    # Split by sentences (Thai typically uses space as a sentence separator in cooking instructions)
    sentences = re.split(r'(?<=[ๆ.]) ', method_text)
    formatted = "<ol>"
    for sentence in sentences:
        if sentence.strip():
            formatted += f"<li>{sentence.strip()}</li>"
    formatted += "</ol>"
    return formatted

def format_nutrition_info(nutrition):
    """Format nutrition information for display"""
    formatted = "<div class='nutrition-info'>"
    formatted += f"<h4>ข้อมูลโภชนาการ (ประมาณการณ์)</h4>"
    formatted += f"<p>พลังงาน: {nutrition['calories']:.1f} แคลอรี่</p>"
    formatted += f"<p>โปรตีน: {nutrition['protein']:.1f} กรัม</p>"
    formatted += f"<p>ไขมัน: {nutrition['fat']:.1f} กรัม</p>"
    formatted += f"<p>คาร์โบไฮเดรต: {nutrition['carbs']:.1f} กรัม</p>"
    formatted += f"<p><small>*ข้อมูลโภชนาการนี้เป็นการประมาณการณ์จากวัตถุดิบในสูตร ({nutrition['matched_ingredients']}/{nutrition['total_ingredients']} วัตถุดิบที่พบในฐานข้อมูล)</small></p>"
    formatted += "</div>"
    return formatted

def main():
    # Load model and data
    model = load_model()
    data = load_data()
    embeddings = get_embeddings(model, data)
    nutrition_db = load_nutrition_database()
    recipe_nutrition = get_recipe_nutrition()
    
    # App title
    st.title("🍲 Thai Food Recipe Chatbot - โภชนาการ")
    st.write("ถามเกี่ยวกับวิธีทำอาหารไทยและข้อมูลโภชนาการได้เลย! Ask about Thai food recipes and nutrition!")
    
    # Sidebar for nutrition filtering
    with st.sidebar:
        st.header("ตัวกรองโภชนาการ (Nutrition Filters)")
        use_filters = st.checkbox("ใช้ตัวกรองโภชนาการ", False)
        
        if use_filters:
            st.subheader("แคลอรี่ (Calories)")
            min_calories = st.slider("แคลอรี่ต่ำสุด (Min)", 0, 2000, 0)
            max_calories = st.slider("แคลอรี่สูงสุด (Max)", 0, 2000, 2000)
            
            st.subheader("โปรตีน (Protein)")
            min_protein = st.slider("โปรตีนต่ำสุด (Min g)", 0, 100, 0)
            max_protein = st.slider("โปรตีนสูงสุด (Max g)", 0, 100, 100)
            
            st.subheader("ไขมัน (Fat)")
            min_fat = st.slider("ไขมันต่ำสุด (Min g)", 0, 100, 0)
            max_fat = st.slider("ไขมันสูงสุด (Max g)", 0, 100, 100)
            
            st.subheader("คาร์โบไฮเดรต (Carbs)")
            min_carbs = st.slider("คาร์บต่ำสุด (Min g)", 0, 200, 0)
            max_carbs = st.slider("คาร์บสูงสุด (Max g)", 0, 200, 200)
            
            strict_filter = st.checkbox("แสดงเฉพาะเมนูที่มีข้อมูลโภชนาการ", False)
            
            nutrition_filter = {
                'min_calories': min_calories if min_calories > 0 else None,
                'max_calories': max_calories if max_calories < 2000 else None,
                'min_protein': min_protein if min_protein > 0 else None,
                'max_protein': max_protein if max_protein < 100 else None,
                'min_fat': min_fat if min_fat > 0 else None,
                'max_fat': max_fat if max_fat < 100 else None,
                'min_carbs': min_carbs if min_carbs > 0 else None,
                'max_carbs': max_carbs if max_carbs < 200 else None,
                'strict': strict_filter
            }
            
            # Remove None values
            nutrition_filter = {k: v for k, v in nutrition_filter.items() if v is not None}
        else:
            nutrition_filter = None

        st.subheader("ข้อมูลฐานข้อมูลโภชนาการ")
        st.write(f"จำนวนวัตถุดิบในฐานข้อมูล: {len(nutrition_db)}")
        
        if st.button("แสดงฐานข้อมูลโภชนาการ"):
            st.dataframe(nutrition_db)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "recipe" in message:
                # Display recipe with nutrition information
                recipe = message["recipe"]
                st.markdown(f"### {recipe['name']}")
                
                # Show nutrition info if available
                recipe_name = recipe['name']
                if recipe_name in recipe_nutrition:
                    nutrition = recipe_nutrition[recipe_name]
                    st.markdown(format_nutrition_info(nutrition), unsafe_allow_html=True)
                
                st.markdown("#### วัตถุดิบ (Ingredients)")
                st.markdown(format_ingredients(recipe["ingredients"]), unsafe_allow_html=True)
                st.markdown("#### วิธีทำ (Method)")
                st.markdown(format_cooking_method(recipe["method"]), unsafe_allow_html=True)
                st.markdown(f"*ความเกี่ยวข้อง (Relevance): {recipe['similarity']:.2f}*")
            else:
                # Display regular message
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("ถามเกี่ยวกับอาหารไทยหรือข้อมูลโภชนาการ..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process the query to extract nutrition criteria
        clean_query, extracted_nutrition_filter = extract_nutrition_criteria_from_query(prompt)
        
        # Combine extracted nutrition criteria with sidebar filters
        if extracted_nutrition_filter and nutrition_filter:
            # Merge the two filters, with extracted criteria taking precedence
            combined_filter = {**nutrition_filter, **extracted_nutrition_filter}
        elif extracted_nutrition_filter:
            combined_filter = extracted_nutrition_filter
        else:
            combined_filter = nutrition_filter
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("กำลังค้นหา..."):
                results = search_recipes(clean_query, model, data, embeddings, combined_filter)
                
                if results:
                    best_match = results[0]
                    
                    # Check if there's a good match
                    if best_match["similarity"] > 0.3:
                        if extracted_nutrition_filter:
                            response = f"ฉันพบสูตรอาหารที่ตรงกับเกณฑ์โภชนาการที่คุณต้องการ: {best_match['name']}"
                        else:
                            response = f"ฉันพบสูตรอาหารที่คุณต้องการ: {best_match['name']}"
                        
                        st.markdown(response)
                        
                        # Display recipe with nutrition information
                        st.markdown(f"### {best_match['name']}")
                        
                        # Show nutrition info if available
                        recipe_name = best_match['name']
                        if recipe_name in recipe_nutrition:
                            nutrition = recipe_nutrition[recipe_name]
                            st.markdown(format_nutrition_info(nutrition), unsafe_allow_html=True)
                        
                        st.markdown("#### วัตถุดิบ (Ingredients)")
                        st.markdown(format_ingredients(best_match["ingredients"]), unsafe_allow_html=True)
                        st.markdown("#### วิธีทำ (Method)")
                        st.markdown(format_cooking_method(best_match["method"]), unsafe_allow_html=True)
                        st.markdown(f"*ความเกี่ยวข้อง (Relevance): {best_match['similarity']:.2f}*")
                        
                        # Add assistant response to chat history with recipe and nutrition data
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response, 
                            "recipe": best_match
                        })
                    else:
                        if extracted_nutrition_filter:
                            response = "ขออภัย ฉันไม่พบสูตรอาหารที่ตรงกับเกณฑ์โภชนาการที่คุณต้องการ กรุณาลองเปลี่ยนเกณฑ์และถามใหม่อีกครั้ง"
                        else:
                            response = "ขออภัย ฉันไม่พบสูตรอาหารที่ตรงกับคำถามของคุณ กรุณาลองถามใหม่อีกครั้ง"
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    if extracted_nutrition_filter or nutrition_filter:
                        response = "ขออภัย ฉันไม่พบสูตรอาหารที่ตรงกับเกณฑ์โภชนาการที่คุณต้องการ กรุณาลองเปลี่ยนเกณฑ์และถามใหม่อีกครั้ง"
                    else:
                        response = "ขออภัย ฉันไม่สามารถค้นหาสูตรอาหารได้ในขณะนี้ กรุณาลองใหม่อีกครั้ง"
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
