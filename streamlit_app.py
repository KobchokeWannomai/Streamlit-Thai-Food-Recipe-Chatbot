import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import pickle
import re

# Page config
st.set_page_config(
    page_title="Thai Food Recipe Chatbot",
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
    # Split by sentences (Thai typically uses space as a sentence separator in cooking instructions)
    sentences = re.split(r'(?<=[ๆ.]) ', method_text)
    formatted = "<ol>"
    for sentence in sentences:
        if sentence.strip():
            formatted += f"<li>{sentence.strip()}</li>"
    formatted += "</ol>"
    return formatted

def search_recipes(query, model, data, embeddings, top_k=3):
    """Search for recipes based on the query"""
    # Encode the query
    query_embedding = model.encode([query])
    
    # Calculate similarity
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # Get top matches
    top_indices = np.argsort(-similarities)[:top_k]
    results = []
    
    for idx in top_indices:
        results.append({
            'name': data.iloc[idx]['name'],
            'similarity': similarities[idx],
            'ingredients': data.iloc[idx]['ingredient'],
            'method': data.iloc[idx]['method']
        })
    
    return results

def main():
    # Load model and data
    model = load_model()
    data = load_data()
    embeddings = get_embeddings(model, data)
    
    # App title
    st.title("🍲 Thai Food Recipe Chatbot")
    st.write("ถามเกี่ยวกับวิธีทำอาหารไทยได้เลย! Ask about Thai food recipes!")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "recipe" in message:
                # Display recipe
                recipe = message["recipe"]
                st.markdown(f"### {recipe['name']}")
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
            with st.spinner("กำลังค้นหา..."):
                results = search_recipes(prompt, model, data, embeddings)
                
                if results:
                    best_match = results[0]
                    
                    # Check if there's a good match
                    if best_match["similarity"] > 0.3:
                        response = f"ฉันพบสูตรอาหารที่คุณต้องการ: {best_match['name']}"
                        st.markdown(response)
                        
                        # Display recipe
                        st.markdown(f"### {best_match['name']}")
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

if __name__ == "__main__":
    main()
