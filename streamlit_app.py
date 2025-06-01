import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import re
import json
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
import time

# Import คลาสที่จำเป็น
from nutrition_api import NutritionAPI
from recipe_search import RecipeSearchEngine
from chatbot_responses import ThaiLoodChatbot

# การตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Thai Food Recipe Chatbot - AI Nutrition Assistant",
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
    
    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: bold;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .chat-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    
    .nutrition-highlight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .recipe-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
    }
    
    .nutrition-metric {
        background: linear-gradient(135deg, #a8e6cf 0%, #dcedc1 100%);
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.3rem 0;
        text-align: center;
        font-weight: bold;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #e17055;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #a8e6cf 0%, #dcedc1 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #00b894;
        margin: 1rem 0;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
    }
    
    .user-message {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #fd79a8 0%, #fdcb6e 100%);
        color: white;
        margin-right: 2rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ฟังก์ชันโหลดข้อมูล
@st.cache_data
def load_data():
    """โหลดข้อมูลอาหารไทยที่มีโภชนาการ"""
    try:
        if os.path.exists("thai_food_processed.csv"):
            df = pd.read_csv("thai_food_processed.csv")
            return df
        else:
            # สร้างข้อมูลตัวอย่างหากไม่มีไฟล์
            return create_sample_data_with_nutrition()
    except Exception as e:
        st.error(f"ข้อผิดพลาดในการโหลดข้อมูล: {str(e)}")
        return create_sample_data_with_nutrition()

@st.cache_resource
def initialize_systems():
    """เริ่มต้นระบบต่างๆ"""
    nutrition_api = NutritionAPI()
    data = load_data()
    search_engine = RecipeSearchEngine(data, nutrition_api)
    chatbot = ThaiLoodChatbot(data, nutrition_api, search_engine)
    return nutrition_api, search_engine, chatbot, data

def create_sample_data_with_nutrition():
    """สร้างข้อมูลตัวอย่างที่มีโภชนาการครบถ้วน"""
    data = {
        'name': [
            'ผัดกะเพราหมูสับ', 'ต้มยำกุ้งน้ำใส', 'ส้มตำไทย', 'แกงเขียวหวานไก่', 'ผัดไทยกุ้งสด',
            'ไข่เจียวฟู', 'ข้าวผัดกุ้ง', 'ยำวุ้นเส้นทะเล', 'ลาบหมูอีสาน', 'มะม่วงข้าวเหนียว',
            'แกงจืดเต้าหู้', 'ผัดพักบุ้งไฟแดง', 'ต้มข่าไก่', 'ผัดซีอิ๊วหมู', 'ยำเนื้อย่าง'
        ],
        'ingredient': [
            '- เนื้อหมูสับ 200 กรัม\n- ใบกะเพรา 1 ถ้วย\n- พริกขี้หนู 5 เม็ด\n- กระเทียม 4 กลีบ\n- น้ำปลา 2 ช้อนโต๊ะ\n- น้ำตาลทราย 1 ช้อนชา\n- น้ำมันพืช 2 ช้อนโต๊ะ',
            '- กุ้งนาง 300 กรัม\n- เห็ดฟาง 100 กรัม\n- มะนาว 3 ผล\n- ใบมะกรูด 5 ใบ\n- ตะไคร้ 3 ต้น\n- ข่า 4 แว่น\n- น้ำปลา 3 ช้อนโต๊ะ\n- พริกขี้หนู 7 เม็ด',
            '- มะละกอดิบ 2 ถ้วย\n- มะเขือเทศ 3 ผล\n- ถั่วฝักยาว 10 เส้น\n- กุ้งแห้ง 2 ช้อนโต๊ะ\n- ถั่วลิสงคั่ว 3 ช้อนโต๊ะ\n- พริกขี้หนู 5 เม็ด\n- กระเทียม 3 กลีบ\n- น้ำปลา 2 ช้อนโต๊ะ\n- น้ำตาลปึก 3 ช้อนโต๊ะ\n- มะนาว 2 ผล',
            '- เนื้อไก่ 400 กรัม\n- กะทิ 2 ถ้วย\n- น้ำพริกแกงเขียวหวาน 3 ช้อนโต๊ะ\n- มะเขือเปราะ 8 ผล\n- ใบโหระพา 1 ถ้วย\n- พริกชี้ฟ้าแดง 3 เม็ด\n- น้ำปลา 2 ช้อนโต๊ะ\n- น้ำตาลปึก 2 ช้อนโต๊ะ',
            '- เส้นจันท์ 200 กรัม\n- กุ้งสด 150 กรัม\n- เต้าหู้ 100 กรัม\n- ไข่ไก่ 2 ฟอง\n- ถั่วงอก 100 กรัม\n- กุ้ยช่าย 50 กรัม\n- น้ำมันพืช 3 ช้อนโต๊ะ\n- น้ำมะขามเปียก 3 ช้อนโต๊ะ\n- น้ำปลา 2 ช้อนโต๊ะ\n- น้ำตาลปึก 3 ช้อนโต๊ะ',
            '- ไข่ไก่ 3 ฟอง\n- น้ำปลา 1 ช้อนชา\n- ต้นหอม 2 ต้น\n- ผักชี 1 ต้น\n- น้ำมันหมู 3 ช้อนโต๊ะ',
            '- ข้าวสวย 3 ถ้วย\n- กุ้งสด 200 กรัม\n- ไข่ไก่ 2 ฟอง\n- หอมใหญ่ 1 หัว\n- แครอท 1 ผล\n- ซีอิ๊วขาว 2 ช้อนโต๊ะ\n- น้ำมันพืช 2 ช้อนโต๊ะ',
            '- วุ้นเส้น 150 กรัม\n- กุ้งสด 150 กรัม\n- หมูสับ 100 กรัม\n- ปลาหมึก 100 กรัม\n- มะนาว 3 ผล\n- ผักชี 3 ต้น\n- น้ำปลา 3 ช้อนโต๊ะ\n- น้ำตาลปึก 2 ช้อนโต๊ะ\n- พริกขี้หนู 6 เม็ด',
            '- เนื้อหมูสับ 300 กรัม\n- ข้าวคั่ว 3 ช้อนโต๊ะ\n- พริกแห้ง 8 เม็ด\n- หอมแดง 5 หัว\n- ใบสะระแหน่ 1 ถ้วย\n- ผักชี 3 ต้น\n- น้ำปลา 4 ช้อนโต๊ะ\n- น้ำมะนาว 3 ช้อนโต๊ะ',
            '- ข้าวเหนียว 2 ถ้วย\n- มะม่วงสุก 2 ผล\n- กะทิ 1 ถ้วย\n- น้ำตาลปึก 3 ช้อนโต๊ะ\n- เกลือ 1/2 ช้อนชา',
            '- เต้าหู้อ่อน 200 กรัม\n- หมูสับ 100 กรัม\n- ต้นหอม 3 ต้น\n- ผักชี 2 ต้น\n- น้ำซุปกระดูก 4 ถ้วย\n- ซีอิ๊วขาว 1 ช้อนโต๊ะ',
            '- ผักบุ้ง 300 กรัม\n- หมูหั่นฝอย 100 กรัม\n- พริกแกง 2 ช้อนโต๊ะ\n- กระเทียม 4 กลีบ\n- น้ำปลา 2 ช้อนโต๊ะ\n- น้ำมันพืช 2 ช้อนโต๊ะ',
            '- เนื้อไก่ 300 กรัม\n- กะทิ 2 ถ้วย\n- ข่า 5 แว่น\n- ตะไคร้ 3 ต้น\n- ใบมะกรูด 5 ใบ\n- เห็ดฟาง 100 กรัม\n- น้ำปลา 2 ช้อนโต๊ะ\n- น้ำตาลปึก 1 ช้อนโต๊ะ\n- น้ำมะนาว 2 ช้อนโต๊ะ',
            '- เนื้อหมูหั่นบาง 250 กรัม\n- ซีอิ๊วดำ 2 ช้อนโต๊ะ\n- ซีอิ๊วขาว 1 ช้อนโต๊ะ\n- กระเทียม 4 กลีบ\n- คะน้า 200 กรัม\n- น้ำมันพืช 2 ช้อนโต๊ะ',
            '- เนื้อวัวย่าง 200 กรัม\n- มะนาว 3 ผล\n- น้ำปลา 3 ช้อนโต๊ะ\n- น้ำตาลปึก 2 ช้อนโต๊ะ\n- พริกขี้หนู 5 เม็ด\n- หอมแดง 3 หัว\n- ใบสะระแหน่ 1 ถ้วย\n- ผักชี 3 ต้น'
        ],
        'method': [
            'โขลกกระเทียมและพริกให้ละเอียด ผัดในน้ำมันร้อนจนหอม ใส่หมูสับผัดจนสุก ปรุงรสด้วยน้ำปลาและน้ำตาล ใส่ใบกะเพราผัดให้เข้ากัน',
            'ต้มน้ำให้เดือด ใส่ตะไคร้ ข่า ใบมะกรูด พริกขี้หนูโขลก ต้มให้เดือดอีกครั้ง ใส่กุ้งและเห็ดฟาง ปรุงรสด้วยน้ำปลา ยกลงจากเตา ใส่น้ำมะนาว',
            'โขลกพริก กระเทียม ถั่วลิสง กุ้งแห้งให้หยาบ ใส่มะละกอ มะเขือเทศ ถั่วฝักยาว ตำให้เข้ากัน ปรุงรสด้วยน้ำปลา น้ำตาลปึก น้ำมะนาว ชิมรสให้เปรี้ยวหวานเค็ม',
            'คั่วน้ำพริกแกงเขียวหวานกับหัวกะทิให้หอม ใส่เนื้อไก่ผัดให้เข้ากัน เติมกะทิที่เหลือ ต้มให้เดือด ใส่มะเขือเปราะ ปรุงรสด้วยน้ำปลาและน้ำตาลปึก ใส่ใบโหระพาและพริกชี้ฟ้า',
            'แช่เส้นจันท์ให้นุ่ม ตั้งกะทะใส่น้ำมัน ผัดกุ้งและเต้าหู้ ใส่ไข่คนให้เข้ากัน ใส่เส้นจันท์และน้ำซอส ผัดให้เข้ากัน ใส่ถั่วงอกและกุ้ยช่าย โรยถั่วลิสงบด',
            'ตอกไข่ใส่ชาม ใส่น้ำปลา ตีให้เข้ากัน ใส่ต้นหอมและผักชีซอย ตั้งกะทะใส่น้ำมัน พอร้อนเทไข่ลงทอดจนฟูเหลืองทั้งสองด้าน',
            'ตั้งกะทะใส่น้ำมัน ผัดกระเทียมและหอมใหญ่ให้หอม ใส่กุ้งผัดจนสุก ใส่ไข่คนให้เข้ากัน ใส่ข้าวและแครอทผัดให้เข้ากัน ปรุงรสด้วยซีอิ๊ว',
            'แช่วุ้นเส้นให้นุ่ม ลวกกุ้ง หมู และปลาหมึกจนสุก ผสมน้ำยำจากมะนาว น้ำปลา น้ำตาลปึก พริกโขลก คลุกทุกอย่างให้เข้ากัน โรยผักชี',
            'คั่วข้าวให้เหลืองหอม โขลกให้หยาบ ย่างพริกแห้งให้หอม โขลกกับหอมแดงให้ละเอียด ผสมเนื้อหมูสับกับข้าวคั่ว พริกโขลก ปรุงรสด้วยน้ำปลาและน้ำมะนาว โรยผักชีและสะระแหน่',
            'นึ่งข้าวเหนียวให้สุก หั่นมะม่วงเป็นชิ้น ต้มกะทิกับน้ำตาลปึกและเกลือ คนจนละลาย เสิร์ฟข้าวเหนียวพร้อมมะม่วงและกะทิ',
            'ตั้งหม้อใส่น้ำซุป เดือดแล้วใส่หมูสับ ต้มจนสุกใส่เต้าหู้ ต้มให้นุ่ม ปรุงรสด้วยซีอิ๊วขาว ใส่ต้นหอมและผักชี โรยพริกไทยป่น',
            'ผัดกระเทียมกับน้ำพริกแกงให้หอม ใส่หมูผัดจนสุก ใส่ผักบุ้งผัดให้เข้ากัน ปรุงรสด้วยน้ำปลา น้ำตาล ซีอิ๊วดำ ผัดจนผักสุก',
            'ต้มกะทิจนเดือด ใส่ข่า ตะไคร้ ใบมะกรูด ใส่เนื้อไก่ต้มจนสุก ใส่เห็ดฟาง ปรุงรสด้วยน้ำปลา น้ำตาลปึก น้ำมะนาว',
            'หมักหมูกับซีอิ๊วดำ ซีอิ๊วขาว แป้ง ตั้งกะทะใส่น้ำมัน ผัดกระเทียมให้หอม ใส่หมูผัดจนสุก ใส่คะน้าผัดให้เข้ากัน',
            'หั่นเนื้อย่างเป็นชิ้นบางๆ ผสมน้ำยำจากมะนาว น้ำปลา น้ำตาลปึก พริกโขลก หอมแดง คลุกให้เข้ากัน โรยสะระแหน่และผักชี'
        ],
        # เพิ่มข้อมูลโภชนาการจริง (ต่อหนึ่งส่วน)
        'calories': [485, 120, 85, 320, 420, 280, 380, 95, 285, 380, 160, 145, 280, 390, 180],
        'protein': [28.5, 18.2, 3.8, 22.0, 20.5, 18.0, 19.5, 15.8, 24.0, 4.2, 12.5, 8.5, 20.0, 25.0, 22.5],
        'carbs': [12.5, 8.5, 18.2, 15.0, 52.0, 2.5, 48.0, 8.0, 8.5, 85.0, 8.0, 12.0, 12.0, 15.0, 8.0],
        'fat': [32.0, 2.5, 1.2, 22.0, 18.5, 22.0, 15.5, 2.8, 18.0, 8.5, 8.5, 6.5, 18.0, 24.5, 8.5],
        'fiber': [2.8, 1.5, 4.2, 3.5, 3.8, 1.2, 2.5, 1.8, 2.2, 2.5, 2.0, 3.8, 2.5, 3.2, 2.8],
        'vitamin_a': [125, 85, 580, 450, 180, 420, 320, 95, 85, 180, 125, 850, 280, 180, 125],
        'vitamin_c': [15, 8, 85, 25, 12, 8, 15, 18, 12, 8, 8, 95, 20, 35, 25],
        'vitamin_b1': [0.28, 0.12, 0.08, 0.15, 0.25, 0.18, 0.22, 0.08, 0.35, 0.12, 0.15, 0.08, 0.18, 0.25, 0.18],
        'vitamin_b2': [0.32, 0.15, 0.12, 0.22, 0.28, 0.35, 0.25, 0.12, 0.28, 0.08, 0.18, 0.12, 0.22, 0.28, 0.25],
        'calcium': [85, 120, 95, 85, 125, 125, 95, 85, 125, 85, 350, 185, 120, 85, 95],
        'iron': [3.2, 2.8, 1.8, 2.5, 3.5, 2.8, 2.5, 2.2, 4.5, 1.2, 2.8, 2.5, 2.8, 3.8, 3.2],
        'potassium': [485, 380, 450, 420, 385, 280, 320, 285, 520, 220, 285, 485, 450, 385, 485],
        'sodium': [980, 1250, 1180, 850, 1420, 520, 980, 1180, 1520, 85, 680, 850, 780, 1250, 1380]
    }
    
    return pd.DataFrame(data)

def display_recipe_with_nutrition(recipe_data: Dict, nutrition_data: Dict = None):
    """แสดงสูตรอาหารพร้อมข้อมูลโภชนาการ"""
    
    # แสดงในรูปแบบการ์ด
    st.markdown(f"""
    <div class="recipe-card">
        <h3>🍽️ {recipe_data['name']}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # แท็บต่างๆ
    tab1, tab2, tab3 = st.tabs(["📝 สูตรอาหาร", "📊 โภชนาการ", "💡 คำแนะนำ"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 🥬 วัตถุดิบ")
            ingredients = recipe_data['ingredient'].split('\n')
            for ingredient in ingredients:
                if ingredient.strip():
                    st.markdown(f"• {ingredient.strip().lstrip('- ')}")
        
        with col2:
            st.markdown("#### 👨‍🍳 วิธีทำ")
            method_steps = recipe_data['method'].split('. ')
            for i, step in enumerate(method_steps, 1):
                if step.strip():
                    st.markdown(f"**{i}.** {step.strip()}")
    
    with tab2:
        if nutrition_data:
            display_nutrition_details(nutrition_data, recipe_data['name'])
        else:
            st.info("ไม่มีข้อมูลโภชนาการ")
    
    with tab3:
        if nutrition_data:
            display_health_recommendations(nutrition_data, recipe_data['name'])

def display_nutrition_details(nutrition_data: Dict, recipe_name: str):
    """แสดงรายละเอียดโภชนาการ"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="nutrition-highlight">
            <h4>🔥 พลังงานและสารอาหารหลัก</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="nutrition-metric">
            <strong>แคลอรี่:</strong> {nutrition_data.get('calories', 0):.1f} kcal
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="nutrition-metric">
            <strong>โปรตีน:</strong> {nutrition_data.get('protein', 0):.1f} g
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="nutrition-metric">
            <strong>คาร์โบไฮเดรต:</strong> {nutrition_data.get('carbs', 0):.1f} g
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="nutrition-metric">
            <strong>ไขมัน:</strong> {nutrition_data.get('fat', 0):.1f} g
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="nutrition-metric">
            <strong>ใยอาหาร:</strong> {nutrition_data.get('fiber', 0):.1f} g
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="nutrition-highlight">
            <h4>💊 วิตามิน</h4>
        </div>
        """, unsafe_allow_html=True)
        
        vitamins = [
            ('วิตามิน A', 'vitamin_a', 'IU'),
            ('วิตามิน C', 'vitamin_c', 'mg'),
            ('วิตามิน B1', 'vitamin_b1', 'mg'),
            ('วิตามิน B2', 'vitamin_b2', 'mg')
        ]
        
        for name, key, unit in vitamins:
            value = nutrition_data.get(key, 0)
            st.markdown(f"""
            <div class="nutrition-metric">
                <strong>{name}:</strong> {value:.2f} {unit}
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="nutrition-highlight">
            <h4>⚡ แร่ธาตุ</h4>
        </div>
        """, unsafe_allow_html=True)
        
        minerals = [
            ('แคลเซียม', 'calcium', 'mg'),
            ('เหล็ก', 'iron', 'mg'),
            ('โปแตสเซียม', 'potassium', 'mg'),
            ('โซเดียม', 'sodium', 'mg')
        ]
        
        for name, key, unit in minerals:
            value = nutrition_data.get(key, 0)
            st.markdown(f"""
            <div class="nutrition-metric">
                <strong>{name}:</strong> {value:.1f} {unit}
            </div>
            """, unsafe_allow_html=True)
    
    # กราฟโภชนาการ
    create_nutrition_charts(nutrition_data)

def create_nutrition_charts(nutrition_data: Dict):
    """สร้างกราฟแสดงข้อมูลโภชนาการ"""
    
    # กราฟวงกลมแสดงสัดส่วนแมโครนิวเทรียนต์
    macro_labels = ['โปรตีน', 'คาร์โบไฮเดรต', 'ไขมัน']
    macro_values = [
        nutrition_data.get('protein', 0) * 4,  # 1g = 4 kcal
        nutrition_data.get('carbs', 0) * 4,    # 1g = 4 kcal
        nutrition_data.get('fat', 0) * 9       # 1g = 9 kcal
    ]
    
    fig_macro = go.Figure(data=[go.Pie(
        labels=macro_labels,
        values=macro_values,
        hole=.3,
        marker_colors=['#ff9999', '#66b3ff', '#99ff99']
    )])
    
    fig_macro.update_layout(
        title="🥧 สัดส่วนแมโครนิวเทรียนต์",
        title_x=0.5,
        font=dict(family="Sarabun, sans-serif"),
        height=400
    )
    
    st.plotly_chart(fig_macro, use_container_width=True)
    
    # กราฟแท่งแสดงวิตามินและแร่ธาตุ
    vitamins_minerals = {
        'วิตามิน A (IU)': nutrition_data.get('vitamin_a', 0),
        'วิตามิน C (mg)': nutrition_data.get('vitamin_c', 0),
        'แคลเซียม (mg)': nutrition_data.get('calcium', 0),
        'เหล็ก (mg)': nutrition_data.get('iron', 0),
        'โปแตสเซียม (mg)': nutrition_data.get('potassium', 0)
    }
    
    fig_vitamins = go.Figure(data=[
        go.Bar(
            x=list(vitamins_minerals.keys()),
            y=list(vitamins_minerals.values()),
            marker_color=['#ffb366', '#66ffb2', '#b366ff', '#ff66b2', '#66b2ff']
        )
    ])
    
    fig_vitamins.update_layout(
        title="📊 วิตามินและแร่ธาตุ",
        title_x=0.5,
        xaxis_title="สารอาหาร",
        yaxis_title="ปริมาณ",
        font=dict(family="Sarabun, sans-serif"),
        height=400
    )
    
    st.plotly_chart(fig_vitamins, use_container_width=True)

def display_health_recommendations(nutrition_data: Dict, recipe_name: str):
    """แสดงคำแนะนำเพื่อสุขภาพ"""
    
    st.markdown("#### 💡 คำแนะนำเพื่อสุขภาพ")
    
    # วิเคราะห์ข้อมูลโภชนาการ
    calories = nutrition_data.get('calories', 0)
    protein = nutrition_data.get('protein', 0)
    sodium = nutrition_data.get('sodium', 0)
    fiber = nutrition_data.get('fiber', 0)
    vitamin_c = nutrition_data.get('vitamin_c', 0)
    
    recommendations = []
    warnings = []
    
    # คำแนะนำตามค่าโภชนาการ
    if calories < 200:
        recommendations.append("✅ แคลอรี่ต่ำ เหมาะสำหรับผู้ที่ต้องการควบคุมน้ำหนัก")
    elif calories > 500:
        warnings.append("⚠️ แคลอรี่สูง ควรทานในปริมาณพอดีและออกกำลังกาย")
    
    if protein >= 20:
        recommendations.append("✅ โปรตีนสูง เหมาะสำหรับผู้ที่ต้องการเสริมสร้างกล้ามเนื้อ")
    elif protein < 10:
        recommendations.append("💡 ควรทานร่วมกับอาหารที่มีโปรตีนสูง เช่น ไข่ เต้าหู้")
    
    if sodium > 1200:
        warnings.append("⚠️ โซเดียมสูง ไม่เหมาะสำหรับผู้ป่วยความดันโลหิตสูงและโรคไต")
    elif sodium < 600:
        recommendations.append("✅ โซเดียมต่ำ เหมาะสำหรับผู้ป่วยความดันโลหิตสูง")
    
    if fiber >= 3:
        recommendations.append("✅ ใยอาหารดี ช่วยระบบย่อยอาหารและลดคอเลสเตอรอล")
    
    if vitamin_c >= 20:
        recommendations.append("✅ วิตามินซีสูง ช่วยเสริมภูมิคุ้มกันและต้านอนุมูลอิสระ")
    
    # แสดงผล
    if recommendations:
        st.markdown("##### ✅ ข้อดีของเมนูนี้")
        for rec in recommendations:
            st.markdown(f"- {rec}")
    
    if warnings:
        st.markdown("##### ⚠️ ข้อควรระวัง")
        for warning in warnings:
            st.markdown(f"- {warning}")
    
    # คำแนะนำทั่วไป
    st.markdown("##### 🔧 คำแนะนำการปรับปรุง")
    general_tips = [
        "🥗 ทานร่วมกับผักใบเขียวเพื่อเพิ่มใยอาหารและวิตามิน",
        "🍊 ทานผลไม้หลังอาหารเพื่อช่วยดูดซึมธาตุเหล็ก",
        "💧 ดื่มน้ำเปล่าอย่างน้อย 8 แก้วต่อวัน",
        "🚶‍♀️ ออกกำลังกายสม่ำเสมอเพื่อเผาผลาญแคลอรี่"
    ]
    
    for tip in general_tips:
        st.markdown(f"- {tip}")

def main():
    """ฟังก์ชันหลักของแอปพลิเคชัน"""
    
    # หัวข้อหลัก
    st.markdown('<h1 class="main-title">🍲 Thai Food Recipe Chatbot</h1>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; font-size: 1.2em; margin-bottom: 2rem;">🤖 ผู้ช่วยอัจฉริยะสำหรับอาหารไทยและโภชนาการ</div>', unsafe_allow_html=True)
    
    # เริ่มต้นระบบ
    with st.spinner("🔄 กำลังเริ่มต้นระบบ AI..."):
        nutrition_api, search_engine, chatbot, data = initialize_systems()
    
    # แสดงสถิติ
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📖 สูตรอาหาร", len(data))
    with col2:
        st.metric("🧮 AI Chatbot", "✅ พร้อม")
    with col3:
        st.metric("📊 ข้อมูลโภชนาการ", "✅ ครบถ้วน")
    with col4:
        st.metric("🔍 ค้นหาอัจฉริยะ", "✅ ใช้งานได้")
    
    # แถบข้าง
    with st.sidebar:
        st.title("⚙️ การตั้งค่า")
        
        st.markdown("### 🎯 โหมดการค้นหา")
        search_mode = st.selectbox(
            "เลือกโหมด",
            ["ทั่วไป", "ตามโภชนาการ", "ตามกลุ่มผู้ป่วย"],
            help="เลือกโหมดการค้นหาที่เหมาะสม"
        )
        
        st.markdown("### 📈 การแสดงผล")
        show_charts = st.checkbox("แสดงกราฟโภชนาการ", value=True)
        show_recommendations = st.checkbox("แสดงคำแนะนำสุขภาพ", value=True)
        
        st.markdown("### 🔧 ตัวเลือกขั้นสูง")
        max_results = st.slider("จำนวนผลลัพธ์สูงสุด", 1, 10, 5)
        
        if st.button("🔄 รีเซ็ตการสนทนา"):
            st.session_state.messages = []
            st.rerun()
    
    # แนะนำการใช้งาน
    st.markdown("#### 💡 ตัวอย่างคำถามที่สามารถถามได้:")
    
    example_cols = st.columns(3)
    
    with example_cols[0]:
        st.markdown("**🍽️ เกี่ยวกับเมนู**")
        menu_examples = [
            "ผัดกะเพราทำยังไง",
            "วิธีทำต้มยำกุ้ง",
            "เมนูไทยอะไรอร่อยบ้าง"
        ]
        for example in menu_examples:
            if st.button(f"💬 {example}", key=f"menu_{example}"):
                st.session_state.example_query = example
    
    with example_cols[1]:
        st.markdown("**📊 เกี่ยวกับโภชนาการ**")
        nutrition_examples = [
            "อาหารแคลอรี่ต่ำมีอะไรบ้าง",
            "เมนูโปรตีนสูงแนะนำอะไร",
            "อาหารไขมันต่ำมีอะไร"
        ]
        for example in nutrition_examples:
            if st.button(f"📈 {example}", key=f"nutrition_{example}"):
                st.session_state.example_query = example
    
    with example_cols[2]:
        st.markdown("**🏥 เกี่ยวกับสุขภาพ**")
        health_examples = [
            "อาหารสำหรับผู้ป่วยเบาหวาน",
            "เมนูเหมาะกับความดันสูง",
            "อาหารบำรุงกระดูก"
        ]
        for example in health_examples:
            if st.button(f"🏥 {example}", key=f"health_{example}"):
                st.session_state.example_query = example
    
    # เริ่มต้น session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "example_query" not in st.session_state:
        st.session_state.example_query = ""
    
    # แสดงประวัติการสนทนา
    if st.session_state.messages:
        st.markdown("#### 💬 ประวัติการสนทนา")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant" and "recipe_data" in message:
                    # แสดงข้อมูลสูตรอาหาร
                    display_recipe_with_nutrition(
                        message["recipe_data"], 
                        message.get("nutrition_data")
                    )
                elif message["role"] == "assistant" and "recommendations" in message:
                    # แสดงคำแนะนำ
                    st.markdown(message["content"])
                    for rec in message["recommendations"]:
                        st.markdown(f"- {rec}")
                else:
                    st.markdown(message["content"])
    
    # ช่องแชท
    example_query = st.session_state.get('example_query', '')
    if prompt := st.chat_input("🔍 ถามเกี่ยวกับอาหารไทย โภชนาการ หรือสุขภาพ..."):
        query = prompt
        st.session_state.example_query = ""
    elif example_query:
        query = example_query
        st.session_state.example_query = ""
    else:
        return
    
    # เพิ่มข้อความของผู้ใช้
    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.chat_message("user"):
        st.markdown(query)
    
    # ประมวลผลและตอบกลับ
    with st.chat_message("assistant"):
        with st.spinner("🤔 กำลังคิด..."):
            response = chatbot.process_query(query, search_mode, max_results)
            
            if response["type"] == "recipe":
                # แสดงสูตรอาหาร
                display_recipe_with_nutrition(
                    response["recipe_data"], 
                    response.get("nutrition_data")
                )
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "recipe_data": response["recipe_data"],
                    "nutrition_data": response.get("nutrition_data"),
                    "content": f"พบสูตร: {response['recipe_data']['name']}"
                })
                
            elif response["type"] == "recommendations":
                # แสดงคำแนะนำ
                st.markdown(response["content"])
                
                if response.get("recommendations"):
                    st.markdown("#### 📋 รายการแนะนำ:")
                    for i, rec in enumerate(response["recommendations"], 1):
                        st.markdown(f"**{i}. {rec['name']}**")
                        if 'reason' in rec:
                            st.markdown(f"   ↳ {rec['reason']}")
                        
                        # แสดงข้อมูลโภชนาการสั้นๆ
                        if 'nutrition' in rec:
                            nutrition = rec['nutrition']
                            st.markdown(f"   📊 แคลอรี่: {nutrition.get('calories', 0):.0f} kcal | "
                                      f"โปรตีน: {nutrition.get('protein', 0):.1f}g | "
                                      f"ไขมัน: {nutrition.get('fat', 0):.1f}g")
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["content"],
                    "recommendations": response.get("recommendations", [])
                })
                
            else:
                # ตอบกลับทั่วไป
                st.markdown(response["content"])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["content"]
                })

if __name__ == "__main__":
    main()
