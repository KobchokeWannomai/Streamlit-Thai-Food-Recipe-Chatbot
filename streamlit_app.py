import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import pickle
import re
import requests
import time
import json
from nutrition_analyzer import NutritionAnalyzer
from config import Config

# การกำหนดค่าหน้าเว็บ
st.set_page_config(
    page_title="Thai Food Recipe Chatbot",
    page_icon="🍲",
    layout="wide"
)

# JavaScript สำหรับ auto-scroll และปุ่มเลื่อน
scroll_js = """
<script>
function scrollToBottom() {
    window.scrollTo(0, document.body.scrollHeight);
}

function scrollToLatestMessage() {
    const messages = document.querySelectorAll('[data-testid="stChatMessage"]');
    if (messages.length > 0) {
        const lastMessage = messages[messages.length - 1];
        lastMessage.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

// Auto-scroll หลังจาก DOM อัปเดต
const observer = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutation) {
        if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
            setTimeout(scrollToLatestMessage, 100);
        }
    });
});

observer.observe(document.body, {
    childList: true,
    subtree: true
});

// เก็บตำแหน่งก่อนเปิด expander
let scrollPosition = 0;
function saveScrollPosition() {
    scrollPosition = window.pageYOffset;
}

function restoreScrollPosition() {
    window.scrollTo(0, scrollPosition);
}
</script>
"""

# ตั้งค่าฟอนต์ไทยและ CSS
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@400;700&display=swap');
    html, body, [class*="st-"] {{
        font-family: 'Sarabun', sans-serif !important;
    }}
    .nutrition-card {{
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 12px;
        border-left: 5px solid #4CAF50;
        margin: 15px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        width: 100%;
        box-sizing: border-box;
    }}
    .nutrition-item {{
        display: inline-block;
        margin: 5px 10px;
        padding: 8px 15px;
        background-color: #e8f5e8;
        border-radius: 20px;
        font-size: 0.9em;
        font-weight: 500;
    }}
    .vitamin-mineral {{
        display: inline;
        color: #555;
        margin-top: 10px;
        line-height: 1.8;
    }}
    .vitamin-mineral-label {{
        display: inline;
        font-weight: 600;
        margin-right: 10px;
    }}
    .recipe-card {{
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }}
    .recipe-title {{
        font-size: 1.8em;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 15px;
    }}
    .section-title {{
        font-size: 1.2em;
        font-weight: 600;
        color: #34495e;
        margin: 15px 0 10px 0;
        padding-bottom: 5px;
        border-bottom: 2px solid #e0e0e0;
    }}
    .status-indicator {{
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }}
    .status-connected {{
        background-color: #4CAF50;
    }}
    .status-disconnected {{
        background-color: #f44336;
    }}
    .status-testing {{
        background-color: #ff9800;
    }}
    .scroll-to-bottom {{
        position: fixed;
        bottom: 100px;
        right: 20px;
        z-index: 999;
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        cursor: pointer;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        font-size: 20px;
    }}
    .scroll-to-bottom:hover {{
        background-color: #45a049;
    }}
    /* ปรับปรุงการแสดงผลรายการ */
    ul, ol {{
        margin-left: 20px;
        line-height: 1.8;
    }}
    /* ปรับปรุงการแสดงผล metric */
    [data-testid="metric-container"] {{
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}
</style>
{scroll_js}
""", unsafe_allow_html=True)

# เส้นทางของไฟล์
DATA_PATH = "thai_food_processed.csv"
EMBEDDINGS_PATH = "embeddings.pkl"
MODEL_PATH = "model"

class APIManager:
    """จัดการการเชื่อมต่อ API ต่างๆ"""
    
    def __init__(self):
        self.usda_status = False
        self.nutritionix_status = False
        self.external_recipe_status = False
        
    def test_usda_connection(self, api_key):
        """ทดสอบการเชื่อมต่อ USDA API"""
        if not api_key or api_key == "your_usda_api_key_here":
            return False
            
        try:
            url = "https://api.nal.usda.gov/fdc/v1/foods/search"
            params = {
                "api_key": api_key,
                "query": "apple",
                "pageSize": 1
            }
            response = requests.get(url, params=params, timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def test_nutritionix_connection(self, app_id, api_key):
        """ทดสอบการเชื่อมต่อ Nutritionix API"""
        if not app_id or not api_key or app_id == "your_nutritionix_app_id_here":
            return False
            
        try:
            url = "https://trackapi.nutritionix.com/v2/search/instant"
            headers = {
                'x-app-id': app_id,
                'x-app-key': api_key,
                'Content-Type': 'application/json'
            }
            params = {'query': 'apple'}
            response = requests.get(url, headers=headers, params=params, timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def get_external_recipe_data(self, recipe_name):
        """ดึงข้อมูลสูตรอาหารจาก API ภายนอก (จำลอง)"""
        # จำลองการดึงข้อมูลจาก API ภายนอก
        # ในการใช้งานจริง ให้เชื่อมต่อกับ API จริง
        external_recipes = {
            "ไข่เจียว": {
                "ingredients": [
                    {"name": "ไข่ไก่", "amount": 2, "unit": "ฟอง"},
                    {"name": "น้ำมันพืช", "amount": 3, "unit": "ช้อนโต๊ะ", "cooking_only": True, "consumed": 0.3},
                    {"name": "เกลือ", "amount": 0.5, "unit": "ช้อนชา"},
                    {"name": "พริกไทย", "amount": 0.25, "unit": "ช้อนชา"}
                ],
                "nutrition_adjustments": {
                    "oil_absorption": 0.1  # ดูดซับน้ำมัน 10% จากปริมาณที่ใช้ทอด
                }
            },
            "ไข่ดาว": {
                "ingredients": [
                    {"name": "ไข่ไก่", "amount": 1, "unit": "ฟอง"},
                    {"name": "น้ำมันพืช", "amount": 2, "unit": "ช้อนโต๊ะ", "cooking_only": True, "consumed": 0.2}
                ],
                "nutrition_adjustments": {
                    "oil_absorption": 0.2
                }
            },
            "ผัดกะเพรา": {
                "ingredients": [
                    {"name": "หมูสับ", "amount": 200, "unit": "กรัม"},
                    {"name": "ใบกะเพรา", "amount": 1, "unit": "ถ้วย"},
                    {"name": "พริกขี้หนู", "amount": 5, "unit": "เม็ด"},
                    {"name": "กระเทียม", "amount": 5, "unit": "กลีบ"},
                    {"name": "น้ำปลา", "amount": 2, "unit": "ช้อนโต๊ะ"},
                    {"name": "น้ำตาล", "amount": 1, "unit": "ช้อนชา"},
                    {"name": "น้ำมันพืช", "amount": 2, "unit": "ช้อนโต๊ะ", "cooking_only": True, "consumed": 0.7}
                ],
                "nutrition_adjustments": {
                    "oil_absorption": 0.7
                }
            }
        }
        
        return external_recipes.get(recipe_name)

class EnhancedNutritionAnalyzer(NutritionAnalyzer):
    """ตัววิเคราะห์โภชนาการที่ปรับปรุงแล้ว"""
    
    def __init__(self, usda_api_key=None, use_api_data=False, api_manager=None):
        super().__init__(usda_api_key)
        self.use_api_data = use_api_data
        self.api_manager = api_manager or APIManager()
        
    def analyze_recipe_enhanced(self, recipe_name, ingredients_text, use_external_data=False):
        """วิเคราะห์โภชนาการแบบปรับปรุง"""
        # ใช้ข้อมูลจาก API ภายนอกถ้าเปิดใช้งาน
        if use_external_data and self.api_manager:
            external_data = self.api_manager.get_external_recipe_data(recipe_name)
            if external_data:
                return self._analyze_with_external_data(recipe_name, external_data)
        
        # วิเคราะห์แบบปกติ
        return self.analyze_recipe(recipe_name, ingredients_text)
    
    def _analyze_with_external_data(self, recipe_name, external_data):
        """วิเคราะห์โภชนาการด้วยข้อมูลจาก API ภายนอก"""
        nutrition_data = {}
        
        for ingredient_info in external_data['ingredients']:
            ingredient_name = ingredient_info['name']
            amount = ingredient_info['amount']
            unit = ingredient_info['unit']
            
            # ตรวจสอบว่าเป็นวัตถุดิบที่ใช้ในการทำอาหารเท่านั้นหรือไม่
            is_cooking_only = ingredient_info.get('cooking_only', False)
            consumed_ratio = ingredient_info.get('consumed', 1.0)
            
            # ดึงข้อมูลโภชนาการพื้นฐาน
            base_nutrition = self.get_ingredient_nutrition(ingredient_name)
            
            if base_nutrition:
                # คำนวณปริมาณที่บริโภคจริง
                if is_cooking_only:
                    # สำหรับวัตถุดิบที่ใช้ในการทำอาหาร เช่น น้ำมันทอด
                    effective_amount = amount * consumed_ratio
                else:
                    effective_amount = amount
                
                # แปลงหน่วยเป็นกรัม
                weight_grams = self.converter.convert_to_grams(effective_amount, unit, ingredient_name)
                multiplier = weight_grams / 100.0
                
                # สร้างข้อมูลโภชนาการที่ปรับแล้ว
                adjusted_nutrition = self._create_adjusted_nutrition(
                    ingredient_name, base_nutrition, multiplier, amount, unit, is_cooking_only
                )
                
                key = f"{ingredient_name} ({amount} {unit})"
                if is_cooking_only:
                    key += f" (ใช้ {consumed_ratio*100:.0f}%)"
                
                nutrition_data[key] = adjusted_nutrition
        
        # คำนวณโภชนาการรวม
        total_nutrition = self.calculate_total_nutrition(nutrition_data)
        
        return {
            'recipe_name': recipe_name,
            'total_nutrition': self._nutrition_to_dict(total_nutrition),
            'ingredients': [{'ingredient': k, 'nutrition': v} for k, v in nutrition_data.items()],
            'ingredient_count': len(nutrition_data),
            'enhanced': True
        }
    
    def _create_adjusted_nutrition(self, name, base_nutrition, multiplier, amount, unit, is_cooking_only):
        """สร้างข้อมูลโภชนาการที่ปรับแล้ว"""
        from nutrition_analyzer import NutritionInfo
        
        return NutritionInfo(
            name=name,
            calories=base_nutrition.calories * multiplier,
            protein=base_nutrition.protein * multiplier,
            carbs=base_nutrition.carbs * multiplier,
            fat=base_nutrition.fat * multiplier,
            fiber=base_nutrition.fiber * multiplier,
            sugar=base_nutrition.sugar * multiplier,
            sodium=base_nutrition.sodium * multiplier,
            vitamin_a=base_nutrition.vitamin_a * multiplier,
            vitamin_c=base_nutrition.vitamin_c * multiplier,
            vitamin_d=base_nutrition.vitamin_d * multiplier,
            vitamin_e=base_nutrition.vitamin_e * multiplier,
            vitamin_k=base_nutrition.vitamin_k * multiplier,
            vitamin_b1=base_nutrition.vitamin_b1 * multiplier,
            vitamin_b2=base_nutrition.vitamin_b2 * multiplier,
            vitamin_b6=base_nutrition.vitamin_b6 * multiplier,
            vitamin_b12=base_nutrition.vitamin_b12 * multiplier,
            folate=base_nutrition.folate * multiplier,
            niacin=base_nutrition.niacin * multiplier,
            calcium=base_nutrition.calcium * multiplier,
            iron=base_nutrition.iron * multiplier,
            magnesium=base_nutrition.magnesium * multiplier,
            phosphorus=base_nutrition.phosphorus * multiplier,
            potassium=base_nutrition.potassium * multiplier,
            zinc=base_nutrition.zinc * multiplier,
            serving_size=f"{amount} {unit}" + (" (ส่วนบริโภค)" if is_cooking_only else "")
        )
    
    def _nutrition_to_dict(self, nutrition):
        """แปลง NutritionInfo เป็น dict"""
        return {
            'calories': nutrition.calories,
            'protein': nutrition.protein,
            'carbs': nutrition.carbs,
            'fat': nutrition.fat,
            'fiber': nutrition.fiber,
            'vitamins': {
                'วิตามิน A': nutrition.vitamin_a,
                'วิตามิน C': nutrition.vitamin_c,
                'วิตามิน D': nutrition.vitamin_d,
                'วิตามิน E': nutrition.vitamin_e,
                'วิตามิน K': nutrition.vitamin_k,
                'วิตามิน B1': nutrition.vitamin_b1,
                'วิตามิน B2': nutrition.vitamin_b2,
                'วิตามิน B6': nutrition.vitamin_b6,
                'วิตามิน B12': nutrition.vitamin_b12,
            },
            'minerals': {
                'แคลเซียม': nutrition.calcium,
                'เหล็ก': nutrition.iron,
                'แมกนีเซียม': nutrition.magnesium,
                'ฟอสฟอรัส': nutrition.phosphorus,
                'โพแทสเซียม': nutrition.potassium,
                'สังกะสี': nutrition.zinc,
                'โซเดียม': nutrition.sodium,
            }
        }

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
def load_api_manager():
    """โหลด API Manager"""
    return APIManager()

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

def create_settings_sidebar():
    """สร้างแถบการตั้งค่า"""
    with st.sidebar:
        st.title("⚙️ การตั้งค่า")
        
        # ส่วน API Configuration
        st.subheader("🔌 การเชื่อมต่อ API")
        
        # USDA API
        st.markdown("**USDA FoodData Central API**")
        usda_enabled = st.checkbox("เปิดใช้งาน USDA API", key="usda_enabled")
        
        if usda_enabled:
            usda_api_key = st.text_input(
                "USDA API Key",
                type="password",
                value=st.session_state.get("usda_api_key", ""),
                key="usda_api_key_input"
            )
            
            if st.button("ทดสอบการเชื่อมต่อ USDA", key="test_usda"):
                api_manager = load_api_manager()
                with st.spinner("กำลังทดสอบ..."):
                    status = api_manager.test_usda_connection(usda_api_key)
                    if status:
                        st.success("✅ เชื่อมต่อสำเร็จ")
                        st.session_state.usda_api_key = usda_api_key
                        st.session_state.usda_status = True
                    else:
                        st.error("❌ การเชื่อมต่อล้มเหลว")
                        st.session_state.usda_status = False
            
            # แสดงสถานะ
            status_usda = st.session_state.get("usda_status", False)
            status_class = "status-connected" if status_usda else "status-disconnected"
            status_text = "เชื่อมต่อแล้ว" if status_usda else "ไม่ได้เชื่อมต่อ"
            st.markdown(f'<div><span class="status-indicator {status_class}"></span>{status_text}</div>', unsafe_allow_html=True)
        
        st.divider()
        
        # Nutritionix API
        st.markdown("**Nutritionix API**")
        nutritionix_enabled = st.checkbox("เปิดใช้งาน Nutritionix API", key="nutritionix_enabled")
        
        if nutritionix_enabled:
            nutritionix_app_id = st.text_input(
                "Nutritionix App ID",
                value=st.session_state.get("nutritionix_app_id", ""),
                key="nutritionix_app_id_input"
            )
            nutritionix_api_key = st.text_input(
                "Nutritionix API Key",
                type="password",
                value=st.session_state.get("nutritionix_api_key", ""),
                key="nutritionix_api_key_input"
            )
            
            if st.button("ทดสอบการเชื่อมต่อ Nutritionix", key="test_nutritionix"):
                api_manager = load_api_manager()
                with st.spinner("กำลังทดสอบ..."):
                    status = api_manager.test_nutritionix_connection(nutritionix_app_id, nutritionix_api_key)
                    if status:
                        st.success("✅ เชื่อมต่อสำเร็จ")
                        st.session_state.nutritionix_app_id = nutritionix_app_id
                        st.session_state.nutritionix_api_key = nutritionix_api_key
                        st.session_state.nutritionix_status = True
                    else:
                        st.error("❌ การเชื่อมต่อล้มเหลว")
                        st.session_state.nutritionix_status = False
            
            # แสดงสถานะ
            status_nutritionix = st.session_state.get("nutritionix_status", False)
            status_class = "status-connected" if status_nutritionix else "status-disconnected"
            status_text = "เชื่อมต่อแล้ว" if status_nutritionix else "ไม่ได้เชื่อมต่อ"
            st.markdown(f'<div><span class="status-indicator {status_class}"></span>{status_text}</div>', unsafe_allow_html=True)
        
        st.divider()
        
        # ตัวเลือกเพิ่มเติม
        st.subheader("🧪 ตัวเลือกขั้นสูง")
        
        use_external_recipe_data = st.checkbox(
            "ใช้ข้อมูลสูตรอาหารจาก API ภายนอก",
            help="ปรับปรุงการคำนวณโภชนาการด้วยข้อมูลเพิ่มเติม เช่น ปริมาณน้ำมันที่บริโภคจริงในการทอด",
            key="use_external_recipe_data"
        )
        
        accurate_cooking_calculation = st.checkbox(
            "คำนวณการบริโภควัตถุดิบในการทำอาหารอย่างแม่นยำ",
            help="ปรับปรุงการคำนวณสำหรับวัตถุดิบที่ใช้ในการปรุงอาหาร เช่น น้ำมันทอด",
            key="accurate_cooking_calculation"
        )
        
        enhanced_search = st.checkbox(
            "เพิ่มขอบเขตการค้นหา",
            help="ค้นหาอาหารในขอบเขตที่กว้างขึ้น รวมถึงวัตถุดิบและคำอธิบาย",
            value=True,
            key="enhanced_search"
        )
        
        st.divider()
        
        # สถานะระบบ
        st.subheader("📊 สถานะระบบ")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("สูตรอาหาร", len(load_data()))
        with col2:
            st.metric("ฐานข้อมูลโภชนาการ", "150+ วัตถุดิบ")
        
        # ข้อมูลการใช้งาน
        if "search_count" not in st.session_state:
            st.session_state.search_count = 0
        
        st.metric("การค้นหาในเซสชันนี้", st.session_state.search_count)
        
        st.divider()
        
        # ข้อมูลเพิ่มเติม
        with st.expander("ℹ️ เกี่ยวกับ API"):
            st.markdown("""
            **USDA FoodData Central API:**
            - ฟรี ไม่มีค่าใช้จ่าย
            - ข้อมูลโภชนาการที่เชื่อถือได้
            - สมัครได้ที่: https://fdc.nal.usda.gov/api-guide.html
            
            **Nutritionix API:**
            - Free Plan: 200 requests/วัน
            - ข้อมูลอาหารที่หลากหลาย
            - สมัครได้ที่: https://www.nutritionix.com/business/api
            """)
        
        return {
            "usda_enabled": usda_enabled and st.session_state.get("usda_status", False),
            "nutritionix_enabled": nutritionix_enabled and st.session_state.get("nutritionix_status", False),
            "usda_api_key": st.session_state.get("usda_api_key", ""),
            "nutritionix_app_id": st.session_state.get("nutritionix_app_id", ""),
            "nutritionix_api_key": st.session_state.get("nutritionix_api_key", ""),
            "use_external_recipe_data": use_external_recipe_data,
            "accurate_cooking_calculation": accurate_cooking_calculation,
            "enhanced_search": enhanced_search
        }

def format_ingredients(ingredients_text):
    """จัดรูปแบบรายการวัตถุดิบเพื่อการแสดงผลที่ดีขึ้น"""
    ingredients = ingredients_text.split('\n')
    formatted = "<ul style='line-height: 1.8;'>"
    for item in ingredients:
        if item.strip():
            # ลบเครื่องหมาย - ที่อยู่ด้านหน้า
            cleaned_item = item.strip()
            if cleaned_item.startswith('- '):
                cleaned_item = cleaned_item[2:]
            formatted += f"<li>{cleaned_item}</li>"
    formatted += "</ul>"
    return formatted

def format_cooking_method(method_text):
    """จัดรูปแบบวิธีการทำอาหารเพื่อการแสดงผลที่ดีขึ้น"""
    # ตรวจสอบว่ามีเลขขั้นตอนอยู่แล้วหรือไม่
    has_numbered_steps = bool(re.search(r'^\s*\d+\.', method_text, re.MULTILINE))
    
    # แยกบรรทัดและจัดการกับหัวข้อย่อย
    lines = method_text.split('\n')
    formatted = ""
    current_section = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # ตรวจสอบหัวข้อย่อย (ขึ้นต้นด้วย ## หรือ #)
        if line.startswith('##'):
            # ถ้ามีเนื้อหาในส่วนก่อนหน้า ให้จัดรูปแบบก่อน
            if current_section:
                formatted += format_section(current_section, has_numbered_steps)
                current_section = []
            # เพิ่มหัวข้อย่อย (ลบ ## ออกและใช้ขนาดเท่ากับหัวข้อหลัก)
            formatted += f"<div class='section-title' style='font-size: 1.0em; margin-top: 15px;'>{line.replace('##', '').strip()}</div>"
        elif line.startswith('#'):
            # ถ้ามีเนื้อหาในส่วนก่อนหน้า ให้จัดรูปแบบก่อน
            if current_section:
                formatted += format_section(current_section, has_numbered_steps)
                current_section = []
            # เพิ่มหัวข้อย่อย
            formatted += f"<div class='section-title' style='font-size: 1.0em; margin-top: 15px;'>{line.replace('#', '').strip()}</div>"
        elif line.startswith('**หมายเหตุ'):
            # ถ้ามีเนื้อหาในส่วนก่อนหน้า ให้จัดรูปแบบก่อน
            if current_section:
                formatted += format_section(current_section, has_numbered_steps)
                current_section = []
            # เพิ่มหมายเหตุ
            formatted += f"<div style='margin-top: 15px; padding: 10px; background-color: #f9f9f9; border-left: 3px solid #ffa500;'><strong>หมายเหตุ</strong> {line.replace('**หมายเหตุ**', '').replace('**หมายเหตุ', '').strip()}</div>"
        else:
            # เก็บเนื้อหาปกติ
            current_section.append(line)
    
    # จัดรูปแบบส่วนสุดท้าย
    if current_section:
        formatted += format_section(current_section, has_numbered_steps)
    
    return formatted

def format_section(lines, has_numbered_steps):
    """จัดรูปแบบส่วนของวิธีทำ"""
    if not lines:
        return ""
    
    # ถ้ามีเลขขั้นตอนอยู่แล้ว ให้แสดงเป็นลิสต์
    if has_numbered_steps:
        formatted = "<ol style='line-height: 1.8;'>"
        for line in lines:
            # ตรวจสอบว่าบรรทัดขึ้นต้นด้วยตัวเลขหรือไม่
            if re.match(r'^\d+\.', line):
                # ลบตัวเลขออกเพราะ <ol> จะใส่ให้อัตโนมัติ
                clean_line = re.sub(r'^\d+\.\s*', '', line)
                formatted += f"<li>{clean_line}</li>"
            else:
                # ถ้าไม่มีตัวเลข ให้เป็นส่วนต่อของ item ก่อนหน้า
                if formatted.endswith("</li>"):
                    formatted = formatted[:-5] + f" {line}</li>"
                else:
                    formatted += f"<li>{line}</li>"
        formatted += "</ol>"
    else:
        # ถ้าไม่มีเลขขั้นตอน ให้แสดงเป็นย่อหน้า
        # รวมประโยคที่แยกกันด้วยช่องว่างเดียว
        text = ' '.join(lines)
        # แบ่งเป็นประโยคตาม ๆ หรือ .
        sentences = re.split(r'(?<=[ๆ.])\s+', text)
        
        formatted = "<div style='line-height: 1.8; text-align: justify;'>"
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                formatted += sentence.strip()
                # เพิ่มช่องว่างหลังประโยค ยกเว้นประโยคสุดท้าย
                if i < len(sentences) - 1:
                    formatted += " "
        formatted += "</div>"
    
    return formatted

def display_nutrition_info(nutrition_data):
    """แสดงข้อมูลโภชนาการในรูปแบบที่สวยงาม"""
    if not nutrition_data:
        return
    
    total_nutrition = nutrition_data.get('total_nutrition', {})
    is_enhanced = nutrition_data.get('enhanced', False)
    
    st.markdown("### 🥗 ข้อมูลโภชนาการ (ต่อหนึ่งที่)")
    
    if is_enhanced:
        st.info("📈 ข้อมูลที่ปรับปรุงแล้วด้วย API ภายนอก - คำนวณปริมาณการบริโภคจริงแล้ว")
    
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
    
    # วิตามินและแร่ธาตุ - แสดงเฉพาะที่มีค่า
    vitamins = total_nutrition.get('vitamins', {})
    minerals = total_nutrition.get('minerals', {})
    
    if vitamins or minerals:
        nutrition_items = []
        
        # แสดงวิตามิน
        vitamin_units = {
            'วิตามิน A': 'mcg',
            'วิตามิน C': 'mg', 
            'วิตามิน D': 'mcg',
            'วิตามิน E': 'mg',
            'วิตามิน K': 'mcg',
            'วิตามิน B1': 'mg',
            'วิตามิน B2': 'mg',
            'วิตามิน B6': 'mg',
            'วิตามิน B12': 'mcg'
        }
        
        # แสดงเฉพาะวิตามินที่มีค่ามากกว่า 0.1
        for vitamin, amount in vitamins.items():
            if amount > 0.1:
                unit = vitamin_units.get(vitamin, 'mg')
                nutrition_items.append(f"<span class='nutrition-item'>{vitamin}: {amount:.1f} {unit}</span>")
        
        # แสดงแร่ธาตุ
        mineral_units = {
            'แคลเซียม': 'mg',
            'เหล็ก': 'mg',
            'แมกนีเซียม': 'mg',
            'ฟอสฟอรัส': 'mg',
            'โพแทสเซียม': 'mg',
            'สังกะสี': 'mg',
            'โซเดียม': 'mg'
        }
        
        # แสดงเฉพาะแร่ธาตุที่มีค่ามากกว่า 0.1
        for mineral, amount in minerals.items():
            if amount > 0.1:
                unit = mineral_units.get(mineral, 'mg')
                # ถ้าเป็นโซเดียม แสดงเป็นจำนวนเต็ม
                if mineral == 'โซเดียม':
                    nutrition_items.append(f"<span class='nutrition-item'>{mineral}: {amount:.0f} {unit}</span>")
                else:
                    nutrition_items.append(f"<span class='nutrition-item'>{mineral}: {amount:.1f} {unit}</span>")
        
        if nutrition_items:
            st.markdown(f"<div><span class='vitamin-mineral-label'>วิตามินและแร่ธาตุ:</span><span class='vitamin-mineral'>{''.join(nutrition_items)}</span></div>", unsafe_allow_html=True)
    
    # รายละเอียดวัตถุดิบ - เพิ่ม JavaScript สำหรับจดจำตำแหน่ง
    with st.expander("📋 รายละเอียดโภชนาการแต่ละวัตถุดิบ"):
        st.markdown("""
        <script>
        document.addEventListener('DOMContentLoaded', function() {
            // หาปุ่ม expander ทั้งหมด
            const expanders = document.querySelectorAll('[data-testid="stExpander"]');
            expanders.forEach(function(expander) {
                const button = expander.querySelector('button');
                if (button) {
                    button.addEventListener('click', function() {
                        setTimeout(function() {
                            if (expander.getAttribute('aria-expanded') === 'false') {
                                // ปิด expander - กลับไปตำแหน่งเดิม
                                restoreScrollPosition();
                            } else {
                                // เปิด expander - บันทึกตำแหน่ง
                                saveScrollPosition();
                            }
                        }, 100);
                    });
                }
            });
        });
        </script>
        """, unsafe_allow_html=True)
        
        for ingredient_info in nutrition_data.get('ingredients', []):
            # แสดงชื่อวัตถุดิบแบบเต็ม (ใช้ key ที่มีปริมาณ)
            ingredient_full_name = ingredient_info['ingredient']
            nutrition = ingredient_info['nutrition']
            
            # แสดงชื่อและขนาดที่ใช้
            st.write(f"**{ingredient_full_name}**")
            
            # แสดงข้อมูลพื้นฐาน
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.write(f"แคลอรี่: {nutrition.calories:.0f} kcal")
            with col2:
                st.write(f"โปรตีน: {nutrition.protein:.1f} g")
            with col3:
                st.write(f"คาร์โบไฮเดรต: {nutrition.carbs:.1f} g")
            with col4:
                st.write(f"ไขมัน: {nutrition.fat:.1f} g")
            
            # แสดงวิตามินและแร่ธาตุที่มีค่ามากกว่า 0.1
            vitamin_mineral_text = []
            
            # วิตามิน
            if hasattr(nutrition, 'vitamin_a') and nutrition.vitamin_a > 0.1:
                vitamin_mineral_text.append(f"วิตามิน A: {nutrition.vitamin_a:.1f} mcg")
            if hasattr(nutrition, 'vitamin_c') and nutrition.vitamin_c > 0.1:
                vitamin_mineral_text.append(f"วิตามิน C: {nutrition.vitamin_c:.1f} mg")
            if hasattr(nutrition, 'vitamin_d') and nutrition.vitamin_d > 0.1:
                vitamin_mineral_text.append(f"วิตามิน D: {nutrition.vitamin_d:.1f} mcg")
            if hasattr(nutrition, 'vitamin_e') and nutrition.vitamin_e > 0.1:
                vitamin_mineral_text.append(f"วิตามิน E: {nutrition.vitamin_e:.1f} mg")
            if hasattr(nutrition, 'vitamin_k') and nutrition.vitamin_k > 0.1:
                vitamin_mineral_text.append(f"วิตามิน K: {nutrition.vitamin_k:.1f} mcg")
            if hasattr(nutrition, 'vitamin_b1') and nutrition.vitamin_b1 > 0.1:
                vitamin_mineral_text.append(f"วิตามิน B1: {nutrition.vitamin_b1:.1f} mg")
            if hasattr(nutrition, 'vitamin_b6') and nutrition.vitamin_b6 > 0.1:
                vitamin_mineral_text.append(f"วิตามิน B6: {nutrition.vitamin_b6:.1f} mg")
            if hasattr(nutrition, 'vitamin_b12') and nutrition.vitamin_b12 > 0.1:
                vitamin_mineral_text.append(f"วิตามิน B12: {nutrition.vitamin_b12:.1f} mcg")
            
            # แร่ธาตุ
            if hasattr(nutrition, 'calcium') and nutrition.calcium > 0.1:
                vitamin_mineral_text.append(f"แคลเซียม: {nutrition.calcium:.1f} mg")
            if hasattr(nutrition, 'iron') and nutrition.iron > 0.1:
                vitamin_mineral_text.append(f"เหล็ก: {nutrition.iron:.1f} mg")
            if hasattr(nutrition, 'sodium') and nutrition.sodium > 0.1:
                vitamin_mineral_text.append(f"โซเดียม: {nutrition.sodium:.0f} mg")
            if hasattr(nutrition, 'potassium') and nutrition.potassium > 0.1:
                vitamin_mineral_text.append(f"โพแทสเซียม: {nutrition.potassium:.0f} mg")
            if hasattr(nutrition, 'zinc') and nutrition.zinc > 0.1:
                vitamin_mineral_text.append(f"สังกะสี: {nutrition.zinc:.1f} mg")
            if hasattr(nutrition, 'phosphorus') and nutrition.phosphorus > 0.1:
                vitamin_mineral_text.append(f"ฟอสฟอรัส: {nutrition.phosphorus:.0f} mg")
            if hasattr(nutrition, 'magnesium') and nutrition.magnesium > 0.1:
                vitamin_mineral_text.append(f"แมกนีเซียม: {nutrition.magnesium:.0f} mg")
            
            # แสดงใยอาหารถ้ามี
            if hasattr(nutrition, 'fiber') and nutrition.fiber > 0.1:
                vitamin_mineral_text.append(f"ใยอาหาร: {nutrition.fiber:.1f} g")
            
            if vitamin_mineral_text:
                st.write("สารอาหารอื่นๆ: " + ", ".join(vitamin_mineral_text))
            
            # แสดงหมายเหตุสำหรับวัตถุดิบที่มีโซเดียมสูง
            if hasattr(nutrition, 'sodium') and nutrition.sodium > 500:
                st.caption("⚠️ มีโซเดียมสูง")
                
            # แสดงหมายเหตุสำหรับวัตถุดิบที่ปรับปรุงแล้ว
            if is_enhanced and "(ใช้" in ingredient_full_name:
                st.caption("🔧 ปรับปรุงการคำนวณปริมาณการบริโภคแล้ว")
                
            st.divider()

def search_recipes_enhanced(query, model, data, embeddings, nutrition_analyzer, settings, top_k=5):
    """ค้นหาสูตรอาหารตามคำค้นหา - ปรับปรุงขอบเขตการค้นหา"""
    query_lower = query.lower()
    
    # ขยายคำค้นหาด้วยคำที่เกี่ยวข้อง
    query_expansions = {
        'ไข่': ['ไข่ไก่', 'ไข่เป็ด', 'ไข่ดาว', 'ไข่เจียว', 'ไข่ต้ม'],
        'หมู': ['เนื้อหมู', 'หมูสับ', 'หมูย่าง', 'หมูทอด'],
        'ไก่': ['เนื้อไก่', 'ไก่ย่าง', 'ไก่ทอด', 'ไก่ต้ม'],
        'กุ้ง': ['กุ้งนาง', 'กุ้งฝอย', 'กุ้งแห้ง'],
        'ผัด': ['ผัดไทย', 'ผัดกะเพรา', 'ผัดซีอิ๊ว'],
        'แกง': ['แกงเขียวหวาน', 'แกงเผ็ด', 'แกงส้ม', 'แกงมัสมั่น'],
        'ต้ม': ['ต้มยำ', 'ต้มข่า', 'ต้มจืด'],
        'ยำ': ['ยำวุ้นเส้น', 'ยำถั่วพู', 'ยำมะม่วง'],
        'ส้ม': ['ส้มตำ', 'ส้มตำไทย', 'ส้มตำปู'],
        'ลาบ': ['ลาบหมู', 'ลาบไก่', 'ลาบเนื้อ']
    }
    
    # ขยายคำค้นหาถ้าเปิดใช้งาน enhanced search
    expanded_query = query_lower
    if settings.get('enhanced_search', True):
        for key, expansions in query_expansions.items():
            if key in query_lower:
                expanded_query += ' ' + ' '.join(expansions)
    
    # ค้นหาแบบตรงตัวก่อน (exact match)
    exact_matches = []
    for idx, row in data.iterrows():
        recipe_name = row['name'].lower()
        ingredients = row['ingredient'].lower()
        method = row['method'].lower()
        
        # ตรวจสอบการตรงกันในหลายส่วน
        name_match = query_lower in recipe_name or recipe_name in query_lower
        ingredient_match = any(word in ingredients for word in query_lower.split())
        method_match = any(word in method for word in query_lower.split())
        
        if name_match:
            # ให้คะแนนสูงสุดสำหรับการตรงกันของชื่อ
            score = 1.0 if query_lower == recipe_name else 0.9
            exact_matches.append({'index': idx, 'score': score})
        elif ingredient_match and method_match:
            # คะแนนกลางสำหรับการตรงกันทั้งวัตถุดิบและวิธีทำ
            exact_matches.append({'index': idx, 'score': 0.7})
        elif ingredient_match:
            # คะแนนต่ำกว่าสำหรับการตรงกันเฉพาะวัตถุดิบ
            exact_matches.append({'index': idx, 'score': 0.5})
    
    # ค้นหาแบบ semantic search
    search_text = expanded_query if settings.get('enhanced_search', True) else query
    query_embedding = model.encode([search_text])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # รวมผลลัพธ์
    results = []
    used_indices = set()
    
    # เพิ่ม exact matches ก่อน
    for match in sorted(exact_matches, key=lambda x: x['score'], reverse=True):
        idx = match['index']
        if idx not in used_indices:
            recipe_name = data.iloc[idx]['name']
            ingredients = data.iloc[idx]['ingredient']
            
            # วิเคราะห์โภชนาการด้วยการตั้งค่าที่เลือก
            if hasattr(nutrition_analyzer, 'analyze_recipe_enhanced'):
                nutrition_data = nutrition_analyzer.analyze_recipe_enhanced(
                    recipe_name, ingredients, 
                    use_external_data=settings.get('use_external_recipe_data', False)
                )
            else:
                nutrition_data = nutrition_analyzer.analyze_recipe(recipe_name, ingredients)
            
            results.append({
                'name': recipe_name,
                'similarity': match['score'],
                'ingredients': ingredients,
                'method': data.iloc[idx]['method'],
                'nutrition': nutrition_data
            })
            used_indices.add(idx)
    
    # เพิ่มผลลัพธ์จาก semantic search
    threshold = 0.25 if settings.get('enhanced_search', True) else 0.3
    top_indices = np.argsort(-similarities)
    for idx in top_indices:
        if idx not in used_indices and len(results) < top_k:
            if similarities[idx] > threshold:
                recipe_name = data.iloc[idx]['name']
                ingredients = data.iloc[idx]['ingredient']
                
                # วิเคราะห์โภชนาการด้วยการตั้งค่าที่เลือก
                if hasattr(nutrition_analyzer, 'analyze_recipe_enhanced'):
                    nutrition_data = nutrition_analyzer.analyze_recipe_enhanced(
                        recipe_name, ingredients,
                        use_external_data=settings.get('use_external_recipe_data', False)
                    )
                else:
                    nutrition_data = nutrition_analyzer.analyze_recipe(recipe_name, ingredients)
                
                results.append({
                    'name': recipe_name,
                    'similarity': similarities[idx],
                    'ingredients': ingredients,
                    'method': data.iloc[idx]['method'],
                    'nutrition': nutrition_data
                })
                used_indices.add(idx)
    
    return results

def search_by_nutrition_criteria(data, nutrition_analyzer, criteria, settings):
    """ค้นหาสูตรอาหารตามเกณฑ์โภชนาการ"""
    results = []
    
    for _, row in data.iterrows():
        recipe_name = row['name']
        ingredients = row['ingredient']
        
        # วิเคราะห์โภชนาการด้วยการตั้งค่าที่เลือก
        if hasattr(nutrition_analyzer, 'analyze_recipe_enhanced'):
            nutrition_data = nutrition_analyzer.analyze_recipe_enhanced(
                recipe_name, ingredients,
                use_external_data=settings.get('use_external_recipe_data', False)
            )
        else:
            nutrition_data = nutrition_analyzer.analyze_recipe(recipe_name, ingredients)
        
        total_nutrition = nutrition_data.get('total_nutrition', {})
        
        # ตรวจสอบเกณฑ์
        match = True
        if 'max_calories' in criteria and total_nutrition.get('calories', 0) > criteria['max_calories']:
            match = False
        if 'min_calories' in criteria and total_nutrition.get('calories', 0) < criteria['min_calories']:
            match = False
        if 'min_protein' in criteria and total_nutrition.get('protein', 0) < criteria['min_protein']:
            match = False
        
        if match:
            results.append({
                'recipe_name': recipe_name,
                'calories': total_nutrition.get('calories', 0),
                'protein': total_nutrition.get('protein', 0),
                'carbs': total_nutrition.get('carbs', 0),
                'fat': total_nutrition.get('fat', 0),
                'fiber': total_nutrition.get('fiber', 0),
                'ingredients': ingredients,
                'method': row['method'],
                'nutrition': nutrition_data
            })
    
    # เรียงลำดับตามแคลอรี่
    results.sort(key=lambda x: x['calories'])
    return results

def detect_nutrition_search(query):
    """ตรวจจับว่าการค้นหาเป็นการค้นหาตามโภชนาการหรือไม่"""
    nutrition_keywords = [
        'แคลอรี่', 'แคลอรี', 'calorie', 'cal', 'kcal',
        'โปรตีน', 'protein',
        'คาร์โบ', 'คาร์โบไฮเดรต', 'carb', 'carbohydrate',
        'ไขมัน', 'fat',
        'ใยอาหาร', 'fiber',
        'ลดน้ำหนัก', 'diet', 'healthy', 'เฮลธ์ตี้',
        'โภชนาการ', 'nutrition',
        'วิตามิน', 'vitamin',
        'แร่ธาตุ', 'mineral',
        'ต่ำ', 'สูง', 'น้อย', 'เยอะ', 'มาก',
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
        r'น้อยกว่า.*?(\d+).*?แคลอรี่',
        r'แคลอรี.*?ไม่เกิน.*?(\d+)',
        r'ไม่เกิน.*?(\d+).*?แคลอรี'
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
        r'โปรตีน.*?เยอะ.*?(\d+)',
        r'โปรตีน.*?อย่างน้อย.*?(\d+)'
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
    elif 'เฮลธ์ตี้' in query_lower or 'healthy' in query_lower:
        criteria['max_calories'] = 350
    
    return criteria

def create_scroll_to_bottom_button():
    """สร้างปุ่มเลื่อนไปข้อความล่าสุด"""
    st.markdown("""
    <button class="scroll-to-bottom" onclick="scrollToLatestMessage()" title="เลื่อนไปข้อความล่าสุด">
        ↓
    </button>
    """, unsafe_allow_html=True)

def main():
    # สร้างแถบการตั้งค่า
    settings = create_settings_sidebar()
    
    # โหลดโมเดลและข้อมูล
    model = load_model()
    data = load_data()
    embeddings = get_embeddings(model, data)
    api_manager = load_api_manager()
    
    # สร้าง nutrition analyzer ที่ปรับปรุงแล้ว
    usda_api_key = settings.get('usda_api_key') if settings.get('usda_enabled') else None
    nutrition_analyzer = EnhancedNutritionAnalyzer(
        usda_api_key=usda_api_key,
        use_api_data=settings.get('usda_enabled') or settings.get('nutritionix_enabled'),
        api_manager=api_manager
    )
    
    # แอปหลัก
    st.title("🍲 Thai Food Recipe Chatbot")
    st.markdown("**ค้นหาสูตรอาหารไทยพร้อมข้อมูลโภชนาการ** - ถามเกี่ยวกับวิธีทำอาหารไทยหรือค้นหาตามโภชนาการได้เลย!")
    
    # ตัวอย่างการค้นหา
    example_queries = [
        "ไข่เจียว", "ต้มยำกุ้ง", "ผัดไทย", "ส้มตำ", "แกงเขียวหวาน",
        "เมนูแคลอรี่ไม่เกิน 300", "อาหารโปรตีนสูง", "เมนูลดน้ำหนัก",
        "อาหารทอดง่ายๆ", "แกงเผ็ด"
    ]
    
    with st.expander("💡 ตัวอย่างการค้นหา", expanded=False):
        cols = st.columns(5)
        for i, query in enumerate(example_queries):
            with cols[i % 5]:
                if st.button(query, key=f"example_{i}"):
                    st.session_state.example_query = query
    
    # เริ่มต้นประวัติการสนทนา
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # แสดงประวัติการสนทนา
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "recipe" in message:
                # แสดงสูตรพร้อมโภชนาการ
                recipe = message["recipe"]
                st.markdown(f'<div class="recipe-title">{recipe["name"]}</div>', unsafe_allow_html=True)
                
                # แสดงข้อมูลโภชนาการ
                display_nutrition_info(recipe['nutrition'])
                
                st.markdown('<div class="section-title">📝 วัตถุดิบ</div>', unsafe_allow_html=True)
                st.markdown(format_ingredients(recipe["ingredients"]), unsafe_allow_html=True)
                
                st.markdown('<div class="section-title">👩‍🍳 วิธีทำ</div>', unsafe_allow_html=True)
                st.markdown(format_cooking_method(recipe["method"]), unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            elif message["role"] == "assistant" and "nutrition_results" in message:
                # แสดงผลลัพธ์การค้นหาตามโภชนาการ
                results = message["nutrition_results"]
                st.markdown(f"พบ **{len(results)}** สูตรอาหารที่ตรงเกณฑ์:")
                
                for i, result in enumerate(results[:5], 1):
                    with st.expander(f"{i}. {result['recipe_name']} ({result['calories']:.0f} kcal)"):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.write(f"🔥 แคลอรี่: {result['calories']:.0f} kcal")
                        with col2:
                            st.write(f"🥩 โปรตีน: {result['protein']:.1f} g")
                        with col3:
                            st.write(f"🍞 คาร์โบไฮเดรต: {result['carbs']:.1f} g")
                        with col4:
                            st.write(f"🧈 ไขมัน: {result['fat']:.1f} g")
                        
                        if st.button(f"ดูสูตรอาหาร", key=f"recipe_{i}_{len(st.session_state.messages)}"):
                            st.write("### วัตถุดิบ")
                            st.write(result['ingredients'])
                            st.write("### วิธีทำ")
                            st.write(result['method'])
                
                if len(results) > 5:
                    st.markdown(f"*และอีก {len(results) - 5} สูตร...*")
            else:
                # แสดงข้อความธรรมดา
                st.markdown(message["content"])
    
    # สร้างปุ่มเลื่อนไปข้อความล่าสุด
    create_scroll_to_bottom_button()
    
    # ตรวจสอบตัวอย่างการค้นหาที่ถูกคลิก
    input_value = st.session_state.get("example_query", "")
    if input_value:
        st.session_state.pop("example_query", None)  # ลบค่าออกหลังใช้
    
    # ช่องใส่ข้อความสำหรับแชท
    if prompt := st.chat_input("ค้นหาสูตรอาหารไทย...", value=input_value):
        # นับการค้นหา
        st.session_state.search_count += 1
        
        # เพิ่มข้อความของผู้ใช้ลงในประวัติการสนทนา
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # แสดงข้อความของผู้ใช้
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # รับการตอบสนอง
        with st.chat_message("assistant"):
            with st.spinner("กำลังค้นหาและวิเคราะห์..."):
                # ตรวจสอบว่าเป็นการค้นหาตามโภชนาการหรือไม่
                if detect_nutrition_search(prompt):
                    # การค้นหาตามโภชนาการ
                    criteria = extract_nutrition_criteria_from_text(prompt)
                    
                    if criteria:
                        nutrition_results = search_by_nutrition_criteria(data, nutrition_analyzer, criteria, settings)
                        
                        if nutrition_results:
                            response = f"พบ {len(nutrition_results)} สูตรอาหารที่ตรงกับเกณฑ์โภชนาการที่ต้องการ"
                            st.markdown(response)
                            
                            # แสดงผลลัพธ์
                            st.markdown("### 🔍 ผลการค้นหา")
                            for i, result in enumerate(nutrition_results[:5], 1):
                                with st.expander(f"{i}. {result['recipe_name']} ({result['calories']:.0f} kcal)"):
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.write(f"🔥 แคลอรี่: {result['calories']:.0f} kcal")
                                    with col2:
                                        st.write(f"🥩 โปรตีน: {result['protein']:.1f} g")
                                    with col3:
                                        st.write(f"🍞 คาร์โบไฮเดรต: {result['carbs']:.1f} g")
                                    with col4:
                                        st.write(f"🧈 ไขมัน: {result['fat']:.1f} g")
                            
                            if len(nutrition_results) > 5:
                                st.markdown(f"*และอีก {len(nutrition_results) - 5} สูตร...*")
                            
                            # เพิ่มการตอบสนองของแอสซิสแตนต์ลงในประวัติการสนทนา
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response,
                                "nutrition_results": nutrition_results
                            })
                        else:
                            response = "ขออภัย ไม่พบสูตรอาหารที่ตรงกับเกณฑ์โภชนาการที่ต้องการ ลองปรับเกณฑ์ใหม่"
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        # ถ้าตรวจพบคีย์เวิร์ดโภชนาการแต่แยกเกณฑ์ไม่ได้
                        response = "กรุณาระบุเกณฑ์โภชนาการให้ชัดเจนขึ้น เช่น 'เมนูแคลอรี่ไม่เกิน 300' หรือ 'อาหารโปรตีนสูงมากกว่า 20 กรัม'"
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    # การค้นหาทั่วไป
                    results = search_recipes_enhanced(prompt, model, data, embeddings, nutrition_analyzer, settings)
                    
                    if results:
                        best_match = results[0]
                        
                        # ตรวจสอบว่ามีผลลัพธ์ที่ดี
                        threshold = 0.25 if settings.get('enhanced_search', True) else 0.3
                        if best_match["similarity"] > threshold:
                            response = f"พบสูตรอาหารที่คุณค้นหา: **{best_match['name']}**"
                            st.markdown(response)
                            
                            # แสดงสูตรพร้อมโภชนาการ
                            st.markdown(f'<div class="recipe-title">{best_match["name"]}</div>', unsafe_allow_html=True)
                            
                            # แสดงข้อมูลโภชนาการ
                            display_nutrition_info(best_match['nutrition'])
                            
                            st.markdown('<div class="section-title">📝 วัตถุดิบ</div>', unsafe_allow_html=True)
                            st.markdown(format_ingredients(best_match["ingredients"]), unsafe_allow_html=True)
                            
                            st.markdown('<div class="section-title">👩‍🍳 วิธีทำ</div>', unsafe_allow_html=True)
                            st.markdown(format_cooking_method(best_match["method"]), unsafe_allow_html=True)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # แสดงเมนูที่เกี่ยวข้อง
                            if len(results) > 1:
                                st.markdown("### 🍽️ เมนูที่เกี่ยวข้อง")
                                for i, related in enumerate(results[1:4], 1):
                                    st.markdown(f"{i}. **{related['name']}** (ความเกี่ยวข้อง: {related['similarity']:.2f})")
                            
                            # เพิ่มการตอบสนองของแอสซิสแตนต์ลงในประวัติการสนทนาพร้อมข้อมูลสูตร
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response, 
                                "recipe": best_match
                            })
                        else:
                            response = "ขออภัย ฉันไม่พบสูตรอาหารที่ตรงกับคำค้นหาของคุณ กรุณาลองคำค้นหาอื่น"
                            st.markdown(response)
                            
                            # แสดงเมนูแนะนำ
                            st.markdown("### 🍽️ เมนูแนะนำ")
                            random_recipes = data.sample(5)
                            for _, recipe in random_recipes.iterrows():
                                st.markdown(f"- {recipe['name']}")
                            
                            st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        response = "ขออภัย ฉันไม่สามารถค้นหาสูตรอาหารได้ในขณะนี้ กรุณาลองใหม่อีกครั้ง"
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Auto-scroll หลังการตอบสนอง
                st.markdown("""
                <script>
                setTimeout(function() {
                    scrollToLatestMessage();
                }, 500);
                </script>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()