import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import re
import requests
import json
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date

# การตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Thai Food Recipe Chatbot - Complete Nutrition System",
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
    
    .feature-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
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
    
    .success-badge {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 0.3rem 0.7rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
        display: inline-block;
        margin-left: 0.5rem;
    }
    
    .warning-badge {
        background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
        color: white;
        padding: 0.3rem 0.7rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
        display: inline-block;
        margin-left: 0.5rem;
    }
    
    .info-highlight {
        background: linear-gradient(135deg, #a8e6cf 0%, #dcedc1 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #00b894;
        margin: 1rem 0;
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

# Import classes จากไฟล์ที่สร้างไว้ก่อนหน้า
from nutrition_api import NutritionAPI

# คลาสหลักสำหรับการค้นหาขั้นสูง
class AdvancedRecipeSearchEngine:
    """เครื่องมือค้นหาสูตรอาหารไทยขั้นสูงพร้อมการวิเคราะห์โภชนาการ"""
    
    def __init__(self, data: pd.DataFrame, nutrition_api):
        self.data = data
        self.nutrition_api = nutrition_api
        self.nutrition_cache = {}
        
        # โหลดข้อมูลโภชนาการจากไฟล์ CSV ถ้ามี
        self.load_nutrition_data_from_csv()
        
        # คำสำคัญสำหรับการค้นหาตามโภชนาการ (ขยายเพิ่มเติม)
        self.nutrition_keywords = {
            # แคลอรี่
            'calories_low': ['แคลอรี่ต่ำ', 'แคลต่ำ', 'ลดน้ำหนัก', 'เบา', 'ไม่อ้วน', 'ไดเอท', 'diet', 'ลดความอ้วน'],
            'calories_high': ['แคลอรี่สูง', 'แคลสูง', 'เพิ่มน้ำหนัก', 'พลังงานสูง', 'เติมแรง', 'นักกีฬา'],
            
            # โปรตีน
            'protein_high': ['โปรตีนสูง', 'โปรตีนมาก', 'เนื้อเยื่อ', 'กล้ามเนื้อ', 'นักกีฬา', 'ออกกำลังกาย', 'ฟิตเนส'],
            'protein_low': ['โปรตีนต่ำ', 'โปรตีนน้อย', 'ไต', 'โรคไต'],
            
            # ไขมัน
            'fat_low': ['ไขมันต่ำ', 'ไขมันน้อย', 'ลดไขมัน', 'ไม่มันเยอะ', 'สุขภาพดี', 'หัวใจ'],
            'fat_high': ['ไขมันสูง', 'ไขมันมาก'],
            
            # คาร์โบไฮเดรต
            'carbs_low': ['คาร์โบต่ำ', 'แป้งน้อย', 'น้ำตาลต่ำ', 'เบาหวาน', 'คีโต', 'keto', 'low carb'],
            'carbs_high': ['คาร์โบสูง', 'แป้งมาก', 'พลังงาน', 'นักกีฬา'],
            
            # ใยอาหาร
            'fiber_high': ['ใยอาหารสูง', 'ใยอาหารมาก', 'ขับถ่าย', 'ท้องผูก', 'ย่อย', 'ระบบย่อย'],
            
            # วิตามิน
            'vitamin_a_high': ['วิตามินเอสูง', 'วิตามินเอ', 'สายตา', 'ผิวพรรณ', 'ตา'],
            'vitamin_c_high': ['วิตามินซีสูง', 'วิตามินซี', 'ภูมิคุ้มกัน', 'ต้านหวัด', 'เสริมภูมิ'],
            'vitamin_b_high': ['วิตามินบี', 'ระบบประสาท', 'เมแทบอลิซึม', 'พลังงาน'],
            
            # แร่ธาตุ
            'calcium_high': ['แคลเซียมสูง', 'แคลเซียม', 'กระดูก', 'ฟัน', 'ผู้สูงอายุ', 'เด็ก'],
            'iron_high': ['เหล็กสูง', 'ธาตุเหล็ก', 'โลหิตจาง', 'เลือดจาง', 'ผู้หญิง'],
            'potassium_high': ['โปแตสเซียมสูง', 'โปแตสเซียม', 'ความดันโลหิต', 'หัวใจ'],
            'sodium_low': ['โซเดียมต่ำ', 'เกลือน้อย', 'ความดันสูง', 'ไต', 'หัวใจ'],
            
            # กลุ่มผู้ป่วยเฉพาะ
            'diabetes': ['เบาหวาน', 'ผู้ป่วยเบาหวาน', 'น้ำตาลต่ำ', 'ควบคุมน้ำตาล', 'เบาหวาน'],
            'hypertension': ['ความดันสูง', 'ผู้ป่วยความดัน', 'โซเดียมต่ำ', 'ความดัน'],
            'elderly': ['ผู้สูงอายุ', 'คนแก่', 'นุ่ม', 'ย่อยง่าย', 'ผู้ใหญ่'],
            'children': ['เด็ก', 'เด็กเล็ก', 'แคลเซียม', 'เจริญเติบโต', 'ลูก'],
            'athletes': ['นักกีฬา', 'ออกกำลังกาย', 'โปรตีนสูง', 'ฟิตเนส', 'กล้ามเนื้อ'],
            'pregnant': ['ตั้งครรภ์', 'คนท้อง', 'โฟเลต', 'เหล็ก', 'แม่ท้อง'],
            
            # ประเภทอาหาร
            'vegetarian': ['มังสวิรัติ', 'เจ', 'ไม่กินเนื้อ', 'ผัก', 'พืช'],
            'low_sodium': ['โซเดียมต่ำ', 'เกลือน้อย', 'จืด'],
            'healthy': ['สุขภาพ', 'สุขภาพดี', 'คลีน', 'clean eating', 'healthy'],
            'weight_loss': ['ลดน้ำหนัก', 'ลดความอ้วน', 'ไดเอท', 'เบา'],
            'detox': ['ดีท็อกซ์', 'ล้างพิษ', 'ล้างลำไส้', 'detox']
        }
        
        # เกณฑ์การจัดกลุ่มโภชนาการ (ปรับปรุงให้แม่นยำขึ้น)
        self.nutrition_thresholds = {
            'calories': {'very_low': 200, 'low': 350, 'medium': 550, 'high': 750, 'very_high': 1000},
            'protein': {'low': 8, 'medium': 15, 'high': 25, 'very_high': 35},
            'fat': {'low': 8, 'medium': 15, 'high': 25, 'very_high': 35},
            'carbs': {'low': 15, 'medium': 35, 'high': 55, 'very_high': 80},
            'fiber': {'low': 2, 'medium': 5, 'high': 8, 'very_high': 12},
            'vitamin_a': {'low': 50, 'medium': 300, 'high': 800, 'very_high': 1500},
            'vitamin_c': {'low': 5, 'medium': 20, 'high': 50, 'very_high': 100},
            'calcium': {'low': 30, 'medium': 100, 'high': 250, 'very_high': 400},
            'iron': {'low': 1, 'medium': 3, 'high': 7, 'very_high': 12},
            'potassium': {'low': 150, 'medium': 350, 'high': 550, 'very_high': 800},
            'sodium': {'very_low': 300, 'low': 600, 'medium': 1000, 'high': 1500, 'very_high': 2000}
        }

    def load_nutrition_data_from_csv(self):
        """โหลดข้อมูลโภชนาการจากไฟล์ CSV ที่อัปเดต"""
        try:
            # ตรวจสอบว่ามีไฟล์ข้อมูลโภชนาการหรือไม่
            if os.path.exists("thai_food_processed.csv"):
                df = pd.read_csv("thai_food_processed.csv")
                
                # ตรวจสอบว่ามีคอลัมน์โภชนาการหรือไม่
                nutrition_columns = ['calories', 'protein', 'carbs', 'fat', 'fiber',
                                   'vitamin_a', 'vitamin_c', 'calcium', 'iron', 'sodium']
                
                if all(col in df.columns for col in nutrition_columns):
                    # ถ้ามีข้อมูลโภชนาการ ให้ใช้จากไฟล์
                    for idx, row in df.iterrows():
                        nutrition_data = {col: row.get(col, 0) for col in nutrition_columns}
                        self.nutrition_cache[idx] = nutrition_data
                    
                    st.info("🔄 โหลดข้อมูลโภชนาการจากไฟล์สำเร็จ")
                else:
                    st.warning("⚠️ ไฟล์ข้อมูลไม่มีคอลัมน์โภชนาการ จะคำนวณใหม่")
                    
        except Exception as e:
            st.warning(f"⚠️ ไม่สามารถโหลดข้อมูลโภชนาการจากไฟล์: {str(e)}")

    def calculate_all_nutrition_enhanced(self, use_api=False, adjust_consumption=True, 
                                       save_to_file=True) -> Dict:
        """คำนวณค่าโภชนาการสำหรับทุกเมนูและบันทึกลงไฟล์"""
        if hasattr(self, '_all_nutrition_cache') and not save_to_file:
            return self._all_nutrition_cache
            
        all_nutrition = {}
        updated_data = self.data.copy()
        
        # เพิ่มคอลัมน์โภชนาการถ้ายังไม่มี
        nutrition_columns = ['calories', 'protein', 'carbs', 'fat', 'fiber',
                           'vitamin_a', 'vitamin_c', 'vitamin_b1', 'vitamin_b2',
                           'calcium', 'iron', 'potassium', 'sodium']
        
        for col in nutrition_columns:
            if col not in updated_data.columns:
                updated_data[col] = 0.0
        
        with st.spinner("🧮 กำลังคำนวณค่าโภชนาการขั้นสูงสำหรับทุกเมนู..."):
            progress_bar = st.progress(0)
            success_count = 0
            
            for idx, recipe in self.data.iterrows():
                try:
                    # ตรวจสอบว่ามีข้อมูลในแคชหรือไม่
                    if idx in self.nutrition_cache:
                        nutrition = self.nutrition_cache[idx]
                    else:
                        # คำนวณใหม่
                        nutrition_data = self.nutrition_api.calculate_recipe_nutrition(
                            recipe['ingredient'], use_api, adjust_consumption
                        )
                        nutrition = nutrition_data['total_nutrition']
                        self.nutrition_cache[idx] = nutrition
                    
                    all_nutrition[idx] = nutrition
                    
                    # อัปเดตข้อมูลในตาราง
                    for nutrient, value in nutrition.items():
                        if nutrient in nutrition_columns:
                            updated_data.at[idx, nutrient] = round(value, 2)
                    
                    success_count += 1
                    progress_bar.progress((idx + 1) / len(self.data))
                    
                except Exception as e:
                    # ใช้ค่าเริ่มต้นหากคำนวณไม่ได้
                    default_nutrition = {col: 0 for col in nutrition_columns}
                    all_nutrition[idx] = default_nutrition
                    
                    for col in nutrition_columns:
                        updated_data.at[idx, col] = 0
            
            progress_bar.empty()
        
        # บันทึกข้อมูลลงไฟล์
        if save_to_file:
            try:
                updated_data.to_csv("thai_food_processed.csv", index=False, encoding='utf-8')
                st.success(f"✅ อัปเดตข้อมูลโภชนาการสำหรับ {success_count}/{len(self.data)} เมนูเรียบร้อย")
            except Exception as e:
                st.error(f"❌ ไม่สามารถบันทึกไฟล์: {str(e)}")
        
        self._all_nutrition_cache = all_nutrition
        return all_nutrition

    def get_enhanced_nutrition_recommendations(self, query: str, limit: int = 8) -> Dict:
        """ระบบแนะนำอาหารตามโภชนาการที่ปรับปรุงแล้ว"""
        intent = self.detect_nutrition_intent_enhanced(query)
        
        if not intent['is_nutrition_query']:
            return {'type': 'not_nutrition_query'}
        
        results = self.search_by_enhanced_criteria(intent['criteria'], limit)
        
        if not results:
            # แนะนำคำค้นหาใกล้เคียง
            similar_queries = self.get_similar_nutrition_queries(query)
            return {
                'type': 'no_results',
                'message': f'ไม่พบเมนูที่ตรงกับเกณฑ์ "{query}"',
                'suggestions': similar_queries
            }
        
        return {
            'type': 'nutrition_recommendations',
            'query': query,
            'criteria': intent['criteria'],
            'results': results,
            'total_found': len(results),
            'search_analysis': self.analyze_search_results(results)
        }

    def detect_nutrition_intent_enhanced(self, query: str) -> Dict[str, any]:
        """ตรวจจับความต้องการค้นหาตามโภชนาการแบบขั้นสูง"""
        query_lower = query.lower()
        detected_criteria = []
        confidence_scores = {}
        
        # ตรวจสอบคำสำคัญพร้อมคะแนนความมั่นใจ
        for category, keywords in self.nutrition_keywords.items():
            max_confidence = 0
            for keyword in keywords:
                if keyword in query_lower:
                    # คำนวณคะแนนความมั่นใจ
                    keyword_ratio = len(keyword) / len(query_lower)
                    position_bonus = 1.0 if query_lower.index(keyword) < len(query_lower) / 2 else 0.8
                    confidence = min(keyword_ratio * position_bonus * 10, 1.0)
                    max_confidence = max(max_confidence, confidence)
            
            if max_confidence > 0.3:  # เกณฑ์ความมั่นใจ
                detected_criteria.append(category)
                confidence_scores[category] = max_confidence
        
        if not detected_criteria:
            return {'is_nutrition_query': False}
        
        return {
            'is_nutrition_query': True,
            'criteria': detected_criteria,
            'confidence_scores': confidence_scores,
            'query_text': query,
            'intent_strength': sum(confidence_scores.values()) / len(confidence_scores)
        }

    def search_by_enhanced_criteria(self, criteria: List[str], limit: int = 10) -> List[Dict]:
        """ค้นหาเมนูตามเกณฑ์โภชนาการแบบขั้นสูง"""
        all_nutrition = self.calculate_all_nutrition_enhanced(save_to_file=False)
        scored_recipes = []
        
        for idx, nutrition in all_nutrition.items():
            total_score = 0
            detailed_reasons = []
            category_scores = {}
            
            for criterion in criteria:
                criterion_score, reason, category = self._evaluate_enhanced_criterion(criterion, nutrition)
                total_score += criterion_score
                category_scores[category] = criterion_score
                if reason:
                    detailed_reasons.append(reason)
            
            if total_score > 0:
                recipe_data = self.data.iloc[idx]
                
                # คำนวณคะแนนโบนัสจากความหลากหลายของสารอาหาร
                diversity_bonus = self._calculate_nutrition_diversity_bonus(nutrition)
                final_score = total_score + diversity_bonus
                
                scored_recipes.append({
                    'index': idx,
                    'name': recipe_data['name'],
                    'score': final_score,
                    'base_score': total_score,
                    'diversity_bonus': diversity_bonus,
                    'nutrition': nutrition,
                    'reasons': detailed_reasons,
                    'category_scores': category_scores,
                    'ingredient': recipe_data['ingredient'],
                    'method': recipe_data['method'],
                    'nutrition_grade': self._calculate_nutrition_grade(nutrition),
                    'health_benefits': self._identify_health_benefits(nutrition)
                })
        
        # เรียงลำดับตามคะแนนรวม
        scored_recipes.sort(key=lambda x: x['score'], reverse=True)
        return scored_recipes[:limit]

    def _evaluate_enhanced_criterion(self, criterion: str, nutrition: Dict) -> Tuple[float, str, str]:
        """ประเมินคะแนนตามเกณฑ์โภชนาการแบบละเอียด"""
        thresholds = self.nutrition_thresholds
        
        # แคลอรี่ต่ำ
        if criterion == 'calories_low':
            calories = nutrition.get('calories', 0)
            if calories <= thresholds['calories']['very_low']:
                return 15.0, f"แคลอรี่ต่ำมาก ({calories:.0f} kcal)", "energy"
            elif calories <= thresholds['calories']['low']:
                return 12.0, f"แคลอรี่ต่ำ ({calories:.0f} kcal)", "energy"
            elif calories <= thresholds['calories']['medium']:
                return 8.0, f"แคลอรี่ปานกลาง ({calories:.0f} kcal)", "energy"
            return 0.0, "", ""
        
        # แคลอรี่สูง
        elif criterion == 'calories_high':
            calories = nutrition.get('calories', 0)
            if calories >= thresholds['calories']['very_high']:
                return 15.0, f"แคลอรี่สูงมาก ({calories:.0f} kcal)", "energy"
            elif calories >= thresholds['calories']['high']:
                return 12.0, f"แคลอรี่สูง ({calories:.0f} kcal)", "energy"
            return 0.0, "", ""
        
        # โปรตีนสูง
        elif criterion == 'protein_high':
            protein = nutrition.get('protein', 0)
            if protein >= thresholds['protein']['very_high']:
                return 15.0, f"โปรตีนสูงมาก ({protein:.1f} g)", "protein"
            elif protein >= thresholds['protein']['high']:
                return 12.0, f"โปรตีนสูง ({protein:.1f} g)", "protein"
            elif protein >= thresholds['protein']['medium']:
                return 8.0, f"โปรตีนปานกลาง ({protein:.1f} g)", "protein"
            return 0.0, "", ""
        
        # ไขมันต่ำ
        elif criterion == 'fat_low':
            fat = nutrition.get('fat', 0)
            if fat <= thresholds['fat']['low']:
                return 12.0, f"ไขมันต่ำ ({fat:.1f} g)", "fat"
            elif fat <= thresholds['fat']['medium']:
                return 8.0, f"ไขมันปานกลาง ({fat:.1f} g)", "fat"
            return 0.0, "", ""
        
        # คาร์โบไฮเดรตต่ำ
        elif criterion == 'carbs_low':
            carbs = nutrition.get('carbs', 0)
            if carbs <= thresholds['carbs']['low']:
                return 12.0, f"คาร์โบไฮเดรตต่ำ ({carbs:.1f} g)", "carbs"
            elif carbs <= thresholds['carbs']['medium']:
                return 8.0, f"คาร์โบไฮเดรตปานกลาง ({carbs:.1f} g)", "carbs"
            return 0.0, "", ""
        
        # ใยอาหารสูง
        elif criterion == 'fiber_high':
            fiber = nutrition.get('fiber', 0)
            if fiber >= thresholds['fiber']['very_high']:
                return 15.0, f"ใยอาหารสูงมาก ({fiber:.1f} g)", "fiber"
            elif fiber >= thresholds['fiber']['high']:
                return 12.0, f"ใยอาหารสูง ({fiber:.1f} g)", "fiber"
            elif fiber >= thresholds['fiber']['medium']:
                return 8.0, f"ใยอาหารปานกลาง ({fiber:.1f} g)", "fiber"
            return 0.0, "", ""
        
        # วิตามินและแร่ธาตุ
        elif criterion == 'vitamin_c_high':
            vitamin_c = nutrition.get('vitamin_c', 0)
            if vitamin_c >= thresholds['vitamin_c']['very_high']:
                return 15.0, f"วิตามินซีสูงมาก ({vitamin_c:.1f} mg)", "vitamins"
            elif vitamin_c >= thresholds['vitamin_c']['high']:
                return 12.0, f"วิตามินซีสูง ({vitamin_c:.1f} mg)", "vitamins"
            return 0.0, "", ""
        
        elif criterion == 'calcium_high':
            calcium = nutrition.get('calcium', 0)
            if calcium >= thresholds['calcium']['very_high']:
                return 15.0, f"แคลเซียมสูงมาก ({calcium:.0f} mg)", "minerals"
            elif calcium >= thresholds['calcium']['high']:
                return 12.0, f"แคลเซียมสูง ({calcium:.0f} mg)", "minerals"
            return 0.0, "", ""
        
        elif criterion == 'iron_high':
            iron = nutrition.get('iron', 0)
            if iron >= thresholds['iron']['very_high']:
                return 15.0, f"เหล็กสูงมาก ({iron:.1f} mg)", "minerals"
            elif iron >= thresholds['iron']['high']:
                return 12.0, f"เหล็กสูง ({iron:.1f} mg)", "minerals"
            return 0.0, "", ""
        
        elif criterion == 'sodium_low':
            sodium = nutrition.get('sodium', 0)
            if sodium <= thresholds['sodium']['very_low']:
                return 15.0, f"โซเดียมต่ำมาก ({sodium:.0f} mg)", "minerals"
            elif sodium <= thresholds['sodium']['low']:
                return 12.0, f"โซเดียมต่ำ ({sodium:.0f} mg)", "minerals"
            return 0.0, "", ""
        
        # เกณฑ์สำหรับผู้ป่วยเฉพาะ
        elif criterion == 'diabetes':
            score = 0
            reasons_list = []
            
            carbs = nutrition.get('carbs', 0)
            fiber = nutrition.get('fiber', 0)
            sodium = nutrition.get('sodium', 0)
            
            if carbs <= thresholds['carbs']['low']:
                score += 10
                reasons_list.append(f"คาร์โบไฮเดรตต่ำ ({carbs:.1f} g)")
            if fiber >= thresholds['fiber']['medium']:
                score += 8
                reasons_list.append(f"ใยอาหารดี ({fiber:.1f} g)")
            if sodium <= thresholds['sodium']['medium']:
                score += 5
                reasons_list.append(f"โซเดียมเหมาะสม ({sodium:.0f} mg)")
            
            reason = "เหมาะสำหรับผู้ป่วยเบาหวาน: " + ", ".join(reasons_list) if reasons_list else ""
            return score, reason, "health_condition"
        
        elif criterion == 'hypertension':
            score = 0
            reasons_list = []
            
            sodium = nutrition.get('sodium', 0)
            potassium = nutrition.get('potassium', 0)
            fiber = nutrition.get('fiber', 0)
            
            if sodium <= thresholds['sodium']['low']:
                score += 12
                reasons_list.append(f"โซเดียมต่ำ ({sodium:.0f} mg)")
            if potassium >= thresholds['potassium']['medium']:
                score += 8
                reasons_list.append(f"โปแตสเซียมดี ({potassium:.0f} mg)")
            if fiber >= thresholds['fiber']['medium']:
                score += 5
                reasons_list.append(f"ใยอาหารดี ({fiber:.1f} g)")
            
            reason = "เหมาะสำหรับผู้ป่วยความดันสูง: " + ", ".join(reasons_list) if reasons_list else ""
            return score, reason, "health_condition"
        
        return 0.0, "", ""

    def _calculate_nutrition_diversity_bonus(self, nutrition: Dict) -> float:
        """คำนวณคะแนนโบนัสจากความหลากหลายของสารอาหาร"""
        important_nutrients = ['protein', 'fiber', 'vitamin_a', 'vitamin_c', 'calcium', 'iron']
        good_nutrients = 0
        
        for nutrient in important_nutrients:
            value = nutrition.get(nutrient, 0)
            threshold = self.nutrition_thresholds.get(nutrient, {}).get('medium', 0)
            if value >= threshold:
                good_nutrients += 1
        
        # โบนัส 0-3 คะแนนตามจำนวนสารอาหารที่ดี
        return (good_nutrients / len(important_nutrients)) * 3

    def _calculate_nutrition_grade(self, nutrition: Dict) -> str:
        """คำนวณเกรดโภชนาการ A-F"""
        total_score = 0
        max_score = 0
        
        nutrient_weights = {
            'protein': 3, 'fiber': 2, 'vitamin_a': 2, 'vitamin_c': 2,
            'calcium': 2, 'iron': 2, 'potassium': 1
        }
        
        for nutrient, weight in nutrient_weights.items():
            value = nutrition.get(nutrient, 0)
            thresholds = self.nutrition_thresholds.get(nutrient, {})
            
            if value >= thresholds.get('high', 0):
                score = 4
            elif value >= thresholds.get('medium', 0):
                score = 3
            elif value >= thresholds.get('low', 0):
                score = 2
            else:
                score = 1
            
            total_score += score * weight
            max_score += 4 * weight
        
        percentage = (total_score / max_score) * 100 if max_score > 0 else 0
        
        if percentage >= 85:
            return 'A'
        elif percentage >= 75:
            return 'B'
        elif percentage >= 65:
            return 'C'
        elif percentage >= 55:
            return 'D'
        else:
            return 'F'

    def _identify_health_benefits(self, nutrition: Dict) -> List[str]:
        """ระบุประโยชน์ต่อสุขภาพจากสารอาหาร"""
        benefits = []
        
        if nutrition.get('fiber', 0) >= self.nutrition_thresholds['fiber']['high']:
            benefits.append("ช่วยระบบย่อยอาหาร")
        
        if nutrition.get('vitamin_c', 0) >= self.nutrition_thresholds['vitamin_c']['high']:
            benefits.append("เสริมภูมิคุ้มกัน")
        
        if nutrition.get('calcium', 0) >= self.nutrition_thresholds['calcium']['high']:
            benefits.append("บำรุงกระดูกและฟัน")
        
        if nutrition.get('iron', 0) >= self.nutrition_thresholds['iron']['high']:
            benefits.append("ป้องกันโลหิตจาง")
        
        if nutrition.get('protein', 0) >= self.nutrition_thresholds['protein']['high']:
            benefits.append("เสริมสร้างกล้ามเนื้อ")
        
        if nutrition.get('sodium', 0) <= self.nutrition_thresholds['sodium']['low']:
            benefits.append("ดีต่อหัวใจและไต")
        
        return benefits

    def analyze_search_results(self, results: List[Dict]) -> Dict:
        """วิเคราะห์ผลการค้นหา"""
        if not results:
            return {}
        
        analysis = {
            'average_score': np.mean([r['score'] for r in results]),
            'score_range': (min(r['score'] for r in results), max(r['score'] for r in results)),
            'nutrition_grades': {},
            'common_benefits': {},
            'top_categories': {}
        }
        
        # นับเกรดโภชนาการ
        for result in results:
            grade = result['nutrition_grade']
            analysis['nutrition_grades'][grade] = analysis['nutrition_grades'].get(grade, 0) + 1
        
        # นับประโยชน์ที่พบบ่อย
        for result in results:
            for benefit in result['health_benefits']:
                analysis['common_benefits'][benefit] = analysis['common_benefits'].get(benefit, 0) + 1
        
        # นับหมวดหมู่ที่ได้คะแนนสูง
        for result in results:
            for category, score in result['category_scores'].items():
                if score > 0:
                    analysis['top_categories'][category] = analysis['top_categories'].get(category, 0) + 1
        
        return analysis

    def get_similar_nutrition_queries(self, query: str) -> List[str]:
        """แนะนำคำค้นหาใกล้เคียง"""
        suggestions = [
            "อาหารแคลอรี่ต่ำ",
            "เมนูโปรตีนสูง",
            "อาหารไขมันต่ำ",
            "เมนูใยอาหารสูง",
            "อาหารโซเดียมต่ำ",
            "เมนูวิตามินซีสูง",
            "อาหารแคลเซียมสูง",
            "เมนูเหล็กสูง",
            "อาหารเหมาะสำหรับเบาหวาน",
            "เมนูเหมาะสำหรับความดันสูง"
        ]
        
        # หาคำแนะนำที่ใกล้เคียงที่สุด
        similar = []
        for suggestion in suggestions:
            similarity = SequenceMatcher(None, query.lower(), suggestion.lower()).ratio()
            if similarity > 0.3:
                similar.append((suggestion, similarity))
        
        similar.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in similar[:3]]

# ตัวแปรไฟล์และโฟลเดอร์
DATA_PATH = "thai_food_processed.csv"
SAMPLE_DATA_PATH = "thai_food_sample.csv"

@st.cache_data
def load_data():
    """โหลดข้อมูลอาหารไทยที่อัปเดต"""
    try:
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            # ตรวจสอบว่ามีข้อมูลโภชนาการหรือไม่
            nutrition_columns = ['calories', 'protein', 'carbs', 'fat', 'fiber']
            if not all(col in df.columns for col in nutrition_columns):
                st.warning("⚠️ ไฟล์ข้อมูลไม่มีข้อมูลโภชนาการ จะเพิ่มคอลัมน์โภชนาการ")
                for col in nutrition_columns + ['vitamin_a', 'vitamin_c', 'calcium', 'iron', 'sodium']:
                    if col not in df.columns:
                        df[col] = 0.0
            return df
        elif os.path.exists(SAMPLE_DATA_PATH):
            return pd.read_csv(SAMPLE_DATA_PATH)
        else:
            return create_enhanced_sample_data()
    except Exception as e:
        st.error(f"ข้อผิดพลาดในการโหลดข้อมูล: {str(e)}")
        return create_enhanced_sample_data()

def create_enhanced_sample_data():
    """สร้างข้อมูลตัวอย่างที่มีโภชนาการ"""
    sample_data = {
        'name': [
            'ผัดกะเพราหมูสับ', 'ต้มยำกุ้งน้ำใส', 'ส้มตำไทย', 'แกงเขียวหวานไก่', 'ผัดไทยกุ้งสด',
            'ไข่เจียวฟู', 'ข้าวผัดกุ้ง', 'ยำวุ้นเส้นทะเล', 'ลาบหมูอีสาน', 'มะม่วงข้าวเหนียว',
            'แกงจืดเต้าหู้', 'ผัดพักบุ้งไฟแดง', 'น้ำพริกปลาร้า', 'ต้มข่าไก่', 'ผัดซีอิ๊วหมู'
        ],
        'ingredient': [
            '- เนื้อหมูสับ 200 กรัม\n- ใบกะเพรา 1 ถ้วย\n- พริกขี้หนู 5 เม็ด\n- กระเทียม 4 กลีบ\n- น้ำปลา 2 ช้อนโต๊ะ\n- น้ำตาลทราย 1 ช้อนชา',
            '- กุ้งนาง 300 กรัม\n- เห็ดฟาง 100 กรัม\n- มะนาว 3 ผล\n- ใบมะกรูด 5 ใบ\n- ตะไคร้ 3 ต้น\n- ข่า 4 แว่น\n- น้ำปลา 3 ช้อนโต๊ะ',
            '- มะละกอดิบ 2 ถ้วย\n- มะเขือเทศ 3 ผล\n- ถั่วฝักยาว 10 เส้น\n- กุ้งแห้ง 2 ช้อนโต๊ะ\n- ถั่วลิสงคั่ว 3 ช้อนโต๊ะ\n- พริกขี้หนู 5 เม็ด',
            '- เนื้อไก่ 400 กรัม\n- กะทิ 2 ถ้วย\n- น้ำพริกแกงเขียวหวาน 3 ช้อนโต๊ะ\n- มะเขือเปราะ 8 ผล\n- ใบโหระพา 1 ถ้วย',
            '- เส้นจันท์ 200 กรัม\n- กุ้งสด 150 กรัม\n- เต้าหู้ 100 กรัม\n- ไข่ไก่ 2 ฟอง\n- ถั่วงอก 100 กรัม\n- กุ้ยช่าย 50 กรัม',
            '- ไข่ไก่ 3 ฟอง\n- น้ำปลา 1 ช้อนชา\n- ต้นหอม 2 ต้น\n- ผักชี 1 ต้น\n- น้ำมันหมู 2 ช้อนโต๊ะ',
            '- ข้าวสวย 3 ถ้วย\n- กุ้งสด 200 กรัม\n- ไข่ไก่ 2 ฟอง\n- หอมใหญ่ 1 หัว\n- แครอท 1 ผล\n- ซีอิ๊วขาว 2 ช้อนโต๊ะ',
            '- วุ้นเส้น 150 กรัม\n- กุ้งสด 150 กรัม\n- หมูสับ 100 กรัม\n- ปลาหมึก 100 กรัม\n- มะนาว 3 ผล\n- ผักชี 3 ต้น',
            '- เนื้อหมูสับ 300 กรัม\n- ข้าวคั่ว 3 ช้อนโต๊ะ\n- พริกแห้ง 8 เม็ด\n- หอมแดง 5 หัว\n- ใบสะระแหน่ 1 ถ้วย\n- น้ำปลา 4 ช้อนโต๊ะ',
            '- ข้าวเหนียว 2 ถ้วย\n- มะม่วงสุก 2 ผล\n- กะทิ 1 ถ้วย\n- น้ำตาลปึก 3 ช้อนโต๊ะ\n- เกลือ 1/2 ช้อนชา',
            '- เต้าหู้อ่อน 200 กรัม\n- หมูสับ 100 กรัม\n- ต้นหอม 3 ต้น\n- ผักชี 2 ต้น\n- น้ำซุปกระดูก 4 ถ้วย',
            '- ผักบุ้ง 300 กรัม\n- หมูหั่นฝอย 100 กรัม\n- พริกแกง 2 ช้อนโต๊ะ\n- กระเทียม 4 กลีบ\n- น้ำปลา 2 ช้อนโต๊ะ',
            '- ปลาร้า 3 ช้อนโต๊ะ\n- พริกแห้ง 10 เม็ด\n- กระเทียม 5 กลีบ\n- หอมแดง 3 หัว\n- มะเขือเทศ 2 ผล',
            '- เนื้อไก่ 300 กรัม\n- กะทิ 2 ถ้วย\n- ข่า 5 แว่น\n- ตะไคร้ 3 ต้น\n- ใบมะกรูด 5 ใบ\n- เห็ดฟาง 100 กรัม',
            '- เนื้อหมูหั่นบาง 250 กรัม\n- ซีอิ๊วดำ 2 ช้อนโต๊ะ\n- ซีอิ๊วขาว 1 ช้อนโต๊ะ\n- กระเทียม 4 กลีบ\n- คะน้า 200 กรัม'
        ],
        'method': [
            'โขลกกระเทียมและพริกให้ละเอียด ผัดในน้ำมันร้อนจนหอม ใส่หมูสับผัดจนสุก ปรุงรสด้วยน้ำปลาและน้ำตาล ใส่ใบกะเพราผัดให้เข้ากัน',
            'ต้มน้ำให้เดือด ใส่ตะไคร้ ข่า ใบมะกรูด พริกขี้หนูโขลก ต้มให้เดือดอีกครั้ง ใส่กุ้งและเห็ดฟาง ปรุงรสด้วยน้ำปลา ยกลงจากเตา ใส่น้ำมะนาว',
            'โขลกพริก กระเทียม ถั่วลิสง กุ้งแห้งให้หยาบ ใส่มะละกอ มะเขือเทศ ถั่วฝักยาว ตำให้เข้ากัน ปรุงรสด้วยน้ำปลา น้ำตาลปึก น้ำมะนาว',
            'คั่วน้ำพริกแกงเขียวหวานกับหัวกะทิให้หอม ใส่เนื้อไก่ผัดให้เข้ากัน เติมกะทิที่เหลือ ต้มให้เดือด ใส่มะเขือเปราะ ปรุงรส ใส่ใบโหระพา',
            'แช่เส้นจันท์ให้นุ่ม ตั้งกะทะใส่น้ำมัน ผัดกุ้งและเต้าหู้ ใส่ไข่คนให้เข้ากัน ใส่เส้นจันท์และน้ำซอส ผัดให้เข้ากัน ใส่ถั่วงอกและกุ้ยช่าย',
            'ตอกไข่ใส่ชาม ใส่น้ำปลา ตีให้เข้ากัน ใส่ต้นหอมและผักชีซอย ตั้งกะทะใส่น้ำมัน พอร้อนเทไข่ลงทอดจนฟูเหลืองทั้งสองด้าน',
            'ตั้งกะทะใส่น้ำมัน ผัดกระเทียมและหอมใหญ่ให้หอม ใส่กุ้งผัดจนสุก ใส่ไข่คนให้เข้ากัน ใส่ข้าวและแครอทผัดให้เข้ากัน ปรุงรสด้วยซีอิ๊ว',
            'แช่วุ้นเส้นให้นุ่ม ลวกกุ้ง หมู และปลาหมึกจนสุก ผสมน้ำยำจากมะนาว น้ำปลา น้ำตาลปึก พริกโขลก คลุกทุกอย่างให้เข้ากัน โรยผักชี',
            'คั่วข้าวให้เหลืองหอม โขลกให้หยาบ ย่างพริกแห้งให้หอม โขลกกับหอมแดงให้ละเอียด ผสมเนื้อหมูสับกับข้าวคั่ว พริกโขลก ปรุงรส โรยสะระแหน่',
            'นึ่งข้าวเหนียวให้สุก หั่นมะม่วงเป็นชิ้น ต้มกะทิกับน้ำตาลปึกและเกลือ คนจนละลาย เสิร์ฟข้าวเหนียวพร้อมมะม่วงและกะทิ',
            'ตั้งหม้อใส่น้ำซุป เดือดแล้วใส่หมูสับ ต้มจนสุกใส่เต้าหู้ ต้มให้นุ่ม ปรุงรสด้วยซีอิ๊วขาว ใส่ต้นหอมและผักชี โรยพริกไทยป่น',
            'ผัดกระเทียมกับน้ำพริกแกงให้หอม ใส่หมูผัดจนสุก ใส่ผักบุ้งผัดให้เข้ากัน ปรุงรสด้วยน้ำปลา น้ำตาล ซีอิ๊วดำ ผัดจนผักสุก',
            'ย่างพริกแห้งให้หอม โขลกกับกระเทียม หอมแดงให้ละเอียด ใส่ปลาร้าโขลกต่อ ใส่มะเขือเทศโขลกหยาบ ปรุงรสตามชอบ',
            'ต้มกะทิจนเดือด ใส่ข่า ตะไคร้ ใบมะกรูด ใส่เนื้อไก่ต้มจนสุก ใส่เห็ดฟาง ปรุงรสด้วยน้ำปลา น้ำตาลปึก น้ำมะนาว',
            'หมักหมูกับซีอิ๊วดำ ซีอิ๊วขาว แป้ง ตั้งกะทะใส่น้ำมัน ผัดกระเทียมให้หอม ใส่หมูผัดจนสุก ใส่คะน้าผัดให้เข้ากัน'
        ]
    }
    return pd.DataFrame(sample_data)

@st.cache_resource
def initialize_nutrition_api():
    """เริ่มต้นระบบข้อมูลโภชนาการ"""
    return NutritionAPI()

def display_enhanced_recipe_with_nutrition(recipe: Dict, nutrition_data: Dict, similarity_score: float = None):
    """แสดงสูตรอาหารพร้อมข้อมูลโภชนาการแบบละเอียด"""
    
    # หัวข้อ
    title_html = f"### 🍽️ {recipe['name']}"
    if similarity_score is not None:
        similarity_percent = similarity_score * 100
        if similarity_percent >= 70:
            badge_class = "success-badge"
        elif similarity_percent >= 50:
            badge_class = "warning-badge"
        else:
            badge_class = "info-highlight"
        title_html += f' <span class="{badge_class}">{similarity_percent:.0f}% ตรง</span>'
    
    st.markdown(title_html, unsafe_allow_html=True)
    
    # แท็บต่างๆ
    tab1, tab2, tab3, tab4 = st.tabs(["📝 สูตรอาหาร", "📊 โภชนาการ", "🔬 วิเคราะห์ละเอียด", "💡 คำแนะนำ"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 🥬 วัตถุดิบ")
            ingredients_formatted = format_ingredients(recipe["ingredient"])
            st.markdown(ingredients_formatted, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 👨‍🍳 วิธีทำ")
            method_formatted = format_cooking_method(recipe["method"])
            st.markdown(method_formatted, unsafe_allow_html=True)
    
    with tab2:
        if nutrition_data:
            display_nutrition_info_enhanced(nutrition_data, recipe['name'])
        else:
            st.info("ไม่มีข้อมูลโภชนาการ")
    
    with tab3:
        if nutrition_data:
            display_detailed_nutrition_analysis(nutrition_data, recipe['name'])
        else:
            st.info("ไม่มีข้อมูลสำหรับการวิเคราะห์")
    
    with tab4:
        if nutrition_data:
            display_nutrition_recommendations_tab(nutrition_data, recipe['name'])
        else:
            st.info("ไม่สามารถให้คำแนะนำได้")

def format_ingredients(ingredients_text):
    """จัดรูปแบบรายการวัตถุดิบ"""
    if not ingredients_text:
        return "<p>ไม่มีข้อมูลวัตถุดิบ</p>"
        
    ingredients = ingredients_text.split('\n')
    formatted = "<ul style='margin: 0; padding-left: 1.5rem;'>"
    for item in ingredients:
        if item.strip():
            clean_item = item.strip().lstrip('- ')
            formatted += f"<li style='margin: 0.2rem 0;'>{clean_item}</li>"
    formatted += "</ul>"
    return formatted

def format_cooking_method(method_text):
    """จัดรูปแบบวิธีทำ"""
    if not method_text:
        return "<p>ไม่มีข้อมูลวิธีทำ</p>"
    
    has_numbers = bool(re.search(r'^\s*\d+\.', method_text, re.MULTILINE))
    lines = method_text.split('\n')
    formatted = "<div style='margin: 0; line-height: 1.6;'>"
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if has_numbers and re.match(r'^\d+\.', line):
            formatted += f"<p style='margin: 0.3rem 0; padding-left: 1rem;'>{line}</p>"
        else:
            formatted += f"<p style='margin: 0.5rem 0;'>{line}</p>"
    
    formatted += "</div>"
    return formatted

def display_nutrition_info_enhanced(nutrition_data: Dict, recipe_name: str):
    """แสดงข้อมูลโภชนาการแบบขั้นสูง"""
    if isinstance(nutrition_data, dict) and 'total_nutrition' in nutrition_data:
        total_nutrition = nutrition_data['total_nutrition']
    else:
        total_nutrition = nutrition_data
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="nutrition-card">
            <h4>🔥 พลังงานและสารอาหารหลัก</h4>
            <p><strong>แคลอรี่:</strong> {total_nutrition.get('calories', 0):.1f} kcal</p>
            <p><strong>โปรตีน:</strong> {total_nutrition.get('protein', 0):.1f} g</p>
            <p><strong>คาร์โบไฮเดรต:</strong> {total_nutrition.get('carbs', 0):.1f} g</p>
            <p><strong>ไขมัน:</strong> {total_nutrition.get('fat', 0):.1f} g</p>
            <p><strong>ใยอาหาร:</strong> {total_nutrition.get('fiber', 0):.1f} g</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="vitamin-card">
            <h4>💊 วิตามิน</h4>
            <p><strong>วิตามิน A:</strong> {total_nutrition.get('vitamin_a', 0):.1f} IU</p>
            <p><strong>วิตามิน C:</strong> {total_nutrition.get('vitamin_c', 0):.1f} mg</p>
            <p><strong>วิตามิน B1:</strong> {total_nutrition.get('vitamin_b1', 0):.2f} mg</p>
            <p><strong>วิตามิน B2:</strong> {total_nutrition.get('vitamin_b2', 0):.2f} mg</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="mineral-card">
            <h4>⚡ แร่ธาตุ</h4>
            <p><strong>แคลเซียม:</strong> {total_nutrition.get('calcium', 0):.1f} mg</p>
            <p><strong>เหล็ก:</strong> {total_nutrition.get('iron', 0):.1f} mg</p>
            <p><strong>โปแตสเซียม:</strong> {total_nutrition.get('potassium', 0):.1f} mg</p>
            <p><strong>โซเดียม:</strong> {total_nutrition.get('sodium', 0):.1f} mg</p>
        </div>
        """, unsafe_allow_html=True)

def display_detailed_nutrition_analysis(nutrition_data: Dict, recipe_name: str):
    """แสดงการวิเคราะห์โภชนาการแบบละเอียด"""
    if isinstance(nutrition_data, dict) and 'total_nutrition' in nutrition_data:
        total_nutrition = nutrition_data['total_nutrition']
    else:
        total_nutrition = nutrition_data
    
    # สร้างกราฟวงกลมแสดงสัดส่วนแมโครนิวเทรียนต์
    fig_macro = go.Figure(data=[go.Pie(
        labels=['โปรตีน', 'คาร์โบไฮเดรต', 'ไขมัน'],
        values=[
            total_nutrition.get('protein', 0) * 4,  # 1g = 4 kcal
            total_nutrition.get('carbs', 0) * 4,    # 1g = 4 kcal
            total_nutrition.get('fat', 0) * 9       # 1g = 9 kcal
        ],
        hole=.3,
        marker_colors=['#FF9999', '#66B2FF', '#99FF99']
    )])
    
    fig_macro.update_layout(
        title="🥧 สัดส่วนแมโครนิวเทรียนต์ (ตามแคลอรี่)",
        title_x=0.5,
        font=dict(family="Sarabun, sans-serif"),
        height=400
    )
    
    st.plotly_chart(fig_macro, use_container_width=True)
    
    # กราฟแท่งแสดงวิตามินและแร่ธาตุ
    vitamins_minerals = {
        'วิตามิน A (IU)': total_nutrition.get('vitamin_a', 0),
        'วิตามิน C (mg)': total_nutrition.get('vitamin_c', 0),
        'แคลเซียม (mg)': total_nutrition.get('calcium', 0),
        'เหล็ก (mg)': total_nutrition.get('iron', 0),
        'โปแตสเซียม (mg)': total_nutrition.get('potassium', 0)
    }
    
    fig_vitamins = go.Figure(data=[
        go.Bar(
            x=list(vitamins_minerals.keys()),
            y=list(vitamins_minerals.values()),
            marker_color=['#FFB366', '#66FFB2', '#B366FF', '#FF66B2', '#66B2FF']
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

def display_nutrition_recommendations_tab(nutrition_data: Dict, recipe_name: str):
    """แสดงคำแนะนำโภชนาการ"""
    if isinstance(nutrition_data, dict) and 'total_nutrition' in nutrition_data:
        total_nutrition = nutrition_data['total_nutrition']
    else:
        total_nutrition = nutrition_data
    
    st.markdown("#### 💡 คำแนะนำสำหรับการบริโภค")
    
    recommendations = []
    warnings = []
    
    # วิเคราะห์และให้คำแนะนำ
    calories = total_nutrition.get('calories', 0)
    protein = total_nutrition.get('protein', 0)
    sodium = total_nutrition.get('sodium', 0)
    fiber = total_nutrition.get('fiber', 0)
    
    if calories < 300:
        recommendations.append("✅ เหมาะสำหรับผู้ที่ต้องการควบคุมน้ำหนัก")
    elif calories > 600:
        warnings.append("⚠️ มีแคลอรี่สูง ควรทานร่วมกับผักใบเขียว")
    
    if protein >= 20:
        recommendations.append("✅ โปรตีนสูง เหมาะสำหรับผู้ที่ออกกำลังกาย")
    elif protein < 10:
        recommendations.append("💡 ควรเพิ่มโปรตีนจากไข่หรือเต้าหู้")
    
    if sodium > 1000:
        warnings.append("⚠️ โซเดียมสูง ไม่เหมาะสำหรับผู้ป่วยความดันสูง")
    elif sodium < 500:
        recommendations.append("✅ โซเดียมต่ำ เหมาะสำหรับผู้ป่วยความดันสูง")
    
    if fiber >= 5:
        recommendations.append("✅ ใยอาหารดี ช่วยระบบย่อยอาหาร")
    
    # แสดงคำแนะนำ
    if recommendations:
        st.markdown("##### ✅ ข้อดี")
        for rec in recommendations:
            st.markdown(f"- {rec}")
    
    if warnings:
        st.markdown("##### ⚠️ ข้อควรระวัง")
        for warning in warnings:
            st.markdown(f"- {warning}")
    
    # คำแนะนำการปรับปรุง
    st.markdown("##### 🔧 คำแนะนำการปรับปรุง")
    improvements = [
        "🥗 เพิ่มผักใบเขียวเป็นเครื่องเคียง",
        "🍊 ทานผลไม้หลังอาหารเพื่อเพิ่มวิตามินซี",
        "🥛 ดื่มนมหรือน้ำมากขึ้น",
        "🌾 เพิ่มข้าวกล้องแทนข้าวขาวเพื่อใยอาหาร"
    ]
    
    for improvement in improvements[:3]:
        st.markdown(f"- {improvement}")

def main():
    """ฟังก์ชันหลักของแอปพลิเคชัน"""
    
    st.markdown('<h1 class="main-title">🍲 Thai Food Recipe Chatbot</h1>')
    st.markdown('<div style="text-align: center; font-size: 1.2em; margin-bottom: 2rem;">🥘 ระบบค้นหาสูตรอาหารไทยขั้นสูงพร้อมวิเคราะห์โภชนาการ</div>')
    
    # เริ่มต้นระบบ
    with st.spinner("กำลังเริ่มต้นระบบขั้นสูง..."):
        data = load_data()
        
        if data.empty:
            st.error("ไม่มีข้อมูลสูตรอาหาร")
            return
        
        nutrition_api = initialize_nutrition_api()
        search_engine = AdvancedRecipeSearchEngine(data, nutrition_api)
    
    # แสดงสถิติ
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📖 จำนวนสูตร", len(data))
    with col2:
        nutrition_count = sum(1 for col in ['calories', 'protein', 'carbs'] if col in data.columns)
        st.metric("📊 ข้อมูลโภชนาการ", f"{nutrition_count}/3 คอลัมน์")
    with col3:
        st.metric("🔍 Enhanced Search", "✅ พร้อม")
    with col4:
        st.metric("🧮 AI Analysis", "✅ ใช้งานได้")
    
    # แถบการตั้งค่า
    with st.sidebar:
        st.title("🔧 การตั้งค่าขั้นสูง")
        
        st.markdown("### 📊 ข้อมูลโภชนาการ")
        update_nutrition = st.checkbox("อัปเดตข้อมูลโภชนาการ", help="คำนวณค่าโภชนาการใหม่สำหรับทุกเมนู")
        use_api = st.checkbox("ใช้ API ภายนอก", help="ใช้ข้อมูลจาก API เพื่อความแม่นยำสูงสุด")
        
        if update_nutrition:
            if st.button("🔄 อัปเดตข้อมูลโภชนาการ"):
                search_engine.calculate_all_nutrition_enhanced(use_api=use_api, save_to_file=True)
                st.rerun()
        
        st.markdown("### 🎯 การค้นหา")
        search_limit = st.slider("จำนวนผลลัพธ์", 3, 10, 5)
        
        st.markdown("### 📈 การแสดงผล")
        show_detailed_analysis = st.checkbox("แสดงการวิเคราะห์ละเอียด", value=True)
        show_nutrition_comparison = st.checkbox("แสดงการเปรียบเทียบโภชนาการ", value=True)
    
    # แนะนำการค้นหาตามโภชนาการ
    st.markdown("#### 💡 ลองค้นหาตามคุณค่าทางโภชนาการ:")
    
    nutrition_suggestions = [
        "🔥 อาหารแคลอรี่ต่ำเพื่อลดน้ำหนัก",
        "💪 เมนูโปรตีนสูงสำหรับนักกีฬา", 
        "🥗 อาหารไขมันต่ำเพื่อสุขภาพหัวใจ",
        "🍃 เมนูใยอาหารสูงช่วยระบบย่อย",
        "🧂 อาหารโซเดียมต่ำสำหรับความดันสูง",
        "🍊 เมนูวิตามินซีสูงเสริมภูมิคุ้มกัน",
        "🦴 อาหารแคลเซียมสูงบำรุงกระดูก",
        "🩸 เมนูเหล็กสูงป้องกันโลหิตจาง",
        "🌿 อาหารเหมาะสำหรับผู้ป่วยเบาหวาน",
        "❤️ เมนูเหมาะสำหรับผู้ป่วยหัวใจ"
    ]
    
    # แสดงปุ่มแนะนำในรูปแบบ grid
    cols = st.columns(3)
    for i, suggestion in enumerate(nutrition_suggestions):
        col_idx = i % 3
        if cols[col_idx].button(suggestion, key=f"nutrition_suggestion_{i}"):
            st.session_state['search_query'] = suggestion
    
    st.markdown("---")
    
    # แสดงตัวอย่างเมนูทั่วไป
    st.markdown("#### 🍽️ หรือเลือกจากเมนูยอดนิยม:")
    sample_recipes = data['name'].head(6).tolist()
    cols = st.columns(3)
    for i, recipe_name in enumerate(sample_recipes):
        col_idx = i % 3
        if cols[col_idx].button(f"🥘 {recipe_name}", key=f"sample_{i}"):
            st.session_state['search_query'] = recipe_name
    
    # เริ่มต้น session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "search_query" not in st.session_state:
        st.session_state.search_query = ""
    
    # แสดงประวัติการสนทนา
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                if "nutrition_recommendations" in message:
                    display_enhanced_nutrition_recommendations(message["nutrition_recommendations"], search_engine, show_detailed_analysis)
                elif "recipe" in message:
                    recipe = message["recipe"]
                    nutrition_data = message.get("nutrition_data")
                    similarity_score = message.get("similarity_score")
                    display_enhanced_recipe_with_nutrition(recipe, nutrition_data, similarity_score)
                else:
                    st.markdown(message["content"])
            else:
                st.markdown(message["content"])
    
    # ช่องค้นหา
    search_query = st.session_state.get('search_query', '')
    if prompt := st.chat_input("🔍 ค้นหาสูตรอาหาร หรือถามเกี่ยวกับโภชนาการ เช่น 'อาหารแคลอรี่ต่ำ' หรือ 'ผัดกะเพรา'...", key="main_chat"):
        search_query = prompt
        st.session_state.search_query = ""
    
    if search_query:
        st.session_state.messages.append({"role": "user", "content": search_query})
        
        with st.chat_message("user"):
            st.markdown(search_query)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 กำลังค้นหาด้วยระบบ AI ขั้นสูง..."):
                
                # ลองค้นหาตามโภชนาการก่อน
                nutrition_recommendations = search_engine.get_enhanced_nutrition_recommendations(
                    search_query, limit=search_limit
                )
                
                if nutrition_recommendations['type'] != 'not_nutrition_query':
                    # พบการค้นหาตามโภชนาการ
                    display_enhanced_nutrition_recommendations(nutrition_recommendations, search_engine, show_detailed_analysis)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "nutrition_recommendations": nutrition_recommendations
                    })
                    
                    # แสดงเมนูที่เกี่ยวข้องเพิ่มเติม
                    if nutrition_recommendations['type'] == 'nutrition_recommendations':
                        st.markdown("#### 🔗 คำแนะนำเพิ่มเติม")
                        related_suggestions = [s for s in nutrition_suggestions if s != search_query][:3]
                        
                        suggestion_cols = st.columns(len(related_suggestions))
                        for i, suggestion in enumerate(related_suggestions):
                            if suggestion_cols[i].button(f"💡 {suggestion}", key=f"related_{i}"):
                                st.session_state.search_query = suggestion
                                st.rerun()
                    
                else:
                    # ค้นหาแบบปกติ (ตามชื่อเมนู)
                    regular_results = search_by_recipe_name(search_query, data, search_limit)
                    
                    if regular_results:
                        best_match = regular_results[0]
                        recipe_name, similarity, recipe_idx = best_match
                        
                        # คำนวณโภชนาการ
                        try:
                            nutrition_data = nutrition_api.calculate_recipe_nutrition(
                                data.iloc[recipe_idx]['ingredient'], use_api, True
                            )
                        except:
                            nutrition_data = None
                        
                        recipe = {
                            'name': recipe_name,
                            'ingredient': data.iloc[recipe_idx]['ingredient'],
                            'method': data.iloc[recipe_idx]['method']
                        }
                        
                        st.markdown(f"🎯 **พบสูตรอาหารที่ตรงกับการค้นหา: {recipe_name}**")
                        
                        display_enhanced_recipe_with_nutrition(recipe, nutrition_data, similarity)
                        
                        # แนะนำเมนูอื่น
                        if len(regular_results) > 1:
                            st.markdown("#### 🔍 เมนูอื่นที่น่าสนใจ:")
                            other_results = regular_results[1:min(4, len(regular_results))]
                            
                            cols = st.columns(len(other_results))
                            for i, (other_name, other_sim, _) in enumerate(other_results):
                                similarity_percent = other_sim * 100
                                if cols[i].button(f"🍽️ {other_name}\n({similarity_percent:.0f}% ตรง)", key=f"other_{i}"):
                                    st.session_state.search_query = other_name
                                    st.rerun()
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "recipe": recipe,
                            "nutrition_data": nutrition_data,
                            "similarity_score": similarity
                        })
                    else:
                        # ไม่พบผลลัพธ์
                        response = f"""
                        ❌ ไม่พบสูตรอาหารที่ตรงกับ '{search_query}'
                        
                        💡 **คำแนะนำ:**
                        - ลองใช้คำค้นหาที่ง่ายกว่า เช่น "ไข่เจียว" 
                        - ตรวจสอบการสะกดคำภาษาไทย
                        - ลองค้นหาตามโภชนาการ เช่น "อาหารแคลอรี่ต่ำ"
                        
                        🍽️ **เมนูที่มีในระบบ:** {', '.join(data['name'].head(8).tolist())}
                        """
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

def search_by_recipe_name(query: str, data: pd.DataFrame, limit: int = 5) -> List[Tuple[str, float, int]]:
    """ค้นหาสูตรอาหารตามชื่อ"""
    query_lower = query.lower().strip()
    matches = []
    
    # ลบคำที่ไม่จำเป็น
    stop_words = ["อาหาร", "เมนู", "สูตร", "วิธีทำ", "ทำ", "ปรุง"]
    query_words = [word for word in query_lower.split() if word not in stop_words and len(word) > 1]
    clean_query = " ".join(query_words) if query_words else query_lower
    
    for idx, recipe_name in enumerate(data['name']):
        recipe_name_lower = recipe_name.lower()
        
        # คำนวณความคล้ายคลึง
        sequence_similarity = SequenceMatcher(None, clean_query, recipe_name_lower).ratio()
        
        # ตรวจสอบคำที่ตรงกันทั้งหมด
        exact_match_score = 0
        if clean_query in recipe_name_lower:
            exact_match_score = min(len(clean_query) / len(recipe_name_lower), 1.0)
        
        # ตรวจสอบคำแยกกัน
        word_scores = []
        for q_word in query_words:
            if len(q_word) <= 1:
                continue
            best_match_score = 0
            for r_word in recipe_name_lower.split():
                word_similarity = SequenceMatcher(None, q_word, r_word).ratio()
                if word_similarity >= 0.8:
                    best_match_score = max(best_match_score, word_similarity)
                elif len(q_word) >= 3 and q_word in r_word:
                    best_match_score = max(best_match_score, 0.7)
            if best_match_score > 0:
                word_scores.append(best_match_score)
        
        word_match_score = sum(word_scores) / len(query_words) if query_words and word_scores else 0
        
        # คำนวณคะแนนรวม
        final_score = max([
            sequence_similarity * 0.3,
            exact_match_score * 0.9,
            word_match_score * 0.7
        ])
        
        if final_score >= 0.3:  # เกณฑ์การค้นหา
            matches.append((recipe_name, final_score, idx))
    
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[:limit]

def display_enhanced_nutrition_recommendations(recommendations: Dict, search_engine, show_detailed: bool = True):
    """แสดงผลคำแนะนำอาหารตามโภชนาการแบบขั้นสูง"""
    if recommendations['type'] == 'no_results':
        st.warning(recommendations['message'])
        if 'suggestions' in recommendations:
            st.markdown("#### 💡 ลองค้นหาคำเหล่านี้แทน:")
            for suggestion in recommendations['suggestions']:
                if st.button(f"🔍 {suggestion}", key=f"suggestion_{suggestion}"):
                    st.session_state.search_query = suggestion
                    st.rerun()
        return
    
    results = recommendations['results']
    
    # แสดงหัวข้อและสถิติ
    st.markdown(f"### 🎯 ผลการค้นหา: {recommendations['query']}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 จำนวนเมนูที่พบ", recommendations['total_found'])
    with col2:
        if 'search_analysis' in recommendations:
            avg_score = recommendations['search_analysis'].get('average_score', 0)
            st.metric("⭐ คะแนนเฉลี่ย", f"{avg_score:.1f}")
    with col3:
        if 'search_analysis' in recommendations:
            nutrition_grades = recommendations['search_analysis'].get('nutrition_grades', {})
            if nutrition_grades:
                best_grade = min(nutrition_grades.keys())
                st.metric("🏆 เกรดดีที่สุด", best_grade)
    
    # แสดงการวิเคราะห์ผลการค้นหา
    if show_detailed and 'search_analysis' in recommendations:
        analysis = recommendations['search_analysis']
        
        if analysis.get('common_benefits'):
            st.markdown("#### 🌟 ประโยชน์ที่พบบ่อยในเมนูที่แนะนำ:")
            benefits = sorted(analysis['common_benefits'].items(), key=lambda x: x[1], reverse=True)
            benefit_cols = st.columns(min(len(benefits), 3))
            for i, (benefit, count) in enumerate(benefits[:3]):
                benefit_cols[i].metric(benefit, f"{count} เมนู")
    
    # แสดงรายการแนะนำ
    for i, recipe in enumerate(results, 1):
        with st.expander(f"🥘 {i}. {recipe['name']} (คะแนน: {recipe['score']:.1f} | เกรด: {recipe['nutrition_grade']})"):
            col1, col2 = st.columns([1.2, 0.8])
            
            with col1:
                st.markdown("#### ✨ เหมาะสมเพราะ")
                for reason in recipe['reasons']:
                    st.markdown(f"• {reason}")
                
                if recipe['health_benefits']:
                    st.markdown("#### 🌿 ประโยชน์ต่อสุขภาพ")
                    for benefit in recipe['health_benefits']:
                        st.markdown(f"• {benefit}")
                
                # วัตถุดิบ (แสดงเฉพาะส่วนหลัก)
                if recipe.get('ingredient'):
                    st.markdown("#### 🥬 วัตถุดิบหลัก")
                    ingredients = recipe['ingredient'].split('\n')[:4]
                    for ingredient in ingredients:
                        if ingredient.strip():
                            st.write(f"• {ingredient.strip().lstrip('- ')}")
            
            with col2:
                st.markdown("#### 📊 ค่าโภชนาการ")
                nutrition = recipe['nutrition']
                
                # แสดงค่าโภชนาการหลัก
                metrics_data = [
                    ("🔥 แคลอรี่", f"{nutrition['calories']:.0f} kcal"),
                    ("🥩 โปรตีน", f"{nutrition['protein']:.1f} g"),
                    ("🍞 คาร์โบ", f"{nutrition['carbs']:.1f} g"),
                    ("🫒 ไขมัน", f"{nutrition['fat']:.1f} g"),
                    ("🌾 ใยอาหาร", f"{nutrition['fiber']:.1f} g"),
                    ("🧂 โซเดียม", f"{nutrition['sodium']:.0f} mg")
                ]
                
                for label, value in metrics_data:
                    st.write(f"**{label}:** {value}")
                
                # คะแนนย่อยตามหมวดหมู่
                if show_detailed and recipe.get('category_scores'):
                    st.markdown("#### 🏅 คะแนนย่อย")
                    for category, score in recipe['category_scores'].items():
                        if score > 0:
                            st.write(f"• {category}: {score:.1f}")
            
            # ปุ่มดูรายละเอียดเพิ่มเติม
            if st.button(f"👁️ ดูรายละเอียด {recipe['name']}", key=f"detail_{i}"):
                st.session_state.search_query = recipe['name']
                st.rerun()

if __name__ == "__main__":
    main()
