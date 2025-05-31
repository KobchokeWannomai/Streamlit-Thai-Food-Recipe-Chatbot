import streamlit as st
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Tuple, Any
from difflib import SequenceMatcher
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class NutritionBasedSearchEngine:
    """เครื่องมือค้นหาแนะนำอาหารตามคุณค่าทางโภชนาการ"""
    
    def __init__(self, data: pd.DataFrame, nutrition_api):
        self.data = data
        self.nutrition_api = nutrition_api
        self.nutrition_cache = {}
        
        # คำสำคัญสำหรับการค้นหาตามโภชนาการ
        self.nutrition_keywords = {
            # แคลอรี่
            'calories_low': ['แคลอรี่ต่ำ', 'แคลต่ำ', 'ลดน้ำหนัก', 'เบา', 'ไม่อ้วน', 'ไดเอท', 'diet'],
            'calories_high': ['แคลอรี่สูง', 'แคลสูง', 'เพิ่มน้ำหนัก', 'พลังงานสูง', 'เติมแรง'],
            
            # โปรตีน
            'protein_high': ['โปรตีนสูง', 'โปรตีนมาก', 'เนื้อเยื่อ', 'กล้ามเนื้อ', 'นักกีฬา', 'ออกกำลังกาย'],
            'protein_low': ['โปรตีนต่ำ', 'โปรตีนน้อย'],
            
            # ไขมัน
            'fat_low': ['ไขมันต่ำ', 'ไขมันน้อย', 'ลดไขมัน', 'ไม่มันเยอะ', 'สุขภาพดี'],
            'fat_high': ['ไขมันสูง', 'ไขมันมาก'],
            
            # คาร์โบไฮเดรต
            'carbs_low': ['คาร์โบต่ำ', 'แป้งน้อย', 'น้ำตาลต่ำ', 'เบาหวาน', 'คีโต', 'keto'],
            'carbs_high': ['คาร์โบสูง', 'แป้งมาก', 'พลังงาน'],
            
            # ใยอาหาร
            'fiber_high': ['ใยอาหารสูง', 'ใยอาหารมาก', 'ขับถ่าย', 'ท้องผูก', 'ย่อย'],
            
            # วิตามิน
            'vitamin_a_high': ['วิตามินเอสูง', 'วิตามินเอ', 'สายตา', 'ผิวพรรณ'],
            'vitamin_c_high': ['วิตามินซีสูง', 'วิตามินซี', 'ภูมิคุ้มกัน', 'ต้านหวัด'],
            'vitamin_b_high': ['วิตามินบี', 'ระบบประสาท', 'เมแทบอลิซึม'],
            
            # แร่ธาตุ
            'calcium_high': ['แคลเซียมสูง', 'แคลเซียม', 'กระดูก', 'ฟัน', 'ผู้สูงอายุ'],
            'iron_high': ['เหล็กสูง', 'ธาตุเหล็ก', 'โลหิตจาง', 'เลือดจาง'],
            'potassium_high': ['โปแตสเซียมสูง', 'โปแตสเซียม', 'ความดันโลหิต'],
            'sodium_low': ['โซเดียมต่ำ', 'เกลือน้อย', 'ความดันสูง', 'ไต'],
            
            # กลุ่มผู้ป่วยเฉพาะ
            'diabetes': ['เบาหวาน', 'ผู้ป่วยเบาหวาน', 'น้ำตาลต่ำ', 'ควบคุมน้ำตาล'],
            'hypertension': ['ความดันสูง', 'ผู้ป่วยความดัน', 'โซเดียมต่ำ'],
            'elderly': ['ผู้สูงอายุ', 'คนแก่', 'นุ่ม', 'ย่อยง่าย'],
            'children': ['เด็ก', 'เด็กเล็ก', 'แคลเซียม', 'เจริญเติบโต'],
            'athletes': ['นักกีฬา', 'ออกกำลังกาย', 'โปรตีนสูง', 'ฟิตเนส'],
            
            # ประเภทอาหาร
            'vegetarian': ['มังสวิรัติ', 'เจ', 'ไม่กินเนื้อ', 'ผัก'],
            'low_sodium': ['โซเดียมต่ำ', 'เกลือน้อย', 'จืด'],
            'healthy': ['สุขภาพ', 'สุขภาพดี', 'คลีน', 'clean eating']
        }
        
        # เกณฑ์การจัดกลุ่มโภชนาการ (ต่อหนึ่งส่วน)
        self.nutrition_thresholds = {
            'calories': {'low': 300, 'medium': 500, 'high': 700},
            'protein': {'low': 10, 'medium': 20, 'high': 30},
            'fat': {'low': 10, 'medium': 20, 'high': 30},
            'carbs': {'low': 20, 'medium': 40, 'high': 60},
            'fiber': {'low': 3, 'medium': 6, 'high': 10},
            'vitamin_a': {'low': 100, 'medium': 500, 'high': 1000},
            'vitamin_c': {'low': 10, 'medium': 30, 'high': 60},
            'calcium': {'low': 50, 'medium': 150, 'high': 300},
            'iron': {'low': 2, 'medium': 5, 'high': 10},
            'potassium': {'low': 200, 'medium': 400, 'high': 600},
            'sodium': {'low': 400, 'medium': 800, 'high': 1200}
        }
    
    def calculate_all_nutrition(self, use_api=False, adjust_consumption=True) -> Dict:
        """คำนวณค่าโภชนาการสำหรับทุกเมนู"""
        if hasattr(self, '_all_nutrition_cache'):
            return self._all_nutrition_cache
            
        all_nutrition = {}
        
        with st.spinner("🧮 กำลังคำนวณค่าโภชนาการสำหรับทุกเมนู..."):
            progress_bar = st.progress(0)
            
            for idx, recipe in self.data.iterrows():
                try:
                    nutrition_data = self.nutrition_api.calculate_recipe_nutrition(
                        recipe['ingredient'], use_api, adjust_consumption
                    )
                    all_nutrition[idx] = nutrition_data['total_nutrition']
                    progress_bar.progress((idx + 1) / len(self.data))
                except:
                    # ใช้ค่าเริ่มต้นหากคำนวณไม่ได้
                    all_nutrition[idx] = {
                        'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0, 'fiber': 0,
                        'vitamin_a': 0, 'vitamin_c': 0, 'vitamin_b1': 0, 'vitamin_b2': 0,
                        'calcium': 0, 'iron': 0, 'potassium': 0, 'sodium': 0
                    }
            
            progress_bar.empty()
        
        self._all_nutrition_cache = all_nutrition
        return all_nutrition
    
    def detect_nutrition_intent(self, query: str) -> Dict[str, Any]:
        """ตรวจจับความต้องการค้นหาตามโภชนาการ"""
        query_lower = query.lower()
        detected_criteria = []
        
        # ตรวจสอบคำสำคัญ
        for category, keywords in self.nutrition_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    detected_criteria.append(category)
                    break
        
        if not detected_criteria:
            return {'is_nutrition_query': False}
        
        # แปลงเป็นเกณฑ์การค้นหา
        search_criteria = {
            'is_nutrition_query': True,
            'criteria': detected_criteria,
            'query_text': query
        }
        
        return search_criteria
    
    def search_by_nutrition_criteria(self, criteria: List[str], limit: int = 10) -> List[Dict]:
        """ค้นหาเมนูตามเกณฑ์โภชนาการ"""
        all_nutrition = self.calculate_all_nutrition()
        scored_recipes = []
        
        for idx, nutrition in all_nutrition.items():
            score = 0
            reasons = []
            
            for criterion in criteria:
                criterion_score, reason = self._evaluate_nutrition_criterion(criterion, nutrition)
                score += criterion_score
                if reason:
                    reasons.append(reason)
            
            if score > 0:
                recipe_data = self.data.iloc[idx]
                scored_recipes.append({
                    'index': idx,
                    'name': recipe_data['name'],
                    'score': score,
                    'nutrition': nutrition,
                    'reasons': reasons,
                    'ingredient': recipe_data['ingredient'],
                    'method': recipe_data['method']
                })
        
        # เรียงลำดับตามคะแนน
        scored_recipes.sort(key=lambda x: x['score'], reverse=True)
        return scored_recipes[:limit]
    
    def _evaluate_nutrition_criterion(self, criterion: str, nutrition: Dict) -> Tuple[float, str]:
        """ประเมินคะแนนตามเกณฑ์โภชนาการ"""
        thresholds = self.nutrition_thresholds
        
        # แคลอรี่ต่ำ
        if criterion == 'calories_low':
            if nutrition['calories'] <= thresholds['calories']['low']:
                return 10.0, f"แคลอรี่ต่ำ ({nutrition['calories']:.0f} kcal)"
            elif nutrition['calories'] <= thresholds['calories']['medium']:
                return 5.0, f"แคลอรี่ปานกลาง ({nutrition['calories']:.0f} kcal)"
        
        # แคลอรี่สูง
        elif criterion == 'calories_high':
            if nutrition['calories'] >= thresholds['calories']['high']:
                return 10.0, f"แคลอรี่สูง ({nutrition['calories']:.0f} kcal)"
        
        # โปรตีนสูง
        elif criterion == 'protein_high':
            if nutrition['protein'] >= thresholds['protein']['high']:
                return 10.0, f"โปรตีนสูง ({nutrition['protein']:.1f} g)"
            elif nutrition['protein'] >= thresholds['protein']['medium']:
                return 7.0, f"โปรตีนปานกลาง ({nutrition['protein']:.1f} g)"
        
        # ไขมันต่ำ
        elif criterion == 'fat_low':
            if nutrition['fat'] <= thresholds['fat']['low']:
                return 10.0, f"ไขมันต่ำ ({nutrition['fat']:.1f} g)"
            elif nutrition['fat'] <= thresholds['fat']['medium']:
                return 5.0, f"ไขมันปานกลาง ({nutrition['fat']:.1f} g)"
        
        # คาร์โบไฮเดรตต่ำ
        elif criterion == 'carbs_low':
            if nutrition['carbs'] <= thresholds['carbs']['low']:
                return 10.0, f"คาร์โบไฮเดรตต่ำ ({nutrition['carbs']:.1f} g)"
            elif nutrition['carbs'] <= thresholds['carbs']['medium']:
                return 5.0, f"คาร์โบไฮเดรตปานกลาง ({nutrition['carbs']:.1f} g)"
        
        # ใยอาหารสูง
        elif criterion == 'fiber_high':
            if nutrition['fiber'] >= thresholds['fiber']['high']:
                return 10.0, f"ใยอาหารสูง ({nutrition['fiber']:.1f} g)"
            elif nutrition['fiber'] >= thresholds['fiber']['medium']:
                return 7.0, f"ใยอาหารปานกลาง ({nutrition['fiber']:.1f} g)"
        
        # วิตามินเอสูง
        elif criterion == 'vitamin_a_high':
            if nutrition['vitamin_a'] >= thresholds['vitamin_a']['high']:
                return 10.0, f"วิตามินเอสูง ({nutrition['vitamin_a']:.0f} IU)"
            elif nutrition['vitamin_a'] >= thresholds['vitamin_a']['medium']:
                return 7.0, f"วิตามินเอปานกลาง ({nutrition['vitamin_a']:.0f} IU)"
        
        # วิตามินซีสูง
        elif criterion == 'vitamin_c_high':
            if nutrition['vitamin_c'] >= thresholds['vitamin_c']['high']:
                return 10.0, f"วิตามินซีสูง ({nutrition['vitamin_c']:.1f} mg)"
            elif nutrition['vitamin_c'] >= thresholds['vitamin_c']['medium']:
                return 7.0, f"วิตามินซีปานกลาง ({nutrition['vitamin_c']:.1f} mg)"
        
        # แคลเซียมสูง
        elif criterion == 'calcium_high':
            if nutrition['calcium'] >= thresholds['calcium']['high']:
                return 10.0, f"แคลเซียมสูง ({nutrition['calcium']:.0f} mg)"
            elif nutrition['calcium'] >= thresholds['calcium']['medium']:
                return 7.0, f"แคลเซียมปานกลาง ({nutrition['calcium']:.0f} mg)"
        
        # เหล็กสูง
        elif criterion == 'iron_high':
            if nutrition['iron'] >= thresholds['iron']['high']:
                return 10.0, f"เหล็กสูง ({nutrition['iron']:.1f} mg)"
            elif nutrition['iron'] >= thresholds['iron']['medium']:
                return 7.0, f"เหล็กปานกลาง ({nutrition['iron']:.1f} mg)"
        
        # โซเดียมต่ำ
        elif criterion == 'sodium_low':
            if nutrition['sodium'] <= thresholds['sodium']['low']:
                return 10.0, f"โซเดียมต่ำ ({nutrition['sodium']:.0f} mg)"
            elif nutrition['sodium'] <= thresholds['sodium']['medium']:
                return 5.0, f"โซเดียมปานกลาง ({nutrition['sodium']:.0f} mg)"
        
        # เกณฑ์สำหรับผู้ป่วยเฉพาะ
        elif criterion == 'diabetes':
            score = 0
            reasons_list = []
            if nutrition['carbs'] <= thresholds['carbs']['low']:
                score += 8
                reasons_list.append(f"คาร์โบไฮเดรตต่ำ ({nutrition['carbs']:.1f} g)")
            if nutrition['fiber'] >= thresholds['fiber']['medium']:
                score += 5
                reasons_list.append(f"ใยอาหารดี ({nutrition['fiber']:.1f} g)")
            if nutrition['sodium'] <= thresholds['sodium']['medium']:
                score += 3
                reasons_list.append(f"โซเดียมพอดี ({nutrition['sodium']:.0f} mg)")
            return score, "เหมาะสำหรับผู้ป่วยเบาหวาน: " + ", ".join(reasons_list)
        
        elif criterion == 'hypertension':
            score = 0
            reasons_list = []
            if nutrition['sodium'] <= thresholds['sodium']['low']:
                score += 10
                reasons_list.append(f"โซเดียมต่ำ ({nutrition['sodium']:.0f} mg)")
            if nutrition['potassium'] >= thresholds['potassium']['medium']:
                score += 5
                reasons_list.append(f"โปแตสเซียมดี ({nutrition['potassium']:.0f} mg)")
            return score, "เหมาะสำหรับผู้ป่วยความดันสูง: " + ", ".join(reasons_list)
        
        return 0.0, ""
    
    def get_nutrition_recommendations(self, query: str, limit: int = 5) -> Dict:
        """ให้คำแนะนำอาหารตามคำถามเกี่ยวกับโภชนาการ"""
        intent = self.detect_nutrition_intent(query)
        
        if not intent['is_nutrition_query']:
            return {'type': 'not_nutrition_query'}
        
        results = self.search_by_nutrition_criteria(intent['criteria'], limit)
        
        if not results:
            return {
                'type': 'no_results',
                'message': f'ไม่พบเมนูที่ตรงกับเกณฑ์ "{query}" โปรดลองคำค้นหาอื่น'
            }
        
        return {
            'type': 'nutrition_recommendations',
            'query': query,
            'criteria': intent['criteria'],
            'results': results,
            'total_found': len(results)
        }
    
    def create_nutrition_comparison_chart(self, recipes: List[Dict], nutrients: List[str] = None) -> go.Figure:
        """สร้างกราฟเปรียบเทียบค่าโภชนาการ"""
        if not nutrients:
            nutrients = ['calories', 'protein', 'carbs', 'fat', 'fiber']
        
        recipe_names = [recipe['name'][:15] + '...' if len(recipe['name']) > 15 
                       else recipe['name'] for recipe in recipes[:5]]
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=['แคลอรี่ (kcal)', 'โปรตีน (g)', 'คาร์โบไฮเดรต (g)', 
                           'ไขมัน (g)', 'ใยอาหาร (g)', 'โซเดียม (mg)'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        nutrients_to_plot = ['calories', 'protein', 'carbs', 'fat', 'fiber', 'sodium']
        positions = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]
        
        for i, nutrient in enumerate(nutrients_to_plot):
            if i < len(recipes):
                values = [recipe['nutrition'][nutrient] for recipe in recipes[:5]]
                
                fig.add_trace(
                    go.Bar(
                        x=recipe_names,
                        y=values,
                        name=nutrient.title(),
                        marker_color=colors[i % len(colors)],
                        showlegend=False
                    ),
                    row=positions[i][0], col=positions[i][1]
                )
        
        fig.update_layout(
            height=600,
            title_text="📊 การเปรียบเทียบค่าโภชนาการ",
            title_x=0.5,
            font=dict(family="Sarabun, sans-serif")
        )
        
        return fig
    
    def display_nutrition_recommendations(self, recommendations: Dict):
        """แสดงผลคำแนะนำอาหารตามโภชนาการ"""
        if recommendations['type'] == 'not_nutrition_query':
            return False
        
        if recommendations['type'] == 'no_results':
            st.warning(recommendations['message'])
            return True
        
        results = recommendations['results']
        
        # แสดงหัวข้อ
        st.markdown(f"### 🎯 ผลการค้นหา: {recommendations['query']}")
        st.markdown(f"**พบ {recommendations['total_found']} เมนูที่เหมาะสม**")
        
        # แสดงกราฟเปรียบเทียบ
        if len(results) > 1:
            fig = self.create_nutrition_comparison_chart(results)
            st.plotly_chart(fig, use_container_width=True)
        
        # แสดงรายการแนะนำ
        for i, recipe in enumerate(results, 1):
            with st.expander(f"🥘 {i}. {recipe['name']} (คะแนน: {recipe['score']:.1f})"):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("#### 🥬 วัตถุดิบ")
                    ingredients_formatted = self._format_ingredients(recipe['ingredient'])
                    st.markdown(ingredients_formatted, unsafe_allow_html=True)
                    
                    st.markdown("#### ✨ เหมาะสำหรับ")
                    for reason in recipe['reasons']:
                        st.markdown(f"• {reason}")
                
                with col2:
                    st.markdown("#### 📊 ค่าโภชนาการ")
                    nutrition = recipe['nutrition']
                    
                    # แสดงค่าโภชนาการหลัก
                    metrics_col1, metrics_col2 = st.columns(2)
                    with metrics_col1:
                        st.metric("🔥 แคลอรี่", f"{nutrition['calories']:.0f} kcal")
                        st.metric("🥩 โปรตีน", f"{nutrition['protein']:.1f} g")
                        st.metric("🍞 คาร์โบ", f"{nutrition['carbs']:.1f} g")
                    
                    with metrics_col2:
                        st.metric("🫒 ไขมัน", f"{nutrition['fat']:.1f} g")
                        st.metric("🌾 ใยอาหาร", f"{nutrition['fiber']:.1f} g")
                        st.metric("🧂 โซเดียม", f"{nutrition['sodium']:.0f} mg")
                    
                    # แสดงวิตามินและแร่ธาตุ
                    with st.expander("วิตามินและแร่ธาตุ"):
                        vit_col1, vit_col2 = st.columns(2)
                        with vit_col1:
                            st.write(f"🅰️ วิตามิน A: {nutrition['vitamin_a']:.0f} IU")
                            st.write(f"🍊 วิตามิน C: {nutrition['vitamin_c']:.1f} mg")
                        with vit_col2:
                            st.write(f"🦴 แคลเซียม: {nutrition['calcium']:.0f} mg")
                            st.write(f"⚡ เหล็ก: {nutrition['iron']:.1f} mg")
                
                # แสดงวิธีทำ
                st.markdown("#### 👨‍🍳 วิธีทำ")
                method_formatted = self._format_cooking_method(recipe['method'])
                st.markdown(method_formatted, unsafe_allow_html=True)
        
        return True
    
    def _format_ingredients(self, ingredients_text: str) -> str:
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
    
    def _format_cooking_method(self, method_text: str) -> str:
        """จัดรูปแบบวิธีทำ"""
        if not method_text:
            return "<p>ไม่มีข้อมูลวิธีทำ</p>"
        
        # ตรวจสอบว่ามีเลขขั้นตอนอยู่แล้วหรือไม่
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
    
    def get_nutrition_suggestions(self) -> List[str]:
        """ให้คำแนะนำการค้นหาตามโภชนาการ"""
        return [
            "🔥 แนะนำอาหารแคลอรี่ต่ำ",
            "💪 เมนูโปรตีนสูงสำหรับนักกีฬา", 
            "🥗 อาหารไขมันต่ำเพื่อสุขภาพ",
            "🍃 เมนูใยอาหารสูงช่วยขับถ่าย",
            "🧂 อาหารโซเดียมต่ำสำหรับผู้ป่วยความดันสูง",
            "🍎 เมนูวิตามินซีสูงเสริมภูมิคุ้มกัน",
            "🦴 อาหารแคลเซียมสูงบำรุงกระดูก",
            "🩸 เมนูเหล็กสูงป้องกันโลหิตจาง",
            "🌿 อาหารเหมาะสำหรับผู้ป่วยเบาหวาน",
            "👶 เมนูเหมาะสำหรับเด็ก"
        ]
