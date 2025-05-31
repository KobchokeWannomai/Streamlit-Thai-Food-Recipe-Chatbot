import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any
import json

class PersonalNutritionManager:
    """ระบบจัดการโภชนาการส่วนบุคคลและติดตามเป้าหมายสุขภาพ"""
    
    def __init__(self):
        self.health_goals = {
            'weight_loss': {
                'name': 'ลดน้ำหนัก',
                'calories_reduction': 500,  # ลดแคลอรี่วันละ 500 kcal
                'protein_ratio': 0.25,     # 25% ของแคลอรี่จากโปรตีน
                'carb_ratio': 0.40,        # 40% จากคาร์โบไฮเดรต
                'fat_ratio': 0.35,         # 35% จากไขมัน
                'fiber_min': 30,           # ใยอาหารขั้นต่ำ 30g
                'water_min': 2500          # น้ำขั้นต่ำ 2.5L
            },
            'weight_gain': {
                'name': 'เพิ่มน้ำหนัก',
                'calories_addition': 500,   # เพิ่มแคลอรี่วันละ 500 kcal
                'protein_ratio': 0.20,     # 20% ของแคลอรี่จากโปรตีน
                'carb_ratio': 0.50,        # 50% จากคาร์โบไฮเดรต
                'fat_ratio': 0.30,         # 30% จากไขมัน
                'protein_min': 80,         # โปรตีนขั้นต่ำ 80g
                'water_min': 3000          # น้ำขั้นต่ำ 3L
            },
            'muscle_building': {
                'name': 'เพิ่มกล้ามเนื้อ',
                'calories_addition': 300,   # เพิ่มแคลอรี่วันละ 300 kcal
                'protein_ratio': 0.30,     # 30% ของแคลอรี่จากโปรตีน
                'carb_ratio': 0.40,        # 40% จากคาร์โบไฮเดรต
                'fat_ratio': 0.30,         # 30% จากไขมัน
                'protein_min': 100,        # โปรตีนขั้นต่ำ 100g
                'water_min': 3500          # น้ำขั้นต่ำ 3.5L
            },
            'maintenance': {
                'name': 'รักษาน้ำหนัก',
                'calories_change': 0,       # ไม่เปลี่ยนแคลอรี่
                'protein_ratio': 0.20,     # 20% ของแคลอรี่จากโปรตีน
                'carb_ratio': 0.50,        # 50% จากคาร์โบไฮเดรต
                'fat_ratio': 0.30,         # 30% จากไขมัน
                'fiber_min': 25,           # ใยอาหารขั้นต่ำ 25g
                'water_min': 2000          # น้ำขั้นต่ำ 2L
            },
            'diabetes_control': {
                'name': 'ควบคุมเบาหวาน',
                'protein_ratio': 0.20,     # 20% ของแคลอรี่จากโปรตีน
                'carb_ratio': 0.45,        # 45% จากคาร์โบไฮเดรต (ต่ำกว่าปกติ)
                'fat_ratio': 0.35,         # 35% จากไขมัน
                'fiber_min': 35,           # ใยอาหารสูง 35g
                'sodium_max': 2000,        # โซเดียมสูงสุด 2000mg
                'sugar_max': 25            # น้ำตาลสูงสุด 25g
            },
            'heart_health': {
                'name': 'ดูแลหัวใจ',
                'protein_ratio': 0.15,     # 15% ของแคลอรี่จากโปรตีน
                'carb_ratio': 0.55,        # 55% จากคาร์โบไฮเดรต
                'fat_ratio': 0.30,         # 30% จากไขมัน (ไขมันดี)
                'fiber_min': 30,           # ใยอาหารสูง 30g
                'sodium_max': 1500,        # โซเดียมต่ำ 1500mg
                'omega3_min': 2            # โอเมก้า 3 ขั้นต่ำ 2g
            }
        }
        
        self.activity_levels = {
            'sedentary': {'name': 'นั่งทำงาน', 'multiplier': 1.2},
            'light': {'name': 'ออกกำลังกายเบา', 'multiplier': 1.375},
            'moderate': {'name': 'ออกกำลังกายปานกลาง', 'multiplier': 1.55},
            'active': {'name': 'ออกกำลังกายหนัก', 'multiplier': 1.725},
            'very_active': {'name': 'ออกกำลังกายหนักมาก', 'multiplier': 1.9}
        }

    def calculate_bmr(self, weight: float, height: float, age: int, gender: str) -> float:
        """คำนวณ BMR (Basal Metabolic Rate) ด้วยสูตร Mifflin-St Jeor"""
        if gender.lower() == 'male':
            bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
        else:
            bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161
        return bmr

    def calculate_tdee(self, bmr: float, activity_level: str) -> float:
        """คำนวณ TDEE (Total Daily Energy Expenditure)"""
        multiplier = self.activity_levels[activity_level]['multiplier']
        return bmr * multiplier

    def calculate_bmi(self, weight: float, height: float) -> Tuple[float, str, str]:
        """คำนวณ BMI และประเมินสถานะ"""
        height_m = height / 100  # แปลงเซนติเมตรเป็นเมตร
        bmi = weight / (height_m ** 2)
        
        if bmi < 18.5:
            status = 'น้ำหนักต่ำกว่าเกณฑ์'
            color = '#ffc107'
        elif 18.5 <= bmi < 25:
            status = 'น้ำหนักปกติ'
            color = '#28a745'
        elif 25 <= bmi < 30:
            status = 'น้ำหนักเกิน'
            color = '#fd7e14'
        else:
            status = 'อ้วน'
            color = '#dc3545'
        
        return bmi, status, color

    def get_personalized_nutrition_target(self, user_profile: Dict, health_goal: str) -> Dict:
        """คำนวณเป้าหมายโภชนาการส่วนบุคคล"""
        bmr = self.calculate_bmr(
            user_profile['weight'], 
            user_profile['height'], 
            user_profile['age'], 
            user_profile['gender']
        )
        
        tdee = self.calculate_tdee(bmr, user_profile['activity_level'])
        
        goal_config = self.health_goals.get(health_goal, self.health_goals['maintenance'])
        
        # คำนวณแคลอรี่เป้าหมาย
        if 'calories_reduction' in goal_config:
            target_calories = tdee - goal_config['calories_reduction']
        elif 'calories_addition' in goal_config:
            target_calories = tdee + goal_config['calories_addition']
        else:
            target_calories = tdee
        
        # คำนวณแมโครนิวเทรียนต์
        protein_calories = target_calories * goal_config['protein_ratio']
        carb_calories = target_calories * goal_config['carb_ratio']
        fat_calories = target_calories * goal_config['fat_ratio']
        
        target_protein = protein_calories / 4  # 1g โปรตีน = 4 kcal
        target_carbs = carb_calories / 4       # 1g คาร์โบไฮเดรต = 4 kcal
        target_fat = fat_calories / 9          # 1g ไขมัน = 9 kcal
        
        targets = {
            'calories': target_calories,
            'protein': max(target_protein, goal_config.get('protein_min', 0)),
            'carbs': target_carbs,
            'fat': target_fat,
            'fiber': goal_config.get('fiber_min', 25),
            'water': goal_config.get('water_min', 2000),
            'sodium': goal_config.get('sodium_max', 2300),
            'goal_name': goal_config['name']
        }
        
        return targets

    def create_user_profile_form(self) -> Dict:
        """สร้างแบบฟอร์มโปรไฟล์ผู้ใช้"""
        st.markdown("### 👤 ข้อมูลส่วนบุคคล")
        
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("ชื่อ", value=st.session_state.get('user_name', ''))
            age = st.number_input("อายุ (ปี)", min_value=10, max_value=100, 
                                value=st.session_state.get('user_age', 30))
            weight = st.number_input("น้ำหนัก (กิโลกรัม)", min_value=30.0, max_value=200.0, 
                                   value=st.session_state.get('user_weight', 60.0), step=0.1)
            
        with col2:
            gender = st.selectbox("เพศ", options=['male', 'female'], 
                                format_func=lambda x: 'ชาย' if x == 'male' else 'หญิง',
                                index=0 if st.session_state.get('user_gender', 'male') == 'male' else 1)
            height = st.number_input("ส่วนสูง (เซนติเมตร)", min_value=100, max_value=220, 
                                   value=st.session_state.get('user_height', 170))
            activity_level = st.selectbox("ระดับกิจกรรม", 
                                        options=list(self.activity_levels.keys()),
                                        format_func=lambda x: self.activity_levels[x]['name'],
                                        index=list(self.activity_levels.keys()).index(
                                            st.session_state.get('user_activity', 'moderate')))
        
        # คำนวณและแสดง BMI
        bmi, bmi_status, bmi_color = self.calculate_bmi(weight, height)
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {bmi_color}22 0%, {bmi_color}11 100%); 
                    padding: 1rem; border-radius: 10px; border-left: 4px solid {bmi_color}; margin: 1rem 0;">
            <h4 style="margin: 0; color: {bmi_color};">📊 ค่าดัชนีมวลกาย (BMI)</h4>
            <p style="margin: 0.5rem 0; font-size: 1.2em;"><strong>BMI: {bmi:.1f}</strong></p>
            <p style="margin: 0; font-style: italic;">สถานะ: {bmi_status}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # เลือกเป้าหมายสุขภาพ
        st.markdown("### 🎯 เป้าหมายสุขภาพ")
        health_goal = st.selectbox("เลือกเป้าหมายของคุณ",
                                 options=list(self.health_goals.keys()),
                                 format_func=lambda x: self.health_goals[x]['name'],
                                 index=list(self.health_goals.keys()).index(
                                     st.session_state.get('user_goal', 'maintenance')))
        
        # โรคประจำตัว
        st.markdown("### 🏥 ข้อมูลสุขภาพ")
        health_conditions = st.multiselect("โรคประจำตัว (หากมี)",
                                         options=['diabetes', 'hypertension', 'heart_disease', 'kidney_disease', 'none'],
                                         format_func=lambda x: {
                                             'diabetes': 'เบาหวาน',
                                             'hypertension': 'ความดันโลหิตสูง',
                                             'heart_disease': 'โรคหัวใจ',
                                             'kidney_disease': 'โรคไต',
                                             'none': 'ไม่มี'
                                         }[x],
                                         default=st.session_state.get('user_conditions', ['none']))
        
        food_allergies = st.text_input("อาหารที่แพ้ (หากมี)", 
                                     value=st.session_state.get('user_allergies', ''))
        
        user_profile = {
            'name': name,
            'age': age,
            'weight': weight,
            'height': height,
            'gender': gender,
            'activity_level': activity_level,
            'health_goal': health_goal,
            'health_conditions': health_conditions,
            'food_allergies': food_allergies,
            'bmi': bmi,
            'bmi_status': bmi_status
        }
        
        # บันทึกลง session state
        for key, value in user_profile.items():
            st.session_state[f'user_{key.replace("health_", "").replace("food_", "")}'] = value
        
        return user_profile

    def display_nutrition_targets(self, user_profile: Dict):
        """แสดงเป้าหมายโภชนาการส่วนบุคคล"""
        targets = self.get_personalized_nutrition_target(user_profile, user_profile['health_goal'])
        
        st.markdown(f"### 🎯 เป้าหมายโภชนาการสำหรับ: {targets['goal_name']}")
        
        # แสดงเป้าหมายในรูปแบบ metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🔥 แคลอรี่", f"{targets['calories']:.0f} kcal")
            st.metric("🥩 โปรตีน", f"{targets['protein']:.0f} g")
            
        with col2:
            st.metric("🍞 คาร์โบไฮเดรต", f"{targets['carbs']:.0f} g")
            st.metric("🫒 ไขมัน", f"{targets['fat']:.0f} g")
            
        with col3:
            st.metric("🌾 ใยอาหาร", f"{targets['fiber']:.0f} g")
            st.metric("💧 น้ำ", f"{targets['water']:.0f} ml")
            
        with col4:
            if 'sodium' in targets:
                st.metric("🧂 โซเดียม (สูงสุด)", f"{targets['sodium']:.0f} mg")
        
        return targets

    def create_daily_nutrition_tracker(self):
        """สร้างระบบติดตามโภชนาการรายวัน"""
        st.markdown("### 📅 บันทึกการบริโภคประจำวัน")
        
        # เลือกวันที่
        selected_date = st.date_input("เลือกวันที่", value=date.today())
        
        # ดึงข้อมูลที่บันทึกไว้
        date_key = selected_date.strftime("%Y-%m-%d")
        daily_meals = st.session_state.get(f'daily_meals_{date_key}', [])
        
        # แสดงมื้ออาหารที่บันทึกแล้ว
        if daily_meals:
            st.markdown("#### 🍽️ มื้ออาหารที่บันทึกแล้ว")
            
            total_nutrition = {
                'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0, 'fiber': 0,
                'vitamin_a': 0, 'vitamin_c': 0, 'calcium': 0, 'iron': 0, 'sodium': 0
            }
            
            for i, meal in enumerate(daily_meals):
                with st.expander(f"{meal['meal_type']}: {meal['name']} ({meal['portion']} ส่วน)"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"🔥 แคลอรี่: {meal['nutrition']['calories']:.0f} kcal")
                        st.write(f"🥩 โปรตีน: {meal['nutrition']['protein']:.1f} g")
                        st.write(f"🍞 คาร์โบ: {meal['nutrition']['carbs']:.1f} g")
                        st.write(f"🫒 ไขมัน: {meal['nutrition']['fat']:.1f} g")
                    
                    with col2:
                        st.write(f"🌾 ใยอาหาร: {meal['nutrition']['fiber']:.1f} g")
                        st.write(f"🅰️ วิตามิน A: {meal['nutrition']['vitamin_a']:.0f} IU")
                        st.write(f"🍊 วิตามิน C: {meal['nutrition']['vitamin_c']:.1f} mg")
                        st.write(f"🧂 โซเดียม: {meal['nutrition']['sodium']:.0f} mg")
                    
                    if st.button(f"ลบ {meal['name']}", key=f"delete_meal_{i}"):
                        daily_meals.pop(i)
                        st.session_state[f'daily_meals_{date_key}'] = daily_meals
                        st.rerun()
                
                # รวมค่าโภชนาการ
                for nutrient in total_nutrition:
                    total_nutrition[nutrient] += meal['nutrition'][nutrient] * meal['portion']
            
            # แสดงสรุปโภชนาการรวม
            self.display_daily_nutrition_summary(total_nutrition, selected_date)
        
        # เพิ่มมื้ออาหารใหม่
        with st.expander("➕ เพิ่มมื้ออาหารใหม่", expanded=not daily_meals):
            meal_type = st.selectbox("ประเภทมื้อ", 
                                   options=['เช้า', 'กลางวัน', 'เย็น', 'ว่าง'],
                                   key="new_meal_type")
            
            meal_name = st.text_input("ชื่ออาหาร", key="new_meal_name")
            portion_size = st.number_input("จำนวนส่วน", min_value=0.1, max_value=10.0, 
                                         value=1.0, step=0.1, key="new_meal_portion")
            
            # สำหรับความง่าย ให้ใส่ข้อมูลโภชนาการโดยประมาณ
            col1, col2 = st.columns(2)
            with col1:
                calories = st.number_input("แคลอรี่ (kcal)", min_value=0, value=0, key="meal_calories")
                protein = st.number_input("โปรตีน (g)", min_value=0.0, value=0.0, step=0.1, key="meal_protein")
                carbs = st.number_input("คาร์โบไฮเดรต (g)", min_value=0.0, value=0.0, step=0.1, key="meal_carbs")
            
            with col2:
                fat = st.number_input("ไขมัน (g)", min_value=0.0, value=0.0, step=0.1, key="meal_fat")
                fiber = st.number_input("ใยอาหาร (g)", min_value=0.0, value=0.0, step=0.1, key="meal_fiber")
                sodium = st.number_input("โซเดียม (mg)", min_value=0, value=0, key="meal_sodium")
            
            if st.button("บันทึกมื้ออาหาร"):
                if meal_name:
                    new_meal = {
                        'meal_type': meal_type,
                        'name': meal_name,
                        'portion': portion_size,
                        'nutrition': {
                            'calories': calories,
                            'protein': protein,
                            'carbs': carbs,
                            'fat': fat,
                            'fiber': fiber,
                            'vitamin_a': 0,  # ค่าเริ่มต้น
                            'vitamin_c': 0,
                            'calcium': 0,
                            'iron': 0,
                            'sodium': sodium
                        },
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    daily_meals.append(new_meal)
                    st.session_state[f'daily_meals_{date_key}'] = daily_meals
                    st.success(f"บันทึก {meal_name} เรียบร้อยแล้ว!")
                    st.rerun()
                else:
                    st.error("กรุณาใส่ชื่ออาหาร")

    def display_daily_nutrition_summary(self, total_nutrition: Dict, selected_date: date):
        """แสดงสรุปโภชนาการประจำวัน"""
        st.markdown("#### 📊 สรุปโภชนาการประจำวัน")
        
        # ดึงเป้าหมายของผู้ใช้
        if 'user_weight' in st.session_state:
            user_profile = {
                'weight': st.session_state.get('user_weight', 60),
                'height': st.session_state.get('user_height', 170),
                'age': st.session_state.get('user_age', 30),
                'gender': st.session_state.get('user_gender', 'male'),
                'activity_level': st.session_state.get('user_activity', 'moderate'),
                'health_goal': st.session_state.get('user_goal', 'maintenance')
            }
            
            targets = self.get_personalized_nutrition_target(user_profile, user_profile['health_goal'])
            
            # สร้างกราฟเปรียบเทียบ
            fig = self.create_daily_progress_chart(total_nutrition, targets)
            st.plotly_chart(fig, use_container_width=True)
            
            # แสดงเปอร์เซ็นต์ความสำเร็จ
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                cal_percent = (total_nutrition['calories'] / targets['calories']) * 100
                st.metric("🔥 แคลอรี่", f"{total_nutrition['calories']:.0f} / {targets['calories']:.0f}",
                         delta=f"{cal_percent:.0f}%")
            
            with col2:
                protein_percent = (total_nutrition['protein'] / targets['protein']) * 100
                st.metric("🥩 โปรตีน", f"{total_nutrition['protein']:.1f} / {targets['protein']:.0f}",
                         delta=f"{protein_percent:.0f}%")
            
            with col3:
                carb_percent = (total_nutrition['carbs'] / targets['carbs']) * 100
                st.metric("🍞 คาร์โบ", f"{total_nutrition['carbs']:.1f} / {targets['carbs']:.0f}",
                         delta=f"{carb_percent:.0f}%")
            
            with col4:
                fat_percent = (total_nutrition['fat'] / targets['fat']) * 100
                st.metric("🫒 ไขมัน", f"{total_nutrition['fat']:.1f} / {targets['fat']:.0f}",
                         delta=f"{fat_percent:.0f}%")

    def create_daily_progress_chart(self, actual: Dict, targets: Dict) -> go.Figure:
        """สร้างกราฟแสดงความก้าวหน้าประจำวัน"""
        nutrients = ['calories', 'protein', 'carbs', 'fat', 'fiber']
        actual_values = [actual.get(nutrient, 0) for nutrient in nutrients]
        target_values = [targets.get(nutrient, 0) for nutrient in nutrients]
        percentages = [(actual.get(nutrient, 0) / targets.get(nutrient, 1)) * 100 
                      for nutrient in nutrients]
        
        nutrient_names = ['แคลอรี่', 'โปรตีน', 'คาร์โบ', 'ไขมัน', 'ใยอาหาร']
        
        # กำหนดสีตามเปอร์เซ็นต์
        colors = []
        for pct in percentages:
            if 80 <= pct <= 120:
                colors.append('#28a745')  # เขียว - ดี
            elif 60 <= pct < 80 or 120 < pct <= 150:
                colors.append('#ffc107')  # เหลือง - ปานกลาง
            else:
                colors.append('#dc3545')  # แดง - ต้องปรับปรุง
        
        fig = go.Figure()
        
        # แท่งกราฟแสดงค่าจริง
        fig.add_trace(go.Bar(
            name='ค่าจริง',
            x=nutrient_names,
            y=actual_values,
            marker_color=colors,
            opacity=0.8,
            text=[f'{val:.0f}' for val in actual_values],
            textposition='outside'
        ))
        
        # เส้นแสดงเป้าหมาย
        fig.add_trace(go.Scatter(
            name='เป้าหมาย',
            x=nutrient_names,
            y=target_values,
            mode='markers+lines',
            line=dict(color='red', width=3, dash='dash'),
            marker=dict(color='red', size=10)
        ))
        
        fig.update_layout(
            title='📈 ความก้าวหน้าโภชนาการประจำวัน',
            title_x=0.5,
            xaxis_title='สารอาหาร',
            yaxis_title='ปริมาณ',
            font=dict(family="Sarabun, sans-serif"),
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig

    def create_weekly_progress_chart(self):
        """สร้างกราฟแสดงความก้าวหน้ารายสัปดาห์"""
        st.markdown("### 📅 ความก้าวหน้ารายสัปดาห์")
        
        # เลือกสัปดาห์
        end_date = st.date_input("สิ้นสุดสัปดาห์", value=date.today())
        start_date = end_date - timedelta(days=6)
        
        daily_calories = []
        daily_protein = []
        daily_carbs = []
        daily_fat = []
        dates = []
        
        for i in range(7):
            current_date = start_date + timedelta(days=i)
            date_key = current_date.strftime("%Y-%m-%d")
            daily_meals = st.session_state.get(f'daily_meals_{date_key}', [])
            
            total_calories = sum(meal['nutrition']['calories'] * meal['portion'] for meal in daily_meals)
            total_protein = sum(meal['nutrition']['protein'] * meal['portion'] for meal in daily_meals)
            total_carbs = sum(meal['nutrition']['carbs'] * meal['portion'] for meal in daily_meals)
            total_fat = sum(meal['nutrition']['fat'] * meal['portion'] for meal in daily_meals)
            
            daily_calories.append(total_calories)
            daily_protein.append(total_protein)
            daily_carbs.append(total_carbs)
            daily_fat.append(total_fat)
            dates.append(current_date.strftime("%m/%d"))
        
        # สร้างกราฟ
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('แคลอรี่ (kcal)', 'โปรตีน (g)', 'คาร์โบไฮเดรต (g)', 'ไขมัน (g)'),
            shared_xaxes=True
        )
        
        fig.add_trace(go.Scatter(x=dates, y=daily_calories, mode='lines+markers', 
                               name='แคลอรี่', line=dict(color='#FF6B6B')), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=daily_protein, mode='lines+markers',
                               name='โปรตีน', line=dict(color='#4ECDC4')), row=1, col=2)
        fig.add_trace(go.Scatter(x=dates, y=daily_carbs, mode='lines+markers',
                               name='คาร์โบ', line=dict(color='#45B7D1')), row=2, col=1)
        fig.add_trace(go.Scatter(x=dates, y=daily_fat, mode='lines+markers',
                               name='ไขมัน', line=dict(color='#96CEB4')), row=2, col=2)
        
        # เพิ่มเส้นเป้าหมาย (ถ้ามีข้อมูลผู้ใช้)
        if 'user_weight' in st.session_state:
            user_profile = {
                'weight': st.session_state.get('user_weight', 60),
                'height': st.session_state.get('user_height', 170),
                'age': st.session_state.get('user_age', 30),
                'gender': st.session_state.get('user_gender', 'male'),
                'activity_level': st.session_state.get('user_activity', 'moderate'),
                'health_goal': st.session_state.get('user_goal', 'maintenance')
            }
            
            targets = self.get_personalized_nutrition_target(user_profile, user_profile['health_goal'])
            
            fig.add_hline(y=targets['calories'], line_dash="dash", line_color="red", 
                         annotation_text=f"เป้าหมาย: {targets['calories']:.0f}", row=1, col=1)
            fig.add_hline(y=targets['protein'], line_dash="dash", line_color="red",
                         annotation_text=f"เป้าหมาย: {targets['protein']:.0f}", row=1, col=2)
            fig.add_hline(y=targets['carbs'], line_dash="dash", line_color="red",
                         annotation_text=f"เป้าหมาย: {targets['carbs']:.0f}", row=2, col=1)
            fig.add_hline(y=targets['fat'], line_dash="dash", line_color="red",
                         annotation_text=f"เป้าหมาย: {targets['fat']:.0f}", row=2, col=2)
        
        fig.update_layout(
            title_text=f"📊 ความก้าวหน้ารายสัปดาห์ ({start_date.strftime('%d/%m')} - {end_date.strftime('%d/%m')})",
            title_x=0.5,
            height=600,
            font=dict(family="Sarabun, sans-serif"),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def generate_meal_recommendations(self, user_profile: Dict, recipes_data: pd.DataFrame) -> List[Dict]:
        """สร้างคำแนะนำเมนูตามโปรไฟล์ผู้ใช้"""
        targets = self.get_personalized_nutrition_target(user_profile, user_profile['health_goal'])
        
        # กรองเมนูตามเงื่อนไข
        recommended_recipes = []
        
        for idx, recipe in recipes_data.iterrows():
            # คำนวณโภชนาการประมาณการ (ใช้ข้อมูลจากฐานข้อมูล)
            # ในที่นี้เราจะใช้ข้อมูลตัวอย่าง
            recipe_nutrition = {
                'calories': np.random.randint(200, 600),
                'protein': np.random.randint(10, 40),
                'carbs': np.random.randint(20, 60),
                'fat': np.random.randint(5, 30),
                'fiber': np.random.randint(2, 15),
                'sodium': np.random.randint(300, 1500)
            }
            
            # ตรวจสอบความเหมาะสม
            suitability_score = 0
            reasons = []
            
            # ตรวจสอบแคลอรี่
            if recipe_nutrition['calories'] <= targets['calories'] / 3:  # เหมาะสำหรับ 1 มื้อ
                suitability_score += 2
                reasons.append("แคลอรี่เหมาะสม")
            
            # ตรวจสอบโปรตีน
            if recipe_nutrition['protein'] >= targets['protein'] / 4:  # อย่างน้อย 1/4 ของเป้าหมาย
                suitability_score += 2
                reasons.append("โปรตีนดี")
            
            # ตรวจสอบโซเดียม
            if 'sodium' in targets and recipe_nutrition['sodium'] <= targets['sodium'] / 3:
                suitability_score += 1
                reasons.append("โซเดียมต่ำ")
            
            # ตรวจสอบใยอาหาร
            if recipe_nutrition['fiber'] >= 5:
                suitability_score += 1
                reasons.append("ใยอาหารสูง")
            
            # ตรวจสอบโรคประจำตัว
            if 'diabetes' in user_profile.get('health_conditions', []):
                if recipe_nutrition['carbs'] <= 30:  # คาร์โบต่ำ
                    suitability_score += 2
                    reasons.append("เหมาะสำหรับเบาหวาน")
            
            if 'hypertension' in user_profile.get('health_conditions', []):
                if recipe_nutrition['sodium'] <= 500:  # โซเดียมต่ำมาก
                    suitability_score += 2
                    reasons.append("เหมาะสำหรับความดันสูง")
            
            if suitability_score >= 3:  # เกณฑ์การแนะนำ
                recommended_recipes.append({
                    'name': recipe['name'],
                    'score': suitability_score,
                    'reasons': reasons,
                    'nutrition': recipe_nutrition,
                    'ingredient': recipe.get('ingredient', ''),
                    'method': recipe.get('method', '')
                })
        
        # เรียงลำดับตามคะแนน
        recommended_recipes.sort(key=lambda x: x['score'], reverse=True)
        
        return recommended_recipes[:10]  # คืนค่า 10 เมนูแรก

    def display_meal_recommendations(self, recommendations: List[Dict]):
        """แสดงคำแนะนำเมนูอาหาร"""
        st.markdown("### 🍽️ เมนูแนะนำสำหรับคุณ")
        
        if not recommendations:
            st.info("ไม่มีเมนูแนะนำในขณะนี้ กรุณาลองปรับเป้าหมายหรือข้อมูลส่วนตัว")
            return
        
        for i, recipe in enumerate(recommendations[:5], 1):
            with st.expander(f"🥘 {i}. {recipe['name']} (คะแนน: {recipe['score']})"):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("#### ✨ เหมาะสมเพราะ")
                    for reason in recipe['reasons']:
                        st.markdown(f"• {reason}")
                    
                    st.markdown("#### 📊 ค่าโภชนาการ (ประมาณ)")
                    nutrition = recipe['nutrition']
                    st.write(f"🔥 แคลอรี่: {nutrition['calories']} kcal")
                    st.write(f"🥩 โปรตีน: {nutrition['protein']} g")
                    st.write(f"🍞 คาร์โบ: {nutrition['carbs']} g")
                    st.write(f"🫒 ไขมัน: {nutrition['fat']} g")
                
                with col2:
                    if recipe.get('ingredient'):
                        st.markdown("#### 🥬 วัตถุดิบ")
                        ingredients = recipe['ingredient'].split('\n')[:5]  # แสดงแค่ 5 รายการแรก
                        for ingredient in ingredients:
                            if ingredient.strip():
                                st.write(f"• {ingredient.strip().lstrip('- ')}")
                
                if st.button(f"เพิ่ม {recipe['name']} ลงในมื้ออาหารวันนี้", key=f"add_recommended_{i}"):
                    # เพิ่มลงในมื้ออาหารวันนี้
                    today = date.today().strftime("%Y-%m-%d")
                    daily_meals = st.session_state.get(f'daily_meals_{today}', [])
                    
                    new_meal = {
                        'meal_type': 'แนะนำ',
                        'name': recipe['name'],
                        'portion': 1.0,
                        'nutrition': recipe['nutrition'],
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    daily_meals.append(new_meal)
                    st.session_state[f'daily_meals_{today}'] = daily_meals
                    st.success(f"เพิ่ม {recipe['name']} ลงในมื้ออาหารวันนี้แล้ว!")
                    st.rerun()
