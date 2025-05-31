import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any
import math

class NutritionAnalyzer:
    """คลาสสำหรับวิเคราะห์และแสดงกราฟข้อมูลโภชนาการขั้นสูง"""
    
    def __init__(self):
        # ค่าแนะนำการบริโภคประจำวัน (RDA) สำหรับผู้ใหญ่
        self.daily_recommendations = {
            'calories': {'male': 2500, 'female': 2000, 'child': 1500, 'elderly': 1800},
            'protein': {'male': 65, 'female': 50, 'child': 30, 'elderly': 55},
            'carbs': {'male': 300, 'female': 250, 'child': 180, 'elderly': 220},
            'fat': {'male': 70, 'female': 55, 'child': 40, 'elderly': 50},
            'fiber': {'male': 30, 'female': 25, 'child': 15, 'elderly': 25},
            'vitamin_a': {'male': 900, 'female': 700, 'child': 400, 'elderly': 800},
            'vitamin_c': {'male': 90, 'female': 75, 'child': 45, 'elderly': 80},
            'calcium': {'male': 1000, 'female': 1200, 'child': 800, 'elderly': 1200},
            'iron': {'male': 8, 'female': 18, 'child': 10, 'elderly': 8},
            'sodium': {'male': 2300, 'female': 2300, 'child': 1500, 'elderly': 2000}
        }
        
        # ข้อมูลสำหรับการให้คำแนะนำเฉพาะโรค
        self.disease_recommendations = {
            'diabetes': {
                'carbs_limit': 45,  # % ของแคลอรี่ทั้งหมด
                'fiber_min': 25,    # กรัม
                'sodium_limit': 2000,  # มิลลิกรัม
                'sugar_limit': 25   # กรัม
            },
            'hypertension': {
                'sodium_limit': 1500,  # มิลลิกรัม
                'potassium_min': 3500,  # มิลลิกรัม
                'fat_limit': 30,    # % ของแคลอรี่
                'fiber_min': 25     # กรัม
            },
            'heart_disease': {
                'saturated_fat_limit': 7,  # % ของแคลอรี่
                'cholesterol_limit': 200,   # มิลลิกรัม
                'sodium_limit': 2000,      # มิลลิกรัม
                'fiber_min': 30            # กรัม
            },
            'kidney_disease': {
                'protein_limit': 0.8,  # กรัม/กก. น้ำหนัก
                'sodium_limit': 2000,   # มิลลิกรัม
                'potassium_limit': 2000, # มิลลิกรัม
                'phosphorus_limit': 800  # มิลลิกรัม
            }
        }
        
        # สีสำหรับกราฟ
        self.color_palette = {
            'primary': '#667eea',
            'secondary': '#764ba2',
            'success': '#28a745',
            'warning': '#ffc107',
            'danger': '#dc3545',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }

    def create_nutrition_radar_chart(self, nutrition_data: Dict, target_group: str = 'male') -> go.Figure:
        """สร้างกราฟเรดาร์แสดงค่าโภชนาการเทียบกับค่าแนะนำ"""
        
        nutrients = ['calories', 'protein', 'carbs', 'fat', 'fiber', 'vitamin_c', 'calcium', 'iron']
        values = []
        recommendations = []
        percentages = []
        
        for nutrient in nutrients:
            value = nutrition_data.get(nutrient, 0)
            recommended = self.daily_recommendations[nutrient][target_group]
            percentage = min((value / recommended) * 100, 200)  # จำกัดไม่เกิน 200%
            
            values.append(value)
            recommendations.append(recommended)
            percentages.append(percentage)
        
        # ชื่อสารอาหารภาษาไทย
        nutrient_names = [
            'แคลอรี่<br>(kcal)',
            'โปรตีน<br>(g)', 
            'คาร์โบ<br>(g)',
            'ไขมัน<br>(g)',
            'ใยอาหาร<br>(g)',
            'วิตามิน C<br>(mg)',
            'แคลเซียม<br>(mg)',
            'เหล็ก<br>(mg)'
        ]
        
        fig = go.Figure()
        
        # เพิ่มเส้นค่าแนะนำ (100%)
        fig.add_trace(go.Scatterpolar(
            r=[100] * len(nutrients),
            theta=nutrient_names,
            fill='toself',
            fillcolor='rgba(40, 167, 69, 0.1)',
            line=dict(color='rgba(40, 167, 69, 0.8)', width=2, dash='dash'),
            name='ค่าแนะนำ (100%)'
        ))
        
        # เพิ่มค่าจริง
        fig.add_trace(go.Scatterpolar(
            r=percentages,
            theta=nutrient_names,
            fill='toself',
            fillcolor='rgba(102, 126, 234, 0.3)',
            line=dict(color='rgba(102, 126, 234, 0.8)', width=3),
            name='ค่าจริง'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 200],
                    ticksuffix='%',
                    tickfont=dict(size=10)
                ),
                angularaxis=dict(
                    tickfont=dict(size=11)
                )
            ),
            title={
                'text': f'📊 การเปรียบเทียบค่าโภชนาการกับค่าแนะนำ<br><sub>กลุ่มเป้าหมาย: {target_group}</sub>',
                'x': 0.5,
                'font': {'size': 16}
            },
            font=dict(family="Sarabun, sans-serif"),
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig

    def create_nutrient_distribution_chart(self, recipes: List[Dict]) -> go.Figure:
        """สร้างกราฟแสดงการกระจายของสารอาหารในหลายเมนู"""
        
        if not recipes:
            return go.Figure()
        
        # เตรียมข้อมูล
        recipe_names = [recipe['name'][:15] + '...' if len(recipe['name']) > 15 
                       else recipe['name'] for recipe in recipes[:8]]
        
        calories = [recipe['nutrition']['calories'] for recipe in recipes[:8]]
        proteins = [recipe['nutrition']['protein'] for recipe in recipes[:8]]
        carbs = [recipe['nutrition']['carbs'] for recipe in recipes[:8]]
        fats = [recipe['nutrition']['fat'] for recipe in recipes[:8]]
        
        # สร้าง subplot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('แคลอรี่ (kcal)', 'โปรตีน (g)', 'คาร์โบไฮเดรต (g)', 'ไขมัน (g)'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # เพิ่มกราฟแต่ละสารอาหาร
        fig.add_trace(
            go.Bar(x=recipe_names, y=calories, name='แคลอรี่',
                   marker_color='#FF6B6B', showlegend=False),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=recipe_names, y=proteins, name='โปรตีน',
                   marker_color='#4ECDC4', showlegend=False),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=recipe_names, y=carbs, name='คาร์โบไฮเดรต',
                   marker_color='#45B7D1', showlegend=False),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(x=recipe_names, y=fats, name='ไขมัน',
                   marker_color='#96CEB4', showlegend=False),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="📈 การเปรียบเทียบสารอาหารหลักระหว่างเมนู",
            title_x=0.5,
            height=600,
            font=dict(family="Sarabun, sans-serif")
        )
        
        # ปรับแต่ง x-axis
        for i in range(1, 3):
            for j in range(1, 3):
                fig.update_xaxes(tickangle=45, row=i, col=j)
        
        return fig

    def create_vitamin_mineral_chart(self, nutrition_data: Dict) -> go.Figure:
        """สร้างกราฟแสดงวิตามินและแร่ธาตุ"""
        
        vitamins = {
            'วิตามิน A': nutrition_data.get('vitamin_a', 0),
            'วิตามิน C': nutrition_data.get('vitamin_c', 0),
            'วิตามิน B1': nutrition_data.get('vitamin_b1', 0) * 1000,  # แปลงเป็น mcg
            'วิตามิน B2': nutrition_data.get('vitamin_b2', 0) * 1000   # แปลงเป็น mcg
        }
        
        minerals = {
            'แคลเซียม': nutrition_data.get('calcium', 0),
            'เหล็ก': nutrition_data.get('iron', 0),
            'โปแตสเซียม': nutrition_data.get('potassium', 0),
            'โซเดียม': nutrition_data.get('sodium', 0)
        }
        
        # สร้าง subplot
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('🍊 วิตามิน', '⚡ แร่ธาตุ'),
            specs=[[{"type": "domain"}, {"type": "domain"}]]
        )
        
        # กราฟวงกลมวิตามิน
        fig.add_trace(
            go.Pie(
                labels=list(vitamins.keys()),
                values=list(vitamins.values()),
                name="วิตามิน",
                marker_colors=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
            ),
            row=1, col=1
        )
        
        # กราฟวงกลมแร่ธาตุ
        fig.add_trace(
            go.Pie(
                labels=list(minerals.keys()),
                values=list(minerals.values()),
                name="แร่ธาตุ",
                marker_colors=['#FFB366', '#66FFB2', '#B366FF', '#FF66B2']
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="🧪 องค์ประกอบวิตามินและแร่ธาตุ",
            title_x=0.5,
            font=dict(family="Sarabun, sans-serif"),
            height=400
        )
        
        return fig

    def create_health_recommendation_chart(self, nutrition_data: Dict, 
                                         disease_type: str = None) -> go.Figure:
        """สร้างกราฟแสดงคำแนะนำสำหรับผู้ป่วยเฉพาะโรค"""
        
        if not disease_type or disease_type not in self.disease_recommendations:
            return go.Figure()
        
        recommendations = self.disease_recommendations[disease_type]
        
        # เตรียมข้อมูลสำหรับแสดงผล
        categories = []
        current_values = []
        recommended_values = []
        status_colors = []
        
        if disease_type == 'diabetes':
            # คาร์โบไฮเดรต (% ของแคลอรี่)
            carb_calories = nutrition_data.get('carbs', 0) * 4
            total_calories = nutrition_data.get('calories', 1)
            carb_percentage = (carb_calories / total_calories) * 100 if total_calories > 0 else 0
            
            categories.extend(['คาร์โบไฮเดรต (%)', 'ใยอาหาร (g)', 'โซเดียม (mg)'])
            current_values.extend([carb_percentage, nutrition_data.get('fiber', 0), 
                                 nutrition_data.get('sodium', 0)])
            recommended_values.extend([recommendations['carbs_limit'], 
                                     recommendations['fiber_min'],
                                     recommendations['sodium_limit']])
            
            # กำหนดสีตามเกณฑ์
            status_colors.extend([
                'green' if carb_percentage <= recommendations['carbs_limit'] else 'red',
                'green' if nutrition_data.get('fiber', 0) >= recommendations['fiber_min'] else 'orange',
                'green' if nutrition_data.get('sodium', 0) <= recommendations['sodium_limit'] else 'red'
            ])
        
        elif disease_type == 'hypertension':
            categories.extend(['โซเดียม (mg)', 'โปแตสเซียม (mg)', 'ใยอาหาร (g)'])
            current_values.extend([nutrition_data.get('sodium', 0),
                                 nutrition_data.get('potassium', 0),
                                 nutrition_data.get('fiber', 0)])
            recommended_values.extend([recommendations['sodium_limit'],
                                     recommendations['potassium_min'],
                                     recommendations['fiber_min']])
            
            status_colors.extend([
                'green' if nutrition_data.get('sodium', 0) <= recommendations['sodium_limit'] else 'red',
                'green' if nutrition_data.get('potassium', 0) >= recommendations['potassium_min'] else 'orange',
                'green' if nutrition_data.get('fiber', 0) >= recommendations['fiber_min'] else 'orange'
            ])
        
        # สร้างกราฟ
        fig = go.Figure()
        
        # แสดงค่าปัจจุบัน
        fig.add_trace(go.Bar(
            name='ค่าปัจจุบัน',
            x=categories,
            y=current_values,
            marker_color=status_colors,
            opacity=0.8
        ))
        
        # แสดงเส้นค่าแนะนำ
        for i, (category, recommended) in enumerate(zip(categories, recommended_values)):
            fig.add_hline(
                y=recommended,
                line_dash="dash",
                line_color="blue",
                annotation_text=f"แนะนำ: {recommended}",
                annotation_position="top right"
            )
        
        disease_names = {
            'diabetes': 'ผู้ป่วยเบาหวาน',
            'hypertension': 'ผู้ป่วยความดันสูง',
            'heart_disease': 'ผู้ป่วยหัวใจ',
            'kidney_disease': 'ผู้ป่วยไต'
        }
        
        fig.update_layout(
            title=f"🏥 คำแนะนำสำหรับ{disease_names.get(disease_type, disease_type)}",
            title_x=0.5,
            xaxis_title="สารอาหาร",
            yaxis_title="ปริมาณ",
            font=dict(family="Sarabun, sans-serif"),
            height=400,
            showlegend=True
        )
        
        return fig

    def calculate_nutrition_score(self, nutrition_data: Dict, target_group: str = 'male') -> Dict:
        """คำนวณคะแนนโภชนาการโดยรวม"""
        
        scores = {}
        total_score = 0
        max_score = 0
        
        important_nutrients = ['calories', 'protein', 'fiber', 'vitamin_c', 'calcium', 'iron']
        
        for nutrient in important_nutrients:
            value = nutrition_data.get(nutrient, 0)
            recommended = self.daily_recommendations[nutrient][target_group]
            
            # คำนวณคะแนนแต่ละสารอาหาร (0-100)
            if nutrient in ['calories']:
                # สำหรับแคลอรี่ คะแนนดีสุดคือ 80-120% ของค่าแนะนำ
                percentage = (value / recommended) * 100
                if 80 <= percentage <= 120:
                    score = 100
                elif 60 <= percentage < 80 or 120 < percentage <= 150:
                    score = 70
                elif 40 <= percentage < 60 or 150 < percentage <= 200:
                    score = 40
                else:
                    score = 10
            else:
                # สำหรับสารอาหารอื่น ยิ่งมากยิ่งดี (จนถึงขีดจำกัด)
                percentage = min((value / recommended) * 100, 200)
                if percentage >= 100:
                    score = 100
                elif percentage >= 75:
                    score = 80
                elif percentage >= 50:
                    score = 60
                elif percentage >= 25:
                    score = 40
                else:
                    score = 20
            
            scores[nutrient] = score
            total_score += score
            max_score += 100
        
        overall_score = (total_score / max_score) * 100 if max_score > 0 else 0
        
        # การประเมินโดยรวม
        if overall_score >= 85:
            grade = 'A'
            comment = 'ดีเยี่ยม! ครบถ้วนสมบูรณ์'
        elif overall_score >= 70:
            grade = 'B'
            comment = 'ดี ควรปรับปรุงบางสารอาหาร'
        elif overall_score >= 55:
            grade = 'C'
            comment = 'ปานกลาง ควรเพิ่มความหลากหลาย'
        elif overall_score >= 40:
            grade = 'D'
            comment = 'ต้องปรับปรุง ขาดสารอาหารสำคัญ'
        else:
            grade = 'F'
            comment = 'ควรเลือกเมนูอื่นที่มีประโยชน์มากกว่า'
        
        return {
            'individual_scores': scores,
            'overall_score': overall_score,
            'grade': grade,
            'comment': comment,
            'total_nutrients_evaluated': len(important_nutrients)
        }

    def create_nutrition_gauge_chart(self, nutrition_score: Dict) -> go.Figure:
        """สร้างกราฟเกจแสดงคะแนนโภชนาการ"""
        
        score = nutrition_score['overall_score']
        grade = nutrition_score['grade']
        
        # กำหนดสีตามคะแนน
        if score >= 85:
            color = "#28a745"  # เขียว
        elif score >= 70:
            color = "#17a2b8"  # ฟ้า
        elif score >= 55:
            color = "#ffc107"  # เหลือง
        elif score >= 40:
            color = "#fd7e14"  # ส้ม
        else:
            color = "#dc3545"  # แดง
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"🎯 คะแนนโภชนาการ<br><span style='font-size:0.8em'>เกรด: {grade}</span>"},
            delta = {'reference': 70, 'increasing': {'color': "#28a745"}, 
                    'decreasing': {'color': "#dc3545"}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 40], 'color': '#ffebee'},
                    {'range': [40, 55], 'color': '#fff3e0'},
                    {'range': [55, 70], 'color': '#fffde7'},
                    {'range': [70, 85], 'color': '#e8f5e8'},
                    {'range': [85, 100], 'color': '#e0f2f1'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        
        fig.update_layout(
            font=dict(family="Sarabun, sans-serif", size=14),
            height=400,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        return fig

    def generate_nutrition_advice(self, nutrition_data: Dict, nutrition_score: Dict, 
                                target_group: str = 'male') -> Dict:
        """สร้างคำแนะนำโภชนาการแบบส่วนบุคคล"""
        
        advice = {
            'strengths': [],
            'improvements': [],
            'specific_recommendations': [],
            'meal_suggestions': []
        }
        
        scores = nutrition_score['individual_scores']
        recommendations = self.daily_recommendations
        
        # วิเคราะห์จุดแข็ง
        for nutrient, score in scores.items():
            if score >= 80:
                advice['strengths'].append(f"✅ {nutrient}: ได้รับเพียงพอ")
        
        # วิเคราะห์จุดที่ต้องปรับปรุง
        for nutrient, score in scores.items():
            current = nutrition_data.get(nutrient, 0)
            recommended = recommendations[nutrient][target_group]
            
            if score < 60:
                if nutrient == 'calories':
                    if current < recommended * 0.8:
                        advice['improvements'].append(f"⚠️ แคลอรี่ต่ำเกินไป (ปัจจุบัน: {current:.0f}, แนะนำ: {recommended})")
                    else:
                        advice['improvements'].append(f"⚠️ แคลอรี่สูงเกินไป (ปัจจุบัน: {current:.0f}, แนะนำ: {recommended})")
                else:
                    advice['improvements'].append(f"📈 ควรเพิ่ม{nutrient} (ปัจจุบัน: {current:.1f}, แนะนำ: {recommended})")
        
        # คำแนะนำเฉพาะ
        if nutrition_data.get('sodium', 0) > 2000:
            advice['specific_recommendations'].append("🧂 ลดการใช้เกลือและซอสต่างๆ")
        
        if nutrition_data.get('fiber', 0) < 20:
            advice['specific_recommendations'].append("🥬 เพิ่มผักใบเขียวและผลไม้")
        
        if nutrition_data.get('protein', 0) < recommendations['protein'][target_group] * 0.8:
            advice['specific_recommendations'].append("🍖 เพิ่มโปรตีนจากเนื้อสัตว์หรือถั่ว")
        
        # แนะนำมื้ออาหาร
        if nutrition_score['overall_score'] < 70:
            advice['meal_suggestions'] = [
                "🥗 เพิ่มสลัดผักใบเขียวเป็นเครื่องเคียง",
                "🍊 ทานผลไม้หลังอาหาร",
                "🥛 ดื่มนมหรือผลิตภัณฑ์จากนม",
                "🐟 เลือกปลาแทนเนื้อแดงบางมื้อ"
            ]
        
        return advice

    def display_comprehensive_nutrition_analysis(self, nutrition_data: Dict, 
                                               recipe_name: str = "เมนูนี้",
                                               target_group: str = 'male'):
        """แสดงการวิเคราะห์โภชนาการแบบครอบคลุม"""
        
        st.markdown(f"## 🔬 การวิเคราะห์โภชนาการแบบละเอียด: {recipe_name}")
        
        # คำนวณคะแนน
        nutrition_score = self.calculate_nutrition_score(nutrition_data, target_group)
        
        # แสดงคะแนนรวม
        col1, col2 = st.columns([1, 2])
        
        with col1:
            gauge_fig = self.create_nutrition_gauge_chart(nutrition_score)
            st.plotly_chart(gauge_fig, use_container_width=True)
            
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); border-radius: 10px; margin: 1rem 0;">
                <h3 style="margin: 0; color: #1976d2;">📊 สรุปผล</h3>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>คะแนน: {nutrition_score['overall_score']:.1f}/100</strong></p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>เกรด: {nutrition_score['grade']}</strong></p>
                <p style="margin: 0; font-style: italic;">{nutrition_score['comment']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            radar_fig = self.create_nutrition_radar_chart(nutrition_data, target_group)
            st.plotly_chart(radar_fig, use_container_width=True)
        
        # แสดงกราฟวิตามินและแร่ธาตุ
        vitamin_mineral_fig = self.create_vitamin_mineral_chart(nutrition_data)
        st.plotly_chart(vitamin_mineral_fig, use_container_width=True)
        
        # คำแนะนำ
        advice = self.generate_nutrition_advice(nutrition_data, nutrition_score, target_group)
        
        if advice['strengths'] or advice['improvements']:
            col1, col2 = st.columns(2)
            
            with col1:
                if advice['strengths']:
                    st.markdown("### ✅ จุดแข็ง")
                    for strength in advice['strengths']:
                        st.markdown(f"- {strength}")
            
            with col2:
                if advice['improvements']:
                    st.markdown("### 📈 ควรปรับปรุง")
                    for improvement in advice['improvements']:
                        st.markdown(f"- {improvement}")
        
        if advice['specific_recommendations']:
            st.markdown("### 💡 คำแนะนำเฉพาะ")
            for rec in advice['specific_recommendations']:
                st.markdown(f"- {rec}")
        
        if advice['meal_suggestions']:
            st.markdown("### 🍽️ แนะนำอาหารเสริม")
            for suggestion in advice['meal_suggestions']:
                st.markdown(f"- {suggestion}")

    def create_weekly_nutrition_tracker(self, weekly_meals: List[Dict]) -> go.Figure:
        """สร้างกราฟติดตามโภชนาการรายสัปดาห์"""
        
        days = ['จันทร์', 'อังคาร', 'พุธ', 'พฤหัสบดี', 'ศุกร์', 'เสาร์', 'อาทิตย์']
        calories_by_day = []
        protein_by_day = []
        
        for day_meals in weekly_meals:
            daily_calories = sum(meal.get('calories', 0) for meal in day_meals)
            daily_protein = sum(meal.get('protein', 0) for meal in day_meals)
            calories_by_day.append(daily_calories)
            protein_by_day.append(daily_protein)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('แคลอรี่รายวัน', 'โปรตีนรายวัน'),
            shared_xaxes=True
        )
        
        # กราฟแคลอรี่
        fig.add_trace(
            go.Scatter(x=days, y=calories_by_day, mode='lines+markers',
                      name='แคลอรี่', line=dict(color='#FF6B6B', width=3),
                      marker=dict(size=8)),
            row=1, col=1
        )
        
        # เส้นค่าแนะนำแคลอรี่
        fig.add_hline(y=2000, line_dash="dash", line_color="red", 
                     annotation_text="แนะนำ: 2000 kcal", row=1, col=1)
        
        # กราฟโปรตีน
        fig.add_trace(
            go.Scatter(x=days, y=protein_by_day, mode='lines+markers',
                      name='โปรตีน', line=dict(color='#4ECDC4', width=3),
                      marker=dict(size=8)),
            row=2, col=1
        )
        
        # เส้นค่าแนะนำโปรตีน
        fig.add_hline(y=50, line_dash="dash", line_color="blue",
                     annotation_text="แนะนำ: 50g", row=2, col=1)
        
        fig.update_layout(
            title_text="📅 การติดตามโภชนาการรายสัปดาห์",
            title_x=0.5,
            height=600,
            font=dict(family="Sarabun, sans-serif"),
            showlegend=False
        )
        
        return fig
