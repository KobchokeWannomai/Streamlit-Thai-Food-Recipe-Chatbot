import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any
import math

class NutritionAnalyzer:
    """คลาสสำหรับวิเคราะห์และแสดงกราฟข้อมูลโภชนาการขั้นสูง - เวอร์ชันปรับปรุงใหม่"""
    
    def __init__(self):
        # ค่าแนะนำการบริโภคประจำวัน (RDA) - ปรับปรุงให้ครอบคลุมมากขึ้น
        self.daily_recommendations = {
            'calories': {
                'male_adult': 2500, 'female_adult': 2000, 'male_elderly': 2000, 'female_elderly': 1600,
                'child_2_3': 1000, 'child_4_8': 1400, 'child_9_13': 1800, 'teen_14_18': 2200,
                'pregnant': 2200, 'breastfeeding': 2500, 'athlete_male': 3000, 'athlete_female': 2400
            },
            'protein': {
                'male_adult': 65, 'female_adult': 50, 'male_elderly': 70, 'female_elderly': 55,
                'child_2_3': 20, 'child_4_8': 25, 'child_9_13': 35, 'teen_14_18': 50,
                'pregnant': 70, 'breastfeeding': 75, 'athlete_male': 120, 'athlete_female': 90
            },
            'carbs': {
                'male_adult': 300, 'female_adult': 250, 'male_elderly': 250, 'female_elderly': 200,
                'child_2_3': 130, 'child_4_8': 160, 'child_9_13': 200, 'teen_14_18': 250,
                'pregnant': 280, 'breastfeeding': 300, 'athlete_male': 400, 'athlete_female': 320
            },
            'fat': {
                'male_adult': 70, 'female_adult': 55, 'male_elderly': 60, 'female_elderly': 50,
                'child_2_3': 35, 'child_4_8': 40, 'child_9_13': 50, 'teen_14_18': 60,
                'pregnant': 65, 'breastfeeding': 70, 'athlete_male': 80, 'athlete_female': 65
            },
            'fiber': {
                'male_adult': 30, 'female_adult': 25, 'male_elderly': 28, 'female_elderly': 22,
                'child_2_3': 14, 'child_4_8': 18, 'child_9_13': 22, 'teen_14_18': 25,
                'pregnant': 28, 'breastfeeding': 30, 'athlete_male': 35, 'athlete_female': 30
            },
            'vitamin_a': {
                'male_adult': 900, 'female_adult': 700, 'male_elderly': 900, 'female_elderly': 700,
                'child_2_3': 300, 'child_4_8': 400, 'child_9_13': 600, 'teen_14_18': 750,
                'pregnant': 770, 'breastfeeding': 1300, 'athlete_male': 1000, 'athlete_female': 800
            },
            'vitamin_c': {
                'male_adult': 90, 'female_adult': 75, 'male_elderly': 90, 'female_elderly': 75,
                'child_2_3': 15, 'child_4_8': 25, 'child_9_13': 45, 'teen_14_18': 65,
                'pregnant': 85, 'breastfeeding': 120, 'athlete_male': 120, 'athlete_female': 100
            },
            'vitamin_b1': {
                'male_adult': 1.2, 'female_adult': 1.1, 'male_elderly': 1.2, 'female_elderly': 1.1,
                'child_2_3': 0.5, 'child_4_8': 0.6, 'child_9_13': 0.9, 'teen_14_18': 1.0,
                'pregnant': 1.4, 'breastfeeding': 1.4, 'athlete_male': 1.5, 'athlete_female': 1.3
            },
            'vitamin_b2': {
                'male_adult': 1.3, 'female_adult': 1.1, 'male_elderly': 1.3, 'female_elderly': 1.1,
                'child_2_3': 0.5, 'child_4_8': 0.6, 'child_9_13': 0.9, 'teen_14_18': 1.0,
                'pregnant': 1.4, 'breastfeeding': 1.6, 'athlete_male': 1.6, 'athlete_female': 1.4
            },
            'calcium': {
                'male_adult': 1000, 'female_adult': 1000, 'male_elderly': 1200, 'female_elderly': 1200,
                'child_2_3': 700, 'child_4_8': 1000, 'child_9_13': 1300, 'teen_14_18': 1300,
                'pregnant': 1000, 'breastfeeding': 1000, 'athlete_male': 1200, 'athlete_female': 1200
            },
            'iron': {
                'male_adult': 8, 'female_adult': 18, 'male_elderly': 8, 'female_elderly': 8,
                'child_2_3': 7, 'child_4_8': 10, 'child_9_13': 8, 'teen_14_18': 15,
                'pregnant': 27, 'breastfeeding': 9, 'athlete_male': 12, 'athlete_female': 22
            },
            'potassium': {
                'male_adult': 3500, 'female_adult': 2600, 'male_elderly': 3500, 'female_elderly': 2600,
                'child_2_3': 2000, 'child_4_8': 2300, 'child_9_13': 2500, 'teen_14_18': 3000,
                'pregnant': 2900, 'breastfeeding': 2800, 'athlete_male': 4000, 'athlete_female': 3200
            },
            'sodium': {
                'male_adult': 2300, 'female_adult': 2300, 'male_elderly': 1500, 'female_elderly': 1500,
                'child_2_3': 1000, 'child_4_8': 1200, 'child_9_13': 1500, 'teen_14_18': 2300,
                'pregnant': 2300, 'breastfeeding': 2300, 'athlete_male': 2300, 'athlete_female': 2300
            }
        }
        
        # ข้อมูลสำหรับการให้คำแนะนำเฉพาะโรค - เพิ่มความละเอียด
        self.disease_recommendations = {
            'diabetes_type1': {
                'carbs_percent': 45,  # % ของแคลอรี่ทั้งหมด
                'fiber_min': 25,      # กรัม
                'sodium_max': 2300,   # มิลลิกรัม
                'sugar_max': 25,      # กรัม
                'protein_percent': 20,
                'fat_percent': 35,
                'gi_foods': 'low'     # ดัชนีน้ำตาลต่ำ
            },
            'diabetes_type2': {
                'carbs_percent': 40,  # ต่ำกว่า type 1
                'fiber_min': 30,      # สูงกว่า
                'sodium_max': 2000,   # ควบคุมเข้มกว่า
                'sugar_max': 20,      # จำกัดมากกว่า
                'protein_percent': 25,
                'fat_percent': 35,
                'weight_control': True
            },
            'hypertension_stage1': {
                'sodium_max': 2300,   # มิลลิกรัม
                'potassium_min': 3500, # มิลลิกรัม
                'fat_percent': 30,    # % ของแคลอรี่
                'fiber_min': 25,      # กรัม
                'alcohol_limit': 'moderate'
            },
            'hypertension_stage2': {
                'sodium_max': 1500,   # เข้มงวดกว่า
                'potassium_min': 4000,
                'fat_percent': 25,
                'fiber_min': 30,
                'dash_diet': True
            },
            'heart_disease': {
                'saturated_fat_max': 7,   # % ของแคลอรี่
                'cholesterol_max': 200,   # มิลลิกรัม
                'sodium_max': 1500,       # มิลลิกรัม
                'fiber_min': 30,          # กรัม
                'omega3_min': 2,          # กรัม
                'trans_fat_max': 0        # ไม่มีเลย
            },
            'kidney_disease_stage3': {
                'protein_limit': 0.8,     # กรัม/กก. น้ำหนัก
                'sodium_max': 2000,       # มิลลิกรัม
                'potassium_max': 3000,    # มิลลิกรัม
                'phosphorus_max': 1000    # มิลลิกรัม
            },
            'kidney_disease_stage4': {
                'protein_limit': 0.6,     # จำกัดมากขึ้น
                'sodium_max': 1500,
                'potassium_max': 2000,
                'phosphorus_max': 800
            },
            'liver_disease': {
                'protein_adjust': True,   # ปรับตามสภาพ
                'sodium_max': 2000,
                'alcohol_forbidden': True,
                'vitamin_supplement': True,
                'fat_soluble_vitamins': True
            },
            'gout': {
                'purine_foods': 'avoid',  # หลีกเลี่ยงอาหารพิวรีนสูง
                'alcohol_limit': 'minimal',
                'water_min': 3000,        # มิลลิลิตร
                'vitamin_c_min': 500,     # มิลลิกรัม
                'cherry_extract': True
            },
            'osteoporosis': {
                'calcium_min': 1200,      # มิลลิกรัม
                'vitamin_d_min': 800,     # IU
                'protein_min': 1.2,       # กรัม/กก. น้ำหนัก
                'phosphorus_balance': True,
                'magnesium_min': 320
            },
            'anemia': {
                'iron_min': 18,           # มิลลิกรัม
                'vitamin_c_with_iron': True,
                'folate_min': 400,        # มิลลิกรัม
                'vitamin_b12_min': 2.4,   # มิลลิกรัม
                'avoid_tea_with_meals': True
            }
        }
        
        # สีสำหรับกราฟ - เพิ่มความหลากหลาย
        self.color_palette = {
            'primary': '#667eea',
            'secondary': '#764ba2',
            'success': '#28a745',
            'warning': '#ffc107',
            'danger': '#dc3545',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40',
            'purple': '#6f42c1',
            'pink': '#e83e8c',
            'orange': '#fd7e14',
            'teal': '#20c997',
            'cyan': '#0dcaf0',
            'indigo': '#6610f2'
        }

        # เกณฑ์การประเมินสุขภาพ
        self.health_grades = {
            'excellent': {'min': 90, 'color': '#28a745', 'emoji': '🌟'},
            'very_good': {'min': 80, 'color': '#20c997', 'emoji': '⭐'},
            'good': {'min': 70, 'color': '#17a2b8', 'emoji': '👍'},
            'fair': {'min': 60, 'color': '#ffc107', 'emoji': '⚠️'},
            'poor': {'min': 40, 'color': '#fd7e14', 'emoji': '😐'},
            'very_poor': {'min': 0, 'color': '#dc3545', 'emoji': '❌'}
        }

    def create_enhanced_nutrition_radar_chart(self, nutrition_data: Dict, 
                                            target_group: str = 'male_adult',
                                            comparison_data: List[Dict] = None) -> go.Figure:
        """สร้างกราฟเรดาร์แสดงค่าโภชนาการเทียบกับค่าแนะนำ - เวอร์ชันปรับปรุง"""
        
        nutrients = ['calories', 'protein', 'carbs', 'fat', 'fiber', 'vitamin_c', 'calcium', 'iron']
        
        # คำนวณเปอร์เซ็นต์จากค่าแนะนำ
        percentages = []
        nutrient_names = []
        actual_values = []
        recommended_values = []
        
        for nutrient in nutrients:
            value = nutrition_data.get(nutrient, 0)
            recommended = self.daily_recommendations[nutrient].get(target_group, 
                         self.daily_recommendations[nutrient].get('male_adult', 100))
            
            percentage = min((value / recommended) * 100, 200)  # จำกัดไม่เกิน 200%
            
            percentages.append(percentage)
            actual_values.append(value)
            recommended_values.append(recommended)
            
            # ชื่อสารอาหารภาษาไทยพร้อมค่าจริง
            thai_name = self._get_nutrient_thai_name(nutrient)
            unit = self._get_nutrient_unit(nutrient)
            nutrient_names.append(f'{thai_name}<br>({value:.1f} {unit})')
        
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
            name='ค่าจริง',
            hovertemplate='<b>%{theta}</b><br>ค่าจริง: %{r:.1f}%<extra></extra>'
        ))
        
        # เพิ่มข้อมูลเปรียบเทียบ (ถ้ามี)
        if comparison_data:
            for i, comp_data in enumerate(comparison_data[:2]):  # จำกัดไม่เกิน 2 รายการ
                comp_percentages = []
                for nutrient in nutrients:
                    value = comp_data['nutrition'].get(nutrient, 0)
                    recommended = self.daily_recommendations[nutrient].get(target_group, 100)
                    percentage = min((value / recommended) * 100, 200)
                    comp_percentages.append(percentage)
                
                color = ['rgba(255, 107, 107, 0.6)', 'rgba(150, 206, 180, 0.6)'][i]
                line_color = ['rgba(255, 107, 107, 0.8)', 'rgba(150, 206, 180, 0.8)'][i]
                
                fig.add_trace(go.Scatterpolar(
                    r=comp_percentages,
                    theta=nutrient_names,
                    fill='toself',
                    fillcolor=color,
                    line=dict(color=line_color, width=2),
                    name=comp_data.get('name', f'เปรียบเทียบ {i+1}')
                ))
        
        # เพิ่มเส้นระดับ 50% และ 150%
        fig.add_trace(go.Scatterpolar(
            r=[50] * len(nutrients),
            theta=nutrient_names,
            line=dict(color='rgba(255, 193, 7, 0.5)', width=1, dash='dot'),
            name='50% ของค่าแนะนำ',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=[150] * len(nutrients),
            theta=nutrient_names,
            line=dict(color='rgba(255, 107, 107, 0.5)', width=1, dash='dot'),
            name='150% ของค่าแนะนำ',
            showlegend=False
        ))
        
        # การตั้งค่าเพิ่มเติม
        target_group_thai = self._get_target_group_thai_name(target_group)
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 200],
                    ticksuffix='%',
                    tickfont=dict(size=10),
                    gridcolor='rgba(0,0,0,0.1)'
                ),
                angularaxis=dict(
                    tickfont=dict(size=11),
                    rotation=90,  # หมุนให้เริ่มจากด้านบน
                    direction='clockwise'
                ),
                bgcolor='rgba(0,0,0,0.02)'
            ),
            title={
                'text': f'📊 การเปรียบเทียบค่าโภชนาการกับค่าแนะนำ<br><sub>กลุ่มเป้าหมาย: {target_group_thai}</sub>',
                'x': 0.5,
                'font': {'size': 16}
            },
            font=dict(family="Sarabun, sans-serif"),
            height=600,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig

    def create_enhanced_nutrient_distribution_chart(self, recipes: List[Dict]) -> go.Figure:
        """สร้างกราฟแสดงการกระจายของสารอาหารในหลายเมนู - เวอร์ชันปรับปรุง"""
        
        if not recipes:
            return go.Figure()
        
        # เตรียมข้อมูล
        recipe_names = [recipe['name'][:12] + '...' if len(recipe['name']) > 12 
                       else recipe['name'] for recipe in recipes[:10]]
        
        nutrients = ['calories', 'protein', 'carbs', 'fat', 'fiber', 'sodium']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        # สร้าง subplot 3x2
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'แคลอรี่ (kcal)', 'โปรตีน (g)', 
                'คาร์โบไฮเดรต (g)', 'ไขมัน (g)', 
                'ใยอาหาร (g)', 'โซเดียม (mg)'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        positions = [(1,1), (1,2), (2,1), (2,2), (3,1), (3,2)]
        
        for i, nutrient in enumerate(nutrients):
            values = [recipe['nutrition'].get(nutrient, 0) for recipe in recipes[:10]]
            row, col = positions[i]
            
            # คำนวณสถิติ
            avg_value = np.mean(values) if values else 0
            max_value = np.max(values) if values else 0
            min_value = np.min(values) if values else 0
            
            # สร้างสีแบบ gradient ตามค่า
            normalized_values = []
            if max_value > min_value:
                normalized_values = [(v - min_value) / (max_value - min_value) for v in values]
            else:
                normalized_values = [0.5] * len(values)
            
            bar_colors = [f'rgba({int(255 * (1-norm))}, {int(255 * norm)}, 150, 0.8)' 
                         for norm in normalized_values]
            
            fig.add_trace(
                go.Bar(
                    x=recipe_names,
                    y=values,
                    name=self._get_nutrient_thai_name(nutrient),
                    marker_color=bar_colors,
                    showlegend=False,
                    text=[f'{val:.1f}' if val < 1000 else f'{val:.0f}' for val in values],
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>' + 
                                 f'{self._get_nutrient_thai_name(nutrient)}: ' +
                                 '%{y:.1f} ' + self._get_nutrient_unit(nutrient) +
                                 '<extra></extra>'
                ),
                row=row, col=col
            )
            
            # เพิ่มเส้นค่าเฉลี่ย
            fig.add_hline(
                y=avg_value,
                line_dash="dash",
                line_color="red",
                annotation_text=f"ค่าเฉลี่ย: {avg_value:.1f}",
                annotation_position="top right",
                row=row, col=col
            )
        
        fig.update_layout(
            title_text="📈 การเปรียบเทียบสารอาหารหลักระหว่างเมนู",
            title_x=0.5,
            height=800,
            font=dict(family="Sarabun, sans-serif"),
            plot_bgcolor='rgba(0,0,0,0.02)'
        )
        
        # ปรับแต่ง x-axis
        for i in range(1, 4):
            for j in range(1, 3):
                fig.update_xaxes(tickangle=45, row=i, col=j)
        
        return fig

    def create_advanced_vitamin_mineral_chart(self, nutrition_data: Dict) -> go.Figure:
        """สร้างกราฟแสดงวิตามินและแร่ธาตุ - เวอร์ชันขั้นสูง"""
        
        # จัดกลุ่มข้อมูล
        vitamins = {
            'วิตามิน A': nutrition_data.get('vitamin_a', 0),
            'วิตามิน C': nutrition_data.get('vitamin_c', 0),
            'วิตามิน B1': nutrition_data.get('vitamin_b1', 0) * 1000,  # แปลงเป็น mcg
            'วิตามิน B2': nutrition_data.get('vitamin_b2', 0) * 1000   # แปลงเป็น mcg
        }
        
        minerals = {
            'แคลเซียม': nutrition_data.get('calcium', 0),
            'เหล็ก': nutrition_data.get('iron', 0) * 10,  # ขยายขนาดเพื่อให้เห็นชัด
            'โปแตสเซียม': nutrition_data.get('potassium', 0) / 10,  # ลดขนาดเพื่อให้พอดี
            'โซเดียม': nutrition_data.get('sodium', 0) / 10
        }
        
        # สร้าง subplot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('🍊 วิตามิน (พาย)', '⚡ แร่ธาตุ (พาย)', 
                          '📊 วิตามิน (แท่ง)', '📊 แร่ธาตุ (แท่ง)'),
            specs=[[{"type": "domain"}, {"type": "domain"}],
                   [{"type": "xy"}, {"type": "xy"}]]
        )
        
        # กราฟวงกลมวิตามิน
        vitamin_colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
        fig.add_trace(
            go.Pie(
                labels=list(vitamins.keys()),
                values=list(vitamins.values()),
                name="วิตามิน",
                marker_colors=vitamin_colors,
                hole=.3,
                textinfo='label+percent',
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # กราฟวงกลมแร่ธาตุ
        mineral_colors = ['#FFB366', '#66FFB2', '#B366FF', '#FF66B2']
        fig.add_trace(
            go.Pie(
                labels=list(minerals.keys()),
                values=list(minerals.values()),
                name="แร่ธาตุ",
                marker_colors=mineral_colors,
                hole=.3,
                textinfo='label+percent',
                textposition='outside'
            ),
            row=1, col=2
        )
        
        # กราฟแท่งวิตามิน
        fig.add_trace(
            go.Bar(
                x=list(vitamins.keys()),
                y=list(vitamins.values()),
                name='วิตามิน',
                marker_color=vitamin_colors,
                showlegend=False,
                text=[f'{val:.1f}' for val in vitamins.values()],
                textposition='outside'
            ),
            row=2, col=1
        )
        
        # กราฟแท่งแร่ธาตุ
        fig.add_trace(
            go.Bar(
                x=list(minerals.keys()),
                y=list(minerals.values()),
                name='แร่ธาตุ',
                marker_color=mineral_colors,
                showlegend=False,
                text=[f'{val:.1f}' for val in minerals.values()],
                textposition='outside'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="🧪 องค์ประกอบวิตามินและแร่ธาตุแบบละเอียด",
            title_x=0.5,
            font=dict(family="Sarabun, sans-serif"),
            height=600,
            plot_bgcolor='rgba(0,0,0,0.02)'
        )
        
        # ปรับแต่ง x-axis สำหรับกราฟแท่ง
        fig.update_xaxes(tickangle=45, row=2, col=1)
        fig.update_xaxes(tickangle=45, row=2, col=2)
        
        return fig

    def create_health_risk_assessment_chart(self, nutrition_data: Dict, 
                                          disease_type: str = None) -> go.Figure:
        """สร้างกราฟแสดงการประเมินความเสี่ยงต่อสุขภาพ"""
        
        if not disease_type or disease_type not in self.disease_recommendations:
            return go.Figure()
        
        recommendations = self.disease_recommendations[disease_type]
        
        # เตรียมข้อมูลสำหรับแสดงผล
        categories = []
        current_values = []
        recommended_values = []
        risk_levels = []
        risk_colors = []
        
        # วิเคราะห์ตามโรค
        if 'diabetes' in disease_type:
            # คาร์โบไฮเดรต (% ของแคลอรี่)
            carb_calories = nutrition_data.get('carbs', 0) * 4
            total_calories = nutrition_data.get('calories', 1)
            carb_percentage = (carb_calories / total_calories) * 100 if total_calories > 0 else 0
            
            categories.extend(['คาร์โบไฮเดรต (%)', 'ใยอาหาร (g)', 'โซเดียม (mg)'])
            current_values.extend([carb_percentage, nutrition_data.get('fiber', 0), 
                                 nutrition_data.get('sodium', 0)])
            recommended_values.extend([recommendations['carbs_percent'], 
                                     recommendations['fiber_min'],
                                     recommendations['sodium_max']])
            
            # ประเมินความเสี่ยง
            for i, (current, recommended, category) in enumerate(zip(current_values, recommended_values, categories)):
                if 'โซเดียม' in category:
                    # สำหรับโซเดียม: สูงกว่าแนะนำ = เสี่ยง
                    if current <= recommended * 0.8:
                        risk_levels.append('ต่ำ')
                        risk_colors.append('#28a745')
                    elif current <= recommended:
                        risk_levels.append('ปานกลาง')
                        risk_colors.append('#ffc107')
                    else:
                        risk_levels.append('สูง')
                        risk_colors.append('#dc3545')
                else:
                    # สำหรับสารอื่น: ต่ำกว่าแนะนำ = เสี่ยง
                    if current >= recommended:
                        risk_levels.append('ต่ำ')
                        risk_colors.append('#28a745')
                    elif current >= recommended * 0.8:
                        risk_levels.append('ปานกลาง')
                        risk_colors.append('#ffc107')
                    else:
                        risk_levels.append('สูง')
                        risk_colors.append('#dc3545')
        
        elif 'hypertension' in disease_type:
            categories.extend(['โซเดียม (mg)', 'โปแตสเซียม (mg)', 'ใยอาหาร (g)'])
            current_values.extend([nutrition_data.get('sodium', 0),
                                 nutrition_data.get('potassium', 0),
                                 nutrition_data.get('fiber', 0)])
            recommended_values.extend([recommendations['sodium_max'],
                                     recommendations['potassium_min'],
                                     recommendations['fiber_min']])
            
            # ประเมินความเสี่ยงสำหรับความดันสูง
            for i, (current, recommended, category) in enumerate(zip(current_values, recommended_values, categories)):
                if 'โซเดียม' in category:
                    if current <= recommended * 0.7:
                        risk_levels.append('ต่ำ')
                        risk_colors.append('#28a745')
                    elif current <= recommended:
                        risk_levels.append('ปานกลาง')
                        risk_colors.append('#ffc107')
                    else:
                        risk_levels.append('สูง')
                        risk_colors.append('#dc3545')
                else:
                    if current >= recommended:
                        risk_levels.append('ต่ำ')
                        risk_colors.append('#28a745')
                    elif current >= recommended * 0.8:
                        risk_levels.append('ปานกลาง')
                        risk_colors.append('#ffc107')
                    else:
                        risk_levels.append('สูง')
                        risk_colors.append('#dc3545')
        
        # สร้างกราฟ
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('📊 ค่าปัจจุบัน vs แนะนำ', '⚠️ ระดับความเสี่ยง'),
            specs=[[{"secondary_y": False}, {"type": "domain"}]]
        )
        
        # กราฟแท่งเปรียบเทียบ
        fig.add_trace(
            go.Bar(
                name='ค่าปัจจุบัน',
                x=categories,
                y=current_values,
                marker_color=risk_colors,
                opacity=0.8,
                text=[f'{val:.1f}' for val in current_values],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # เส้นค่าแนะนำ
        for i, (category, recommended) in enumerate(zip(categories, recommended_values)):
            fig.add_hline(
                y=recommended,
                line_dash="dash",
                line_color="blue",
                annotation_text=f"แนะนำ: {recommended}",
                annotation_position="top right",
                row=1, col=1
            )
        
        # กราฟวงกลมความเสี่ยง
        risk_counts = {level: risk_levels.count(level) for level in ['ต่ำ', 'ปานกลาง', 'สูง']}
        risk_chart_colors = ['#28a745', '#ffc107', '#dc3545']
        
        fig.add_trace(
            go.Pie(
                labels=list(risk_counts.keys()),
                values=list(risk_counts.values()),
                name="ความเสี่ยง",
                marker_colors=risk_chart_colors,
                hole=.4,
                textinfo='label+percent',
                textposition='outside'
            ),
            row=1, col=2
        )
        
        disease_names = {
            'diabetes_type1': 'ผู้ป่วยเบาหวานชนิดที่ 1',
            'diabetes_type2': 'ผู้ป่วยเบาหวานชนิดที่ 2',
            'hypertension_stage1': 'ผู้ป่วยความดันสูงระยะที่ 1',
            'hypertension_stage2': 'ผู้ป่วยความดันสูงระยะที่ 2',
            'heart_disease': 'ผู้ป่วยโรคหัวใจ',
            'kidney_disease_stage3': 'ผู้ป่วยไตเรื้อรังระยะที่ 3',
            'kidney_disease_stage4': 'ผู้ป่วยไตเรื้อรังระยะที่ 4'
        }
        
        fig.update_layout(
            title=f"🏥 การประเมินความเสี่ยงสำหรับ{disease_names.get(disease_type, disease_type)}",
            title_x=0.5,
            xaxis_title="สารอาหาร",
            yaxis_title="ปริมาณ",
            font=dict(family="Sarabun, sans-serif"),
            height=500,
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0.02)'
        )
        
        return fig

    def calculate_comprehensive_nutrition_score(self, nutrition_data: Dict, 
                                              target_group: str = 'male_adult') -> Dict:
        """คำนวณคะแนนโภชนาการโดยรวมแบบครอบคลุม - เวอร์ชันปรับปรุง"""
        
        scores = {}
        total_score = 0
        max_score = 0
        
        # สารอาหารสำคัญและน้ำหนักคะแนน
        important_nutrients = {
            'calories': 3,    # น้ำหนักสูง
            'protein': 4,     # น้ำหนักสูงสุด
            'fiber': 3,       # น้ำหนักสูง
            'vitamin_c': 2,   # น้ำหนักปานกลาง
            'calcium': 2,     # น้ำหนักปานกลาง
            'iron': 2,        # น้ำหนักปานกลาง
            'sodium': -3      # น้ำหนักติดลบ (ต่ำกว่าดีกว่า)
        }
        
        for nutrient, weight in important_nutrients.items():
            value = nutrition_data.get(nutrient, 0)
            recommended = self.daily_recommendations[nutrient].get(target_group, 
                         self.daily_recommendations[nutrient].get('male_adult', 100))
            
            # คำนวณคะแนนแต่ละสารอาหาร (0-100)
            if nutrient == 'calories':
                # สำหรับแคลอรี่ คะแนนดีสุดคือ 80-120% ของค่าแนะนำ
                percentage = (value / recommended) * 100
                if 80 <= percentage <= 120:
                    score = 100
                elif 60 <= percentage < 80 or 120 < percentage <= 150:
                    score = 80 - abs(percentage - 100) * 0.5
                elif 40 <= percentage < 60 or 150 < percentage <= 200:
                    score = 60 - abs(percentage - 100) * 0.3
                else:
                    score = max(20 - abs(percentage - 100) * 0.1, 0)
                    
            elif nutrient == 'sodium':
                # สำหรับโซเดียม ยิ่งต่ำยิ่งดี
                percentage = (value / recommended) * 100
                if percentage <= 50:
                    score = 100
                elif percentage <= 75:
                    score = 90 - (percentage - 50) * 0.8
                elif percentage <= 100:
                    score = 70 - (percentage - 75) * 0.8
                else:
                    score = max(50 - (percentage - 100) * 0.5, 0)
                    
            else:
                # สำหรับสารอาหารอื่น ยิ่งมากยิ่งดี (จนถึงขีดจำกัด)
                percentage = min((value / recommended) * 100, 200)
                if percentage >= 100:
                    score = 100
                elif percentage >= 80:
                    score = 80 + (percentage - 80) * 1
                elif percentage >= 60:
                    score = 60 + (percentage - 60) * 1
                elif percentage >= 40:
                    score = 40 + (percentage - 40) * 0.5
                else:
                    score = percentage
            
            weighted_score = score * abs(weight)
            scores[nutrient] = {
                'raw_score': score,
                'weighted_score': weighted_score,
                'weight': weight,
                'percentage': (value / recommended) * 100 if recommended > 0 else 0,
                'value': value,
                'recommended': recommended
            }
            
            total_score += weighted_score
            max_score += 100 * abs(weight)
        
        overall_score = (total_score / max_score) * 100 if max_score > 0 else 0
        
        # การประเมินโดยรวม
        grade_info = self._get_grade_info(overall_score)
        
        # คำแนะนำเฉพาะ
        recommendations = self._generate_detailed_recommendations(scores, nutrition_data)
        
        # การวิเคราะห์ความเสี่ยง
        health_risks = self._assess_health_risks(nutrition_data)
        
        return {
            'individual_scores': scores,
            'overall_score': overall_score,
            'grade': grade_info['grade'],
            'grade_color': grade_info['color'],
            'grade_emoji': grade_info['emoji'],
            'grade_description': grade_info['description'],
            'recommendations': recommendations,
            'health_risks': health_risks,
            'total_nutrients_evaluated': len(important_nutrients),
            'strengths': self._identify_strengths(scores),
            'weaknesses': self._identify_weaknesses(scores)
        }

    def _get_grade_info(self, score: float) -> Dict:
        """ได้ข้อมูลเกรดตามคะแนน"""
        for grade, info in self.health_grades.items():
            if score >= info['min']:
                descriptions = {
                    'excellent': 'ดีเยี่ยม! โภชนาการครบถ้วนสมบูรณ์',
                    'very_good': 'ดีมาก! โภชนาการแทบสมบูรณ์แบบ',
                    'good': 'ดี! โภชนาการอยู่ในเกณฑ์ที่พอใจ',
                    'fair': 'พอใช้ ควรปรับปรุงบางด้าน',
                    'poor': 'ต้องปรับปรุง! ขาดสารอาหารสำคัญ',
                    'very_poor': 'ต้องปรับปรุงอย่างเร่งด่วน!'
                }
                return {
                    'grade': grade.upper().replace('_', ' '),
                    'color': info['color'],
                    'emoji': info['emoji'],
                    'description': descriptions[grade]
                }
        
        return {
            'grade': 'UNKNOWN',
            'color': '#6c757d',
            'emoji': '❓',
            'description': 'ไม่สามารถประเมินได้'
        }

    def _generate_detailed_recommendations(self, scores: Dict, nutrition_data: Dict) -> List[str]:
        """สร้างคำแนะนำโดยละเอียด"""
        recommendations = []
        
        for nutrient, score_info in scores.items():
            score = score_info['raw_score']
            percentage = score_info['percentage']
            
            if score < 60:  # ต้องปรับปรุง
                if nutrient == 'calories':
                    if percentage < 80:
                        recommendations.append("🔥 เพิ่มแคลอรี่โดยทานอาหารที่มีสารอาหารครบถ้วน")
                    else:
                        recommendations.append("🔥 ลดแคลอรี่โดยเลือกอาหารที่มีคุณค่า")
                        
                elif nutrient == 'protein':
                    recommendations.append("🥩 เพิ่มโปรตีนจากเนื้อสัตว์ไม่ติดมัน ไข่ ถั่ว เต้าหู้")
                    
                elif nutrient == 'fiber':
                    recommendations.append("🥬 เพิ่มผักใบเขียว ผลไม้ และธัญพืชเต็มเมล็ด")
                    
                elif nutrient == 'vitamin_c':
                    recommendations.append("🍊 เพิ่มผลไม้เปรี้ยว มะเขือเทศ พริก ผักใบเขียว")
                    
                elif nutrient == 'calcium':
                    recommendations.append("🥛 เพิ่มนม โยเกิร์ต เต้าหู้ ปลาเล็กปลาน้อย งา")
                    
                elif nutrient == 'iron':
                    recommendations.append("🩸 เพิ่มเนื้อแดง ตับ ผักใบเขียวเข้ม ถั่วเขียว")
                    
                elif nutrient == 'sodium':
                    if percentage > 100:
                        recommendations.append("🧂 ลดเกลือ ซอส น้ำปลา อาหารดอง อาหารแปรรูป")
        
        # คำแนะนำทั่วไป
        if len(recommendations) == 0:
            recommendations.append("✅ โภชนาการอยู่ในเกณฑ์ดี ควรรักษาไว้")
        else:
            recommendations.append("💧 ดื่มน้ำเปล่าอย่างน้อย 8 แก้วต่อวัน")
            recommendations.append("🚶‍♀️ ออกกำลังกายสม่ำเสมออย่างน้อยวันละ 30 นาที")
        
        return recommendations[:5]  # จำกัดไม่เกิน 5 ข้อ

    def _assess_health_risks(self, nutrition_data: Dict) -> List[str]:
        """ประเมินความเสี่ยงต่อสุขภาพ"""
        risks = []
        
        calories = nutrition_data.get('calories', 0)
        sodium = nutrition_data.get('sodium', 0)
        fat = nutrition_data.get('fat', 0)
        fiber = nutrition_data.get('fiber', 0)
        protein = nutrition_data.get('protein', 0)
        
        # ความเสี่ยงจากแคลอรี่สูง
        if calories > 600:
            risks.append("⚠️ แคลอรี่สูงมาก อาจทำให้น้ำหนักเพิ่ม")
            
        # ความเสี่ยงจากโซเดียมสูง
        if sodium > 1500:
            risks.append("⚠️ โซเดียมสูงเสี่ยงต่อความดันโลหิตสูง")
            
        # ความเสี่ยงจากไขมันสูง
        if fat > 25:
            risks.append("⚠️ ไขมันสูงเสี่ยงต่อโรคหัวใจ")
            
        # ความเสี่ยงจากใยอาหารต่ำ
        if fiber < 2:
            risks.append("⚠️ ใยอาหารต่ำอาจทำให้ย่อยลำบาก")
            
        # ความเสี่ยงจากโปรตีนต่ำ
        if protein < 10:
            risks.append("⚠️ โปรตีนต่ำอาจส่งผลต่อกล้ามเนื้อ")
        
        if not risks:
            risks.append("✅ ไม่พบความเสี่ยงเด่นชัดด้านโภชนาการ")
        
        return risks

    def _identify_strengths(self, scores: Dict) -> List[str]:
        """ระบุจุดแข็งทางโภชนาการ"""
        strengths = []
        
        for nutrient, score_info in scores.items():
            if score_info['raw_score'] >= 80:
                thai_name = self._get_nutrient_thai_name(nutrient)
                strengths.append(f"✅ {thai_name}อยู่ในเกณฑ์ดี")
        
        return strengths

    def _identify_weaknesses(self, scores: Dict) -> List[str]:
        """ระบุจุดอ่อนทางโภชนาการ"""
        weaknesses = []
        
        for nutrient, score_info in scores.items():
            if score_info['raw_score'] < 60:
                thai_name = self._get_nutrient_thai_name(nutrient)
                if nutrient == 'sodium' and score_info['percentage'] > 100:
                    weaknesses.append(f"❌ {thai_name}สูงเกินไป")
                elif nutrient != 'sodium':
                    weaknesses.append(f"❌ {thai_name}ต่ำเกินไป")
        
        return weaknesses

    def create_advanced_nutrition_gauge_chart(self, nutrition_score: Dict) -> go.Figure:
        """สร้างกราฟเกจแสดงคะแนนโภชนาการ - เวอร์ชันขั้นสูง"""
        
        score = nutrition_score['overall_score']
        grade_info = {
            'grade': nutrition_score['grade'],
            'color': nutrition_score['grade_color'],
            'emoji': nutrition_score['grade_emoji'],
            'description': nutrition_score['grade_description']
        }
        
        fig = go.Figure()
        
        # เพิ่มเกจหลัก
        fig.add_trace(go.Indicator(
            mode = "gauge+number+delta",
            value = score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {
                'text': f"🎯 คะแนนโภชนาการ<br><span style='font-size:0.8em'>{grade_info['emoji']} เกรด: {grade_info['grade']}</span>",
                'font': {'size': 20}
            },
            delta = {
                'reference': 70, 
                'increasing': {'color': "#28a745"}, 
                'decreasing': {'color': "#dc3545"},
                'font': {'size': 16}
            },
            number = {
                'font': {'size': 40, 'color': grade_info['color']},
                'suffix': '/100'
            },
            gauge = {
                'axis': {
                    'range': [None, 100], 
                    'tickwidth': 1, 
                    'tickcolor': "darkblue",
                    'tickmode': 'linear',
                    'tick0': 0,
                    'dtick': 20
                },
                'bar': {'color': grade_info['color'], 'thickness': 0.3},
                'bgcolor': "white",
                'borderwidth': 3,
                'bordercolor': grade_info['color'],
                'steps': [
                    {'range': [0, 40], 'color': '#ffebee'},
                    {'range': [40, 60], 'color': '#fff3e0'},
                    {'range': [60, 70], 'color': '#fffde7'},
                    {'range': [70, 80], 'color': '#f3e5f5'},
                    {'range': [80, 90], 'color': '#e8f5e8'},
                    {'range': [90, 100], 'color': '#e0f2f1'}
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
            height=450,
            margin=dict(l=20, r=20, t=80, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            annotations=[
                dict(
                    text=grade_info['description'],
                    x=0.5, y=0.1,
                    xref="paper", yref="paper",
                    showarrow=False,
                    font=dict(size=14, color=grade_info['color']),
                    align="center"
                )
            ]
        )
        
        return fig

    def display_comprehensive_nutrition_analysis(self, nutrition_data: Dict, 
                                               recipe_name: str = "เมนูนี้",
                                               target_group: str = 'male_adult',
                                               comparison_data: List[Dict] = None):
        """แสดงการวิเคราะห์โภชนาการแบบครอบคลุม - เวอร์ชันปรับปรุง"""
        
        st.markdown(f"## 🔬 การวิเคราะห์โภชนาการแบบละเอียด: {recipe_name}")
        
        # คำนวณคะแนน
        nutrition_score = self.calculate_comprehensive_nutrition_score(nutrition_data, target_group)
        
        # แถวแรก: คะแนนรวมและสรุป
        col1, col2 = st.columns([1, 2])
        
        with col1:
            gauge_fig = self.create_advanced_nutrition_gauge_chart(nutrition_score)
            st.plotly_chart(gauge_fig, use_container_width=True)
            
            # แสดงสรุปผล
            grade_color = nutrition_score['grade_color']
            grade_emoji = nutrition_score['grade_emoji']
            grade_description = nutrition_score['grade_description']
            
            st.markdown(f"""
            <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, {grade_color}22 0%, {grade_color}11 100%); 
                        border-radius: 15px; border-left: 5px solid {grade_color}; margin: 1rem 0;">
                <h3 style="margin: 0; color: {grade_color};">{grade_emoji} สรุปผลการประเมิน</h3>
                <p style="margin: 0.5rem 0; font-size: 1.2em;"><strong>คะแนน: {nutrition_score['overall_score']:.1f}/100</strong></p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>เกรด: {nutrition_score['grade']}</strong></p>
                <p style="margin: 0; font-style: italic; line-height: 1.4;">{grade_description}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            radar_fig = self.create_enhanced_nutrition_radar_chart(
                nutrition_data, target_group, comparison_data
            )
            st.plotly_chart(radar_fig, use_container_width=True)
        
        # แถวที่สอง: วิตามินและแร่ธาตุ
        st.markdown("### 🧪 วิตามินและแร่ธาตุ")
        vitamin_mineral_fig = self.create_advanced_vitamin_mineral_chart(nutrition_data)
        st.plotly_chart(vitamin_mineral_fig, use_container_width=True)
        
        # แถวที่สาม: จุดแข็งและจุดอ่อน
        col1, col2 = st.columns(2)
        
        with col1:
            if nutrition_score['strengths']:
                st.markdown("### ✅ จุดแข็ง")
                for strength in nutrition_score['strengths']:
                    st.markdown(f"- {strength}")
            
            if nutrition_score['recommendations']:
                st.markdown("### 💡 คำแนะนำเฉพาะ")
                for rec in nutrition_score['recommendations']:
                    st.markdown(f"- {rec}")
        
        with col2:
            if nutrition_score['weaknesses']:
                st.markdown("### 📈 ควรปรับปรุง")
                for weakness in nutrition_score['weaknesses']:
                    st.markdown(f"- {weakness}")
            
            if nutrition_score['health_risks']:
                st.markdown("### ⚠️ ความเสี่ยงต่อสุขภาพ")
                for risk in nutrition_score['health_risks']:
                    st.markdown(f"- {risk}")
        
        # แถวที่สี่: คะแนนรายละเอียด
        with st.expander("📊 ดูคะแนนรายละเอียด", expanded=False):
            scores_df = pd.DataFrame([
                {
                    'สารอาหาร': self._get_nutrient_thai_name(nutrient),
                    'ค่าจริง': f"{info['value']:.1f} {self._get_nutrient_unit(nutrient)}",
                    'ค่าแนะนำ': f"{info['recommended']:.1f} {self._get_nutrient_unit(nutrient)}",
                    'เปอร์เซ็นต์': f"{info['percentage']:.1f}%",
                    'คะแนน': f"{info['raw_score']:.1f}/100"
                }
                for nutrient, info in nutrition_score['individual_scores'].items()
            ])
            st.dataframe(scores_df, use_container_width=True)

    def create_meal_planning_recommendations(self, nutrition_data: Dict, 
                                           target_calories: int, 
                                           target_group: str = 'male_adult') -> Dict:
        """สร้างคำแนะนำการวางแผนมื้อเมนู"""
        
        current_calories = nutrition_data.get('calories', 0)
        remaining_calories = target_calories - current_calories
        
        recommendations = {
            'remaining_calories': remaining_calories,
            'meal_suggestions': [],
            'snack_suggestions': [],
            'nutrition_gaps': [],
            'warnings': []
        }
        
        # วิเคราะห์สารอาหารที่ขาดแคลน
        target_recommendations = self.daily_recommendations
        
        for nutrient in ['protein', 'fiber', 'vitamin_c', 'calcium', 'iron']:
            current_value = nutrition_data.get(nutrient, 0)
            recommended = target_recommendations[nutrient].get(target_group, 0)
            
            if current_value < recommended * 0.7:  # ขาดแคลนมากกว่า 30%
                recommendations['nutrition_gaps'].append({
                    'nutrient': nutrient,
                    'thai_name': self._get_nutrient_thai_name(nutrient),
                    'gap': recommended - current_value,
                    'percentage': (current_value / recommended) * 100
                })
        
        # คำแนะนำมื้ออาหาร
        if remaining_calories > 400:
            recommendations['meal_suggestions'] = [
                "🍽️ ข้าวผัดกุ้งใส่ผัก (350 kcal)",
                "🍜 ก๋วยเตี๋ยวน้ำใสเนื้อ (320 kcal)",
                "🥗 สลัดไก่ย่างพร้อมแป้ง (380 kcal)"
            ]
        elif remaining_calories > 200:
            recommendations['meal_suggestions'] = [
                "🥪 แซนด์วิชไข่ต้ม (280 kcal)",
                "🍚 ข้าวหน้าไก่เทอริยากิ (320 kcal)",
                "🍲 แกงจืดเต้าหู้ใส่ผัก (250 kcal)"
            ]
        elif remaining_calories > 0:
            recommendations['snack_suggestions'] = [
                "🍎 แอปเปิ้ล 1 ลูก (80 kcal)",
                "🥛 นมไขมันต่ำ 1 แก้ว (120 kcal)",
                "🥜 ถั่วอัลมอนด์ 10 เม็ด (70 kcal)"
            ]
        else:
            recommendations['warnings'].append("⚠️ แคลอรี่เกินเป้าหมายแล้ว ควรออกกำลังกาย")
        
        return recommendations

    def _get_nutrient_thai_name(self, nutrient: str) -> str:
        """แปลงชื่อสารอาหารเป็นภาษาไทย"""
        thai_names = {
            'calories': 'แคลอรี่',
            'protein': 'โปรตีน',
            'carbs': 'คาร์โบไฮเดรต',
            'fat': 'ไขมัน',
            'fiber': 'ใยอาหาร',
            'vitamin_a': 'วิตามินเอ',
            'vitamin_c': 'วิตามินซี',
            'vitamin_b1': 'วิตามินบี1',
            'vitamin_b2': 'วิตามินบี2',
            'calcium': 'แคลเซียม',
            'iron': 'เหล็ก',
            'potassium': 'โปแตสเซียม',
            'sodium': 'โซเดียม'
        }
        return thai_names.get(nutrient, nutrient)

    def _get_nutrient_unit(self, nutrient: str) -> str:
        """ได้หน่วยของสารอาหาร"""
        units = {
            'calories': 'kcal',
            'protein': 'g',
            'carbs': 'g',
            'fat': 'g',
            'fiber': 'g',
            'vitamin_a': 'IU',
            'vitamin_c': 'mg',
            'vitamin_b1': 'mg',
            'vitamin_b2': 'mg',
            'calcium': 'mg',
            'iron': 'mg',
            'potassium': 'mg',
            'sodium': 'mg'
        }
        return units.get(nutrient, '')

    def _get_target_group_thai_name(self, target_group: str) -> str:
        """แปลงกลุ่มเป้าหมายเป็นภาษาไทย"""
        thai_names = {
            'male_adult': 'ชายวัยผู้ใหญ่',
            'female_adult': 'หญิงวัยผู้ใหญ่',
            'male_elderly': 'ชายผู้สูงอายุ',
            'female_elderly': 'หญิงผู้สูงอายุ',
            'child_2_3': 'เด็ก 2-3 ปี',
            'child_4_8': 'เด็ก 4-8 ปี',
            'child_9_13': 'เด็ก 9-13 ปี',
            'teen_14_18': 'วัยรุ่น 14-18 ปี',
            'pregnant': 'หญิงตั้งครรภ์',
            'breastfeeding': 'หญิงให้นม',
            'athlete_male': 'นักกีฬาชาย',
            'athlete_female': 'นักกีฬาหญิง'
        }
        return thai_names.get(target_group, target_group)

    def create_weekly_nutrition_comparison(self, weekly_meals: List[List[Dict]]) -> go.Figure:
        """สร้างกราฟเปรียบเทียบโภชนาการรายสัปดาห์ - เวอร์ชันขั้นสูง"""
        
        days = ['จันทร์', 'อังคาร', 'พุธ', 'พฤหัสบดี', 'ศุกร์', 'เสาร์', 'อาทิตย์']
        
        # คำนวณค่าโภชนาการรายวัน
        daily_nutrition = []
        for day_meals in weekly_meals:
            day_total = {
                'calories': sum(meal.get('calories', 0) for meal in day_meals),
                'protein': sum(meal.get('protein', 0) for meal in day_meals),
                'carbs': sum(meal.get('carbs', 0) for meal in day_meals),
                'fat': sum(meal.get('fat', 0) for meal in day_meals),
                'fiber': sum(meal.get('fiber', 0) for meal in day_meals),
                'sodium': sum(meal.get('sodium', 0) for meal in day_meals)
            }
            daily_nutrition.append(day_total)
        
        # สร้าง subplot
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('แคลอรี่รายวัน', 'โปรตีนรายวัน', 'คาร์โบไฮเดรตรายวัน', 
                          'ไขมันรายวัน', 'ใยอาหารรายวัน', 'โซเดียมรายวัน'),
            shared_xaxes=True
        )
        
        nutrients = ['calories', 'protein', 'carbs', 'fat', 'fiber', 'sodium']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        targets = [2000, 50, 250, 55, 25, 2000]  # ค่าแนะนำ
        positions = [(1,1), (1,2), (2,1), (2,2), (3,1), (3,2)]
        
        for i, (nutrient, color, target) in enumerate(zip(nutrients, colors, targets)):
            values = [day[nutrient] for day in daily_nutrition]
            row, col = positions[i]
            
            # กราฟเส้น
            fig.add_trace(
                go.Scatter(
                    x=days, 
                    y=values, 
                    mode='lines+markers',
                    name=self._get_nutrient_thai_name(nutrient), 
                    line=dict(color=color, width=3),
                    marker=dict(size=8, symbol='circle'),
                    showlegend=False,
                    hovertemplate=f'<b>%{{x}}</b><br>{self._get_nutrient_thai_name(nutrient)}: %{{y:.1f}} {self._get_nutrient_unit(nutrient)}<extra></extra>'
                ),
                row=row, col=col
            )
            
            # เส้นค่าแนะนำ
            fig.add_hline(
                y=target, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"แนะนำ: {target}",
                annotation_position="top right",
                row=row, col=col
            )
            
            # พื้นที่เป้าหมาย (±20%)
            fig.add_hrect(
                y0=target * 0.8, 
                y1=target * 1.2,
                fillcolor="rgba(0,255,0,0.1)",
                layer="below",
                line_width=0,
                row=row, col=col
            )
        
        fig.update_layout(
            title_text="📅 การติดตามโภชนาการรายสัปดาห์แบบละเอียด",
            title_x=0.5,
            height=800,
            font=dict(family="Sarabun, sans-serif"),
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0.02)'
        )
        
        return fig
