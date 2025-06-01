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
    """เครื่องมือค้นหาแนะนำอาหารตามคุณค่าทางโภชนาการ - เวอร์ชันปรับปรุงใหม่"""
    
    def __init__(self, data: pd.DataFrame, nutrition_api):
        self.data = data
        self.nutrition_api = nutrition_api
        self.nutrition_cache = {}
        
        # คำสำคัญสำหรับการค้นหาตามโภชนาการ - เพิ่มความครอบคลุม
        self.nutrition_keywords = {
            # แคลอรี่ - เพิ่มรูปแบบใหม่
            'calories_very_low': [
                'แคลอรี่ต่ำมาก', 'แคลต่ำมาก', 'แคลอรี่น้อยมาก', 
                'ลดน้ำหนักเร่งด่วน', 'ไดเอทเข้มข้น', 'แคลอรี่เกือบศูนย์',
                'พลังงานต่ำสุด', 'ไม่มีแคลอรี่', 'แคลอรี่เหลือน้อย'
            ],
            'calories_low': [
                'แคลอรี่ต่ำ', 'แคลต่ำ', 'ลดน้ำหนัก', 'เบา', 'ไม่อ้วน', 
                'ไดเอท', 'diet', 'ลดความอ้วน', 'คุมน้ำหนัก', 'แคลอรี่น้อย',
                'พลังงานต่ำ', 'เบาๆ', 'ไม่เยอะ', 'ใส่ใจรูปร่าง',
                'คาโลรี่ต่ำ', 'kcal ต่ำ', 'น้อยแคลอรี่'
            ],
            'calories_high': [
                'แคลอรี่สูง', 'แคลสูง', 'เพิ่มน้ำหนัก', 'พลังงานสูง', 
                'เติมแรง', 'นักกีฬา', 'กำลัง', 'แคลอรี่มาก',
                'พลังงานเยอะ', 'อิ่มท้อง', 'เสริมพลัง', 'เติมแคลอรี่'
            ],
            
            # โปรตีน - ขยายคำสำคัญ
            'protein_very_high': [
                'โปรตีนสูงมาก', 'โปรสูงมาก', 'เพาะกาย', 'bodybuilding',
                'นักกีฬาระดับสูง', 'โปรตีนเข้มข้น', 'โปรตีนเยอะมาก',
                'สร้างกล้าม', 'เสริมโปรตีน', 'protein max'
            ],
            'protein_high': [
                'โปรตีนสูง', 'โปรตีนมาก', 'เนื้อเยื่อ', 'กล้ามเนื้อ', 
                'นักกีฬา', 'ออกกำลังกาย', 'ฟิตเนส', 'เสริมสร้าง',
                'โปรสูง', 'protein high', 'โปรตีนเยอะ', 'เสริมกล้าม',
                'สารโปรตีน', 'เนื้อสัตว์', 'ไข่', 'ถั่ว', 'โปรตีนดี'
            ],
            'protein_low': [
                'โปรตีนต่ำ', 'โปรตีนน้อย', 'ไต', 'โรคไต', 'โปรต่ำ',
                'จำกัดโปรตีน', 'ลดโปรตีน', 'โปรตีนจำกัด'
            ],
            
            # ไขมัน - ปรับปรุงความครอบคลุม
            'fat_very_low': [
                'ไขมันต่ำมาก', 'ไม่มีไขมัน', 'fat free', 'โรคหัวใจ',
                'ไขมันน้อยที่สุด', 'ไขมัน 0', 'ไร้ไขมัน', 'ไขมันเกือบศูนย์',
                'ไม่มีมัน', 'ไม่มันเลย', 'ปลอดไขมัน'
            ],
            'fat_low': [
                'ไขมันต่ำ', 'ไขมันน้อย', 'ลดไขมัน', 'ไม่มันเยอะ', 
                'สุขภาพดี', 'หัวใจ', 'โรคหัวใจ', 'ไขมันต่ำ',
                'ไม่มัน', 'ไม่เลี่ยน', 'อ่อนๆ', 'ไม่เยอะ', 'low fat',
                'ลดมัน', 'คุมไขมัน', 'น้อยมัน', 'ไขมันดี'
            ],
            'fat_high': [
                'ไขมันสูง', 'ไขมันมาก', 'มันเยอะ', 'ไขมันดี',
                'ไขมันเพิ่ม', 'มันๆ', 'เลี่ยน', 'high fat'
            ],
            
            # คาร์โบไฮเดรต - เพิ่มความหลากหลาย
            'carbs_very_low': [
                'คาร์โบต่ำมาก', 'keto', 'ketogenic', 'คีโตเจนิค',
                'คาร์โบเกือบศูนย์', 'no carb', 'ไร้คาร์โบ', 'ไม่มีแป้ง',
                'ไม่มีน้ำตาล', 'คาร์โบ 0', 'low carb มาก'
            ],
            'carbs_low': [
                'คาร์โบต่ำ', 'แป้งน้อย', 'น้ำตาลต่ำ', 'เบาหวาน', 
                'คีโต', 'low carb', 'ลดแป้ง', 'คาร์โบน้อย',
                'ลดคาร์โบ', 'คุมน้ำตาล', 'แป้งต่ำ', 'ลดน้ำตาล',
                'carbohydrate ต่ำ', 'คาร์โบไฮเดรตต่ำ'
            ],
            'carbs_high': [
                'คาร์โบสูง', 'แป้งมาก', 'พลังงาน', 'นักกีฬา', 'ข้าว',
                'คาร์โบมาก', 'แป้งเยอะ', 'น้ำตาลมาก', 'high carb'
            ],
            
            # ใยอาหาร - เพิ่มคำเฉพาะ
            'fiber_very_high': [
                'ใยอาหารสูงมาก', 'ใยมากๆ', 'ท้องผูกมาก', 'ใยเข้มข้น',
                'ใยอาหารเยอะมาก', 'fiber สูงมาก', 'ใยมหาศาล'
            ],
            'fiber_high': [
                'ใยอาหารสูง', 'ใยอาหารมาก', 'ขับถ่าย', 'ท้องผูก', 
                'ย่อย', 'ระบบย่อย', 'ช่วยย่อย', 'ผัก', 'ใยสูง',
                'fiber high', 'ใยอาหารเยอะ', 'ช่วยขับถ่าย',
                'ลำไส้ดี', 'ใยอาหารดี', 'ผักใบเขียว', 'ใยธรรมชาติ'
            ],
            
            # วิตามิน - เพิ่มรายละเอียด
            'vitamin_a_very_high': [
                'วิตามินเอสูงมาก', 'วิตเอสูงมาก', 'สายตาดีมาก',
                'ต้านอนุมูลอิสระสูง', 'วิตามินเอเข้มข้น', 'beta carotene สูง'
            ],
            'vitamin_a_high': [
                'วิตามินเอสูง', 'วิตามินเอ', 'วิตเอ', 'สายตา', 'ผิวพรรณ', 
                'ตา', 'บำรุงตา', 'vitamin a', 'วิตามินเอมาก',
                'บำรุงผิว', 'ต้านอนุมูลอิสระ', 'แครอท', 'ผักสีส้ม',
                'เบต้าแคโรทีน', 'ใสใส', 'ผิวดี'
            ],
            'vitamin_c_very_high': [
                'วิตามินซีสูงมาก', 'วิตซีสูงมาก', 'ภูมิคุ้มกันแข็งแรงมาก',
                'ต้านหวัดแรง', 'วิตามินซีเข้มข้น', 'ascorbic acid สูง'
            ],
            'vitamin_c_high': [
                'วิตามินซีสูง', 'วิตามินซี', 'วิตซี', 'ภูมิคุ้มกัน', 'ต้านหวัด', 
                'เสริมภูมิ', 'ต้านอนุมูลอิสระ', 'vitamin c', 'วิตามินซีมาก',
                'ผลไม้เปรี้ยว', 'มะนาว', 'ส้ม', 'มะเขือเทศ', 'ต้านโรค',
                'เสริมภูมิคุ้มกัน', 'ใสใส', 'สดใส'
            ],
            'vitamin_b_high': [
                'วิตามินบี', 'วิตบี', 'ระบบประสาท', 'เมแทบอลิซึม', 'พลังงาน',
                'vitamin b', 'วิตามินบีมาก', 'วิตามินบีคอมเพล็กซ์',
                'ประสาทดี', 'เผาผลาญ', 'สดชื่น'
            ],
            'vitamin_d_high': [
                'วิตามินดี', 'วิตดี', 'กระดูกแข็งแรง', 'vitamin d',
                'วิตามินดีมาก', 'แสงแดด', 'กระดูกดี'
            ],
            'vitamin_e_high': [
                'วิตามินอี', 'วิตอี', 'ต้านอนุมูลอิสระ', 'vitamin e',
                'ผิวพรรณ', 'ชะลอวัย', 'สวยใส'
            ],
            
            # แร่ธาตุ - เพิ่มความครอบคลุม
            'calcium_very_high': [
                'แคลเซียมสูงมาก', 'กระดูกแข็งแรงมาก', 'ป้องกันกระดูกพรุน',
                'แคลเซียมเข้มข้น', 'calcium สูงมาก', 'แคลเซียมมหาศาล'
            ],
            'calcium_high': [
                'แคลเซียมสูง', 'แคลเซียม', 'กระดูก', 'ฟัน', 
                'ผู้สูงอายุ', 'เด็ก', 'บำรุงกระดูก', 'calcium',
                'แคลเซียมมาก', 'กระดูกแข็ง', 'ฟันแข็ง', 'นม', 'โยเกิร์ต',
                'ปลาเล็ก', 'เต้าหู้', 'งา', 'กระดูกดี', 'ฟันดี'
            ],
            'iron_very_high': [
                'เหล็กสูงมาก', 'ธาตุเหล็กสูงมาก', 'รักษาโลหิตจาง',
                'เหล็กเข้มข้น', 'iron สูงมาก', 'เหล็กมหาศาล'
            ],
            'iron_high': [
                'เหล็กสูง', 'ธาตุเหล็ก', 'โลหิตจาง', 'เลือดจาง', 
                'ผู้หญิง', 'ประจำเดือน', 'iron', 'เหล็กมาก',
                'เลือดแดง', 'ออกซิเจน', 'เม็ดเลือด', 'เลือดดี',
                'ผักใบเขียวเข้ม', 'เนื้อแดง', 'ตับ'
            ],
            'potassium_very_high': [
                'โปแตสเซียมสูงมาก', 'หัวใจแข็งแรงมาก', 'ความดันต่ำมาก',
                'โปแตสเซียมเข้มข้น', 'potassium สูงมาก'
            ],
            'potassium_high': [
                'โปแตสเซียมสูง', 'โปแตสเซียม', 'ความดันโลหิต', 
                'หัวใจ', 'กล้ามเนื้อหัวใจ', 'potassium', 'โปแตสเซียมมาก',
                'หัวใจดี', 'ความดันปกติ', 'กล้วย', 'มันฝรั่ง', 'ผลไม้'
            ],
            'sodium_very_low': [
                'โซเดียมต่ำมาก', 'เกลือน้อยมาก', 'ไม่มีเกลือ',
                'ความดันสูงมาก', 'โซเดียมเกือบศูนย์', 'sodium free',
                'ไร้เกลือ', 'ปลอดเกลือ'
            ],
            'sodium_low': [
                'โซเดียมต่ำ', 'เกลือน้อย', 'ความดันสูง', 'ไต', 
                'หัวใจ', 'จืด', 'ไม่เค็ม', 'sodium low', 'โซเดียมน้อย',
                'ลดเกลือ', 'คุมเกลือ', 'ไม่เค็มมาก', 'เค็มน้อย',
                'ลดโซเดียม', 'โซเดียมต่ำ', 'เกลือต่ำ'
            ],
            'zinc_high': [
                'สังกะสีสูง', 'สังกะสี', 'ภูมิคุ้มกัน', 'แผลหาย',
                'zinc', 'สังกะสีมาก', 'ต้านเชื้อ', 'ฟื้นฟู'
            ],
            'magnesium_high': [
                'แมกนีเซียมสูง', 'แมกนีเซียม', 'กล้ามเนื้อ', 'ประสาท',
                'magnesium', 'แมกนีเซียมมาก', 'ผ่อนคลาย', 'นอนหลับ'
            ],
            'phosphorus_high': [
                'ฟอสฟอรัสสูง', 'ฟอสฟอรัส', 'กระดูก', 'ฟัน',
                'phosphorus', 'ฟอสฟอรัสมาก'
            ],
            
            # กลุ่มผู้ป่วยเฉพาะ - เพิ่มรายละเอียด
            'diabetes': [
                'เบาหวาน', 'ผู้ป่วยเบาหวาน', 'น้ำตาลต่ำ', 
                'ควบคุมน้ำตาล', 'เบาหวาน', 'ดัชนีน้ำตาล',
                'diabetes', 'diabetic', 'น้ำตาลในเลือด',
                'ระดับน้ำตาล', 'glucose', 'insulin', 'DM',
                'คุมเบาหวาน', 'เบาหวานทำ', 'เบาหวานกิน'
            ],
            'hypertension': [
                'ความดันสูง', 'ผู้ป่วยความดัน', 'โซเดียมต่ำ', 
                'ความดัน', 'ไฮเปอร์เทนชั่น', 'hypertension',
                'ความดันโลหิต', 'blood pressure', 'HT',
                'คุมความดัน', 'ความดันสูงกิน', 'ความดันสูงทำ'
            ],
            'heart_disease': [
                'โรคหัวใจ', 'หัวใจ', 'โคเลสเตอรอล', 'หลอดเลือด',
                'heart disease', 'cardiovascular', 'หัวใจวาย',
                'หลอดเลือดหัวใจ', 'CAD', 'CVD', 'หัวใจขาดเลือด'
            ],
            'kidney_disease': [
                'โรคไต', 'ไต', 'ล้างไต', 'ไตเสื่อม', 'kidney disease',
                'renal', 'ไตวาย', 'dialysis', 'CKD', 'ไตรรับ'
            ],
            'liver_disease': [
                'โรคตับ', 'ตับ', 'ตับแข็ง', 'liver disease',
                'hepatitis', 'ตับอักเสบ', 'ตับบวม', 'ไขมันพอกตับ'
            ],
            'gout': [
                'เก๊าต์', 'กรดยูริก', 'ข้อเสื่อม', 'ข้อปวด',
                'uric acid', 'โรคเก๊าต์', 'เก๊าท์'
            ],
            'elderly': [
                'ผู้สูงอายุ', 'คนแก่', 'นุ่ม', 'ย่อยง่าย', 
                'ผู้ใหญ่', 'วัยชรา', 'elderly', 'senior',
                'วัยเก๋า', 'ผู้เฒ่า', 'คนชรา', 'อายุมาก'
            ],
            'children': [
                'เด็ก', 'เด็กเล็ก', 'แคลเซียม', 'เจริญเติบโต', 
                'ลูก', 'วัยรุ่น', 'children', 'kid', 'เด็กโต',
                'น้อง', 'หลาน', 'เด็กนักเรียน', 'เด็กอนุบาล'
            ],
            'athletes': [
                'นักกีฬา', 'ออกกำลังกาย', 'โปรตีนสูง', 'ฟิตเนส', 
                'กล้ามเนื้อ', 'เล่นกีฬา', 'athlete', 'fitness',
                'เพาะกาย', 'bodybuilding', 'วิ่ง', 'ยิม', 'gym'
            ],
            'pregnant': [
                'ตั้งครรภ์', 'คนท้อง', 'โฟเลต', 'เหล็ก', 
                'แม่ท้อง', 'มีครรภ์', 'pregnant', 'pregnancy',
                'ให้นม', 'breastfeeding', 'คุณแม่', 'แม่ลูกอ่อน'
            ],
            'menopause': [
                'วัยทอง', 'หมดประจำเดือน', 'ผู้หญิงวัยกลางคน',
                'menopause', 'ฮอร์โมน', 'เอสโตรเจน'
            ],
            
            # ประเภทอาหาร - เพิ่มรายละเอียด
            'vegetarian': [
                'มังสวิรัติ', 'เจ', 'ไม่กินเนื้อ', 'ผัก', 'พืช', 'เจ',
                'vegetarian', 'vegan', 'plant based', 'ไม่กินสัตว์',
                'อาหารเจ', 'จังจืด', 'ไม่กินหมู', 'ไม่กินไก่'
            ],
            'low_sodium': [
                'โซเดียมต่ำ', 'เกลือน้อย', 'จืด', 'ไม่เค็ม',
                'เกลือต่ำ', 'ลดเกลือ', 'sodium ต่ำ'
            ],
            'healthy': [
                'สุขภาพ', 'สุขภาพดี', 'คลีน', 'clean eating', 
                'healthy', 'เพื่อสุขภาพ', 'organic', 'ธรรมชาติ',
                'ดีต่อสุขภาพ', 'บำรุงร่างกาย', 'สุขภาพแข็งแรง'
            ],
            'weight_loss': [
                'ลดน้ำหนัก', 'ลดความอ้วน', 'ไดเอท', 'เบา', 
                'คุมน้ำหนัก', 'weight loss', 'diet', 'slim',
                'ผอม', 'ลดพุง', 'รูปร่างดี'
            ],
            'weight_gain': [
                'เพิ่มน้ำหนัก', 'อ้วน', 'น้ำหนักขึ้น', 'ผอมเกินไป',
                'weight gain', 'bulk', 'mass', 'เสริมน้ำหนัก'
            ],
            'detox': [
                'ดีท็อกซ์', 'ล้างพิษ', 'ล้างลำไส้', 'detox', 
                'ขับสารพิษ', 'cleanse', 'ล้างตับ', 'ล้างไต'
            ],
            'anti_aging': [
                'ต้านอนุมูลอิสระ', 'แอนตี้เอจจิ้ง', 'anti aging',
                'ชะลอวัย', 'อายุยืน', 'antioxidant', 'ต้านแก่'
            ],
            'immune_boost': [
                'เสริมภูมิคุ้มกัน', 'ภูมิต้านทาน', 'ต้านโรค',
                'เสริมภูมิ', 'immune', 'ไม่ป่วย', 'แข็งแรง'
            ],
            'energy_boost': [
                'เติมแรง', 'เสริมพลัง', 'ไม่เหนื่อย', 'มีแรง',
                'energy', 'กำลัง', 'สดชื่น', 'ไม่ง่วง'
            ]
        }
        
        # เกณฑ์การจัดกลุ่มโภชนาการ - ปรับปรุงให้ละเอียดและแม่นยำขึ้น
        self.nutrition_thresholds = {
            'calories': {
                'very_low': 100, 'low': 200, 'medium': 350, 
                'high': 500, 'very_high': 700
            },
            'protein': {
                'very_low': 3, 'low': 8, 'medium': 15, 'high': 25, 'very_high': 35
            },
            'fat': {
                'very_low': 2, 'low': 6, 'medium': 12, 'high': 20, 'very_high': 30
            },
            'carbs': {
                'very_low': 3, 'low': 10, 'medium': 25, 'high': 45, 'very_high': 65
            },
            'fiber': {
                'very_low': 0.3, 'low': 1.5, 'medium': 3, 'high': 6, 'very_high': 10
            },
            'vitamin_a': {
                'very_low': 10, 'low': 50, 'medium': 200, 'high': 600, 'very_high': 1200
            },
            'vitamin_c': {
                'very_low': 1, 'low': 5, 'medium': 15, 'high': 35, 'very_high': 70
            },
            'vitamin_b1': {
                'very_low': 0.02, 'low': 0.05, 'medium': 0.12, 'high': 0.25, 'very_high': 0.45
            },
            'vitamin_b2': {
                'very_low': 0.02, 'low': 0.06, 'medium': 0.15, 'high': 0.30, 'very_high': 0.50
            },
            'calcium': {
                'very_low': 15, 'low': 50, 'medium': 100, 'high': 180, 'very_high': 300
            },
            'iron': {
                'very_low': 0.3, 'low': 1, 'medium': 2.5, 'high': 4.5, 'very_high': 7
            },
            'potassium': {
                'very_low': 80, 'low': 200, 'medium': 350, 'high': 550, 'very_high': 800
            },
            'sodium': {
                'very_low': 150, 'low': 400, 'medium': 800, 
                'high': 1300, 'very_high': 2000
            }
        }

    def calculate_all_nutrition(self, use_api=False, adjust_consumption=True) -> Dict:
        """คำนวณค่าโภชนาการสำหรับทุกเมนู - ปรับปรุงประสิทธิภาพ"""
        if hasattr(self, '_all_nutrition_cache'):
            return self._all_nutrition_cache
            
        all_nutrition = {}
        
        with st.spinner("🧮 กำลังคำนวณค่าโภชนาการสำหรับทุกเมนู..."):
            progress_bar = st.progress(0)
            
            for idx, recipe in self.data.iterrows():
                try:
                    # ตรวจสอบว่ามีข้อมูลโภชนาการในไฟล์แล้วหรือไม่
                    if self._has_nutrition_columns(recipe):
                        nutrition_data = self._extract_nutrition_from_columns(recipe)
                    else:
                        # คำนวณจากวัตถุดิบ
                        nutrition_calculation = self.nutrition_api.calculate_recipe_nutrition(
                            recipe['ingredient'], use_api, adjust_consumption
                        )
                        nutrition_data = nutrition_calculation['total_nutrition']
                    
                    all_nutrition[idx] = nutrition_data
                    progress_bar.progress((idx + 1) / len(self.data))
                except Exception as e:
                    # ใช้ค่าเริ่มต้นหากคำนวณไม่ได้
                    all_nutrition[idx] = self._get_default_nutrition()
            
            progress_bar.empty()
        
        self._all_nutrition_cache = all_nutrition
        return all_nutrition

    def _has_nutrition_columns(self, recipe) -> bool:
        """ตรวจสอบว่ามีคอลัมน์โภชนาการหรือไม่"""
        nutrition_columns = ['calories', 'protein', 'carbs', 'fat', 'fiber', 
                           'vitamin_a', 'vitamin_c', 'calcium', 'iron', 'sodium']
        
        for col in nutrition_columns:
            if col in recipe.index and pd.notna(recipe[col]) and recipe[col] > 0:
                return True
        return False

    def _extract_nutrition_from_columns(self, recipe) -> Dict:
        """ดึงข้อมูลโภชนาการจากคอลัมน์ในไฟล์"""
        nutrition_columns = [
            'calories', 'protein', 'carbs', 'fat', 'fiber',
            'vitamin_a', 'vitamin_c', 'vitamin_b1', 'vitamin_b2',
            'calcium', 'iron', 'potassium', 'sodium'
        ]
        
        nutrition_data = {}
        for col in nutrition_columns:
            if col in recipe.index and pd.notna(recipe[col]):
                nutrition_data[col] = float(recipe[col])
            else:
                nutrition_data[col] = 0.0
        
        return nutrition_data

    def _get_default_nutrition(self) -> Dict:
        """ข้อมูลโภชนาการเริ่มต้น"""
        return {
            'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0, 'fiber': 0,
            'vitamin_a': 0, 'vitamin_c': 0, 'vitamin_b1': 0, 'vitamin_b2': 0,
            'calcium': 0, 'iron': 0, 'potassium': 0, 'sodium': 0
        }
    
    def detect_enhanced_nutrition_intent(self, query: str) -> Dict[str, Any]:
        """ตรวจจับความต้องการค้นหาตามโภชนาการ - เวอร์ชันปรับปรุง"""
        query_lower = query.lower()
        detected_criteria = []
        confidence_scores = {}
        
        # ตรวจสอบคำสำคัญแบบละเอียด
        for category, keywords in self.nutrition_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    detected_criteria.append(category)
                    # ให้คะแนนความมั่นใจตามความยาวและความเฉพาะเจาะจง
                    confidence_scores[category] = len(keyword) * 2
                    break
        
        if not detected_criteria:
            return {'is_nutrition_query': False}
        
        # จัดอันดับตามความมั่นใจ
        sorted_criteria = sorted(detected_criteria, 
                               key=lambda x: confidence_scores.get(x, 0), 
                               reverse=True)
        
        # แปลงเป็นเกณฑ์การค้นหา
        search_criteria = {
            'is_nutrition_query': True,
            'criteria': sorted_criteria[:3],  # เอาแค่ 3 อันดับแรก
            'confidence_scores': confidence_scores,
            'query_text': query,
            'primary_criterion': sorted_criteria[0] if sorted_criteria else None
        }
        
        return search_criteria
    
    def search_by_enhanced_nutrition_criteria(self, criteria: List[str], limit: int = 10) -> List[Dict]:
        """ค้นหาเมนูตามเกณฑ์โภชนาการ - เวอร์ชันปรับปรุงใหม่"""
        all_nutrition = self.calculate_all_nutrition()
        scored_recipes = []
        
        for idx, nutrition in all_nutrition.items():
            total_score = 0
            detailed_reasons = []
            match_categories = []
            
            for criterion in criteria:
                criterion_score, reason, category = self._evaluate_enhanced_nutrition_criterion(
                    criterion, nutrition
                )
                
                if criterion_score > 0:
                    total_score += criterion_score
                    detailed_reasons.append(reason)
                    match_categories.append(category)
            
            if total_score > 0:
                recipe_data = self.data.iloc[idx]
                
                # คำนวณคะแนนโบนัสจากคุณภาพโภชนาการโดยรวม
                quality_bonus = self._calculate_nutrition_quality_score(nutrition)
                final_score = total_score + quality_bonus
                
                # สร้างข้อมูลสรุป
                summary = self._create_nutrition_summary(nutrition, match_categories)
                
                scored_recipes.append({
                    'index': idx,
                    'name': recipe_data['name'],
                    'score': final_score,
                    'match_score': total_score,
                    'quality_score': quality_bonus,
                    'nutrition': nutrition,
                    'reasons': detailed_reasons,
                    'categories': match_categories,
                    'summary': summary,
                    'ingredient': recipe_data['ingredient'],
                    'method': recipe_data['method'],
                    'health_benefits': self._generate_health_benefits(nutrition, match_categories)
                })
        
        # เรียงลำดับตามคะแนนรวมและคุณภาพ
        scored_recipes.sort(key=lambda x: (x['score'], x['quality_score']), reverse=True)
        return scored_recipes[:limit]
    
    def _evaluate_enhanced_nutrition_criterion(self, criterion: str, nutrition: Dict) -> Tuple[float, str, str]:
        """ประเมินเกณฑ์โภชนาการแบบละเอียด - เวอร์ชันปรับปรุง"""
        
        # แยกประเภทและระดับ
        parts = criterion.split('_')
        if len(parts) < 2:
            return 0.0, "", ""
        
        nutrient = parts[0]
        level = '_'.join(parts[1:])
        
        if nutrient not in self.nutrition_thresholds:
            return 0.0, "", ""
        
        value = nutrition.get(nutrient, 0)
        thresholds = self.nutrition_thresholds[nutrient]
        
        # ประเมินตามระดับที่ต้องการ
        if level == 'very_low':
            return self._evaluate_very_low_criterion(nutrient, value, thresholds)
        elif level == 'low':
            return self._evaluate_low_criterion(nutrient, value, thresholds)
        elif level == 'high':
            return self._evaluate_high_criterion(nutrient, value, thresholds)
        elif level == 'very_high':
            return self._evaluate_very_high_criterion(nutrient, value, thresholds)
        else:
            return self._evaluate_medium_criterion(nutrient, value, thresholds)

    def _evaluate_very_low_criterion(self, nutrient: str, value: float, thresholds: Dict) -> Tuple[float, str, str]:
        """ประเมินเกณฑ์ต่ำมาก"""
        if value <= thresholds['very_low']:
            score = 20.0
            reason = f"{self._get_nutrient_thai_name(nutrient)}ต่ำมาก ({value:.1f})"
            category = f"{nutrient}_very_low"
        elif value <= thresholds['low']:
            score = 15.0
            reason = f"{self._get_nutrient_thai_name(nutrient)}ต่ำ ({value:.1f})"
            category = f"{nutrient}_low"
        elif value <= thresholds['medium']:
            score = 8.0
            reason = f"{self._get_nutrient_thai_name(nutrient)}ปานกลาง ({value:.1f})"
            category = f"{nutrient}_medium"
        else:
            return 0.0, "", ""
        
        return score, reason, category

    def _evaluate_low_criterion(self, nutrient: str, value: float, thresholds: Dict) -> Tuple[float, str, str]:
        """ประเมินเกณฑ์ต่ำ"""
        if value <= thresholds['low']:
            score = 18.0
            reason = f"{self._get_nutrient_thai_name(nutrient)}ต่ำ ({value:.1f})"
            category = f"{nutrient}_low"
        elif value <= thresholds['medium']:
            score = 12.0
            reason = f"{self._get_nutrient_thai_name(nutrient)}ปานกลาง ({value:.1f})"
            category = f"{nutrient}_medium"
        else:
            return 0.0, "", ""
        
        return score, reason, category

    def _evaluate_high_criterion(self, nutrient: str, value: float, thresholds: Dict) -> Tuple[float, str, str]:
        """ประเมินเกณฑ์สูง"""
        if value >= thresholds['very_high']:
            score = 20.0
            reason = f"{self._get_nutrient_thai_name(nutrient)}สูงมาก ({value:.1f})"
            category = f"{nutrient}_very_high"
        elif value >= thresholds['high']:
            score = 18.0
            reason = f"{self._get_nutrient_thai_name(nutrient)}สูง ({value:.1f})"
            category = f"{nutrient}_high"
        elif value >= thresholds['medium']:
            score = 12.0
            reason = f"{self._get_nutrient_thai_name(nutrient)}ปานกลาง ({value:.1f})"
            category = f"{nutrient}_medium"
        else:
            return 0.0, "", ""
        
        return score, reason, category

    def _evaluate_very_high_criterion(self, nutrient: str, value: float, thresholds: Dict) -> Tuple[float, str, str]:
        """ประเมินเกณฑ์สูงมาก"""
        if value >= thresholds['very_high']:
            score = 25.0
            reason = f"{self._get_nutrient_thai_name(nutrient)}สูงมาก ({value:.1f})"
            category = f"{nutrient}_very_high"
        elif value >= thresholds['high']:
            score = 15.0
            reason = f"{self._get_nutrient_thai_name(nutrient)}สูง ({value:.1f})"
            category = f"{nutrient}_high"
        else:
            return 0.0, "", ""
        
        return score, reason, category

    def _evaluate_medium_criterion(self, nutrient: str, value: float, thresholds: Dict) -> Tuple[float, str, str]:
        """ประเมินเกณฑ์ปานกลาง"""
        if thresholds['low'] <= value <= thresholds['high']:
            score = 15.0
            reason = f"{self._get_nutrient_thai_name(nutrient)}ปานกลาง ({value:.1f})"
            category = f"{nutrient}_medium"
            return score, reason, category
        
        return 0.0, "", ""

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

    def _calculate_nutrition_quality_score(self, nutrition: Dict) -> float:
        """คำนวณคะแนนคุณภาพโภชนาการโดยรวม - เวอร์ชันปรับปรุง"""
        score = 0
        
        # คะแนนจากความสมดุลของแมโครนิวเทรียนต์
        calories = nutrition.get('calories', 0)
        protein = nutrition.get('protein', 0)
        carbs = nutrition.get('carbs', 0)
        fat = nutrition.get('fat', 0)
        
        # แคลอรี่เหมาะสม (200-400)
        if 200 <= calories <= 400:
            score += 3
        elif 150 <= calories <= 500:
            score += 2
        elif calories > 0:
            score += 1
        
        # โปรตีนเพียงพอ (>15g)
        if protein >= 25:
            score += 4
        elif protein >= 20:
            score += 3
        elif protein >= 15:
            score += 2
        elif protein >= 10:
            score += 1
        
        # ไมโครนิวเทรียนต์ที่ดี
        fiber = nutrition.get('fiber', 0)
        vitamin_c = nutrition.get('vitamin_c', 0)
        calcium = nutrition.get('calcium', 0)
        iron = nutrition.get('iron', 0)
        vitamin_a = nutrition.get('vitamin_a', 0)
        
        if fiber >= 5:
            score += 3
        elif fiber >= 3:
            score += 2
        elif fiber >= 1:
            score += 1
        
        if vitamin_c >= 25:
            score += 2
        elif vitamin_c >= 15:
            score += 1
        
        if calcium >= 150:
            score += 2
        elif calcium >= 100:
            score += 1
        
        if iron >= 3:
            score += 2
        elif iron >= 2:
            score += 1
        
        if vitamin_a >= 500:
            score += 2
        elif vitamin_a >= 200:
            score += 1
        
        # หักคะแนนจากสิ่งที่ไม่ดี
        sodium = nutrition.get('sodium', 0)
        if sodium > 1500:
            score -= 4
        elif sodium > 1200:
            score -= 3
        elif sodium > 1000:
            score -= 2
        elif sodium > 800:
            score -= 1
        
        if fat > 25:
            score -= 2
        elif fat > 20:
            score -= 1
        
        return max(score, 0)

    def _create_nutrition_summary(self, nutrition: Dict, categories: List[str]) -> str:
        """สร้างสรุปข้อมูลโภชนาการ"""
        summary_parts = []
        
        # แคลอรี่
        calories = nutrition.get('calories', 0)
        if calories > 0:
            if calories <= 200:
                summary_parts.append("🔥แคลอรี่ต่ำ")
            elif calories <= 350:
                summary_parts.append("🔥แคลอรี่ปานกลาง")
            else:
                summary_parts.append("🔥แคลอรี่สูง")
        
        # โปรตีน
        protein = nutrition.get('protein', 0)
        if protein >= 20:
            summary_parts.append("💪โปรตีนสูง")
        elif protein >= 15:
            summary_parts.append("💪โปรตีนดี")
        
        # ไขมัน
        fat = nutrition.get('fat', 0)
        if fat <= 8:
            summary_parts.append("🫒ไขมันต่ำ")
        elif fat <= 15:
            summary_parts.append("🫒ไขมันปานกลาง")
        
        # โซเดียม
        sodium = nutrition.get('sodium', 0)
        if sodium <= 400:
            summary_parts.append("🧂โซเดียมต่ำ")
        elif sodium <= 800:
            summary_parts.append("🧂โซเดียมปานกลาง")
        
        # ใยอาหาร
        fiber = nutrition.get('fiber', 0)
        if fiber >= 4:
            summary_parts.append("🌾ใยอาหารสูง")
        elif fiber >= 2:
            summary_parts.append("🌾ใยอาหารดี")
        
        # วิตามิน
        vitamin_c = nutrition.get('vitamin_c', 0)
        if vitamin_c >= 20:
            summary_parts.append("🍊วิตซีสูง")
        
        vitamin_a = nutrition.get('vitamin_a', 0)
        if vitamin_a >= 500:
            summary_parts.append("🥕วิตเอสูง")
        
        # แร่ธาตุ
        calcium = nutrition.get('calcium', 0)
        if calcium >= 150:
            summary_parts.append("🦴แคลเซียมสูง")
        
        iron = nutrition.get('iron', 0)
        if iron >= 3:
            summary_parts.append("🩸เหล็กสูง")
        
        return " ".join(summary_parts[:4])  # จำกัดไม่เกิน 4 รายการ

    def _generate_health_benefits(self, nutrition: Dict, categories: List[str]) -> List[str]:
        """สร้างข้อความประโยชน์ต่อสุขภาพ"""
        benefits = []
        
        # ประโยชน์จากแคลอรี่
        calories = nutrition.get('calories', 0)
        if calories <= 250:
            benefits.append("ช่วยควบคุมน้ำหนัก")
        
        # ประโยชน์จากโปรตีน
        protein = nutrition.get('protein', 0)
        if protein >= 20:
            benefits.append("เสริมสร้างกล้ามเนื้อ")
        
        # ประโยชน์จากไขมันต่ำ
        fat = nutrition.get('fat', 0)
        if fat <= 10:
            benefits.append("ดีต่อหัวใจและหลอดเลือด")
        
        # ประโยชน์จากโซเดียมต่ำ
        sodium = nutrition.get('sodium', 0)
        if sodium <= 600:
            benefits.append("ช่วยควบคุมความดันโลหิต")
        
        # ประโยชน์จากใยอาหาร
        fiber = nutrition.get('fiber', 0)
        if fiber >= 3:
            benefits.append("ส่งเสริมระบบย่อยอาหาร")
        
        # ประโยชน์จากวิตามินซี
        vitamin_c = nutrition.get('vitamin_c', 0)
        if vitamin_c >= 15:
            benefits.append("เสริมภูมิคุ้มกัน")
        
        # ประโยชน์จากแคลเซียม
        calcium = nutrition.get('calcium', 0)
        if calcium >= 100:
            benefits.append("บำรุงกระดูกและฟัน")
        
        # ประโยชน์จากเหล็ก
        iron = nutrition.get('iron', 0)
        if iron >= 2:
            benefits.append("ป้องกันโลหิตจาง")
        
        return benefits[:3]  # จำกัดไม่เกิน 3 ข้อ

    def create_enhanced_nutrition_comparison_chart(self, recipes: List[Dict], 
                                                 nutrients: List[str] = None) -> go.Figure:
        """สร้างกราฟเปรียบเทียบค่าโภชนาการ - เวอร์ชันปรับปรุง"""
        if not nutrients:
            nutrients = ['calories', 'protein', 'carbs', 'fat', 'fiber', 'sodium']
        
        if not recipes:
            return go.Figure()
        
        recipe_names = [recipe['name'][:12] + '...' if len(recipe['name']) > 12 
                       else recipe['name'] for recipe in recipes[:6]]
        
        # สร้าง subplot แบบ 2x3
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'แคลอรี่ (kcal)', 'โปรตีน (g)', 'คาร์โบไฮเดรต (g)', 
                'ไขมัน (g)', 'ใยอาหาร (g)', 'โซเดียม (mg)'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        positions = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]
        
        for i, nutrient in enumerate(nutrients):
            if i >= len(positions):
                break
                
            values = [recipe['nutrition'].get(nutrient, 0) for recipe in recipes[:6]]
            row, col = positions[i]
            
            # เลือกสีตามค่า
            bar_colors = []
            for value in values:
                if nutrient == 'sodium':
                    # โซเดียม: ต่ำ = เขียว, สูง = แดง
                    if value <= 400:
                        bar_colors.append('#28a745')
                    elif value <= 800:
                        bar_colors.append('#ffc107')
                    else:
                        bar_colors.append('#dc3545')
                elif nutrient in ['calories', 'fat']:
                    # แคลอรี่และไขมัน: ต่ำ = เขียว
                    thresholds = self.nutrition_thresholds[nutrient]
                    if value <= thresholds['low']:
                        bar_colors.append('#28a745')
                    elif value <= thresholds['medium']:
                        bar_colors.append('#ffc107')
                    else:
                        bar_colors.append('#fd7e14')
                else:
                    # สารอาหารอื่น: สูง = เขียว
                    thresholds = self.nutrition_thresholds.get(nutrient, {})
                    high_threshold = thresholds.get('high', 999)
                    medium_threshold = thresholds.get('medium', 999)
                    
                    if value >= high_threshold:
                        bar_colors.append('#28a745')
                    elif value >= medium_threshold:
                        bar_colors.append('#ffc107')
                    else:
                        bar_colors.append('#fd7e14')
            
            fig.add_trace(
                go.Bar(
                    x=recipe_names,
                    y=values,
                    name=self._get_nutrient_thai_name(nutrient),
                    marker_color=bar_colors,
                    showlegend=False,
                    text=[f'{val:.1f}' for val in values],
                    textposition='outside'
                ),
                row=row, col=col
            )
            
            # เพิ่มเส้นแนะนำ
            if nutrient in self.nutrition_thresholds:
                thresholds = self.nutrition_thresholds[nutrient]
                if nutrient == 'sodium':
                    # เส้นแนะนำสำหรับโซเดียม (ควรต่ำกว่า)
                    fig.add_hline(
                        y=thresholds['low'], 
                        line_dash="dash", 
                        line_color="green",
                        annotation_text="แนะนำ",
                        row=row, col=col
                    )
                else:
                    # เส้นแนะนำสำหรับสารอาหารอื่น
                    target = thresholds.get('medium', 0)
                    if target > 0:
                        fig.add_hline(
                            y=target, 
                            line_dash="dash", 
                            line_color="blue",
                            annotation_text="เป้าหมาย",
                            row=row, col=col
                        )
        
        fig.update_layout(
            height=700,
            title_text="📊 การเปรียบเทียบค่าโภชนาการแบบละเอียด",
            title_x=0.5,
            font=dict(family="Sarabun, sans-serif"),
            showlegend=False
        )
        
        # ปรับแต่ง x-axis
        for i in range(1, 3):
            for j in range(1, 4):
                fig.update_xaxes(tickangle=45, row=i, col=j)
        
        return fig

    def display_enhanced_nutrition_recommendations(self, recommendations: Dict):
        """แสดงผลคำแนะนำอาหารตามโภชนาการ - เวอร์ชันปรับปรุง"""
        if recommendations['type'] == 'not_nutrition_query':
            return False
        
        if recommendations['type'] == 'no_results':
            st.warning(recommendations['message'])
            return True
        
        results = recommendations['results']
        
        # แสดงหัวข้อและสถิติ
        st.markdown(f"### 🎯 ผลการค้นหา: {recommendations['query']}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📋 พบเมนู", f"{len(results)} รายการ")
        with col2:
            avg_score = sum(r['score'] for r in results) / len(results) if results else 0
            st.metric("⭐ คะแนนเฉลี่ย", f"{avg_score:.1f}")
        with col3:
            high_quality = sum(1 for r in results if r.get('quality_score', 0) >= 5)
            st.metric("🏆 คุณภาพสูง", f"{high_quality} เมนู")
        with col4:
            matched_criteria = recommendations.get('criteria', [])
            st.metric("🔍 เกณฑ์ตรงกัน", f"{len(matched_criteria)} เกณฑ์")
        
        # แสดงกราฟเปรียบเทียบ
        if len(results) > 1:
            st.markdown("#### 📊 การเปรียบเทียบค่าโภชนาการ")
            fig = self.create_enhanced_nutrition_comparison_chart(results)
            st.plotly_chart(fig, use_container_width=True)
        
        # แสดงรายการแนะนำ
        st.markdown("#### 🍽️ รายการเมนูแนะนำ")
        
        for i, recipe in enumerate(results, 1):
            with st.expander(f"🥘 {i}. {recipe['name']} (คะแนน: {recipe['score']:.1f})", 
                           expanded=(i <= 2)):
                
                # แถวแรก: ข้อมูลหลัก
                main_col1, main_col2 = st.columns([1, 1])
                
                with main_col1:
                    st.markdown("#### 🥬 วัตถุดิบ")
                    ingredients_formatted = self._format_ingredients(recipe['ingredient'])
                    st.markdown(ingredients_formatted, unsafe_allow_html=True)
                    
                    st.markdown("#### ✨ เหมาะสมเพราะ")
                    for reason in recipe['reasons']:
                        st.markdown(f"• {reason}")
                    
                    if recipe.get('health_benefits'):
                        st.markdown("#### 💊 ประโยชน์ต่อสุขภาพ")
                        for benefit in recipe['health_benefits']:
                            st.markdown(f"• {benefit}")
                
                with main_col2:
                    st.markdown("#### 📊 ค่าโภชนาการโดยละเอียด")
                    nutrition = recipe['nutrition']
                    
                    # แสดงข้อมูลสรุป
                    if recipe.get('summary'):
                        st.markdown(f"**สรุป:** {recipe['summary']}")
                        st.markdown("---")
                    
                    # แสดงค่าโภชนาการหลักในรูปแบบ metrics
                    metrics_col1, metrics_col2 = st.columns(2)
                    with metrics_col1:
                        st.metric("🔥 แคลอรี่", f"{nutrition['calories']:.0f} kcal")
                        st.metric("🥩 โปรตีน", f"{nutrition['protein']:.1f} g")
                        st.metric("🍞 คาร์โบ", f"{nutrition['carbs']:.1f} g")
                        st.metric("🫒 ไขมัน", f"{nutrition['fat']:.1f} g")
                    
                    with metrics_col2:
                        st.metric("🌾 ใยอาหาร", f"{nutrition['fiber']:.1f} g")
                        st.metric("🧂 โซเดียม", f"{nutrition['sodium']:.0f} mg")
                        st.metric("🦴 แคลเซียม", f"{nutrition['calcium']:.0f} mg")
                        st.metric("⚡ เหล็ก", f"{nutrition['iron']:.1f} mg")
                    
                    # แสดงวิตามินและแร่ธาตุอื่นๆ
                    with st.expander("🧪 วิตามินและแร่ธาตุเพิ่มเติม"):
                        vit_col1, vit_col2 = st.columns(2)
                        with vit_col1:
                            st.write(f"🅰️ วิตามิน A: {nutrition['vitamin_a']:.0f} IU")
                            st.write(f"🍊 วิตามิน C: {nutrition['vitamin_c']:.1f} mg")
                        with vit_col2:
                            st.write(f"🫀 โปแตสเซียม: {nutrition['potassium']:.0f} mg")
                            st.write(f"🧠 วิตามิน B1: {nutrition.get('vitamin_b1', 0):.2f} mg")
                
                # แถวที่สอง: วิธีทำ
                st.markdown("#### 👨‍🍳 วิธีทำ")
                method_formatted = self._format_cooking_method(recipe['method'])
                st.markdown(method_formatted, unsafe_allow_html=True)
                
                # แถวที่สาม: การประเมินและคำแนะนำ
                eval_col1, eval_col2 = st.columns([1, 1])
                
                with eval_col1:
                    # แสดงคะแนนประเมิน
                    st.markdown("#### 🏅 การประเมิน")
                    score_bar_value = min(recipe['score'] / 25 * 100, 100)
                    st.progress(score_bar_value / 100)
                    st.write(f"คะแนนรวม: {recipe['score']:.1f}/25")
                    st.write(f"คะแนนจากเกณฑ์: {recipe.get('match_score', 0):.1f}")
                    st.write(f"คะแนนคุณภาพ: {recipe.get('quality_score', 0):.1f}")
                
                with eval_col2:
                    # คำแนะนำการบริโภค
                    st.markdown("#### 💡 คำแนะนำการบริโภค")
                    
                    calories = nutrition['calories']
                    if calories <= 200:
                        st.info("🍽️ เหมาะเป็นเมนูเบาๆ หรืออาหารว่าง")
                    elif calories <= 400:
                        st.success("🍽️ เหมาะเป็นอาหารมื้อหลัก")
                    else:
                        st.warning("🍽️ แคลอรี่สูง ควรระมัดระวังปริมาณ")
                    
                    sodium = nutrition['sodium']
                    if sodium > 1000:
                        st.warning("🧂 โซเดียมสูง ไม่เหมาะสำหรับผู้ป่วยความดันสูง")
                    elif sodium <= 400:
                        st.success("🧂 โซเดียมต่ำ เหมาะสำหรับทุกคน")
                    
                    protein = nutrition['protein']
                    if protein >= 20:
                        st.success("💪 โปรตีนสูง เหมาะสำหรับผู้ออกกำลังกาย")
        
        return True
    
    def _format_ingredients(self, ingredients_text: str) -> str:
        """จัดรูปแบบรายการวัตถุดิบ - เวอร์ชันปรับปรุง"""
        if not ingredients_text:
            return "<p>ไม่มีข้อมูลวัตถุดิบ</p>"
        
        ingredients = ingredients_text.split('\n')
        formatted = "<ul style='margin: 0; padding-left: 1.5rem; line-height: 1.6;'>"
        for item in ingredients:
            if item.strip():
                clean_item = item.strip().lstrip('- ')
                # เน้นปริมาณและหน่วย
                highlighted_item = re.sub(
                    r'(\d+(?:\.\d+)?)\s*([ก-๙a-zA-Z]+)', 
                    r'<strong>\1 \2</strong>', 
                    clean_item
                )
                formatted += f"<li style='margin: 0.3rem 0;'>{highlighted_item}</li>"
        formatted += "</ul>"
        return formatted
    
    def _format_cooking_method(self, method_text: str) -> str:
        """จัดรูปแบบวิธีทำ - เวอร์ชันปรับปรุง"""
        if not method_text:
            return "<p>ไม่มีข้อมูลวิธีทำ</p>"
        
        # ตรวจสอบว่ามีเลขขั้นตอนอยู่แล้วหรือไม่
        has_numbers = bool(re.search(r'^\s*\d+\.', method_text, re.MULTILINE))
        
        sentences = re.split(r'[.!?]', method_text)
        formatted = "<div style='margin: 0; line-height: 1.8;'>"
        
        step_num = 1
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if has_numbers and re.match(r'^\d+', sentence):
                formatted += f"<p style='margin: 0.5rem 0; padding: 0.3rem; background: #f8f9fa; border-left: 3px solid #007bff; border-radius: 3px;'><strong>{sentence}</strong></p>"
            else:
                # เพิ่มหมายเลขขั้นตอนถ้ายังไม่มี
                if not has_numbers and len(sentence) > 10:
                    formatted += f"<p style='margin: 0.5rem 0; padding: 0.3rem; background: #f8f9fa; border-left: 3px solid #28a745; border-radius: 3px;'><strong>{step_num}.</strong> {sentence}</p>"
                    step_num += 1
                else:
                    formatted += f"<p style='margin: 0.5rem 0;'>{sentence}</p>"
        
        formatted += "</div>"
        return formatted
    
    def get_enhanced_nutrition_suggestions(self) -> List[str]:
        """ให้คำแนะนำการค้นหาตามโภชนาการ - เวอร์ชันปรับปรุง"""
        return [
            "🔥 แนะนำอาหารแคลอรี่ต่ำมาก (น้อยกว่า 100 kcal)",
            "💪 เมนูโปรตีนสูงมากสำหรับนักกีฬา (มากกว่า 35g)", 
            "🥗 อาหารไขมันต่ำมากเพื่อสุขภาพหัวใจ (น้อยกว่า 2g)",
            "🍃 เมนูใยอาหารสูงมากช่วยขับถ่าย (มากกว่า 10g)",
            "🧂 อาหารโซเดียมต่ำมากสำหรับความดันสูง (น้อยกว่า 150mg)",
            "🍊 เมนูวิตามินซีสูงมากเสริมภูมิคุ้มกัน (มากกว่า 70mg)",
            "🦴 อาหารแคลเซียมสูงมากบำรุงกระดูก (มากกว่า 300mg)",
            "🩸 เมนูเหล็กสูงมากป้องกันโลหิตจาง (มากกว่า 7mg)",
            "🥕 อาหารวิตามินเอสูงมากบำรุงสายตา (มากกว่า 1200 IU)",
            "🌿 อาหารเหมาะสำหรับผู้ป่วยเบาหวาน",
            "❤️ เมนูเหมาะสำหรับผู้ป่วยโรคหัวใจ",
            "🧠 อาหารบำรุงสมองและระบบประสาท",
            "👶 เมนูเหมาะสำหรับเด็กและการเจริญเติบโต",
            "🏃‍♀️ อาหารสำหรับนักกีฬาและผู้ออกกำลังกาย",
            "🤰 เมนูสำหรับหญิงตั้งครรภ์และให้นม"
        ]

    def analyze_nutrition_trends(self, recipes: List[Dict]) -> Dict:
        """วิเคราะห์แนวโน้มโภชนาการจากเมนูที่พบ"""
        if not recipes:
            return {}
        
        nutrients_data = {
            'calories': [], 'protein': [], 'carbs': [], 'fat': [], 
            'fiber': [], 'sodium': [], 'calcium': [], 'iron': [],
            'vitamin_a': [], 'vitamin_c': []
        }
        
        for recipe in recipes:
            nutrition = recipe.get('nutrition', {})
            for nutrient in nutrients_data:
                value = nutrition.get(nutrient, 0)
                nutrients_data[nutrient].append(value)
        
        analysis = {}
        for nutrient, values in nutrients_data.items():
            if values:
                analysis[nutrient] = {
                    'average': np.mean(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'std': np.std(values),
                    'median': np.median(values)
                }
        
        return analysis

    def create_nutrition_distribution_chart(self, recipes: List[Dict], nutrient: str) -> go.Figure:
        """สร้างกราฟการกระจายของสารอาหารเฉพาะ"""
        if not recipes:
            return go.Figure()
        
        values = [recipe['nutrition'].get(nutrient, 0) for recipe in recipes]
        names = [recipe['name'] for recipe in recipes]
        
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=values,
            nbinsx=10,
            name=f'การกระจาย{self._get_nutrient_thai_name(nutrient)}',
            marker_color='rgba(102, 126, 234, 0.7)',
            yaxis='y'
        ))
        
        # Box plot
        fig.add_trace(go.Box(
            y=values,
            name=f'{self._get_nutrient_thai_name(nutrient)}',
            marker_color='rgba(255, 107, 107, 0.7)',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title=f'📈 การกระจายของ{self._get_nutrient_thai_name(nutrient)}ในเมนูที่พบ',
            xaxis_title=f'{self._get_nutrient_thai_name(nutrient)}',
            yaxis_title='จำนวนเมนู',
            yaxis2=dict(
                title='ค่า',
                overlaying='y',
                side='right'
            ),
            font=dict(family="Sarabun, sans-serif"),
            height=400
        )
        
        return fig
