import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Tuple, Any
from difflib import SequenceMatcher
import json

class RecipeSearchEngine:
    """ระบบค้นหาสูตรอาหารไทยขั้นสูงพร้อมการวิเคราะห์โภชนาการ - เวอร์ชันปรับปรุง"""
    
    def __init__(self, data: pd.DataFrame, nutrition_api):
        self.data = data
        self.nutrition_api = nutrition_api
        
        # คำสำคัญสำหรับการค้นหาตามโภชนาการ - เพิ่มความครอบคลุม
        self.nutrition_keywords = {
            # แคลอรี่
            'calories_very_low': [
                'แคลอรี่ต่ำมาก', 'แคลต่ำมาก', 'ลดน้ำหนักเร่งด่วน', 
                'ไดเอทเข้มข้น', 'แคลอรี่น้อยที่สุด'
            ],
            'calories_low': [
                'แคลอรี่ต่ำ', 'แคลต่ำ', 'ลดน้ำหนัก', 'เบา', 'ไม่อ้วน', 
                'ไดเอท', 'diet', 'ลดความอ้วน', 'คุมน้ำหนัก', 'แคลอรี่น้อย'
            ],
            'calories_high': [
                'แคลอรี่สูง', 'แคลสูง', 'เพิ่มน้ำหนัก', 'พลังงานสูง', 
                'เติมแรง', 'นักกีฬา', 'กำลัง', 'แคลอรี่มาก'
            ],
            
            # โปรตีน
            'protein_very_high': [
                'โปรตีนสูงมาก', 'โปรสูงมาก', 'เพาะกาย', 'bodybuilding',
                'นักกีฬาระดับสูง', 'โปรตีนเข้มข้น'
            ],
            'protein_high': [
                'โปรตีนสูง', 'โปรตีนมาก', 'เนื้อเยื่อ', 'กล้ามเนื้อ', 
                'นักกีฬา', 'ออกกำลังกาย', 'ฟิตเนส', 'เสริมสร้าง',
                'โปรสูง', 'protein high'
            ],
            'protein_low': [
                'โปรตีนต่ำ', 'โปรตีนน้อย', 'ไต', 'โรคไต', 'โปรต่ำ'
            ],
            
            # ไขมัน
            'fat_very_low': [
                'ไขมันต่ำมาก', 'ไม่มีไขมัน', 'fat free', 'โรคหัวใจ',
                'ไขมันน้อยที่สุด', 'ไขมัน 0'
            ],
            'fat_low': [
                'ไขมันต่ำ', 'ไขมันน้อย', 'ลดไขมัน', 'ไม่มันเยอะ', 
                'สุขภาพดี', 'หัวใจ', 'โรคหัวใจ', 'ไขมันต่ำ'
            ],
            'fat_high': [
                'ไขมันสูง', 'ไขมันมาก', 'มันเยอะ', 'ไขมันดี'
            ],
            
            # คาร์โบไฮเดรต
            'carbs_very_low': [
                'คาร์โบต่ำมาก', 'keto', 'ketogenic', 'คีโตเจนิค',
                'คาร์โบเกือบศูนย์', 'no carb'
            ],
            'carbs_low': [
                'คาร์โบต่ำ', 'แป้งน้อย', 'น้ำตาลต่ำ', 'เบาหวาน', 
                'คีโต', 'low carb', 'ลดแป้ง', 'คาร์โบน้อย'
            ],
            'carbs_high': [
                'คาร์โบสูง', 'แป้งมาก', 'พลังงาน', 'นักกีฬา', 'ข้าว',
                'คาร์โบมาก', 'แป้งเยอะ'
            ],
            
            # ใยอาหาร
            'fiber_very_high': [
                'ใยอาหารสูงมาก', 'ใยมากๆ', 'ท้องผูกมาก', 'ใยเข้มข้น',
                'ใยอาหารเยอะมาก'
            ],
            'fiber_high': [
                'ใยอาหารสูง', 'ใยอาหารมาก', 'ขับถ่าย', 'ท้องผูก', 
                'ย่อย', 'ระบบย่อย', 'ช่วยย่อย', 'ผัก', 'ใยสูง',
                'fiber high', 'ใยอาหารเยอะ'
            ],
            
            # วิตามิน
            'vitamin_a_very_high': [
                'วิตามินเอสูงมาก', 'วิตเอสูงมาก', 'สายตาดีมาก',
                'ต้านอนุมูลอิสระสูง'
            ],
            'vitamin_a_high': [
                'วิตามินเอสูง', 'วิตามินเอ', 'วิตเอ', 'สายตา', 'ผิวพรรณ', 
                'ตา', 'บำรุงตา', 'vitamin a', 'วิตามินเอมาก'
            ],
            'vitamin_c_very_high': [
                'วิตามินซีสูงมาก', 'วิตซีสูงมาก', 'ภูมิคุ้มกันแข็งแรงมาก',
                'ต้านหวัดแรง'
            ],
            'vitamin_c_high': [
                'วิตามินซีสูง', 'วิตามินซี', 'วิตซี', 'ภูมิคุ้มกัน', 'ต้านหวัด', 
                'เสริมภูมิ', 'ต้านอนุมูลอิสระ', 'vitamin c', 'วิตามินซีมาก'
            ],
            'vitamin_b_high': [
                'วิตามินบี', 'วิตบี', 'ระบบประสาท', 'เมแทบอลิซึม', 'พลังงาน',
                'vitamin b', 'วิตามินบีมาก'
            ],
            'vitamin_d_high': [
                'วิตามินดี', 'วิตดี', 'กระดูกแข็งแรง', 'vitamin d',
                'วิตามินดีมาก'
            ],
            
            # แร่ธาตุ
            'calcium_very_high': [
                'แคลเซียมสูงมาก', 'กระดูกแข็งแรงมาก', 'ป้องกันกระดูกพรุน',
                'แคลเซียมเข้มข้น'
            ],
            'calcium_high': [
                'แคลเซียมสูง', 'แคลเซียม', 'กระดูก', 'ฟัน', 
                'ผู้สูงอายุ', 'เด็ก', 'บำรุงกระดูก', 'calcium',
                'แคลเซียมมาก'
            ],
            'iron_very_high': [
                'เหล็กสูงมาก', 'ธาตุเหล็กสูงมาก', 'รักษาโลหิตจาง',
                'เหล็กเข้มข้น'
            ],
            'iron_high': [
                'เหล็กสูง', 'ธาตุเหล็ก', 'โลหิตจาง', 'เลือดจาง', 
                'ผู้หญิง', 'ประจำเดือน', 'iron', 'เหล็กมาก'
            ],
            'potassium_very_high': [
                'โปแตสเซียมสูงมาก', 'หัวใจแข็งแรงมาก', 'ความดันต่ำมาก'
            ],
            'potassium_high': [
                'โปแตสเซียมสูง', 'โปแตสเซียม', 'ความดันโลหิต', 
                'หัวใจ', 'กล้ามเนื้อหัวใจ', 'potassium', 'โปแตสเซียมมาก'
            ],
            'sodium_very_low': [
                'โซเดียมต่ำมาก', 'เกลือน้อยมาก', 'ไม่มีเกลือ',
                'ความดันสูงมาก', 'โซเดียมเกือบศูนย์'
            ],
            'sodium_low': [
                'โซเดียมต่ำ', 'เกลือน้อย', 'ความดันสูง', 'ไต', 
                'หัวใจ', 'จืด', 'ไม่เค็ม', 'sodium low', 'โซเดียมน้อย'
            ],
            'zinc_high': [
                'สังกะสีสูง', 'สังกะสี', 'ภูมิคุ้มกัน', 'แผลหาย',
                'zinc', 'สังกะสีมาก'
            ],
            'magnesium_high': [
                'แมกนีเซียมสูง', 'แมกนีเซียม', 'กล้ามเนื้อ', 'ประสาท',
                'magnesium', 'แมกนีเซียมมาก'
            ],
            
            # กลุ่มผู้ป่วยเฉพาะ - เพิ่มคำสำคัญ
            'diabetes': [
                'เบาหวาน', 'ผู้ป่วยเบาหวาน', 'น้ำตาลต่ำ', 
                'ควบคุมน้ำตาล', 'เบาหวาน', 'ดัชนีน้ำตาล',
                'diabetes', 'diabetic', 'น้ำตาลในเลือด'
            ],
            'hypertension': [
                'ความดันสูง', 'ผู้ป่วยความดัน', 'โซเดียมต่ำ', 
                'ความดัน', 'ไฮเปอร์เทนชั่น', 'hypertension',
                'ความดันโลหิต'
            ],
            'heart_disease': [
                'โรคหัวใจ', 'หัวใจ', 'โคเลสเตอรอล', 'หลอดเลือด',
                'heart disease', 'cardiovascular', 'หัวใจวาย'
            ],
            'kidney_disease': [
                'โรคไต', 'ไต', 'ล้างไต', 'ไตเสื่อม', 'kidney disease',
                'renal', 'ไตวาย'
            ],
            'liver_disease': [
                'โรคตับ', 'ตับ', 'ตับแข็ง', 'liver disease',
                'hepatitis', 'ตับอักเสบ'
            ],
            'elderly': [
                'ผู้สูงอายุ', 'คนแก่', 'นุ่ม', 'ย่อยง่าย', 
                'ผู้ใหญ่', 'วัยชรา', 'elderly', 'senior'
            ],
            'children': [
                'เด็ก', 'เด็กเล็ก', 'แคลเซียม', 'เจริญเติบโต', 
                'ลูก', 'วัยรุ่น', 'children', 'kid', 'เด็กโต'
            ],
            'athletes': [
                'นักกีฬา', 'ออกกำลังกาย', 'โปรตีนสูง', 'ฟิตเนส', 
                'กล้ามเนื้อ', 'เล่นกีฬา', 'athlete', 'fitness',
                'เพาะกาย', 'bodybuilding'
            ],
            'pregnant': [
                'ตั้งครรภ์', 'คนท้อง', 'โฟเลต', 'เหล็ก', 
                'แม่ท้อง', 'มีครรภ์', 'pregnant', 'pregnancy',
                'ให้นม', 'breastfeeding'
            ],
            
            # ประเภทอาหาร - เพิ่มคำสำคัญ
            'vegetarian': [
                'มังสวิรัติ', 'เจ', 'ไม่กินเนื้อ', 'ผัก', 'พืช', 'เจ',
                'vegetarian', 'vegan', 'plant based'
            ],
            'healthy': [
                'สุขภาพ', 'สุขภาพดี', 'คลีน', 'clean eating', 
                'healthy', 'เพื่อสุขภาพ', 'organic', 'ธรรมชาติ'
            ],
            'weight_loss': [
                'ลดน้ำหนัก', 'ลดความอ้วน', 'ไดเอท', 'เบา', 
                'คุมน้ำหนัก', 'weight loss', 'diet', 'slim'
            ],
            'weight_gain': [
                'เพิ่มน้ำหนัก', 'อ้วน', 'น้ำหนักขึ้น', 'ผอมเกินไป',
                'weight gain', 'bulk', 'mass'
            ],
            'detox': [
                'ดีท็อกซ์', 'ล้างพิษ', 'ล้างลำไส้', 'detox', 
                'ขับสารพิษ', 'cleanse', 'ล้างตับ'
            ],
            'anti_aging': [
                'ต้านอนุมูลอิสระ', 'แอนตี้เอจจิ้ง', 'anti aging',
                'ชะลอวัย', 'อายุยืน', 'antioxidant'
            ]
        }
        
        # เกณฑ์การจัดกลุ่มโภชนาการ - ปรับปรุงให้แม่นยำและครอบคลุมขึ้น
        self.nutrition_thresholds = {
            'calories': {
                'very_low': 120, 'low': 250, 'medium': 400, 
                'high': 600, 'very_high': 800
            },
            'protein': {
                'very_low': 5, 'low': 10, 'medium': 18, 'high': 28, 'very_high': 40
            },
            'fat': {
                'very_low': 3, 'low': 8, 'medium': 15, 'high': 25, 'very_high': 35
            },
            'carbs': {
                'very_low': 5, 'low': 15, 'medium': 30, 'high': 50, 'very_high': 70
            },
            'fiber': {
                'very_low': 0.5, 'low': 2, 'medium': 4, 'high': 7, 'very_high': 12
            },
            'vitamin_a': {
                'very_low': 20, 'low': 100, 'medium': 300, 'high': 700, 'very_high': 1500
            },
            'vitamin_c': {
                'very_low': 2, 'low': 8, 'medium': 20, 'high': 40, 'very_high': 80
            },
            'calcium': {
                'very_low': 20, 'low': 60, 'medium': 120, 'high': 200, 'very_high': 400
            },
            'iron': {
                'very_low': 0.5, 'low': 1.5, 'medium': 3, 'high': 5, 'very_high': 8
            },
            'potassium': {
                'very_low': 100, 'low': 250, 'medium': 400, 'high': 600, 'very_high': 900
            },
            'sodium': {
                'very_low': 200, 'low': 500, 'medium': 900, 
                'high': 1400, 'very_high': 2000
            }
        }
        
        # คำสำคัญประเภทการปรุง - เพิ่มความครอบคลุม
        self.cooking_method_keywords = {
            'ทอด': ['ทอด', 'เจียว', 'กรอบ', 'ทอดกรอบ', 'ทอดแกง'],
            'ต้ม': ['ต้ม', 'แกง', 'น้ำซุป', 'ซุป', 'แกงจืด'],
            'ผัด': ['ผัด', 'คั่ว', 'ผัดไฟแรง'],
            'ย่าง': ['ย่าง', 'ปิ้ง', 'เผา', 'ย่างถ่าน', 'บาร์บีคิว'],
            'ยำ': ['ยำ', 'ลาบ', 'ตำ', 'ส้มตำ'],
            'นึ่ง': ['นึ่ง', 'อบ', 'นึ่งไฟอ่อน'],
            'ลวก': ['ลวก', 'ต้มสุก', 'ลวกเบาๆ'],
            'แช่': ['แช่', 'ดอง', 'หมัก', 'แช่น้ำแข็ง'],
            'อบ': ['อบ', 'เบค', 'อบเตาอบ'],
            'ทอดแกง': ['ทอดแกง', 'แกงป่า'],
            'ปนหุง': ['ปนหุง', 'หุงข้าว']
        }

        # คำสำคัญวัตถุดิบ - เพิ่มความครอบคลุม
        self.ingredient_keywords = {
            'เนื้อสัตว์': {
                'หมู': ['หมู', 'สันใน', 'สันนอก', 'หมูสับ', 'หมูแผ่น'],
                'ไก่': ['ไก่', 'อกไก่', 'ขาไก่', 'ไก่สับ', 'ไก่ทั้งตัว'],
                'เนื้อ': ['เนื้อ', 'เนื้อวัว', 'เนื้อควาย', 'เนื้อสับ'],
                'กุ้ง': ['กุ้ง', 'กุ้งนาง', 'กุ้งแม่น้ำ', 'กุ้งฝอย'],
                'ปลา': ['ปลา', 'ปลาช่อน', 'ปลาทู', 'ปลาหมึก'],
                'ไข่': ['ไข่', 'ไข่ไก่', 'ไข่เป็ด', 'ไข่เค็ม']
            },
            'ผัก': {
                'ผักใบเขียว': ['ผักบุ้ง', 'คะน้า', 'ผักกาด', 'ผักชี'],
                'ผักผล': ['มะเขือเทศ', 'มะเขือเปราะ', 'ฟักทอง', 'แตงกวา'],
                'ผักรส': ['กะเพรา', 'โหระพา', 'ใบมะกรูด', 'ตะไคร้']
            },
            'เครื่องปรุง': {
                'เครื่องปรุงพื้นฐาน': ['น้ำปลา', 'ซีอิ๊ว', 'เกลือ', 'น้ำตาล'],
                'เครื่องแกง': ['พริกแกง', 'กะทิ', 'ข่า', 'ตะไคร้'],
                'น้ำมัน': ['น้ำมันพืช', 'น้ำมันหมู', 'น้ำมันมะพร้าว']
            }
        }

    def search_recipes(self, query: str, max_results: int = 10, 
                      search_mode: str = "comprehensive") -> List[Dict]:
        """ค้นหาสูตรอาหารแบบครอบคลุม - เวอร์ชันปรับปรุง"""
        
        query_lower = query.lower().strip()
        
        # 1. ลองค้นหาตามโภชนาการก่อน (ให้ความสำคัญสูงสุด)
        nutrition_results = self.search_by_enhanced_nutrition(query_lower, max_results)
        if nutrition_results:
            return self._rank_and_filter_results(nutrition_results, max_results)
        
        # 2. ค้นหาตามชื่อเมนู (ความสำคัญรองลงมา)
        name_results = self.search_by_enhanced_name(query_lower, max_results)
        if name_results:
            return self._rank_and_filter_results(name_results, max_results)
        
        # 3. ค้นหาตามวัตถุดิบ
        ingredient_results = self.search_by_enhanced_ingredient(query_lower, max_results)
        if ingredient_results:
            return self._rank_and_filter_results(ingredient_results, max_results)
        
        # 4. ค้นหาตามวิธีการปรุง
        method_results = self.search_by_enhanced_cooking_method(query_lower, max_results)
        if method_results:
            return self._rank_and_filter_results(method_results, max_results)
        
        # 5. ค้นหาแบบคลุมเครือ (สุดท้าย)
        return self.enhanced_fuzzy_search(query_lower, max_results)

    def search_by_enhanced_nutrition(self, query: str, max_results: int) -> List[Dict]:
        """ค้นหาตามเกณฑ์โภชนาการ - เวอร์ชันปรับปรุง"""
        
        # ตรวจสอบคำสำคัญโภชนาการแบบละเอียด
        detected_criteria = []
        criteria_weights = {}  # น้ำหนักความสำคัญของแต่ละเกณฑ์
        
        for category, keywords in self.nutrition_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    detected_criteria.append(category)
                    # ให้น้ำหนักตามความเฉพาะเจาะจง
                    criteria_weights[category] = len(keyword)
                    break
        
        if not detected_criteria:
            return []
        
        results = []
        
        for _, row in self.data.iterrows():
            nutrition = self._get_enhanced_nutrition_data(row)
            if not nutrition:
                continue
            
            total_score = 0
            matched_reasons = []
            
            # ประเมินตามเกณฑ์ที่ตรวจพบ
            for criterion in detected_criteria:
                criterion_score, reason = self._evaluate_enhanced_nutrition_criterion(
                    criterion, nutrition
                )
                
                # ใช้น้ำหนักในการคำนวณคะแนน
                weight = criteria_weights.get(criterion, 1)
                weighted_score = criterion_score * (weight / 10)
                total_score += weighted_score
                
                if reason:
                    matched_reasons.append(reason)
            
            if total_score > 0:
                # คำนวณคะแนนความเหมาะสมโดยรวม
                overall_nutrition_score = self._calculate_comprehensive_nutrition_score(nutrition)
                final_score = total_score + (overall_nutrition_score * 0.3)
                
                results.append({
                    'name': row['name'],
                    'ingredient': row['ingredient'],
                    'method': row['method'],
                    'score': final_score,
                    'similarity': min(final_score / 20, 1.0),
                    'reasons': matched_reasons,
                    'nutrition': nutrition,
                    'search_type': 'enhanced_nutrition',
                    'matched_criteria': detected_criteria
                })
        
        # เรียงลำดับตามคะแนนและความเหมาะสม
        results.sort(key=lambda x: (x['score'], x['similarity']), reverse=True)
        return results

    def search_by_enhanced_name(self, query: str, max_results: int) -> List[Dict]:
        """ค้นหาตามชื่อเมนู - เวอร์ชันปรับปรุง"""
        
        results = []
        
        for _, row in self.data.iterrows():
            similarity_score = self._calculate_enhanced_name_similarity(query, row['name'])
            
            if similarity_score > 0.25:  # ลดเกณฑ์เพื่อให้ครอบคลุมมากขึ้น
                nutrition = self._get_enhanced_nutrition_data(row)
                
                # คำนวณคะแนนเพิ่มเติมจากการตรงกันของคำ
                word_bonus = self._calculate_word_match_bonus(query, row['name'])
                final_similarity = similarity_score + word_bonus
                
                results.append({
                    'name': row['name'],
                    'ingredient': row['ingredient'],
                    'method': row['method'],
                    'score': final_similarity * 15,
                    'similarity': min(final_similarity, 1.0),
                    'reasons': [f"ชื่อตรงกัน {final_similarity*100:.0f}%"],
                    'nutrition': nutrition,
                    'search_type': 'enhanced_name'
                })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results

    def search_by_enhanced_ingredient(self, query: str, max_results: int) -> List[Dict]:
        """ค้นหาตามวัตถุดิบ - เวอร์ชันปรับปรุง"""
        
        # สกัดชื่อวัตถุดิบจากคำค้นหาแบบละเอียด
        ingredients_to_find = self._extract_enhanced_ingredients_from_query(query)
        if not ingredients_to_find:
            return []
        
        results = []
        
        for _, row in self.data.iterrows():
            ingredient_text = row['ingredient'].lower()
            
            # คำนวณคะแนนการตรงกันแบบละเอียด
            match_score, matched_ingredients = self._calculate_ingredient_match_score(
                ingredients_to_find, ingredient_text
            )
            
            if match_score > 0:
                nutrition = self._get_enhanced_nutrition_data(row)
                
                # คะแนนเพิ่มเติมจากความครอบคลุมของวัตถุดิบ
                coverage_bonus = self._calculate_ingredient_coverage_bonus(
                    matched_ingredients, ingredient_text
                )
                
                final_score = match_score + coverage_bonus
                reasons = [f"มีวัตถุดิบ: {', '.join(matched_ingredients)}"]
                
                results.append({
                    'name': row['name'],
                    'ingredient': row['ingredient'],
                    'method': row['method'],
                    'score': final_score * 8,
                    'similarity': min(final_score / len(ingredients_to_find), 1.0),
                    'reasons': reasons,
                    'nutrition': nutrition,
                    'search_type': 'enhanced_ingredient'
                })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results

    def search_by_enhanced_cooking_method(self, query: str, max_results: int) -> List[Dict]:
        """ค้นหาตามวิธีการปรุง - เวอร์ชันปรับปรุง"""
        
        detected_methods = []
        method_weights = {}
        
        for method, keywords in self.cooking_method_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    detected_methods.append(method)
                    method_weights[method] = len(keyword)  # น้ำหนักตามความเฉพาะเจาะจง
                    break
        
        if not detected_methods:
            return []
        
        results = []
        
        for _, row in self.data.iterrows():
            method_text = row['method'].lower()
            name_text = row['name'].lower()
            
            total_score = 0
            matched_methods = []
            
            for method in detected_methods:
                method_keywords = self.cooking_method_keywords[method]
                method_score = 0
                
                for keyword in method_keywords:
                    if keyword in method_text:
                        method_score += 3  # คะแนนจากวิธีทำ
                    elif keyword in name_text:
                        method_score += 2  # คะแนนจากชื่อเมนู
                
                if method_score > 0:
                    weight = method_weights.get(method, 1)
                    weighted_score = method_score * (weight / 5)
                    total_score += weighted_score
                    matched_methods.append(method)
            
            if total_score > 0:
                nutrition = self._get_enhanced_nutrition_data(row)
                match_ratio = len(matched_methods) / len(detected_methods)
                
                reasons = [f"วิธีการปรุง: {', '.join(matched_methods)}"]
                
                results.append({
                    'name': row['name'],
                    'ingredient': row['ingredient'],
                    'method': row['method'],
                    'score': total_score,
                    'similarity': match_ratio,
                    'reasons': reasons,
                    'nutrition': nutrition,
                    'search_type': 'enhanced_cooking_method'
                })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results

    def enhanced_fuzzy_search(self, query: str, max_results: int) -> List[Dict]:
        """ค้นหาแบบคลุมเครือ - เวอร์ชันปรับปรุง"""
        
        results = []
        query_words = [word for word in query.split() if len(word) > 2]
        
        for _, row in self.data.iterrows():
            # รวมข้อความทั้งหมดของเมนู
            combined_text = f"{row['name']} {row['ingredient']} {row['method']}".lower()
            
            # คำนวณคะแนนความเกี่ยวข้องแบบละเอียด
            relevance_score = self._calculate_enhanced_relevance_score(
                query_words, combined_text, row
            )
            
            if relevance_score > 1.5:  # เกณฑ์ที่ปรับปรุงแล้ว
                nutrition = self._get_enhanced_nutrition_data(row)
                
                # สร้างเหตุผลจากคำที่ตรงกัน
                matched_words = self._find_matched_words(query_words, combined_text)
                reasons = [f"พบคำ: {', '.join(matched_words[:3])}"] if matched_words else ["คล้ายคลึง"]
                
                results.append({
                    'name': row['name'],
                    'ingredient': row['ingredient'],
                    'method': row['method'],
                    'score': relevance_score,
                    'similarity': min(relevance_score / len(query_words), 1.0),
                    'reasons': reasons,
                    'nutrition': nutrition,
                    'search_type': 'enhanced_fuzzy'
                })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results

    def _get_enhanced_nutrition_data(self, row: pd.Series) -> Optional[Dict]:
        """ดึงข้อมูลโภชนาการของเมนู - เวอร์ชันปรับปรุง"""
        
        nutrition_columns = [
            'calories', 'protein', 'carbs', 'fat', 'fiber',
            'vitamin_a', 'vitamin_c', 'vitamin_b1', 'vitamin_b2',
            'calcium', 'iron', 'potassium', 'sodium'
        ]
        
        nutrition_data = {}
        has_nutrition = False
        
        # ตรวจสอบข้อมูลในคอลัมน์
        for col in nutrition_columns:
            if col in row.index and pd.notna(row[col]):
                nutrition_data[col] = float(row[col])
                if row[col] > 0:
                    has_nutrition = True
            else:
                nutrition_data[col] = 0.0
        
        if has_nutrition:
            return nutrition_data
        
        # คำนวณจากวัตถุดิบหากไม่มีข้อมูล
        try:
            calculated_nutrition = self.nutrition_api.calculate_recipe_nutrition(
                row['ingredient'], use_api=False, adjust_consumption=True
            )
            return calculated_nutrition.get('total_nutrition', nutrition_data)
        except:
            return nutrition_data

    def _evaluate_enhanced_nutrition_criterion(self, criterion: str, nutrition: Dict) -> Tuple[float, str]:
        """ประเมินเกณฑ์โภชนาการ - เวอร์ชันปรับปรุง"""
        
        thresholds = self.nutrition_thresholds
        
        # แยกประเภทและระดับ
        parts = criterion.split('_')
        nutrient = parts[0]
        level = '_'.join(parts[1:]) if len(parts) > 1 else 'medium'
        
        if nutrient not in thresholds:
            return 0.0, ""
        
        value = nutrition.get(nutrient, 0)
        nutrient_thresholds = thresholds[nutrient]
        
        # ประเมินตามระดับที่ต้องการ
        if level in ['very_low', 'low'] and nutrient in ['calories', 'fat', 'carbs', 'sodium']:
            # สารอาหารที่ต่ำกว่าดีกว่า
            if level == 'very_low' and value <= nutrient_thresholds['very_low']:
                return 20.0, f"{nutrient} ต่ำมาก ({value:.1f})"
            elif level == 'low' and value <= nutrient_thresholds['low']:
                return 15.0, f"{nutrient} ต่ำ ({value:.1f})"
            elif value <= nutrient_thresholds['medium']:
                return 8.0, f"{nutrient} ปานกลาง ({value:.1f})"
                
        elif level in ['high', 'very_high']:
            # สารอาหารที่สูงกว่าดีกว่า
            if level == 'very_high' and value >= nutrient_thresholds['very_high']:
                return 20.0, f"{nutrient} สูงมาก ({value:.1f})"
            elif level == 'high' and value >= nutrient_thresholds['high']:
                return 15.0, f"{nutrient} สูง ({value:.1f})"
            elif value >= nutrient_thresholds['medium']:
                return 8.0, f"{nutrient} ปานกลาง ({value:.1f})"
        
        return 0.0, ""

    def _calculate_comprehensive_nutrition_score(self, nutrition: Dict) -> float:
        """คำนวณคะแนนโภชนาการโดยรวม - เวอร์ชันปรับปรุง"""
        
        score = 0
        max_score = 20
        
        # คะแนนจากความสมดุลของแมโครนิวเทรียนต์
        calories = nutrition.get('calories', 0)
        protein = nutrition.get('protein', 0)
        carbs = nutrition.get('carbs', 0)
        fat = nutrition.get('fat', 0)
        
        # คะแนนแคลอรี่ที่เหมาะสม (200-400)
        if 250 <= calories <= 350:
            score += 4
        elif 200 <= calories < 250 or 350 < calories <= 400:
            score += 3
        elif 150 <= calories < 200 or 400 < calories <= 500:
            score += 2
        
        # คะแนนโปรตีนที่เพียงพอ (15-30g)
        if protein >= 25:
            score += 4
        elif protein >= 20:
            score += 3
        elif protein >= 15:
            score += 2
        elif protein >= 10:
            score += 1
        
        # คะแนนจากไมโครนิวเทรียนต์
        fiber = nutrition.get('fiber', 0)
        vitamin_c = nutrition.get('vitamin_c', 0)
        calcium = nutrition.get('calcium', 0)
        iron = nutrition.get('iron', 0)
        
        if fiber >= 5:
            score += 2
        elif fiber >= 3:
            score += 1
        
        if vitamin_c >= 20:
            score += 2
        elif vitamin_c >= 10:
            score += 1
        
        if calcium >= 150:
            score += 2
        elif calcium >= 100:
            score += 1
        
        if iron >= 3:
            score += 2
        elif iron >= 2:
            score += 1
        
        # หักคะแนนจากโซเดียมสูง
        sodium = nutrition.get('sodium', 0)
        if sodium > 1500:
            score -= 3
        elif sodium > 1200:
            score -= 2
        elif sodium > 1000:
            score -= 1
        
        return max(score / max_score * 10, 0)

    def _calculate_enhanced_name_similarity(self, query: str, recipe_name: str) -> float:
        """คำนวณความคล้ายคลึงของชื่อเมนู - เวอร์ชันปรับปรุง"""
        
        query_clean = self._clean_text_for_enhanced_comparison(query)
        name_clean = self._clean_text_for_enhanced_comparison(recipe_name)
        
        # ความคล้ายคลึงโดยรวม
        overall_similarity = SequenceMatcher(None, query_clean, name_clean).ratio()
        
        # ตรวจสอบการตรงกันของคำแบบละเอียด
        query_words = query_clean.split()
        name_words = name_clean.split()
        
        exact_matches = 0
        partial_matches = 0
        
        for q_word in query_words:
            if len(q_word) > 1:
                for n_word in name_words:
                    if q_word == n_word:
                        exact_matches += 2
                        break
                    elif q_word in n_word or n_word in q_word:
                        partial_matches += 1
                        break
        
        word_similarity = (exact_matches + partial_matches) / max(len(query_words) * 2, 1)
        
        # คะแนนจากตำแหน่งของคำ
        position_bonus = 0
        if query_words and name_words:
            if query_words[0] in name_words[0]:  # คำแรกตรงกัน
                position_bonus = 0.2
        
        # รวมคะแนน
        final_similarity = (overall_similarity * 0.3) + (word_similarity * 0.6) + position_bonus
        
        return min(final_similarity, 1.0)

    def _calculate_word_match_bonus(self, query: str, recipe_name: str) -> float:
        """คำนวณคะแนนเพิ่มเติมจากการตรงกันของคำ"""
        
        query_words = set(query.lower().split())
        name_words = set(recipe_name.lower().split())
        
        # คำที่สำคัญให้คะแนนเพิ่ม
        important_words = {'ผัด', 'ต้ม', 'ทอด', 'ย่าง', 'ยำ', 'แกง', 'กะเพรา', 'ส้มตำ'}
        
        bonus = 0
        for word in query_words:
            if word in name_words:
                if word in important_words:
                    bonus += 0.3
                else:
                    bonus += 0.1
        
        return min(bonus, 0.5)

    def _extract_enhanced_ingredients_from_query(self, query: str) -> List[str]:
        """สกัดชื่อวัตถุดิบจากคำค้นหา - เวอร์ชันปรับปรุง"""
        
        found_ingredients = []
        
        # ตรวจสอบวัตถุดิบแต่ละหมวดหมู่
        for category, subcategories in self.ingredient_keywords.items():
            for subcategory, ingredients in subcategories.items():
                for ingredient in ingredients:
                    if ingredient in query:
                        found_ingredients.append(ingredient)
        
        # เพิ่มวัตถุดิบพื้นฐานที่อาจไม่อยู่ในรายการ
        basic_ingredients = [
            'มะนาว', 'พริก', 'หอม', 'กระเทียม', 'ข่า', 'ตะไคร้',
            'ถั่ว', 'เห็ด', 'มะเขือ', 'ข้าว', 'เส้น', 'วุ้น'
        ]
        
        for ingredient in basic_ingredients:
            if ingredient in query and ingredient not in found_ingredients:
                found_ingredients.append(ingredient)
        
        return list(set(found_ingredients))  # ลบรายการซ้ำ

    def _calculate_ingredient_match_score(self, ingredients_to_find: List[str], 
                                        ingredient_text: str) -> Tuple[float, List[str]]:
        """คำนวณคะแนนการตรงกันของวัตถุดิบ"""
        
        matched_ingredients = []
        total_score = 0
        
        for ingredient in ingredients_to_find:
            if ingredient in ingredient_text:
                matched_ingredients.append(ingredient)
                # ให้คะแนนตามความสำคัญของวัตถุดิบ
                if len(ingredient) > 4:  # วัตถุดิบที่มีชื่อยาว (เฉพาะเจาะจงมากกว่า)
                    total_score += 2
                else:
                    total_score += 1
        
        return total_score, matched_ingredients

    def _calculate_ingredient_coverage_bonus(self, matched_ingredients: List[str], 
                                           ingredient_text: str) -> float:
        """คำนวณคะแนนเพิ่มเติมจากความครอบคลุมของวัตถุดิบ"""
        
        total_ingredients = len([line for line in ingredient_text.split('\n') if line.strip()])
        match_ratio = len(matched_ingredients) / max(total_ingredients, 1)
        
        return match_ratio * 2  # คะแนนเพิ่มเติมสูงสุด 2

    def _calculate_enhanced_relevance_score(self, query_words: List[str], 
                                          combined_text: str, row: pd.Series) -> float:
        """คำนวณคะแนนความเกี่ยวข้องแบบละเอียด"""
        
        score = 0
        matched_words = []
        
        for word in query_words:
            if word in combined_text:
                # คะแนนพื้นฐานจากการตรงกันโดยตรง
                score += 3
                matched_words.append(word)
                
                # คะแนนเพิ่มเติมตามตำแหน่งที่พบ
                if word in row['name'].lower():
                    score += 2  # พบในชื่อเมนู
                elif word in row['ingredient'].lower():
                    score += 1  # พบในวัตถุดิบ
            else:
                # ค้นหาคำที่คล้ายคลึง
                for text_word in combined_text.split():
                    similarity = SequenceMatcher(None, word, text_word).ratio()
                    if similarity > 0.75:
                        score += similarity * 2
                        matched_words.append(f"{word}~{text_word}")
                        break
        
        # คะแนนเพิ่มเติมจากความครอบคลุม
        coverage_score = len(matched_words) / len(query_words)
        score += coverage_score * 2
        
        return score

    def _find_matched_words(self, query_words: List[str], combined_text: str) -> List[str]:
        """ค้นหาคำที่ตรงกันระหว่างคำค้นหาและข้อความ"""
        
        matched_words = []
        
        for word in query_words:
            if word in combined_text:
                matched_words.append(word)
            else:
                # ค้นหาคำที่คล้ายคลึง
                for text_word in combined_text.split():
                    if len(text_word) > 2:
                        similarity = SequenceMatcher(None, word, text_word).ratio()
                        if similarity > 0.7:
                            matched_words.append(text_word)
                            break
        
        return list(set(matched_words))

    def _clean_text_for_enhanced_comparison(self, text: str) -> str:
        """ทำความสะอาดข้อความสำหรับการเปรียบเทียบ - เวอร์ชันปรับปรุง"""
        
        # ลบคำที่ไม่จำเป็นแบบละเอียด
        stop_words = [
            'ทำ', 'ปรุง', 'เตรียม', 'วิธี', 'สูตร', 'อะไร', 'ยังไง', 
            'อย่างไร', 'ครับ', 'ค่ะ', 'หา', 'ต้องการ', 'อยาก', 'จะ',
            'แนะนำ', 'บอก', 'มี', 'ได้', 'ไหม', 'หรือ', 'และ', 'กับ',
            'ของ', 'ใน', 'ที่', 'เป็น', 'คือ', 'นี้', 'นั้น'
        ]
        
        words = text.lower().split()
        filtered_words = []
        
        for word in words:
            if word not in stop_words and len(word) > 1:
                # ลบเครื่องหมายวรรคตอน
                clean_word = re.sub(r'[^\u0E00-\u0E7Fa-zA-Z0-9]', '', word)
                if clean_word:
                    filtered_words.append(clean_word)
        
        return ' '.join(filtered_words)

    def _rank_and_filter_results(self, results: List[Dict], max_results: int) -> List[Dict]:
        """จัดอันดับและกรองผลลัพธ์ - เวอร์ชันปรับปรุง"""
        
        if not results:
            return []
        
        # ลบรายการซ้ำ
        unique_results = {}
        for result in results:
            name = result['name']
            if name not in unique_results or result['score'] > unique_results[name]['score']:
                unique_results[name] = result
        
        filtered_results = list(unique_results.values())
        
        # จัดอันดับตามคะแนนรวมและคุณภาพโภชนาการ
        for result in filtered_results:
            nutrition_quality = self._assess_nutrition_quality(result['nutrition'])
            result['final_score'] = result['score'] + (nutrition_quality * 0.2)
        
        filtered_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return filtered_results[:max_results]

    def _assess_nutrition_quality(self, nutrition: Dict) -> float:
        """ประเมินคุณภาพโภชนาการ"""
        
        quality_score = 0
        
        # ประเมินความสมดุล
        calories = nutrition.get('calories', 0)
        protein = nutrition.get('protein', 0)
        fat = nutrition.get('fat', 0)
        carbs = nutrition.get('carbs', 0)
        fiber = nutrition.get('fiber', 0)
        sodium = nutrition.get('sodium', 0)
        
        # คะแนนจากสัดส่วนที่ดี
        if 200 <= calories <= 400:
            quality_score += 2
        if protein >= 15:
            quality_score += 2
        if 5 <= fat <= 20:
            quality_score += 1
        if fiber >= 3:
            quality_score += 2
        if sodium <= 1000:
            quality_score += 2
        
        # หักคะแนนจากโซเดียมสูงมาก
        if sodium > 1500:
            quality_score -= 2
        
        return max(quality_score, 0)

    def get_recipe_recommendations_by_nutrition(self, nutrition_targets: Dict, 
                                               max_results: int = 5) -> List[Dict]:
        """แนะนำเมนูตามเป้าหมายโภชนาการเฉพาะ - เวอร์ชันปรับปรุง"""
        
        recommendations = []
        
        for _, row in self.data.iterrows():
            nutrition = self._get_enhanced_nutrition_data(row)
            if not nutrition:
                continue
            
            # คำนวณคะแนนความเหมาะสมแบบละเอียด
            suitability_score = self._calculate_detailed_suitability_score(
                nutrition, nutrition_targets
            )
            
            if suitability_score >= 0.6:  # เกณฑ์ความเหมาะสม 60%
                reasons = self._generate_suitability_reasons(nutrition, nutrition_targets)
                
                recommendations.append({
                    'name': row['name'],
                    'ingredient': row['ingredient'],
                    'method': row['method'],
                    'score': suitability_score * 10,
                    'similarity': suitability_score,
                    'reasons': reasons,
                    'nutrition': nutrition,
                    'search_type': 'nutrition_target'
                })
        
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:max_results]

    def _calculate_detailed_suitability_score(self, nutrition: Dict, targets: Dict) -> float:
        """คำนวณคะแนนความเหมาะสมแบบละเอียด"""
        
        total_score = 0
        max_score = 0
        
        for target_nutrient, target_range in targets.items():
            current_value = nutrition.get(target_nutrient, 0)
            min_val = target_range.get('min', 0)
            max_val = target_range.get('max', float('inf'))
            optimal_val = target_range.get('optimal', (min_val + max_val) / 2)
            
            max_score += 1
            
            if min_val <= current_value <= max_val:
                # คำนวณคะแนนตามความใกล้เคียงกับค่าที่เหมาะสมที่สุด
                if optimal_val != 0:
                    distance_ratio = abs(current_value - optimal_val) / optimal_val
                    score = max(1 - distance_ratio, 0.5)  # คะแนนขั้นต่ำ 0.5
                else:
                    score = 1
                total_score += score
            elif current_value < min_val:
                # ให้คะแนนบางส่วนหากต่ำกว่าเป้าหมาย
                if min_val > 0:
                    ratio = current_value / min_val
                    total_score += max(ratio * 0.5, 0)
            # ไม่ให้คะแนนหากสูงกว่าเป้าหมาย
        
        return total_score / max_score if max_score > 0 else 0

    def _generate_suitability_reasons(self, nutrition: Dict, targets: Dict) -> List[str]:
        """สร้างเหตุผลความเหมาะสม"""
        
        reasons = []
        
        for target_nutrient, target_range in targets.items():
            current_value = nutrition.get(target_nutrient, 0)
            min_val = target_range.get('min', 0)
            max_val = target_range.get('max', float('inf'))
            
            if min_val <= current_value <= max_val:
                if target_nutrient == 'calories':
                    reasons.append(f"แคลอรี่เหมาะสม {current_value:.0f} kcal")
                elif target_nutrient == 'protein':
                    reasons.append(f"โปรตีนดี {current_value:.1f} g")
                elif target_nutrient == 'sodium':
                    reasons.append(f"โซเดียมเหมาะสม {current_value:.0f} mg")
                else:
                    reasons.append(f"{target_nutrient}: {current_value:.1f}")
        
        return reasons[:3]  # จำกัดเหตุผลไม่เกิน 3 ข้อ
