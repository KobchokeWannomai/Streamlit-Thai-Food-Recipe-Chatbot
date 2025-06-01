import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Tuple, Any
from difflib import SequenceMatcher
import json

class RecipeSearchEngine:
    """ระบบค้นหาสูตรอาหารไทยขั้นสูงพร้อมการวิเคราะห์โภชนาการ"""
    
    def __init__(self, data: pd.DataFrame, nutrition_api):
        self.data = data
        self.nutrition_api = nutrition_api
        
        # คำสำคัญสำหรับการค้นหาตามโภชนาการ
        self.nutrition_keywords = {
            # แคลอรี่
            'calories_low': [
                'แคลอรี่ต่ำ', 'แคลต่ำ', 'ลดน้ำหนัก', 'เบา', 'ไม่อ้วน', 
                'ไดเอท', 'diet', 'ลดความอ้วน', 'คุมน้ำหนัก'
            ],
            'calories_high': [
                'แคลอรี่สูง', 'แคลสูง', 'เพิ่มน้ำหนัก', 'พลังงานสูง', 
                'เติมแรง', 'นักกีฬา', 'กำลัง'
            ],
            
            # โปรตีน
            'protein_high': [
                'โปรตีนสูง', 'โปรตีนมาก', 'เนื้อเยื่อ', 'กล้ามเนื้อ', 
                'นักกีฬา', 'ออกกำลังกาย', 'ฟิตเนส', 'เสริมสร้าง'
            ],
            'protein_low': [
                'โปรตีนต่ำ', 'โปรตีนน้อย', 'ไต', 'โรคไต'
            ],
            
            # ไขมัน
            'fat_low': [
                'ไขมันต่ำ', 'ไขมันน้อย', 'ลดไขมัน', 'ไม่มันเยอะ', 
                'สุขภาพดี', 'หัวใจ', 'โรคหัวใจ'
            ],
            'fat_high': [
                'ไขมันสูง', 'ไขมันมาก', 'มันเยอะ'
            ],
            
            # คาร์โบไฮเดรต
            'carbs_low': [
                'คาร์โบต่ำ', 'แป้งน้อย', 'น้ำตาลต่ำ', 'เบาหวาน', 
                'คีโต', 'keto', 'low carb', 'ลดแป้ง'
            ],
            'carbs_high': [
                'คาร์โบสูง', 'แป้งมาก', 'พลังงาน', 'นักกีฬา', 'ข้าว'
            ],
            
            # ใยอาหาร
            'fiber_high': [
                'ใยอาหารสูง', 'ใยอาหารมาก', 'ขับถ่าย', 'ท้องผูก', 
                'ย่อย', 'ระบบย่อย', 'ช่วยย่อย', 'ผัก'
            ],
            
            # วิตามิน
            'vitamin_a_high': [
                'วิตามินเอสูง', 'วิตามินเอ', 'สายตา', 'ผิวพรรณ', 
                'ตา', 'บำรุงตา'
            ],
            'vitamin_c_high': [
                'วิตามินซีสูง', 'วิตามินซี', 'ภูมิคุ้มกัน', 'ต้านหวัด', 
                'เสริมภูมิ', 'ต้านอนุมูลอิสระ'
            ],
            'vitamin_b_high': [
                'วิตามินบี', 'ระบบประสาท', 'เมแทบอลิซึม', 'พลังงาน'
            ],
            
            # แร่ธาตุ
            'calcium_high': [
                'แคลเซียมสูง', 'แคลเซียม', 'กระดูก', 'ฟัน', 
                'ผู้สูงอายุ', 'เด็ก', 'บำรุงกระดูก'
            ],
            'iron_high': [
                'เหล็กสูง', 'ธาตุเหล็ก', 'โลหิตจาง', 'เลือดจาง', 
                'ผู้หญิง', 'ประจำเดือน'
            ],
            'potassium_high': [
                'โปแตสเซียมสูง', 'โปแตสเซียม', 'ความดันโลหิต', 
                'หัวใจ', 'กล้ามเนื้อหัวใจ'
            ],
            'sodium_low': [
                'โซเดียมต่ำ', 'เกลือน้อย', 'ความดันสูง', 'ไต', 
                'หัวใจ', 'จืด', 'ไม่เค็ม'
            ],
            
            # กลุ่มผู้ป่วยเฉพาะ
            'diabetes': [
                'เบาหวาน', 'ผู้ป่วยเบาหวาน', 'น้ำตาลต่ำ', 
                'ควบคุมน้ำตาล', 'เบาหวาน', 'ดัชนีน้ำตาล'
            ],
            'hypertension': [
                'ความดันสูง', 'ผู้ป่วยความดัน', 'โซเดียมต่ำ', 
                'ความดัน', 'ไฮเปอร์เทนชั่น'
            ],
            'heart_disease': [
                'โรคหัวใจ', 'หัวใจ', 'โคเลสเตอรอล', 'หลอดเลือด'
            ],
            'kidney_disease': [
                'โรคไต', 'ไต', 'ล้างไต', 'ไตเสื่อม'
            ],
            'elderly': [
                'ผู้สูงอายุ', 'คนแก่', 'นุ่ม', 'ย่อยง่าย', 
                'ผู้ใหญ่', 'วัยชรา'
            ],
            'children': [
                'เด็ก', 'เด็กเล็ก', 'แคลเซียม', 'เจริญเติบโต', 
                'ลูก', 'วัยรุ่น'
            ],
            'athletes': [
                'นักกีฬา', 'ออกกำลังกาย', 'โปรตีนสูง', 'ฟิตเนส', 
                'กล้ามเนื้อ', 'เล่นกีฬา'
            ],
            'pregnant': [
                'ตั้งครรภ์', 'คนท้อง', 'โฟเลต', 'เหล็ก', 
                'แม่ท้อง', 'มีครรภ์'
            ],
            
            # ประเภทอาหาร
            'vegetarian': [
                'มังสวิรัติ', 'เจ', 'ไม่กินเนื้อ', 'ผัก', 'พืช', 'เจ'
            ],
            'healthy': [
                'สุขภาพ', 'สุขภาพดี', 'คลีน', 'clean eating', 
                'healthy', 'เพื่อสุขภาพ'
            ],
            'weight_loss': [
                'ลดน้ำหนัก', 'ลดความอ้วน', 'ไดเอท', 'เบา', 
                'คุมน้ำหนัก'
            ],
            'detox': [
                'ดีท็อกซ์', 'ล้างพิษ', 'ล้างลำไส้', 'detox', 
                'ขับสารพิษ'
            ]
        }
        
        # เกณฑ์การจัดกลุ่มโภชนาการ (ปรับปรุงให้แม่นยำขึ้น)
        self.nutrition_thresholds = {
            'calories': {
                'very_low': 150, 'low': 250, 'medium': 400, 
                'high': 600, 'very_high': 800
            },
            'protein': {
                'low': 8, 'medium': 15, 'high': 25, 'very_high': 35
            },
            'fat': {
                'low': 8, 'medium': 15, 'high': 25, 'very_high': 35
            },
            'carbs': {
                'low': 10, 'medium': 25, 'high': 45, 'very_high': 70
            },
            'fiber': {
                'low': 1, 'medium': 3, 'high': 6, 'very_high': 10
            },
            'vitamin_a': {
                'low': 50, 'medium': 200, 'high': 500, 'very_high': 1000
            },
            'vitamin_c': {
                'low': 5, 'medium': 15, 'high': 30, 'very_high': 60
            },
            'calcium': {
                'low': 50, 'medium': 100, 'high': 200, 'very_high': 400
            },
            'iron': {
                'low': 1, 'medium': 2.5, 'high': 5, 'very_high': 8
            },
            'potassium': {
                'low': 200, 'medium': 350, 'high': 500, 'very_high': 700
            },
            'sodium': {
                'very_low': 300, 'low': 600, 'medium': 1000, 
                'high': 1500, 'very_high': 2000
            }
        }
        
        # คำสำคัญประเภทการปรุง
        self.cooking_method_keywords = {
            'ทอด': ['ทอด', 'เจียว', 'กรอบ'],
            'ต้ม': ['ต้ม', 'แกง', 'น้ำซุป'],
            'ผัด': ['ผัด', 'คั่ว'],
            'ย่าง': ['ย่าง', 'ปิ้ง', 'เผา'],
            'ยำ': ['ยำ', 'ลาบ', 'ตำ'],
            'นึ่ง': ['นึ่ง', 'อบ'],
            'ลวก': ['ลวก', 'ต้ม']
        }

    def search_recipes(self, query: str, max_results: int = 10, 
                      search_mode: str = "comprehensive") -> List[Dict]:
        """ค้นหาสูตรอาหารแบบครอบคลุม"""
        
        query_lower = query.lower().strip()
        
        # ลองค้นหาตามโภชนาการก่อน
        nutrition_results = self.search_by_nutrition(query_lower, max_results)
        if nutrition_results:
            return nutrition_results
        
        # ค้นหาตามชื่อเมนู
        name_results = self.search_by_name(query_lower, max_results)
        if name_results:
            return name_results
        
        # ค้นหาตามวัตถุดิบ
        ingredient_results = self.search_by_ingredient(query_lower, max_results)
        if ingredient_results:
            return ingredient_results
        
        # ค้นหาตามวิธีการปรุง
        method_results = self.search_by_cooking_method(query_lower, max_results)
        if method_results:
            return method_results
        
        # ค้นหาแบบคลุมเครือ
        return self.fuzzy_search(query_lower, max_results)

    def search_by_nutrition(self, query: str, max_results: int) -> List[Dict]:
        """ค้นหาตามเกณฑ์โภชนาการ"""
        
        # ตรวจสอบคำสำคัญโภชนาการ
        detected_criteria = []
        for category, keywords in self.nutrition_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    detected_criteria.append(category)
                    break
        
        if not detected_criteria:
            return []
        
        results = []
        
        for _, row in self.data.iterrows():
            # คำนวณโภชนาการหรือใช้ข้อมูลที่มีอยู่
            nutrition = self._get_nutrition_data(row)
            if not nutrition:
                continue
            
            score = 0
            reasons = []
            
            # ประเมินตามเกณฑ์ที่ตรวจพบ
            for criterion in detected_criteria:
                criterion_score, reason = self._evaluate_nutrition_criterion(
                    criterion, nutrition
                )
                score += criterion_score
                if reason:
                    reasons.append(reason)
            
            if score > 0:
                results.append({
                    'name': row['name'],
                    'ingredient': row['ingredient'],
                    'method': row['method'],
                    'score': score,
                    'similarity': score / len(detected_criteria),
                    'reasons': reasons,
                    'nutrition': nutrition,
                    'search_type': 'nutrition'
                })
        
        # เรียงลำดับตามคะแนน
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:max_results]

    def search_by_name(self, query: str, max_results: int) -> List[Dict]:
        """ค้นหาตามชื่อเมนู"""
        
        results = []
        
        for _, row in self.data.iterrows():
            similarity = self._calculate_name_similarity(query, row['name'])
            
            if similarity > 0.3:  # เกณฑ์ความคล้ายคลึง
                nutrition = self._get_nutrition_data(row)
                
                results.append({
                    'name': row['name'],
                    'ingredient': row['ingredient'],
                    'method': row['method'],
                    'score': similarity * 10,
                    'similarity': similarity,
                    'reasons': [f"ชื่อตรงกัน {similarity*100:.0f}%"],
                    'nutrition': nutrition,
                    'search_type': 'name'
                })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:max_results]

    def search_by_ingredient(self, query: str, max_results: int) -> List[Dict]:
        """ค้นหาตามวัตถุดิบ"""
        
        # สกัดชื่อวัตถุดิบจากคำค้นหา
        ingredients_to_find = self._extract_ingredients_from_query(query)
        if not ingredients_to_find:
            return []
        
        results = []
        
        for _, row in self.data.iterrows():
            ingredient_text = row['ingredient'].lower()
            
            # นับจำนวนวัตถุดิบที่ตรงกัน
            matches = 0
            matched_ingredients = []
            
            for ingredient in ingredients_to_find:
                if ingredient in ingredient_text:
                    matches += 1
                    matched_ingredients.append(ingredient)
            
            if matches > 0:
                match_ratio = matches / len(ingredients_to_find)
                nutrition = self._get_nutrition_data(row)
                
                reasons = [f"มีวัตถุดิบ: {', '.join(matched_ingredients)}"]
                
                results.append({
                    'name': row['name'],
                    'ingredient': row['ingredient'],
                    'method': row['method'],
                    'score': match_ratio * 10,
                    'similarity': match_ratio,
                    'reasons': reasons,
                    'nutrition': nutrition,
                    'search_type': 'ingredient'
                })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:max_results]

    def search_by_cooking_method(self, query: str, max_results: int) -> List[Dict]:
        """ค้นหาตามวิธีการปรุง"""
        
        detected_methods = []
        for method, keywords in self.cooking_method_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    detected_methods.append(method)
                    break
        
        if not detected_methods:
            return []
        
        results = []
        
        for _, row in self.data.iterrows():
            method_text = row['method'].lower()
            name_text = row['name'].lower()
            
            matches = 0
            matched_methods = []
            
            for method in detected_methods:
                method_keywords = self.cooking_method_keywords[method]
                for keyword in method_keywords:
                    if keyword in method_text or keyword in name_text:
                        matches += 1
                        matched_methods.append(method)
                        break
            
            if matches > 0:
                match_ratio = matches / len(detected_methods)
                nutrition = self._get_nutrition_data(row)
                
                reasons = [f"วิธีการปรุง: {', '.join(matched_methods)}"]
                
                results.append({
                    'name': row['name'],
                    'ingredient': row['ingredient'],
                    'method': row['method'],
                    'score': match_ratio * 8,
                    'similarity': match_ratio,
                    'reasons': reasons,
                    'nutrition': nutrition,
                    'search_type': 'cooking_method'
                })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:max_results]

    def fuzzy_search(self, query: str, max_results: int) -> List[Dict]:
        """ค้นหาแบบคลุมเครือ"""
        
        results = []
        query_words = query.split()
        
        for _, row in self.data.iterrows():
            # รวมข้อความทั้งหมดของเมนู
            combined_text = f"{row['name']} {row['ingredient']} {row['method']}".lower()
            
            # คำนวณคะแนนความเกี่ยวข้อง
            relevance_score = 0
            matched_words = []
            
            for word in query_words:
                if len(word) > 2:  # ข้ามคำสั้นๆ
                    if word in combined_text:
                        relevance_score += 2
                        matched_words.append(word)
                    else:
                        # ค้นหาคำที่คล้ายคลึง
                        for text_word in combined_text.split():
                            similarity = SequenceMatcher(None, word, text_word).ratio()
                            if similarity > 0.7:
                                relevance_score += similarity
                                matched_words.append(f"{word}~{text_word}")
                                break
            
            if relevance_score > 1:
                nutrition = self._get_nutrition_data(row)
                
                reasons = [f"พบคำ: {', '.join(matched_words[:3])}"]
                
                results.append({
                    'name': row['name'],
                    'ingredient': row['ingredient'],
                    'method': row['method'],
                    'score': relevance_score,
                    'similarity': min(relevance_score / len(query_words), 1.0),
                    'reasons': reasons,
                    'nutrition': nutrition,
                    'search_type': 'fuzzy'
                })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:max_results]

    def _get_nutrition_data(self, row: pd.Series) -> Optional[Dict]:
        """ดึงข้อมูลโภชนาการของเมนู"""
        
        # ตรวจสอบว่ามีข้อมูลโภชนาการในคอลัมน์หรือไม่
        nutrition_columns = [
            'calories', 'protein', 'carbs', 'fat', 'fiber',
            'vitamin_a', 'vitamin_c', 'vitamin_b1', 'vitamin_b2',
            'calcium', 'iron', 'potassium', 'sodium'
        ]
        
        nutrition_data = {}
        has_nutrition = False
        
        for col in nutrition_columns:
            if col in row.index and pd.notna(row[col]):
                nutrition_data[col] = float(row[col])
                if row[col] > 0:
                    has_nutrition = True
            else:
                nutrition_data[col] = 0.0
        
        if has_nutrition:
            return nutrition_data
        
        # หากไม่มีข้อมูลในคอลัมน์ ให้คำนวณจากวัตถุดิบ
        try:
            calculated_nutrition = self.nutrition_api.calculate_recipe_nutrition(
                row['ingredient'], use_api=False, adjust_consumption=True
            )
            return calculated_nutrition.get('total_nutrition', nutrition_data)
        except:
            return nutrition_data

    def _calculate_name_similarity(self, query: str, recipe_name: str) -> float:
        """คำนวณความคล้ายคลึงของชื่อเมนู"""
        
        query_clean = self._clean_text_for_comparison(query)
        name_clean = self._clean_text_for_comparison(recipe_name)
        
        # ความคล้ายคลึงโดยรวม
        overall_similarity = SequenceMatcher(None, query_clean, name_clean).ratio()
        
        # ตรวจสอบการตรงกันของคำ
        query_words = query_clean.split()
        name_words = name_clean.split()
        
        word_matches = 0
        for q_word in query_words:
            if len(q_word) > 1:
                for n_word in name_words:
                    if q_word in n_word or n_word in q_word:
                        word_matches += 1
                        break
        
        word_similarity = word_matches / max(len(query_words), 1)
        
        # รวมคะแนน
        final_similarity = (overall_similarity * 0.4) + (word_similarity * 0.6)
        
        return final_similarity

    def _clean_text_for_comparison(self, text: str) -> str:
        """ทำความสะอาดข้อความสำหรับการเปรียบเทียบ"""
        
        # ลบคำที่ไม่จำเป็น
        stop_words = [
            'ทำ', 'ปรุง', 'เตรียม', 'วิธี', 'สูตร', 'อะไร', 'ยังไง', 
            'อย่างไร', 'ครับ', 'ค่ะ', 'หา', 'ต้องการ', 'อยาก', 'จะ'
        ]
        
        words = text.lower().split()
        filtered_words = [word for word in words if word not in stop_words and len(word) > 1]
        
        return ' '.join(filtered_words)

    def _extract_ingredients_from_query(self, query: str) -> List[str]:
        """สกัดชื่อวัตถุดิบจากคำค้นหา"""
        
        common_ingredients = [
            'หมู', 'ไก่', 'เนื้อ', 'กุ้ง', 'ปลา', 'ไข่', 'เต้าหู้',
            'ผัก', 'ผักบุ้ง', 'คะน้า', 'กะเพรา', 'โหระพา',
            'มะเขือเทศ', 'หอม', 'กระเทียม', 'พริก',
            'มะนาว', 'มะขาม', 'กะทิ', 'ข่า', 'ตะไคร้',
            'วุ้นเส้น', 'เส้นจันท์', 'ข้าว'
        ]
        
        found_ingredients = []
        
        for ingredient in common_ingredients:
            if ingredient in query:
                found_ingredients.append(ingredient)
        
        return found_ingredients

    def _evaluate_nutrition_criterion(self, criterion: str, nutrition: Dict) -> Tuple[float, str]:
        """ประเมินคะแนนตามเกณฑ์โภชนาการ"""
        
        thresholds = self.nutrition_thresholds
        
        # แคลอรี่ต่ำ
        if criterion == 'calories_low':
            calories = nutrition.get('calories', 0)
            if calories <= thresholds['calories']['very_low']:
                return 15.0, f"แคลอรี่ต่ำมาก ({calories:.0f} kcal)"
            elif calories <= thresholds['calories']['low']:
                return 12.0, f"แคลอรี่ต่ำ ({calories:.0f} kcal)"
            elif calories <= thresholds['calories']['medium']:
                return 8.0, f"แคลอรี่ปานกลาง ({calories:.0f} kcal)"
            return 0.0, ""
        
        # แคลอรี่สูง
        elif criterion == 'calories_high':
            calories = nutrition.get('calories', 0)
            if calories >= thresholds['calories']['very_high']:
                return 15.0, f"แคลอรี่สูงมาก ({calories:.0f} kcal)"
            elif calories >= thresholds['calories']['high']:
                return 12.0, f"แคลอรี่สูง ({calories:.0f} kcal)"
            return 0.0, ""
        
        # โปรตีนสูง
        elif criterion == 'protein_high':
            protein = nutrition.get('protein', 0)
            if protein >= thresholds['protein']['very_high']:
                return 15.0, f"โปรตีนสูงมาก ({protein:.1f} g)"
            elif protein >= thresholds['protein']['high']:
                return 12.0, f"โปรตีนสูง ({protein:.1f} g)"
            elif protein >= thresholds['protein']['medium']:
                return 8.0, f"โปรตีนปานกลาง ({protein:.1f} g)"
            return 0.0, ""
        
        # โปรตีนต่ำ
        elif criterion == 'protein_low':
            protein = nutrition.get('protein', 0)
            if protein <= thresholds['protein']['low']:
                return 12.0, f"โปรตีนต่ำ ({protein:.1f} g)"
            return 0.0, ""
        
        # ไขมันต่ำ
        elif criterion == 'fat_low':
            fat = nutrition.get('fat', 0)
            if fat <= thresholds['fat']['low']:
                return 12.0, f"ไขมันต่ำ ({fat:.1f} g)"
            elif fat <= thresholds['fat']['medium']:
                return 8.0, f"ไขมันปานกลาง ({fat:.1f} g)"
            return 0.0, ""
        
        # ไขมันสูง
        elif criterion == 'fat_high':
            fat = nutrition.get('fat', 0)
            if fat >= thresholds['fat']['very_high']:
                return 12.0, f"ไขมันสูง ({fat:.1f} g)"
            return 0.0, ""
        
        # คาร์โบไฮเดรตต่ำ
        elif criterion == 'carbs_low':
            carbs = nutrition.get('carbs', 0)
            if carbs <= thresholds['carbs']['low']:
                return 12.0, f"คาร์โบไฮเดรตต่ำ ({carbs:.1f} g)"
            elif carbs <= thresholds['carbs']['medium']:
                return 8.0, f"คาร์โบไฮเดรตปานกลาง ({carbs:.1f} g)"
            return 0.0, ""
        
        # คาร์โบไฮเดรตสูง
        elif criterion == 'carbs_high':
            carbs = nutrition.get('carbs', 0)
            if carbs >= thresholds['carbs']['very_high']:
                return 12.0, f"คาร์โบไฮเดรตสูง ({carbs:.1f} g)"
            return 0.0, ""
        
        # ใยอาหารสูง
        elif criterion == 'fiber_high':
            fiber = nutrition.get('fiber', 0)
            if fiber >= thresholds['fiber']['very_high']:
                return 15.0, f"ใยอาหารสูงมาก ({fiber:.1f} g)"
            elif fiber >= thresholds['fiber']['high']:
                return 12.0, f"ใยอาหารสูง ({fiber:.1f} g)"
            elif fiber >= thresholds['fiber']['medium']:
                return 8.0, f"ใยอาหารปานกลาง ({fiber:.1f} g)"
            return 0.0, ""
        
        # วิตามินและแร่ธาตุ
        elif criterion == 'vitamin_a_high':
            vitamin_a = nutrition.get('vitamin_a', 0)
            if vitamin_a >= thresholds['vitamin_a']['very_high']:
                return 15.0, f"วิตามินเอสูงมาก ({vitamin_a:.0f} IU)"
            elif vitamin_a >= thresholds['vitamin_a']['high']:
                return 12.0, f"วิตามินเอสูง ({vitamin_a:.0f} IU)"
            return 0.0, ""
        
        elif criterion == 'vitamin_c_high':
            vitamin_c = nutrition.get('vitamin_c', 0)
            if vitamin_c >= thresholds['vitamin_c']['very_high']:
                return 15.0, f"วิตามินซีสูงมาก ({vitamin_c:.1f} mg)"
            elif vitamin_c >= thresholds['vitamin_c']['high']:
                return 12.0, f"วิตามินซีสูง ({vitamin_c:.1f} mg)"
            return 0.0, ""
        
        elif criterion == 'calcium_high':
            calcium = nutrition.get('calcium', 0)
            if calcium >= thresholds['calcium']['very_high']:
                return 15.0, f"แคลเซียมสูงมาก ({calcium:.0f} mg)"
            elif calcium >= thresholds['calcium']['high']:
                return 12.0, f"แคลเซียมสูง ({calcium:.0f} mg)"
            return 0.0, ""
        
        elif criterion == 'iron_high':
            iron = nutrition.get('iron', 0)
            if iron >= thresholds['iron']['very_high']:
                return 15.0, f"เหล็กสูงมาก ({iron:.1f} mg)"
            elif iron >= thresholds['iron']['high']:
                return 12.0, f"เหล็กสูง ({iron:.1f} mg)"
            return 0.0, ""
        
        elif criterion == 'potassium_high':
            potassium = nutrition.get('potassium', 0)
            if potassium >= thresholds['potassium']['very_high']:
                return 15.0, f"โปแตสเซียมสูงมาก ({potassium:.0f} mg)"
            elif potassium >= thresholds['potassium']['high']:
                return 12.0, f"โปแตสเซียมสูง ({potassium:.0f} mg)"
            return 0.0, ""
        
        elif criterion == 'sodium_low':
            sodium = nutrition.get('sodium', 0)
            if sodium <= thresholds['sodium']['very_low']:
                return 15.0, f"โซเดียมต่ำมาก ({sodium:.0f} mg)"
            elif sodium <= thresholds['sodium']['low']:
                return 12.0, f"โซเดียมต่ำ ({sodium:.0f} mg)"
            return 0.0, ""
        
        # เกณฑ์สำหรับผู้ป่วยเฉพาะ
        elif criterion == 'diabetes':
            score = 0
            reasons_list = []
            
            carbs = nutrition.get('carbs', 0)
            fiber = nutrition.get('fiber', 0)
            sodium = nutrition.get('sodium', 0)
            fat = nutrition.get('fat', 0)
            
            if carbs <= thresholds['carbs']['low']:
                score += 10
                reasons_list.append(f"คาร์โบไฮเดรตต่ำ ({carbs:.1f} g)")
            if fiber >= thresholds['fiber']['medium']:
                score += 8
                reasons_list.append(f"ใยอาหารดี ({fiber:.1f} g)")
            if sodium <= thresholds['sodium']['medium']:
                score += 5
                reasons_list.append(f"โซเดียมเหมาะสม ({sodium:.0f} mg)")
            if fat <= thresholds['fat']['medium']:
                score += 5
                reasons_list.append(f"ไขมันพอดี ({fat:.1f} g)")
            
            reason = "เหมาะสำหรับผู้ป่วยเบาหวาน: " + ", ".join(reasons_list) if reasons_list else ""
            return score, reason
        
        elif criterion == 'hypertension':
            score = 0
            reasons_list = []
            
            sodium = nutrition.get('sodium', 0)
            potassium = nutrition.get('potassium', 0)
            fiber = nutrition.get('fiber', 0)
            
            if sodium <= thresholds['sodium']['low']:
                score += 12
                reasons_list.append(f"โซเดียมต่ำ ({sodium:.0f} mg)")
            elif sodium <= thresholds['sodium']['medium']:
                score += 6
                reasons_list.append(f"โซเดียมปานกลาง ({sodium:.0f} mg)")
            
            if potassium >= thresholds['potassium']['medium']:
                score += 8
                reasons_list.append(f"โปแตสเซียมดี ({potassium:.0f} mg)")
            if fiber >= thresholds['fiber']['medium']:
                score += 5
                reasons_list.append(f"ใยอาหารดี ({fiber:.1f} g)")
            
            reason = "เหมาะสำหรับผู้ป่วยความดันสูง: " + ", ".join(reasons_list) if reasons_list else ""
            return score, reason
        
        elif criterion == 'heart_disease':
            score = 0
            reasons_list = []
            
            fat = nutrition.get('fat', 0)
            sodium = nutrition.get('sodium', 0)
            fiber = nutrition.get('fiber', 0)
            
            if fat <= thresholds['fat']['low']:
                score += 10
                reasons_list.append(f"ไขมันต่ำ ({fat:.1f} g)")
            if sodium <= thresholds['sodium']['low']:
                score += 10
                reasons_list.append(f"โซเดียมต่ำ ({sodium:.0f} mg)")
            if fiber >= thresholds['fiber']['medium']:
                score += 8
                reasons_list.append(f"ใยอาหารสูง ({fiber:.1f} g)")
            
            reason = "เหมาะสำหรับโรคหัวใจ: " + ", ".join(reasons_list) if reasons_list else ""
            return score, reason
        
        elif criterion == 'kidney_disease':
            score = 0
            reasons_list = []
            
            protein = nutrition.get('protein', 0)
            sodium = nutrition.get('sodium', 0)
            potassium = nutrition.get('potassium', 0)
            
            if protein <= thresholds['protein']['medium']:
                score += 10
                reasons_list.append(f"โปรตีนปานกลาง ({protein:.1f} g)")
            if sodium <= thresholds['sodium']['low']:
                score += 10
                reasons_list.append(f"โซเดียมต่ำ ({sodium:.0f} mg)")
            if potassium <= thresholds['potassium']['medium']:
                score += 8
                reasons_list.append(f"โปแตสเซียมไม่สูง ({potassium:.0f} mg)")
            
            reason = "เหมาะสำหรับโรคไต: " + ", ".join(reasons_list) if reasons_list else ""
            return score, reason
        
        elif criterion == 'weight_loss':
            score = 0
            reasons_list = []
            
            calories = nutrition.get('calories', 0)
            protein = nutrition.get('protein', 0)
            fiber = nutrition.get('fiber', 0)
            fat = nutrition.get('fat', 0)
            
            if calories <= thresholds['calories']['low']:
                score += 12
                reasons_list.append(f"แคลอรี่ต่ำ ({calories:.0f} kcal)")
            if protein >= thresholds['protein']['medium']:
                score += 8
                reasons_list.append(f"โปรตีนดี ({protein:.1f} g)")
            if fiber >= thresholds['fiber']['medium']:
                score += 6
                reasons_list.append(f"ใยอาหารช่วยอิ่ม ({fiber:.1f} g)")
            if fat <= thresholds['fat']['medium']:
                score += 5
                reasons_list.append(f"ไขมันพอดี ({fat:.1f} g)")
            
            reason = "เหมาะสำหรับลดน้ำหนัก: " + ", ".join(reasons_list) if reasons_list else ""
            return score, reason
        
        return 0.0, ""

    def get_recipe_recommendations_by_nutrition(self, nutrition_targets: Dict, 
                                               max_results: int = 5) -> List[Dict]:
        """แนะนำเมนูตามเป้าหมายโภชนาการเฉพาะ"""
        
        recommendations = []
        
        for _, row in self.data.iterrows():
            nutrition = self._get_nutrition_data(row)
            if not nutrition:
                continue
            
            # คำนวณคะแนนความเหมาะสม
            score = 0
            reasons = []
            
            for target_nutrient, target_range in nutrition_targets.items():
                current_value = nutrition.get(target_nutrient, 0)
                min_val = target_range.get('min', 0)
                max_val = target_range.get('max', float('inf'))
                
                if min_val <= current_value <= max_val:
                    score += 1
                    reasons.append(f"{target_nutrient}: {current_value:.1f} (เหมาะสม)")
                elif current_value < min_val:
                    reasons.append(f"{target_nutrient}: {current_value:.1f} (ต่ำเกินไป)")
                else:
                    reasons.append(f"{target_nutrient}: {current_value:.1f} (สูงเกินไป)")
            
            if score >= len(nutrition_targets) * 0.6:  # เกณฑ์ 60% ของเป้าหมาย
                recommendations.append({
                    'name': row['name'],
                    'ingredient': row['ingredient'],
                    'method': row['method'],
                    'score': score,
                    'similarity': score / len(nutrition_targets),
                    'reasons': reasons,
                    'nutrition': nutrition,
                    'search_type': 'nutrition_target'
                })
        
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:max_results]
