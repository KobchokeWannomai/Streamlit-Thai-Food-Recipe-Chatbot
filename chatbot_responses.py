import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Tuple, Any
from difflib import SequenceMatcher
import random

class ThaiLoodChatbot:
    """ระบบ Chatbot อัจฉริยะสำหรับอาหารไทยและโภชนาการ"""
    
    def __init__(self, data: pd.DataFrame, nutrition_api, search_engine):
        self.data = data
        self.nutrition_api = nutrition_api
        self.search_engine = search_engine
        
        # รูปแบบคำถามและการตอบกลับ
        self.question_patterns = {
            # คำถามเกี่ยวกับสูตรอาหาร
            'recipe_how': {
                'patterns': [
                    r'(ทำ|ปรุง|เตรียม).*(ยังไง|อย่างไร|วิธี)',
                    r'วิธี(ทำ|ปรุง|เตรียม)',
                    r'สูตร.*อะไร',
                    r'.*ทำยังไง'
                ],
                'type': 'recipe_search'
            },
            
            # คำถามเกี่ยวกับโภชนาการ
            'nutrition_query': {
                'patterns': [
                    r'แคลอรี่.*เท่าไหร่',
                    r'โปรตีน.*มาก.*น้อย',
                    r'(ไขมัน|คาร์โบ|ใยอาหาร).*(สูง|ต่ำ|มาก|น้อย)',
                    r'คุณค่า.*โภชนาการ',
                    r'วิตามิน.*แร่ธาตุ',
                    r'(โซเดียม|แคลเซียม|เหล็ก).*(สูง|ต่ำ)'
                ],
                'type': 'nutrition_analysis'
            },
            
            # คำถามเกี่ยวกับสุขภาพ
            'health_query': {
                'patterns': [
                    r'(เบาหวาน|ความดัน|หัวใจ|ไต)',
                    r'ผู้ป่วย.*กิน.*ได้',
                    r'เหมาะ.*(คนแก่|เด็ก|ผู้สูงอายุ)',
                    r'(ลดน้ำหนัก|เพิ่มน้ำหนัก)',
                    r'(ออกกำลังกาย|กล้ามเนื้อ)',
                    r'(ท้อง|ขับถ่าย|ย่อย)'
                ],
                'type': 'health_recommendation'
            },
            
            # คำถามเกี่ยวกับการแนะนำ
            'recommendation_query': {
                'patterns': [
                    r'แนะนำ.*อะไร',
                    r'มี.*อะไรบ้าง',
                    r'ควร.*กิน.*อะไร',
                    r'อยาก.*กิน.*อะไร',
                    r'เมนู.*อะไร.*ดี',
                    r'อาหาร.*อะไร.*อร่อย'
                ],
                'type': 'general_recommendation'
            },
            
            # คำถามเปรียบเทียบ
            'comparison_query': {
                'patterns': [
                    r'.*เปรียบเทียบ.*',
                    r'.*แตกต่าง.*',
                    r'.*ดีกว่า.*',
                    r'.*เลือก.*',
                    r'.*กับ.*อะไร.*ดี'
                ],
                'type': 'comparison'
            },
            
            # คำถามทั่วไป
            'general_query': {
                'patterns': [
                    r'(สวัสดี|หวัดดี|ดี)',
                    r'ช่วย.*ได้.*ไหม',
                    r'คือ.*อะไร',
                    r'บอก.*เกี่ยวกับ',
                    r'รู้.*เรื่อง',
                    r'ขอบคุณ'
                ],
                'type': 'general_chat'
            }
        }
        
        # คำสำคัญโภชนาการ
        self.nutrition_keywords = {
            'แคลอรี่': ['แคลอรี่', 'แคล', 'kcal', 'พลังงาน', 'calories'],
            'โปรตีน': ['โปรตีน', 'protein', 'เนื้อ', 'กล้ามเนื้อ'],
            'คาร์โบไฮเดรต': ['คาร์โบ', 'carbs', 'แป้ง', 'น้ำตาล', 'คาร์โบไฮเดรต'],
            'ไขมัน': ['ไขมัน', 'fat', 'น้ำมัน', 'เนย'],
            'ใยอาหาร': ['ใยอาหาร', 'fiber', 'ใย', 'ผัก'],
            'วิตามิน': ['วิตามิน', 'vitamin'],
            'แร่ธาตุ': ['แร่ธาตุ', 'mineral', 'แคลเซียม', 'เหล็ก', 'โซเดียม']
        }
        
        # คำตอบมาตรฐาน
        self.standard_responses = {
            'greeting': [
                "สวัสดีครับ! ผมเป็น AI ที่พร้อมช่วยเหลือเรื่องอาหารไทยและโภชนาการ 😊",
                "หวัดดีครับ! มีอะไรให้ช่วยเกี่ยวกับอาหารไทยหรือโภชนาการไหมครับ? 🍲",
                "ยินดีต้อนรับครับ! ถามเรื่องอาหารไทย สูตรการทำ หรือโภชนาการได้เลยนะครับ 👨‍🍳"
            ],
            'thanks': [
                "ยินดีครับ! มีอะไรให้ช่วยอีกไหม? 😊",
                "เป็นประโยชน์ก็ดีครับ! ถามต่อได้เสมอนะครับ 🙂",
                "ไม่เป็นไรครับ! พร้อมช่วยเสมอ 💪"
            ],
            'not_understand': [
                "ขออภัยครับ ผมไม่เข้าใจคำถามนี้ ลองถามใหม่หรือถามเรื่องอาหารไทยได้นะครับ 🤔",
                "ผมยังไม่เข้าใจครับ ลองถามเกี่ยวกับสูตรอาหาร โภชนาการ หรือสุขภาพดูครับ 😅",
                "ขอโทษครับ คำถามนี้ยากไปหน่อย ลองถามเรื่องอาหารไทยแทนไหมครับ? 🍜"
            ],
            'no_results': [
                "ขออภัยครับ ไม่พบข้อมูลที่คุณต้องการ ลองถามอีกรูปแบบดูครับ",
                "ไม่เจอข้อมูลเลยครับ ลองเปลี่ยนคำค้นหาใหม่ดูไหมครับ?",
                "ขอโทษครับ ยังไม่มีข้อมูลที่ตรงกัน ถามอย่างอื่นได้ไหมครับ?"
            ]
        }
        
        # เกณฑ์โภชนาการสำหรับการแนะนำ
        self.nutrition_thresholds = {
            'low_calorie': 250,      # แคลอรี่ต่ำ
            'high_protein': 20,      # โปรตีนสูง
            'low_fat': 10,          # ไขมันต่ำ
            'low_carb': 15,         # คาร์โบต่ำ
            'high_fiber': 3,        # ใยอาหารสูง
            'low_sodium': 800,      # โซเดียมต่ำ
            'high_vitamin_c': 20,   # วิตามินซีสูง
            'high_calcium': 100,    # แคลเซียมสูง
            'high_iron': 3          # เหล็กสูง
        }

    def process_query(self, query: str, mode: str = "ทั่วไป", max_results: int = 5) -> Dict:
        """ประมวลผลคำถามและส่งคืนคำตอบ"""
        
        query_clean = query.strip().lower()
        
        # วิเคราะห์ประเภทคำถาม
        question_type = self.classify_question(query_clean)
        
        if question_type == 'recipe_search':
            return self.handle_recipe_search(query, max_results)
        elif question_type == 'nutrition_analysis':
            return self.handle_nutrition_query(query, max_results)
        elif question_type == 'health_recommendation':
            return self.handle_health_query(query, max_results)
        elif question_type == 'general_recommendation':
            return self.handle_recommendation_query(query, mode, max_results)
        elif question_type == 'comparison':
            return self.handle_comparison_query(query, max_results)
        elif question_type == 'general_chat':
            return self.handle_general_chat(query)
        else:
            return self.handle_fallback(query, max_results)

    def classify_question(self, query: str) -> str:
        """จำแนกประเภทคำถาม"""
        
        for question_type, config in self.question_patterns.items():
            for pattern in config['patterns']:
                if re.search(pattern, query):
                    return config['type']
        
        return 'unknown'

    def handle_recipe_search(self, query: str, max_results: int) -> Dict:
        """จัดการคำถามเกี่ยวกับสูตรอาหาร"""
        
        # ดึงชื่อเมนูจากคำถาม
        menu_name = self.extract_menu_name(query)
        
        if not menu_name:
            return {
                "type": "recommendations",
                "content": "ลองถามสูตรเมนูเฉพาะ เช่น 'วิธีทำผัดกะเพรา' หรือ 'สูตรต้มยำกุ้ง' ครับ",
                "recommendations": self.get_popular_recipes(5)
            }
        
        # ค้นหาเมนู
        matches = self.search_engine.search_recipes(menu_name, max_results)
        
        if matches:
            best_match = matches[0]
            recipe_data = {
                'name': best_match['name'],
                'ingredient': best_match['ingredient'],
                'method': best_match['method']
            }
            
            # คำนวณโภชนาการ
            nutrition_data = self.get_nutrition_for_recipe(best_match['name'])
            
            return {
                "type": "recipe",
                "recipe_data": recipe_data,
                "nutrition_data": nutrition_data,
                "content": f"พบสูตร {recipe_data['name']} ครับ"
            }
        else:
            return {
                "type": "recommendations",
                "content": f"ไม่พบสูตร '{menu_name}' ครับ ลองดูเมนูอื่นไหมครับ?",
                "recommendations": self.get_popular_recipes(max_results)
            }

    def handle_nutrition_query(self, query: str, max_results: int) -> Dict:
        """จัดการคำถามเกี่ยวกับโภชนาการ"""
        
        # วิเคราะห์คำถามโภชนาการ
        nutrition_intent = self.analyze_nutrition_intent(query)
        
        if not nutrition_intent:
            return {
                "type": "general",
                "content": "ลองถามเรื่องโภชนาการเฉพาะ เช่น 'อาหารแคลอรี่ต่ำ' หรือ 'เมนูโปรตีนสูง' ครับ"
            }
        
        # ค้นหาเมนูตามเกณฑ์โภชนาการ
        recommendations = self.get_nutrition_based_recommendations(nutrition_intent, max_results)
        
        if recommendations:
            content = f"แนะนำเมนู{nutrition_intent['description']} ครับ:"
            return {
                "type": "recommendations",
                "content": content,
                "recommendations": recommendations
            }
        else:
            return {
                "type": "general",
                "content": f"ไม่พบเมนูที่ตรงกับเกณฑ์ {nutrition_intent['description']} ครับ"
            }

    def handle_health_query(self, query: str, max_results: int) -> Dict:
        """จัดการคำถามเกี่ยวกับสุขภาพ"""
        
        health_condition = self.identify_health_condition(query)
        
        if health_condition:
            recommendations = self.get_health_based_recommendations(health_condition, max_results)
            
            if recommendations:
                content = f"แนะนำเมนูสำหรับ{health_condition['description']} ครับ:"
                return {
                    "type": "recommendations",
                    "content": content,
                    "recommendations": recommendations
                }
        
        return {
            "type": "general",
            "content": "ลองถามเรื่องสุขภาพเฉพาะ เช่น 'อาหารสำหรับเบาหวาน' หรือ 'เมนูเหมาะกับความดันสูง' ครับ"
        }

    def handle_recommendation_query(self, query: str, mode: str, max_results: int) -> Dict:
        """จัดการคำถามแนะนำทั่วไป"""
        
        if mode == "ตามโภชนาการ":
            # แนะนำตามโภชนาการ
            recommendations = self.get_balanced_nutrition_recommendations(max_results)
        elif mode == "ตามกลุ่มผู้ป่วย":
            # แนะนำตามกลุ่มสุขภาพ
            recommendations = self.get_health_conscious_recommendations(max_results)
        else:
            # แนะนำทั่วไป
            recommendations = self.get_popular_recipes(max_results)
        
        return {
            "type": "recommendations",
            "content": "แนะนำเมนูเหล่านี้ครับ:",
            "recommendations": recommendations
        }

    def handle_comparison_query(self, query: str, max_results: int) -> Dict:
        """จัดการคำถามเปรียบเทียบ"""
        
        # ดึงชื่อเมนูที่ต้องการเปรียบเทียบ
        menus = self.extract_comparison_menus(query)
        
        if len(menus) >= 2:
            comparison_data = self.compare_recipes(menus)
            return {
                "type": "comparison",
                "content": f"เปรียบเทียบ {' และ '.join(menus)} ครับ:",
                "comparison_data": comparison_data
            }
        
        return {
            "type": "general",
            "content": "ลองถามเปรียบเทียบ 2 เมนู เช่น 'ผัดกะเพรากับผัดไทยอะไรดีกว่า' ครับ"
        }

    def handle_general_chat(self, query: str) -> Dict:
        """จัดการคำถามทั่วไป"""
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['สวัสดี', 'หวัดดี', 'ดี']):
            response = random.choice(self.standard_responses['greeting'])
        elif any(word in query_lower for word in ['ขอบคุณ', 'thanks', 'thank you']):
            response = random.choice(self.standard_responses['thanks'])
        else:
            response = "มีอะไรให้ช่วยเรื่องอาหารไทยหรือโภชนาการไหมครับ? 😊"
        
        return {
            "type": "general",
            "content": response
        }

    def handle_fallback(self, query: str, max_results: int) -> Dict:
        """จัดการกรณีไม่เข้าใจคำถาม"""
        
        # ลองค้นหาคำใกล้เคียงในชื่อเมนู
        possible_matches = self.find_similar_menu_names(query, 3)
        
        if possible_matches:
            suggestions = [{"name": match, "reason": "ชื่อใกล้เคียง"} for match in possible_matches]
            return {
                "type": "recommendations",
                "content": "ไม่แน่ใจว่าหาอะไร ลองดูเมนูที่ใกล้เคียงไหมครับ?",
                "recommendations": suggestions
            }
        
        response = random.choice(self.standard_responses['not_understand'])
        return {
            "type": "general",
            "content": response
        }

    def extract_menu_name(self, query: str) -> Optional[str]:
        """ดึงชื่อเมนูจากคำถาม"""
        
        # ลบคำที่ไม่จำเป็น
        stop_words = ['วิธี', 'ทำ', 'ปรุง', 'สูตร', 'ยังไง', 'อย่างไร', 'ครับ', 'ค่ะ', 'หา', 'ต้องการ']
        words = query.split()
        
        filtered_words = []
        for word in words:
            if word not in stop_words and len(word) > 1:
                filtered_words.append(word)
        
        menu_candidate = ' '.join(filtered_words)
        
        # ตรวจสอบว่าใกล้เคียงกับเมนูในฐานข้อมูลไหม
        best_match = None
        best_score = 0
        
        for _, row in self.data.iterrows():
            similarity = SequenceMatcher(None, menu_candidate.lower(), row['name'].lower()).ratio()
            if similarity > best_score and similarity > 0.3:
                best_score = similarity
                best_match = row['name']
        
        return menu_candidate if menu_candidate else None

    def analyze_nutrition_intent(self, query: str) -> Optional[Dict]:
        """วิเคราะห์ความต้องการทางโภชนาการ"""
        
        intents = {
            'แคลอรี่ต่ำ': {
                'keywords': ['แคลอรี่ต่ำ', 'แคลต่ำ', 'ลดน้ำหนัก', 'diet'],
                'criteria': 'low_calorie',
                'description': 'แคลอรี่ต่ำ (น้อยกว่า 250 kcal)'
            },
            'โปรตีนสูง': {
                'keywords': ['โปรตีนสูง', 'โปรตีนมาก', 'กล้ามเนื้อ', 'ออกกำลังกาย'],
                'criteria': 'high_protein',
                'description': 'โปรตีนสูง (มากกว่า 20g)'
            },
            'ไขมันต่ำ': {
                'keywords': ['ไขมันต่ำ', 'ไขมันน้อย', 'ลดไขมัน'],
                'criteria': 'low_fat',
                'description': 'ไขมันต่ำ (น้อยกว่า 10g)'
            },
            'คาร์โบต่ำ': {
                'keywords': ['คาร์โบต่ำ', 'แป้งน้อย', 'น้ำตาลต่ำ', 'keto'],
                'criteria': 'low_carb',
                'description': 'คาร์โบไฮเดรตต่ำ (น้อยกว่า 15g)'
            },
            'ใยอาหารสูง': {
                'keywords': ['ใยอาหารสูง', 'ใยมาก', 'ช่วยย่อย', 'ขับถ่าย'],
                'criteria': 'high_fiber',
                'description': 'ใยอาหารสูง (มากกว่า 3g)'
            },
            'โซเดียมต่ำ': {
                'keywords': ['โซเดียมต่ำ', 'เกลือน้อย', 'ความดัน'],
                'criteria': 'low_sodium',
                'description': 'โซเดียมต่ำ (น้อยกว่า 800mg)'
            }
        }
        
        for intent_name, intent_config in intents.items():
            for keyword in intent_config['keywords']:
                if keyword in query:
                    return intent_config
        
        return None

    def identify_health_condition(self, query: str) -> Optional[Dict]:
        """ระบุภาวะสุขภาพจากคำถาม"""
        
        conditions = {
            'เบาหวาน': {
                'keywords': ['เบาหวาน', 'ผู้ป่วยเบาหวาน', 'น้ำตาลสูง'],
                'description': 'ผู้ป่วยเบาหวาน'
            },
            'ความดันสูง': {
                'keywords': ['ความดันสูง', 'ความดัน', 'ไฮเปอร์เทนชั่น'],
                'description': 'ผู้ป่วยความดันโลหิตสูง'
            },
            'ลดน้ำหนัก': {
                'keywords': ['ลดน้ำหนัก', 'ลดความอ้วน', 'diet', 'ไดเอท'],
                'description': 'ผู้ที่ต้องการลดน้ำหนัก'
            },
            'ผู้สูงอายุ': {
                'keywords': ['ผู้สูงอายุ', 'คนแก่', 'อายุมาก'],
                'description': 'ผู้สูงอายุ'
            },
            'เด็ก': {
                'keywords': ['เด็ก', 'เด็กเล็ก', 'ลูก'],
                'description': 'เด็ก'
            }
        }
        
        for condition_name, condition_config in conditions.items():
            for keyword in condition_config['keywords']:
                if keyword in query:
                    return condition_config
        
        return None

    def get_nutrition_based_recommendations(self, nutrition_intent: Dict, max_results: int) -> List[Dict]:
        """แนะนำเมนูตามเกณฑ์โภชนาการ"""
        
        criteria = nutrition_intent['criteria']
        threshold = self.nutrition_thresholds.get(criteria)
        
        if not threshold:
            return []
        
        recommendations = []
        
        for _, row in self.data.iterrows():
            nutrition = self.get_nutrition_for_recipe(row['name'])
            if not nutrition:
                continue
            
            match = False
            reason = ""
            
            if criteria == 'low_calorie' and nutrition.get('calories', 999) < threshold:
                match = True
                reason = f"แคลอรี่เพียง {nutrition['calories']:.0f} kcal"
            elif criteria == 'high_protein' and nutrition.get('protein', 0) > threshold:
                match = True
                reason = f"โปรตีน {nutrition['protein']:.1f}g สูง"
            elif criteria == 'low_fat' and nutrition.get('fat', 999) < threshold:
                match = True
                reason = f"ไขมันเพียง {nutrition['fat']:.1f}g"
            elif criteria == 'low_carb' and nutrition.get('carbs', 999) < threshold:
                match = True
                reason = f"คาร์โบไฮเดรตเพียง {nutrition['carbs']:.1f}g"
            elif criteria == 'high_fiber' and nutrition.get('fiber', 0) > threshold:
                match = True
                reason = f"ใยอาหาร {nutrition['fiber']:.1f}g สูง"
            elif criteria == 'low_sodium' and nutrition.get('sodium', 999) < threshold:
                match = True
                reason = f"โซเดียมเพียง {nutrition['sodium']:.0f}mg"
            
            if match:
                recommendations.append({
                    'name': row['name'],
                    'reason': reason,
                    'nutrition': nutrition
                })
            
            if len(recommendations) >= max_results:
                break
        
        return recommendations

    def get_health_based_recommendations(self, health_condition: Dict, max_results: int) -> List[Dict]:
        """แนะนำเมนูตามภาวะสุขภาพ"""
        
        condition_name = health_condition['description']
        recommendations = []
        
        for _, row in self.data.iterrows():
            nutrition = self.get_nutrition_for_recipe(row['name'])
            if not nutrition:
                continue
            
            suitable = False
            reason = ""
            
            if 'เบาหวาน' in condition_name:
                # เกณฑ์สำหรับเบาหวาน: คาร์โบต่ำ, ไขมันต่ำ, ใยอาหารสูง
                if (nutrition.get('carbs', 999) < 20 and 
                    nutrition.get('fat', 999) < 15 and 
                    nutrition.get('fiber', 0) > 2):
                    suitable = True
                    reason = "เหมาะสำหรับเบาหวาน: คาร์โบต่ำ ไขมันต่ำ"
                    
            elif 'ความดัน' in condition_name:
                # เกณฑ์สำหรับความดันสูง: โซเดียมต่ำ
                if nutrition.get('sodium', 999) < 600:
                    suitable = True
                    reason = f"เหมาะสำหรับความดันสูง: โซเดียมต่ำ ({nutrition['sodium']:.0f}mg)"
                    
            elif 'ลดน้ำหนัก' in condition_name:
                # เกณฑ์สำหรับลดน้ำหนัก: แคลอรี่ต่ำ, โปรตีนสูง
                if (nutrition.get('calories', 999) < 300 and 
                    nutrition.get('protein', 0) > 15):
                    suitable = True
                    reason = f"เหมาะสำหรับลดน้ำหนัก: {nutrition['calories']:.0f} kcal, โปรตีน {nutrition['protein']:.1f}g"
                    
            elif 'ผู้สูงอายุ' in condition_name:
                # เกณฑ์สำหรับผู้สูงอายุ: โซเดียมต่ำ, แคลเซียมสูง
                if (nutrition.get('sodium', 999) < 800 and 
                    nutrition.get('calcium', 0) > 80):
                    suitable = True
                    reason = f"เหมาะสำหรับผู้สูงอายุ: แคลเซียม {nutrition['calcium']:.0f}mg"
                    
            elif 'เด็ก' in condition_name:
                # เกณฑ์สำหรับเด็ก: แคลเซียมสูง, วิตามินสูง
                if (nutrition.get('calcium', 0) > 100 or 
                    nutrition.get('vitamin_c', 0) > 15):
                    suitable = True
                    reason = "เหมาะสำหรับเด็ก: แคลเซียมและวิตามิน"
            
            if suitable:
                recommendations.append({
                    'name': row['name'],
                    'reason': reason,
                    'nutrition': nutrition
                })
            
            if len(recommendations) >= max_results:
                break
        
        return recommendations

    def get_balanced_nutrition_recommendations(self, max_results: int) -> List[Dict]:
        """แนะนำเมนูที่มีโภชนาการสมดุล"""
        
        recommendations = []
        
        for _, row in self.data.iterrows():
            nutrition = self.get_nutrition_for_recipe(row['name'])
            if not nutrition:
                continue
            
            # คำนวณคะแนนความสมดุล
            balance_score = 0
            
            # ตรวจสอบแคลอรี่ปานกลาง (200-400)
            calories = nutrition.get('calories', 0)
            if 200 <= calories <= 400:
                balance_score += 2
            
            # ตรวจสอบโปรตีนดี (15-30g)
            protein = nutrition.get('protein', 0)
            if 15 <= protein <= 30:
                balance_score += 2
            
            # ตรวจสอบไขมันปานกลาง (5-20g)
            fat = nutrition.get('fat', 0)
            if 5 <= fat <= 20:
                balance_score += 1
            
            # ตรวจสอบใยอาหาร (>2g)
            fiber = nutrition.get('fiber', 0)
            if fiber > 2:
                balance_score += 1
            
            # ตรวจสอบโซเดียมไม่สูงเกินไป (<1200mg)
            sodium = nutrition.get('sodium', 0)
            if sodium < 1200:
                balance_score += 1
            
            if balance_score >= 4:  # เกณฑ์ความสมดุล
                recommendations.append({
                    'name': row['name'],
                    'reason': f"โภชนาการสมดุล (คะแนน {balance_score}/7)",
                    'nutrition': nutrition
                })
            
            if len(recommendations) >= max_results:
                break
        
        return recommendations

    def get_health_conscious_recommendations(self, max_results: int) -> List[Dict]:
        """แนะนำเมนูเพื่อสุขภาพ"""
        
        recommendations = []
        
        for _, row in self.data.iterrows():
            nutrition = self.get_nutrition_for_recipe(row['name'])
            if not nutrition:
                continue
            
            # เกณฑ์เพื่อสุขภาพ
            health_score = 0
            
            if nutrition.get('calories', 999) < 350:
                health_score += 1
            if nutrition.get('fat', 999) < 15:
                health_score += 1
            if nutrition.get('sodium', 999) < 1000:
                health_score += 1
            if nutrition.get('fiber', 0) > 2:
                health_score += 1
            if nutrition.get('vitamin_c', 0) > 10:
                health_score += 1
            
            if health_score >= 3:
                recommendations.append({
                    'name': row['name'],
                    'reason': f"เพื่อสุขภาพ (คะแนน {health_score}/5)",
                    'nutrition': nutrition
                })
            
            if len(recommendations) >= max_results:
                break
        
        return recommendations

    def get_popular_recipes(self, max_results: int) -> List[Dict]:
        """แนะนำเมนูยอดนิยม"""
        
        popular_names = [
            'ผัดกะเพราหมูสับ', 'ต้มยำกุ้งน้ำใส', 'ส้มตำไทย', 
            'แกงเขียวหวานไก่', 'ผัดไทยกุ้งสด', 'ไข่เจียวฟู'
        ]
        
        recommendations = []
        
        for name in popular_names[:max_results]:
            if name in self.data['name'].values:
                row = self.data[self.data['name'] == name].iloc[0]
                nutrition = self.get_nutrition_for_recipe(name)
                
                recommendations.append({
                    'name': name,
                    'reason': 'เมนูยอดนิยม',
                    'nutrition': nutrition
                })
        
        return recommendations

    def get_nutrition_for_recipe(self, recipe_name: str) -> Optional[Dict]:
        """ดึงข้อมูลโภชนาการสำหรับเมนู"""
        
        try:
            recipe_row = self.data[self.data['name'] == recipe_name]
            if recipe_row.empty:
                return None
            
            row = recipe_row.iloc[0]
            
            # ดึงข้อมูลโภชนาการจากคอลัมน์ (ถ้ามี)
            nutrition_columns = ['calories', 'protein', 'carbs', 'fat', 'fiber', 
                               'vitamin_a', 'vitamin_c', 'vitamin_b1', 'vitamin_b2',
                               'calcium', 'iron', 'potassium', 'sodium']
            
            nutrition_data = {}
            for col in nutrition_columns:
                if col in row.index:
                    nutrition_data[col] = row[col]
                else:
                    nutrition_data[col] = 0
            
            return nutrition_data
            
        except Exception as e:
            return None

    def find_similar_menu_names(self, query: str, max_results: int) -> List[str]:
        """หาชื่อเมนูที่คล้ายคลึง"""
        
        similarities = []
        
        for _, row in self.data.iterrows():
            similarity = SequenceMatcher(None, query.lower(), row['name'].lower()).ratio()
            if similarity > 0.3:
                similarities.append((row['name'], similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in similarities[:max_results]]

    def extract_comparison_menus(self, query: str) -> List[str]:
        """ดึงชื่อเมนูที่ต้องการเปรียบเทียบ"""
        
        # รูปแบบการเปรียบเทียบ เช่น "A กับ B", "A และ B"
        patterns = [
            r'(\w+)\s*กับ\s*(\w+)',
            r'(\w+)\s*และ\s*(\w+)',
            r'(\w+)\s*หรือ\s*(\w+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                menu1, menu2 = match.groups()
                # ตรวจสอบว่าเป็นชื่อเมนูจริงไหม
                if (self.find_similar_menu_names(menu1, 1) and 
                    self.find_similar_menu_names(menu2, 1)):
                    return [menu1, menu2]
        
        return []

    def compare_recipes(self, menu_names: List[str]) -> Dict:
        """เปรียบเทียบเมนูอาหาร"""
        
        comparison_data = {}
        
        for menu_name in menu_names:
            nutrition = self.get_nutrition_for_recipe(menu_name)
            if nutrition:
                comparison_data[menu_name] = nutrition
        
        return comparison_data
