import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Tuple, Any
from difflib import SequenceMatcher
import random

class ThaiLoodChatbot:
    """ระบบ Chatbot อัจฉริยะสำหรับอาหารไทยและโภชนาการ - เวอร์ชันปรับปรุง"""
    
    def __init__(self, data: pd.DataFrame, nutrition_api, search_engine):
        self.data = data
        self.nutrition_api = nutrition_api
        self.search_engine = search_engine
        
        # รูปแบบคำถามและการตอบกลับที่ปรับปรุงแล้ว
        self.question_patterns = {
            # คำถามเกี่ยวกับสูตรอาหาร
            'recipe_how': {
                'patterns': [
                    r'(ทำ|ปรุง|เตรียม).*(ยังไง|อย่างไร|วิธี)',
                    r'วิธี(ทำ|ปรุง|เตรียม)',
                    r'สูตร.*อะไร',
                    r'.*ทำยังไง',
                    r'สอน.*ทำ',
                    r'วิธีการ.*ปรุง'
                ],
                'type': 'recipe_search'
            },
            
            # คำถามเกี่ยวกับโภชนาการ - เพิ่มรูปแบบใหม่
            'nutrition_query': {
                'patterns': [
                    r'แคลอรี่.*เท่าไหร่',
                    r'โปรตีน.*(มาก|น้อย|สูง|ต่ำ)',
                    r'(ไขมัน|คาร์โบ|ใยอาหาร).*(สูง|ต่ำ|มาก|น้อย)',
                    r'คุณค่า.*โภชนาการ',
                    r'วิตามิน.*(เอ|ซี|บี|ดี)',
                    r'(โซเดียม|แคลเซียม|เหล็ก|โปแตสเซียม).*(สูง|ต่ำ)',
                    r'สารอาหาร.*อะไร',
                    r'โภชนาการ.*มี.*อะไร',
                    r'ค่าโภชนาการ.*ของ',
                    r'(วิตามิน|แร่ธาตุ|สารอาหาร)',
                    r'(มีประโยชน์|ดีต่อ).*อะไร'
                ],
                'type': 'nutrition_analysis'
            },
            
            # คำถามเกี่ยวกับสุขภาพ - เพิ่มความครอบคลุม
            'health_query': {
                'patterns': [
                    r'(เบาหวาน|ความดัน|หัวใจ|ไต|ตับ)',
                    r'ผู้ป่วย.*กิน.*ได้',
                    r'เหมาะ.*(คนแก่|เด็ก|ผู้สูงอายุ|เด็กเล็ก)',
                    r'(ลดน้ำหนัก|เพิ่มน้ำหนัก|คุมน้ำหนัก)',
                    r'(ออกกำลังกาย|กล้ามเนื้อ|นักกีฬา)',
                    r'(ท้อง|ขับถ่าย|ย่อย|ท้องผูก)',
                    r'(ตั้งครรภ์|คนท้อง|ให้นม)',
                    r'(โคเลสเตอรอล|ไตรกลีเซอร์ไรด์)',
                    r'(ภูมิแพ้|แพ้อาหาร)',
                    r'(กระดูก|ฟัน|ผิวพรรณ|สายตา)',
                    r'สุขภาพ.*ดี'
                ],
                'type': 'health_recommendation'
            },
            
            # คำถามเกี่ยวกับการแนะนำ - ปรับปรุงให้ละเอียดขึ้น
            'recommendation_query': {
                'patterns': [
                    r'แนะนำ.*อะไร',
                    r'มี.*อะไรบ้าง',
                    r'ควร.*กิน.*อะไร',
                    r'อยาก.*กิน.*อะไร',
                    r'เมนู.*อะไร.*ดี',
                    r'อาหาร.*อะไร.*อร่อย',
                    r'ช่วย.*เลือก',
                    r'แนะนำ.*เมนู',
                    r'(อะไร.*ดี|ดี.*อะไร)',
                    r'หา.*เมนู',
                    r'อยาก.*ลอง',
                    r'ทำ.*อะไร.*กิน'
                ],
                'type': 'general_recommendation'
            },
            
            # คำถามเปรียบเทียบ - เพิ่มรูปแบบ
            'comparison_query': {
                'patterns': [
                    r'.*เปรียบเทียบ.*',
                    r'.*แตกต่าง.*',
                    r'.*ดีกว่า.*',
                    r'.*เลือก.*',
                    r'.*กับ.*อะไร.*ดี',
                    r'(.*หรือ.*)',
                    r'ใคร.*ดีกว่า',
                    r'แบบไหน.*ดี',
                    r'อะไร.*เหมาะ.*กว่า'
                ],
                'type': 'comparison'
            },
            
            # คำถามทั่วไป
            'general_query': {
                'patterns': [
                    r'(สวัสดี|หวัดดี|ดี|ฮัลโหล)',
                    r'ช่วย.*ได้.*ไหม',
                    r'คือ.*อะไร',
                    r'บอก.*เกี่ยวกับ',
                    r'รู้.*เรื่อง',
                    r'(ขอบคุณ|thanks|thank you)',
                    r'(ไม่เข้าใจ|ไม่รู้)',
                    r'ทำไม'
                ],
                'type': 'general_chat'
            }
        }
        
        # คำสำคัญโภชนาการที่ปรับปรุงแล้ว
        self.nutrition_keywords = {
            # แคลอรี่
            'แคลอรี่': ['แคลอรี่', 'แคล', 'kcal', 'พลังงาน', 'calories', 'cal'],
            'โปรตีน': ['โปรตีน', 'protein', 'เนื้อ', 'กล้ามเนื้อ', 'โปร'],
            'คาร์โบไฮเดรต': ['คาร์โบ', 'carbs', 'แป้ง', 'น้ำตาล', 'คาร์โบไฮเดรต', 'คาโบ'],
            'ไขมัน': ['ไขมัน', 'fat', 'น้ำมัน', 'เนย', 'มัน'],
            'ใยอาหาร': ['ใยอาหาร', 'fiber', 'ใย', 'ผัก', 'ไฟเบอร์'],
            
            # วิตามิน
            'วิตามินเอ': ['วิตามินเอ', 'vitamin a', 'วิตเอ', 'สายตา'],
            'วิตามินซี': ['วิตามินซี', 'vitamin c', 'วิตซี', 'ภูมิคุ้มกัน'],
            'วิตามินบี': ['วิตามินบี', 'vitamin b', 'วิตบี', 'ประสาท'],
            'วิตามินดี': ['วิตามินดี', 'vitamin d', 'วิตดี', 'กระดูก'],
            
            # แร่ธาตุ
            'แคลเซียม': ['แคลเซียม', 'calcium', 'กระดูก', 'ฟัน'],
            'เหล็ก': ['เหล็ก', 'iron', 'โลหิตจาง', 'เลือดจาง'],
            'โซเดียม': ['โซเดียม', 'sodium', 'เกลือ', 'ความดัน'],
            'โปแตสเซียม': ['โปแตสเซียม', 'potassium', 'หัวใจ', 'กล้ามเนื้อ'],
            'สังกะสี': ['สังกะสี', 'zinc', 'ภูมิคุ้มกัน', 'แผลหาย'],
            'แมกนีเซียม': ['แมกนีเซียม', 'magnesium', 'กล้ามเนื้อ', 'ประสาท']
        }
        
        # คำตอบมาตรฐานที่ปรับปรุง
        self.standard_responses = {
            'greeting': [
                "สวัสดีครับ! ผมเป็น AI ที่พร้อมช่วยเหลือเรื่องอาหารไทยและโภชนาการ 😊 ถามเรื่องสูตรอาหาร โภชนาการ หรือคำแนะนำสุขภาพได้เลยครับ",
                "หวัดดีครับ! มีอะไรให้ช่วยเกี่ยวกับอาหารไทยหรือโภชนาการไหมครับ? 🍲 ผมรู้เรื่องสูตรอาหาร คุณค่าทางโภชนาการ และคำแนะนำสุขภาพ",
                "ยินดีต้อนรับครับ! ถามเรื่องอาหารไทย สูตรการทำ โภชนาการ หรือคำแนะนำสุขภาพได้เลยนะครับ 👨‍🍳"
            ],
            'thanks': [
                "ยินดีครับ! มีอะไรให้ช่วยอีกไหม? ถามเรื่องอาหารไทยหรือโภชนาการต่อได้เลยครับ 😊",
                "เป็นประโยชน์ก็ดีครับ! พร้อมช่วยเรื่องสูตรอาหารและโภชนาการได้เสมอนะครับ 🙂",
                "ไม่เป็นไรครับ! อยากรู้เรื่องอาหารไทยหรือสุขภาพอะไรเพิ่มเติมไหมครับ? 💪"
            ],
            'not_understand': [
                "ขออภัยครับ ผมไม่เข้าใจคำถามนี้ ลองถามใหม่หรือถามเรื่องอาหารไทย โภชนาการ หรือสุขภาพได้นะครับ 🤔",
                "ผมยังไม่เข้าใจครับ ลองถามเกี่ยวกับ:\n- สูตรอาหารไทย (เช่น วิธีทำผัดกะเพรา)\n- โภชนาการ (เช่น อาหารแคลอรี่ต่ำ)\n- สุขภาพ (เช่น อาหารสำหรับเบาหวาน) 😅",
                "ขอโทษครับ คำถามนี้ยากไปหน่อย ลองถามเรื่องอาหารไทย โภชนาการ หรือสุขภาพแทนไหมครับ? 🍜"
            ],
            'no_results': [
                "ขออภัยครับ ไม่พบข้อมูลที่คุณต้องการ ลองถามอีกรูปแบบหรือใช้คำที่เฉพาะเจาะจงมากขึ้นครับ",
                "ไม่เจอข้อมูลเลยครับ ลองเปลี่ยนคำค้นหาใหม่ หรือถามเรื่องโภชนาการเฉพาะดูไหมครับ?",
                "ขอโทษครับ ยังไม่มีข้อมูลที่ตรงกัน ลองถามเรื่องวิตามิน แร่ธาตุ หรือสารอาหารเฉพาะดูไหมครับ?"
            ]
        }
        
        # เกณฑ์โภชนาการที่ปรับปรุงใหม่
        self.nutrition_thresholds = {
            'very_low_calorie': 150,    # แคลอรี่ต่ำมาก
            'low_calorie': 250,         # แคลอรี่ต่ำ
            'high_protein': 20,         # โปรตีนสูง
            'very_high_protein': 30,    # โปรตีนสูงมาก
            'low_fat': 8,               # ไขมันต่ำ
            'very_low_fat': 5,          # ไขมันต่ำมาก
            'low_carb': 15,             # คาร์โบต่ำ
            'very_low_carb': 10,        # คาร์โบต่ำมาก
            'high_fiber': 4,            # ใยอาหารสูง
            'very_high_fiber': 7,       # ใยอาหารสูงมาก
            'low_sodium': 600,          # โซเดียมต่ำ
            'very_low_sodium': 300,     # โซเดียมต่ำมาก
            'high_vitamin_c': 20,       # วิตามินซีสูง
            'very_high_vitamin_c': 40,  # วิตามินซีสูงมาก
            'high_vitamin_a': 500,      # วิตามินเอสูง
            'very_high_vitamin_a': 1000,# วิตามินเอสูงมาก
            'high_calcium': 150,        # แคลเซียมสูง
            'very_high_calcium': 300,   # แคลเซียมสูงมาก
            'high_iron': 3,             # เหล็กสูง
            'very_high_iron': 6,        # เหล็กสูงมาก
            'high_potassium': 400,      # โปแตสเซียมสูง
            'very_high_potassium': 600  # โปแตสเซียมสูงมาก
        }

    def process_query(self, query: str, mode: str = "ทั่วไป", max_results: int = 5) -> Dict:
        """ประมวลผลคำถามและส่งคืนคำตอบ - เวอร์ชันปรับปรุง"""
        
        query_clean = query.strip().lower()
        
        # วิเคราะห์ประเภทคำถาม
        question_type = self.classify_question(query_clean)
        
        if question_type == 'recipe_search':
            return self.handle_recipe_search(query, max_results)
        elif question_type == 'nutrition_analysis':
            return self.handle_enhanced_nutrition_query(query, max_results)
        elif question_type == 'health_recommendation':
            return self.handle_enhanced_health_query(query, max_results)
        elif question_type == 'general_recommendation':
            return self.handle_enhanced_recommendation_query(query, mode, max_results)
        elif question_type == 'comparison':
            return self.handle_enhanced_comparison_query(query, max_results)
        elif question_type == 'general_chat':
            return self.handle_general_chat(query)
        else:
            return self.handle_enhanced_fallback(query, max_results)

    def classify_question(self, query: str) -> str:
        """จำแนกประเภทคำถาม - เวอร์ชันปรับปรุง"""
        
        # เพิ่มคะแนนถ่วงน้ำหนักสำหรับการจำแนกที่แม่นยำขึ้น
        scores = {}
        
        for question_type, config in self.question_patterns.items():
            score = 0
            for pattern in config['patterns']:
                if re.search(pattern, query):
                    score += 1
            
            if score > 0:
                scores[config['type']] = score
        
        if not scores:
            return 'unknown'
        
        # คืนค่าประเภทที่มีคะแนนสูงสุด
        return max(scores, key=scores.get)

    def handle_enhanced_nutrition_query(self, query: str, max_results: int) -> Dict:
        """จัดการคำถามเกี่ยวกับโภชนาการ - เวอร์ชันปรับปรุง"""
        
        # วิเคราะห์คำถามโภชนาการแบบละเอียด
        nutrition_analysis = self.analyze_detailed_nutrition_intent(query)
        
        if not nutrition_analysis:
            return {
                "type": "general",
                "content": self.generate_nutrition_help_message()
            }
        
        # ค้นหาเมนูตามเกณฑ์โภชนาการที่วิเคราะห์ได้
        recommendations = self.get_enhanced_nutrition_recommendations(
            nutrition_analysis, max_results
        )
        
        if recommendations:
            content = self.format_nutrition_response(nutrition_analysis, recommendations)
            return {
                "type": "recommendations",
                "content": content,
                "recommendations": recommendations,
                "nutrition_analysis": nutrition_analysis
            }
        else:
            return {
                "type": "general",
                "content": f"ขออภัยครับ ไม่พบเมนูที่ตรงกับเกณฑ์ '{nutrition_analysis['description']}' \nลองปรับเกณฑ์หรือถามเรื่องอื่นดูครับ"
            }

    def analyze_detailed_nutrition_intent(self, query: str) -> Optional[Dict]:
        """วิเคราะห์ความต้องการทางโภชนาการแบบละเอียด"""
        
        intents = {
            # แคลอรี่
            'แคลอรี่ต่ำมาก': {
                'keywords': ['แคลอรี่ต่ำมาก', 'แคลต่ำมาก', 'ลดน้ำหนักเร่งด่วน'],
                'criteria': 'very_low_calorie',
                'description': 'แคลอรี่ต่ำมาก (น้อยกว่า 150 kcal)',
                'health_benefit': 'เหมาะสำหรับการลดน้ำหนักอย่างเข้มข้น'
            },
            'แคลอรี่ต่ำ': {
                'keywords': ['แคลอรี่ต่ำ', 'แคลต่ำ', 'ลดน้ำหนัก', 'diet', 'ไดเอท'],
                'criteria': 'low_calorie',
                'description': 'แคลอรี่ต่ำ (น้อยกว่า 250 kcal)',
                'health_benefit': 'ช่วยควบคุมน้ำหนักและลดความอ้วน'
            },
            
            # โปรตีน
            'โปรตีนสูงมาก': {
                'keywords': ['โปรตีนสูงมาก', 'โปรสูงมาก', 'นักกีฬา', 'เพาะกาย'],
                'criteria': 'very_high_protein',
                'description': 'โปรตีนสูงมาก (มากกว่า 30g)',
                'health_benefit': 'เหมาะสำหรับนักกีฬาและผู้ต้องการเพิ่มกล้ามเนื้อ'
            },
            'โปรตีนสูง': {
                'keywords': ['โปรตีนสูง', 'โปรตีนมาก', 'กล้ามเนื้อ', 'ออกกำลังกาย'],
                'criteria': 'high_protein',
                'description': 'โปรตีนสูง (มากกว่า 20g)',
                'health_benefit': 'ช่วยสร้างและซ่อมแซมกล้ามเนื้อ'
            },
            
            # ไขมัน
            'ไขมันต่ำมาก': {
                'keywords': ['ไขมันต่ำมาก', 'ไม่มีไขมัน', 'โรคหัวใจ'],
                'criteria': 'very_low_fat',
                'description': 'ไขมันต่ำมาก (น้อยกว่า 5g)',
                'health_benefit': 'เหมาะสำหรับผู้ป่วยโรคหัวใจและโคเลสเตอรอลสูง'
            },
            'ไขมันต่ำ': {
                'keywords': ['ไขมันต่ำ', 'ไขมันน้อย', 'ลดไขมัน', 'สุขภาพดี'],
                'criteria': 'low_fat',
                'description': 'ไขมันต่ำ (น้อยกว่า 8g)',
                'health_benefit': 'ช่วยลดความเสี่ยงโรคหัวใจและโคเลสเตอรอล'
            },
            
            # คาร์โบไฮเดรต
            'คาร์โบต่ำมาก': {
                'keywords': ['คาร์โบต่ำมาก', 'keto', 'คีโตเจนิค'],
                'criteria': 'very_low_carb',
                'description': 'คาร์โบไฮเดรตต่ำมาก (น้อยกว่า 10g)',
                'health_benefit': 'เหมาะสำหรับการควบคุมน้ำตาลในเลือดอย่างเข้มข้น'
            },
            'คาร์โบต่ำ': {
                'keywords': ['คาร์โบต่ำ', 'แป้งน้อย', 'น้ำตาลต่ำ', 'เบาหวาน'],
                'criteria': 'low_carb',
                'description': 'คาร์โบไฮเดรตต่ำ (น้อยกว่า 15g)',
                'health_benefit': 'ช่วยควบคุมระดับน้ำตาลในเลือด'
            },
            
            # ใยอาหาร
            'ใยอาหารสูงมาก': {
                'keywords': ['ใยอาหารสูงมาก', 'ใยมากๆ', 'ท้องผูกมาก'],
                'criteria': 'very_high_fiber',
                'description': 'ใยอาหารสูงมาก (มากกว่า 7g)',
                'health_benefit': 'ช่วยระบบย่อยอาหารและลดคอเลสเตอรอลอย่างมีประสิทธิภาพ'
            },
            'ใยอาหารสูง': {
                'keywords': ['ใยอาหารสูง', 'ใยมาก', 'ช่วยย่อย', 'ขับถ่าย'],
                'criteria': 'high_fiber',
                'description': 'ใยอาหารสูง (มากกว่า 4g)',
                'health_benefit': 'ส่งเสริมระบบย่อยอาหารและช่วยขับถ่าย'
            },
            
            # โซเดียม
            'โซเดียมต่ำมาก': {
                'keywords': ['โซเดียมต่ำมาก', 'เกลือน้อยมาก', 'ไม่มีเกลือ'],
                'criteria': 'very_low_sodium',
                'description': 'โซเดียมต่ำมาก (น้อยกว่า 300mg)',
                'health_benefit': 'เหมาะสำหรับผู้ป่วยความดันสูงและโรคไตอย่างเข้มข้น'
            },
            'โซเดียมต่ำ': {
                'keywords': ['โซเดียมต่ำ', 'เกลือน้อย', 'ความดันสูง', 'ไต'],
                'criteria': 'low_sodium',
                'description': 'โซเดียมต่ำ (น้อยกว่า 600mg)',
                'health_benefit': 'ช่วยควบคุมความดันโลหิตและลดภาระไต'
            },
            
            # วิตามินซี
            'วิตามินซีสูงมาก': {
                'keywords': ['วิตามินซีสูงมาก', 'วิตซีสูงมาก', 'ภูมิคุ้มกันแข็งแรง'],
                'criteria': 'very_high_vitamin_c',
                'description': 'วิตามินซีสูงมาก (มากกว่า 40mg)',
                'health_benefit': 'เสริมภูมิคุ้มกันและต้านอนุมูลอิสระอย่างมีประสิทธิภาพ'
            },
            'วิตามินซีสูง': {
                'keywords': ['วิตามินซีสูง', 'วิตซีสูง', 'ภูมิคุ้มกัน', 'ต้านหวัด'],
                'criteria': 'high_vitamin_c',
                'description': 'วิตามินซีสูง (มากกว่า 20mg)',
                'health_benefit': 'เสริมสร้างภูมิคุ้มกันและช่วยดูดซึมเหล็ก'
            },
            
            # วิตามินเอ
            'วิตามินเอสูงมาก': {
                'keywords': ['วิตามินเอสูงมาก', 'วิตเอสูงมาก', 'สายตาดีมาก'],
                'criteria': 'very_high_vitamin_a',
                'description': 'วิตามินเอสูงมาก (มากกว่า 1000 IU)',
                'health_benefit': 'บำรุงสายตาและผิวพรรณอย่างเข้มข้น'
            },
            'วิตามินเอสูง': {
                'keywords': ['วิตามินเอสูง', 'วิตเอสูง', 'สายตา', 'ผิวพรรณ'],
                'criteria': 'high_vitamin_a',
                'description': 'วิตามินเอสูง (มากกว่า 500 IU)',
                'health_benefit': 'ช่วยบำรุงสายตาและสุขภาพผิวพรรณ'
            },
            
            # แคลเซียม
            'แคลเซียมสูงมาก': {
                'keywords': ['แคลเซียมสูงมาก', 'กระดูกแข็งแรงมาก', 'ป้องกันกระดูกพรุน'],
                'criteria': 'very_high_calcium',
                'description': 'แคลเซียมสูงมาก (มากกว่า 300mg)',
                'health_benefit': 'เสริมสร้างกระดูกและฟันให้แข็งแรง ป้องกันกระดูกพรุน'
            },
            'แคลเซียมสูง': {
                'keywords': ['แคลเซียมสูง', 'กระดูก', 'ฟัน', 'ผู้สูงอายุ', 'เด็ก'],
                'criteria': 'high_calcium',
                'description': 'แคลเซียมสูง (มากกว่า 150mg)',
                'health_benefit': 'ช่วยบำรุงกระดูกและฟันให้แข็งแรง'
            },
            
            # เหล็ก
            'เหล็กสูงมาก': {
                'keywords': ['เหล็กสูงมาก', 'ธาตุเหล็กสูงมาก', 'รักษาโลหิตจาง'],
                'criteria': 'very_high_iron',
                'description': 'เหล็กสูงมาก (มากกว่า 6mg)',
                'health_benefit': 'ป้องกันและรักษาโลหิตจางอย่างมีประสิทธิภาพ'
            },
            'เหล็กสูง': {
                'keywords': ['เหล็กสูง', 'ธาตุเหล็ก', 'โลหิตจาง', 'เลือดจาง'],
                'criteria': 'high_iron',
                'description': 'เหล็กสูง (มากกว่า 3mg)',
                'health_benefit': 'ช่วยป้องกันโลหิตจางและเสริมสร้างเม็ดเลือดแดง'
            },
            
            # โปแตสเซียม
            'โปแตสเซียมสูงมาก': {
                'keywords': ['โปแตสเซียมสูงมาก', 'หัวใจแข็งแรงมาก'],
                'criteria': 'very_high_potassium',
                'description': 'โปแตสเซียมสูงมาก (มากกว่า 600mg)',
                'health_benefit': 'ช่วยควบคุมความดันโลหิตและบำรุงหัวใจอย่างเข้มข้น'
            },
            'โปแตสเซียมสูง': {
                'keywords': ['โปแตสเซียมสูง', 'โปแตสเซียม', 'ความดันโลหิต', 'หัวใจ'],
                'criteria': 'high_potassium',
                'description': 'โปแตสเซียมสูง (มากกว่า 400mg)',
                'health_benefit': 'ช่วยควบคุมความดันโลหิตและส่งเสริมสุขภาพหัวใจ'
            }
        }
        
        # ค้นหาเกณฑ์ที่ตรงกับคำค้นหา
        for intent_name, intent_config in intents.items():
            for keyword in intent_config['keywords']:
                if keyword in query:
                    return intent_config
        
        return None

    def get_enhanced_nutrition_recommendations(self, nutrition_analysis: Dict, 
                                             max_results: int) -> List[Dict]:
        """แนะนำเมนูตามเกณฑ์โภชนาการ - เวอร์ชันปรับปรุง"""
        
        criteria = nutrition_analysis['criteria']
        threshold = self.nutrition_thresholds.get(criteria)
        
        if not threshold:
            return []
        
        recommendations = []
        
        for _, row in self.data.iterrows():
            nutrition = self._get_recipe_nutrition(row)
            if not nutrition:
                continue
            
            match_score, reason = self._evaluate_enhanced_nutrition_criterion(
                criteria, nutrition, threshold
            )
            
            if match_score > 0:
                # คำนวณคะแนนความเหมาะสมโดยรวม
                overall_score = self._calculate_overall_nutrition_score(nutrition)
                
                recommendations.append({
                    'name': row['name'],
                    'reason': reason,
                    'nutrition': nutrition,
                    'match_score': match_score,
                    'overall_score': overall_score,
                    'health_benefit': nutrition_analysis.get('health_benefit', ''),
                    'ingredient': row.get('ingredient', ''),
                    'method': row.get('method', '')
                })
        
        # เรียงลำดับตามคะแนนการตรงกันและคะแนนโดยรวม
        recommendations.sort(
            key=lambda x: (x['match_score'], x['overall_score']), 
            reverse=True
        )
        
        return recommendations[:max_results]

    def _evaluate_enhanced_nutrition_criterion(self, criteria: str, nutrition: Dict, 
                                             threshold: float) -> Tuple[float, str]:
        """ประเมินเกณฑ์โภชนาการ - เวอร์ชันปรับปรุง"""
        
        if criteria.endswith('_calorie'):
            value = nutrition.get('calories', 0)
            if value <= threshold:
                score = 15 - (value / threshold) * 5  # คะแนน 10-15
                return score, f"แคลอรี่เพียง {value:.0f} kcal"
                
        elif criteria.endswith('_protein'):
            value = nutrition.get('protein', 0)
            if value >= threshold:
                score = 10 + min((value - threshold) / 5 * 5, 5)  # คะแนน 10-15
                return score, f"โปรตีนสูง {value:.1f}g"
                
        elif criteria.endswith('_fat'):
            value = nutrition.get('fat', 0)
            if value <= threshold:
                score = 15 - (value / threshold) * 5
                return score, f"ไขมันต่ำ {value:.1f}g"
                
        elif criteria.endswith('_carb'):
            value = nutrition.get('carbs', 0)
            if value <= threshold:
                score = 15 - (value / threshold) * 5
                return score, f"คาร์โบไฮเดรตต่ำ {value:.1f}g"
                
        elif criteria.endswith('_fiber'):
            value = nutrition.get('fiber', 0)
            if value >= threshold:
                score = 10 + min((value - threshold) / 2 * 5, 5)
                return score, f"ใยอาหารสูง {value:.1f}g"
                
        elif criteria.endswith('_sodium'):
            value = nutrition.get('sodium', 0)
            if value <= threshold:
                score = 15 - (value / threshold) * 5
                return score, f"โซเดียมต่ำ {value:.0f}mg"
                
        elif criteria.endswith('_vitamin_c'):
            value = nutrition.get('vitamin_c', 0)
            if value >= threshold:
                score = 10 + min((value - threshold) / 10 * 5, 5)
                return score, f"วิตามินซีสูง {value:.1f}mg"
                
        elif criteria.endswith('_vitamin_a'):
            value = nutrition.get('vitamin_a', 0)
            if value >= threshold:
                score = 10 + min((value - threshold) / 200 * 5, 5)
                return score, f"วิตามินเอสูง {value:.0f} IU"
                
        elif criteria.endswith('_calcium'):
            value = nutrition.get('calcium', 0)
            if value >= threshold:
                score = 10 + min((value - threshold) / 50 * 5, 5)
                return score, f"แคลเซียมสูง {value:.0f}mg"
                
        elif criteria.endswith('_iron'):
            value = nutrition.get('iron', 0)
            if value >= threshold:
                score = 10 + min((value - threshold) / 1 * 5, 5)
                return score, f"เหล็กสูง {value:.1f}mg"
                
        elif criteria.endswith('_potassium'):
            value = nutrition.get('potassium', 0)
            if value >= threshold:
                score = 10 + min((value - threshold) / 100 * 5, 5)
                return score, f"โปแตสเซียมสูง {value:.0f}mg"
        
        return 0.0, ""

    def _calculate_overall_nutrition_score(self, nutrition: Dict) -> float:
        """คำนวณคะแนนโภชนาการโดยรวม"""
        
        score = 0
        
        # คะแนนจากแคลอรี่ที่เหมาะสม (200-400)
        calories = nutrition.get('calories', 0)
        if 200 <= calories <= 400:
            score += 2
        elif 150 <= calories < 200 or 400 < calories <= 500:
            score += 1
        
        # คะแนนจากโปรตีนที่ดี (>15g)
        protein = nutrition.get('protein', 0)
        if protein >= 20:
            score += 3
        elif protein >= 15:
            score += 2
        elif protein >= 10:
            score += 1
        
        # คะแนนจากใยอาหาร (>3g)
        fiber = nutrition.get('fiber', 0)
        if fiber >= 5:
            score += 2
        elif fiber >= 3:
            score += 1
        
        # คะแนนจากวิตามินซี (>15mg)
        vitamin_c = nutrition.get('vitamin_c', 0)
        if vitamin_c >= 20:
            score += 2
        elif vitamin_c >= 10:
            score += 1
        
        # คะแนนจากแคลเซียม (>100mg)
        calcium = nutrition.get('calcium', 0)
        if calcium >= 150:
            score += 2
        elif calcium >= 100:
            score += 1
        
        # คะแนนจากเหล็ก (>2mg)
        iron = nutrition.get('iron', 0)
        if iron >= 3:
            score += 2
        elif iron >= 2:
            score += 1
        
        # หักคะแนนจากโซเดียมสูง
        sodium = nutrition.get('sodium', 0)
        if sodium > 1500:
            score -= 2
        elif sodium > 1000:
            score -= 1
        
        return max(score, 0)

    def format_nutrition_response(self, nutrition_analysis: Dict, 
                                recommendations: List[Dict]) -> str:
        """จัดรูปแบบการตอบกลับเรื่องโภชนาการ"""
        
        description = nutrition_analysis['description']
        health_benefit = nutrition_analysis.get('health_benefit', '')
        
        response = f"พบเมนู{description} ทั้งหมด {len(recommendations)} รายการ\n\n"
        
        if health_benefit:
            response += f"💡 **ประโยชน์:** {health_benefit}\n\n"
        
        response += "📋 **รายการแนะนำ:**"
        
        return response

    def generate_nutrition_help_message(self) -> str:
        """สร้างข้อความช่วยเหลือเรื่องโภชนาการ"""
        
        return """ลองถามเรื่องโภชนาการเฉพาะ เช่น:

🔥 **แคลอรี่:** "อาหารแคลอรี่ต่ำ", "เมนูแคลอรี่ต่ำมาก"
🥩 **โปรตีน:** "อาหารโปรตีนสูง", "เมนูโปรตีนสูงมาก"
🫒 **ไขมัน:** "อาหารไขมันต่ำ", "เมนูไขมันต่ำมาก"
🍞 **คาร์โบไฮเดรต:** "อาหารคาร์โบต่ำ", "เมนูเคโต"
🌾 **ใยอาหาร:** "อาหารใยอาหารสูง", "เมนูช่วยย่อย"
🧂 **โซเดียม:** "อาหารโซเดียมต่ำ", "เมนูเกลือน้อย"
🍊 **วิตามิน:** "อาหารวิตามินซีสูง", "เมนูวิตามินเอสูง"
🦴 **แร่ธาตุ:** "อาหารแคลเซียมสูง", "เมนูเหล็กสูง", "อาหารโปแตสเซียมสูง"

หรือถามเรื่องสุขภาพเฉพาะ เช่น "อาหารสำหรับเบาหวาน", "เมนูสำหรับความดันสูง" ครับ"""

    def handle_enhanced_health_query(self, query: str, max_results: int) -> Dict:
        """จัดการคำถามเกี่ยวกับสุขภาพ - เวอร์ชันปรับปรุง"""
        
        health_condition = self.identify_enhanced_health_condition(query)
        
        if health_condition:
            recommendations = self.get_enhanced_health_recommendations(
                health_condition, max_results
            )
            
            if recommendations:
                content = f"แนะนำเมนูสำหรับ{health_condition['description']} ครับ:\n\n"
                content += f"💡 **คำแนะนำ:** {health_condition.get('advice', '')}"
                
                return {
                    "type": "recommendations",
                    "content": content,
                    "recommendations": recommendations,
                    "health_condition": health_condition
                }
        
        return {
            "type": "general",
            "content": self.generate_health_help_message()
        }

    def identify_enhanced_health_condition(self, query: str) -> Optional[Dict]:
        """ระบุภาวะสุขภาพจากคำถาม - เวอร์ชันปรับปรุง"""
        
        conditions = {
            'เบาหวาน': {
                'keywords': ['เบาหวาน', 'ผู้ป่วยเบาหวาน', 'น้ำตาลสูง', 'ดัชนีน้ำตาล'],
                'description': 'ผู้ป่วยเบาหวาน',
                'advice': 'ควรเลือกอาหารคาร์โบไฮเดรตต่ำ ใยอาหารสูง และโซเดียมต่ำ',
                'criteria': {
                    'carbs_max': 20,
                    'fiber_min': 3,
                    'sodium_max': 800,
                    'fat_max': 15
                }
            },
            'ความดันสูง': {
                'keywords': ['ความดันสูง', 'ความดัน', 'ไฮเปอร์เทนชั่น', 'โซเดียมต่ำ'],
                'description': 'ผู้ป่วยความดันโลหิตสูง',
                'advice': 'ควรเลือกอาหารโซเดียมต่ำ โปแตสเซียมสูง และใยอาหารสูง',
                'criteria': {
                    'sodium_max': 600,
                    'potassium_min': 400,
                    'fiber_min': 3
                }
            },
            'โรคหัวใจ': {
                'keywords': ['โรคหัวใจ', 'หัวใจ', 'โคเลสเตอรอล', 'หลอดเลือด'],
                'description': 'ผู้ป่วยโรคหัวใจ',
                'advice': 'ควรเลือกอาหารไขมันต่ำ โซเดียมต่ำ และใยอาหารสูง',
                'criteria': {
                    'fat_max': 8,
                    'sodium_max': 600,
                    'fiber_min': 4
                }
            },
            'โรคไต': {
                'keywords': ['โรคไต', 'ไต', 'ล้างไต', 'ไตเสื่อม'],
                'description': 'ผู้ป่วยโรคไต',
                'advice': 'ควรเลือกอาหารโปรตีนปานกลาง โซเดียมต่ำ และโปแตสเซียมไม่สูง',
                'criteria': {
                    'protein_max': 20,
                    'sodium_max': 600,
                    'potassium_max': 400
                }
            },
            'ลดน้ำหนัก': {
                'keywords': ['ลดน้ำหนัก', 'ลดความอ้วน', 'diet', 'ไดเอท', 'ผอม'],
                'description': 'ผู้ที่ต้องการลดน้ำหนัก',
                'advice': 'ควรเลือกอาหารแคลอรี่ต่ำ โปรตีนสูง และใยอาหารสูง',
                'criteria': {
                    'calories_max': 300,
                    'protein_min': 15,
                    'fiber_min': 3
                }
            },
            'เพิ่มน้ำหนัก': {
                'keywords': ['เพิ่มน้ำหนัก', 'อ้วน', 'น้ำหนักขึ้น', 'ผอมเกินไป'],
                'description': 'ผู้ที่ต้องการเพิ่มน้ำหนัก',
                'advice': 'ควรเลือกอาหารแคลอรี่สูง โปรตีนสูง และไขมันดี',
                'criteria': {
                    'calories_min': 400,
                    'protein_min': 20,
                    'fat_min': 15
                }
            },
            'ผู้สูงอายุ': {
                'keywords': ['ผู้สูงอายุ', 'คนแก่', 'อายุมาก', 'วัยเก๋า'],
                'description': 'ผู้สูงอายุ',
                'advice': 'ควรเลือกอาหารแคลเซียมสูง โซเดียมต่ำ และโปรตีนเพียงพอ',
                'criteria': {
                    'calcium_min': 150,
                    'sodium_max': 800,
                    'protein_min': 15
                }
            },
            'เด็ก': {
                'keywords': ['เด็ก', 'เด็กเล็ก', 'ลูก', 'วัยรุ่น'],
                'description': 'เด็ก',
                'advice': 'ควรเลือกอาหารแคลเซียมสูง วิตามินครบถ้วน และโซเดียมไม่สูง',
                'criteria': {
                    'calcium_min': 100,
                    'vitamin_c_min': 15,
                    'sodium_max': 800
                }
            },
            'นักกีฬา': {
                'keywords': ['นักกีฬา', 'ออกกำลังกาย', 'ฟิตเนส', 'เล่นกีฬา', 'เพาะกาย'],
                'description': 'นักกีฬา',
                'advice': 'ควรเลือกอาหารโปรตีนสูง แคลอรี่เพียงพอ และโปแตสเซียมสูง',
                'criteria': {
                    'protein_min': 25,
                    'calories_min': 350,
                    'potassium_min': 400
                }
            },
            'ตั้งครรภ์': {
                'keywords': ['ตั้งครรภ์', 'คนท้อง', 'มีครรภ์', 'แม่ท้อง'],
                'description': 'หญิงตั้งครรภ์',
                'advice': 'ควรเลือกอาหารเหล็กสูง แคลเซียมสูง และโฟเลตสูง',
                'criteria': {
                    'iron_min': 3,
                    'calcium_min': 150,
                    'vitamin_c_min': 20
                }
            }
        }
        
        for condition_name, condition_config in conditions.items():
            for keyword in condition_config['keywords']:
                if keyword in query:
                    return condition_config
        
        return None

    def get_enhanced_health_recommendations(self, health_condition: Dict, 
                                          max_results: int) -> List[Dict]:
        """แนะนำเมนูตามภาวะสุขภาพ - เวอร์ชันปรับปรุง"""
        
        criteria = health_condition['criteria']
        recommendations = []
        
        for _, row in self.data.iterrows():
            nutrition = self._get_recipe_nutrition(row)
            if not nutrition:
                continue
            
            score = 0
            reasons = []
            
            # ตรวจสอบเกณฑ์ต่างๆ
            for criterion, value in criteria.items():
                nutrient = criterion.split('_')[0]
                limit_type = criterion.split('_')[1]
                
                current_value = nutrition.get(nutrient, 0)
                
                if limit_type == 'max' and current_value <= value:
                    score += 3
                    reasons.append(f"{nutrient} {current_value:.1f} (เหมาะสม)")
                elif limit_type == 'min' and current_value >= value:
                    score += 3
                    reasons.append(f"{nutrient} {current_value:.1f} (ดี)")
                elif limit_type == 'max' and current_value > value:
                    reasons.append(f"{nutrient} {current_value:.1f} (สูงไป)")
                elif limit_type == 'min' and current_value < value:
                    reasons.append(f"{nutrient} {current_value:.1f} (ต่ำไป)")
            
            # คำนวณเปอร์เซ็นต์ความเหมาะสม
            max_score = len(criteria) * 3
            suitability_percent = (score / max_score * 100) if max_score > 0 else 0
            
            if suitability_percent >= 60:  # เกณฑ์ 60% ขึ้นไป
                recommendations.append({
                    'name': row['name'],
                    'reason': f"เหมาะสม {suitability_percent:.0f}%: " + ", ".join(reasons[:3]),
                    'nutrition': nutrition,
                    'suitability_score': score,
                    'suitability_percent': suitability_percent,
                    'ingredient': row.get('ingredient', ''),
                    'method': row.get('method', '')
                })
        
        recommendations.sort(key=lambda x: x['suitability_score'], reverse=True)
        return recommendations[:max_results]

    def generate_health_help_message(self) -> str:
        """สร้างข้อความช่วยเหลือเรื่องสุขภาพ"""
        
        return """ลองถามเรื่องสุขภาพเฉพาะ เช่น:

🏥 **โรคเรื้อรัง:**
- "อาหารสำหรับผู้ป่วยเบาหวาน"
- "เมนูสำหรับความดันโลหิตสูง"
- "อาหารสำหรับโรคหัวใจ"
- "เมนูสำหรับโรคไต"

⚖️ **การควบคุมน้ำหนัก:**
- "อาหารลดน้ำหนัก"
- "เมนูเพิ่มน้ำหนัก"

👨‍👩‍👧‍👦 **ตามช่วงวัย:**
- "อาหารสำหรับผู้สูงอายุ"
- "เมนูสำหรับเด็ก"
- "อาหารสำหรับหญิงตั้งครรภ์"

💪 **ไลฟ์สไตล์:**
- "อาหารสำหรับนักกีฬา"
- "เมนูสำหรับออกกำลังกาย"

ครับ"""

    def handle_enhanced_recommendation_query(self, query: str, mode: str, 
                                           max_results: int) -> Dict:
        """จัดการคำถามแนะนำทั่วไป - เวอร์ชันปรับปรุง"""
        
        if mode == "ตามโภชนาการ":
            recommendations = self.get_balanced_nutrition_recommendations(max_results)
            content = "แนะนำเมนูที่มีโภชนาการสมดุล:"
        elif mode == "ตามกลุ่มผู้ป่วย":
            recommendations = self.get_health_conscious_recommendations(max_results)
            content = "แนะนำเมนูเพื่อสุขภาพ:"
        else:
            # วิเคราะห์คำถามเพื่อให้การแนะนำที่เฉพาะเจาะจงขึ้น
            specific_intent = self.analyze_recommendation_intent(query)
            if specific_intent:
                recommendations = self.get_specific_recommendations(specific_intent, max_results)
                content = f"แนะนำเมนู{specific_intent['description']}:"
            else:
                recommendations = self.get_popular_recipes(max_results)
                content = "แนะนำเมนูยอดนิยม:"
        
        return {
            "type": "recommendations",
            "content": content,
            "recommendations": recommendations
        }

    def analyze_recommendation_intent(self, query: str) -> Optional[Dict]:
        """วิเคราะห์ความต้องการการแนะนำเฉพาะ"""
        
        intents = {
            'ง่าย': {
                'keywords': ['ง่าย', 'ไม่ยาก', 'เรียนทำ', 'มือใหม่'],
                'description': 'ที่ทำง่าย',
                'type': 'difficulty'
            },
            'เร็ว': {
                'keywords': ['เร็ว', 'ไว', 'รีบ', 'ประหยัดเวลา'],
                'description': 'ที่ทำเร็ว',
                'type': 'time'
            },
            'ประหยัด': {
                'keywords': ['ประหยัด', 'ถูก', 'ไม่แพง', 'ราคาไม่สูง'],
                'description': 'ที่ประหยัด',
                'type': 'cost'
            },
            'อร่อย': {
                'keywords': ['อร่อย', 'เด็ด', 'สุดยอด', 'เยี่ยม'],
                'description': 'ที่อร่อย',
                'type': 'taste'
            },
            'ผัก': {
                'keywords': ['ผัก', 'เจ', 'มังสวิรัติ', 'ไม่กินเนื้อ'],
                'description': 'จากผัก',
                'type': 'vegetarian'
            },
            'เผ็ด': {
                'keywords': ['เผ็ด', 'เผ็ดๆ', 'รสจัด'],
                'description': 'รสเผ็ด',
                'type': 'spicy'
            },
            'หวาน': {
                'keywords': ['หวาน', 'ของหวาน', 'ขนม'],
                'description': 'รสหวาน',
                'type': 'sweet'
            }
        }
        
        for intent_name, intent_config in intents.items():
            for keyword in intent_config['keywords']:
                if keyword in query:
                    return intent_config
        
        return None

    def get_specific_recommendations(self, intent: Dict, max_results: int) -> List[Dict]:
        """แนะนำเมนูตามความต้องการเฉพาะ"""
        
        recommendations = []
        intent_type = intent['type']
        
        for _, row in self.data.iterrows():
            score = 0
            reason = ""
            
            if intent_type == 'difficulty':
                # ประเมินความง่ายจากจำนวนวัตถุดิบและขั้นตอน
                ingredient_count = len(row['ingredient'].split('\n'))
                method_length = len(row['method'])
                if ingredient_count <= 6 and method_length <= 200:
                    score = 15 - ingredient_count
                    reason = f"ใช้วัตถุดิบเพียง {ingredient_count} อย่าง ทำง่าย"
                    
            elif intent_type == 'time':
                # ประเมินเวลาจากวิธีการทำ
                method = row['method'].lower()
                if any(word in method for word in ['ผัด', 'เจียว', 'ลวก']):
                    score = 12
                    reason = "ทำได้รวดเร็ว ไม่เกิน 15 นาที"
                elif any(word in method for word in ['ต้ม', 'คั่ว']):
                    score = 8
                    reason = "ทำได้เร็ว ประมาณ 20 นาที"
                    
            elif intent_type == 'vegetarian':
                # ตรวจสอบว่าเป็นอาหารเจหรือไม่
                ingredient = row['ingredient'].lower()
                if not any(meat in ingredient for meat in ['หมู', 'ไก่', 'เนื้อ', 'กุ้ง', 'ปลา']):
                    score = 15
                    reason = "อาหารจากผัก เหมาะสำหรับคนเจ"
            
            if score > 0:
                nutrition = self._get_recipe_nutrition(row)
                recommendations.append({
                    'name': row['name'],
                    'reason': reason,
                    'nutrition': nutrition,
                    'score': score,
                    'ingredient': row.get('ingredient', ''),
                    'method': row.get('method', '')
                })
        
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:max_results]

    def handle_enhanced_comparison_query(self, query: str, max_results: int) -> Dict:
        """จัดการคำถามเปรียบเทียบ - เวอร์ชันปรับปรุง"""
        
        menus = self.extract_comparison_menus(query)
        
        if len(menus) >= 2:
            comparison_data = self.enhanced_compare_recipes(menus)
            if comparison_data:
                return {
                    "type": "comparison",
                    "content": f"เปรียบเทียบ {' กับ '.join(menus)} ครับ:",
                    "comparison_data": comparison_data
                }
        
        return {
            "type": "general",
            "content": "ลองถามเปรียบเทียบ 2 เมนู เช่น:\n- 'ผัดกะเพรากับผัดไทยอะไรดีกว่า'\n- 'ต้มยำกับแกงเขียวหวานเปรียบเทียบ'\n- 'ส้มตำหรือยำวุ้นเส้นดีกว่า' ครับ"
        }

    def enhanced_compare_recipes(self, menu_names: List[str]) -> Optional[Dict]:
        """เปรียบเทียบเมนูอาหารแบบละเอียด"""
        
        comparison_data = {'menus': {}, 'analysis': {}}
        
        for menu_name in menu_names:
            # ค้นหาเมนูที่ใกล้เคียงที่สุด
            best_match = self.find_best_matching_recipe(menu_name)
            if best_match:
                nutrition = self._get_recipe_nutrition(best_match)
                comparison_data['menus'][menu_name] = {
                    'nutrition': nutrition,
                    'recipe_data': {
                        'name': best_match['name'],
                        'ingredient': best_match.get('ingredient', ''),
                        'method': best_match.get('method', '')
                    }
                }
        
        if len(comparison_data['menus']) >= 2:
            comparison_data['analysis'] = self.analyze_recipe_comparison(
                comparison_data['menus']
            )
            return comparison_data
        
        return None

    def find_best_matching_recipe(self, menu_name: str) -> Optional[pd.Series]:
        """หาเมนูที่ตรงกันที่สุด"""
        
        best_match = None
        best_score = 0
        
        for _, row in self.data.iterrows():
            similarity = SequenceMatcher(None, menu_name.lower(), row['name'].lower()).ratio()
            if similarity > best_score and similarity > 0.3:
                best_score = similarity
                best_match = row
        
        return best_match

    def analyze_recipe_comparison(self, menus_data: Dict) -> Dict:
        """วิเคราะห์การเปรียบเทียบเมนู"""
        
        analysis = {
            'winner': {},
            'summary': [],
            'recommendations': []
        }
        
        menu_names = list(menus_data.keys())
        nutritions = [menus_data[name]['nutrition'] for name in menu_names]
        
        # เปรียบเทียบแต่ละสารอาหาร
        nutrients = ['calories', 'protein', 'carbs', 'fat', 'fiber', 'vitamin_c', 'calcium', 'iron', 'sodium']
        
        for nutrient in nutrients:
            values = [nutrition.get(nutrient, 0) for nutrition in nutritions]
            
            if nutrient in ['sodium']:  # สารอาหารที่ต่ำกว่าดีกว่า
                winner_idx = values.index(min(values))
                analysis['winner'][nutrient] = menu_names[winner_idx]
            else:  # สารอาหารที่สูงกว่าดีกว่า
                winner_idx = values.index(max(values))
                analysis['winner'][nutrient] = menu_names[winner_idx]
        
        # สรุปผลการเปรียบเทียบ
        winner_counts = {}
        for menu in menu_names:
            winner_counts[menu] = list(analysis['winner'].values()).count(menu)
        
        overall_winner = max(winner_counts, key=winner_counts.get)
        
        analysis['summary'].append(f"🏆 โดยรวมแล้ว '{overall_winner}' มีคุณค่าทางโภชนาการดีกว่า")
        
        # คำแนะนำเฉพาะ
        for menu_name in menu_names:
            nutrition = menus_data[menu_name]['nutrition']
            
            if nutrition.get('calories', 0) < 250:
                analysis['recommendations'].append(f"✅ '{menu_name}' เหมาะสำหรับลดน้ำหนัก (แคลอรี่ต่ำ)")
            
            if nutrition.get('protein', 0) > 20:
                analysis['recommendations'].append(f"💪 '{menu_name}' เหมาะสำหรับเพิ่มกล้ามเนื้อ (โปรตีนสูง)")
            
            if nutrition.get('sodium', 0) < 600:
                analysis['recommendations'].append(f"❤️ '{menu_name}' เหมาะสำหรับผู้ป่วยความดันสูง (โซเดียมต่ำ)")
        
        return analysis

    def handle_enhanced_fallback(self, query: str, max_results: int) -> Dict:
        """จัดการกรณีไม่เข้าใจคำถาม - เวอร์ชันปรับปรุง"""
        
        # ลองวิเคราะห์คำถามในมุมต่างๆ
        possible_intents = self.analyze_fallback_query(query)
        
        if possible_intents:
            suggestions = []
            for intent in possible_intents:
                suggestions.extend(self.get_suggestions_for_intent(intent, 2))
            
            return {
                "type": "recommendations",
                "content": "ไม่แน่ใจว่าคุณต้องการอะไร ลองดูเมนูเหล่านี้ไหมครับ?",
                "recommendations": suggestions[:max_results]
            }
        
        # ลองค้นหาคำใกล้เคียงในชื่อเมนู
        similar_menus = self.find_similar_menu_names(query, 3)
        
        if similar_menus:
            suggestions = []
            for menu_name in similar_menus:
                row = self.data[self.data['name'] == menu_name].iloc[0]
                nutrition = self._get_recipe_nutrition(row)
                suggestions.append({
                    "name": menu_name,
                    "reason": "ชื่อใกล้เคียงกับที่ค้นหา",
                    "nutrition": nutrition,
                    "ingredient": row.get('ingredient', ''),
                    "method": row.get('method', '')
                })
            
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

    def analyze_fallback_query(self, query: str) -> List[str]:
        """วิเคราะห์คำถามที่ไม่เข้าใจเพื่อหาความเป็นไปได้"""
        
        possible_intents = []
        
        # ตรวจสอบคำสำคัญต่างๆ
        if any(word in query for word in ['ผัก', 'เจ', 'มังสวิรัติ']):
            possible_intents.append('vegetarian')
        
        if any(word in query for word in ['เผ็ด', 'รสจัด']):
            possible_intents.append('spicy')
        
        if any(word in query for word in ['ง่าย', 'ไม่ยาก']):
            possible_intents.append('easy')
        
        if any(word in query for word in ['เร็ว', 'ไว', 'รีบ']):
            possible_intents.append('quick')
        
        if any(word in query for word in ['สุขภาพ', 'ดีต่อสุขภาพ']):
            possible_intents.append('healthy')
        
        return possible_intents

    def get_suggestions_for_intent(self, intent: str, count: int) -> List[Dict]:
        """ให้คำแนะนำตามความเป็นไปได้ที่วิเคราะห์ได้"""
        
        suggestions = []
        
        for _, row in self.data.iterrows():
            match = False
            reason = ""
            
            if intent == 'vegetarian':
                ingredient = row['ingredient'].lower()
                if not any(meat in ingredient for meat in ['หมู', 'ไก่', 'เนื้อ', 'กุ้ง', 'ปลา']):
                    match = True
                    reason = "อาหารจากผัก"
            
            elif intent == 'spicy':
                ingredient = row['ingredient'].lower()
                method = row['method'].lower()
                if any(spice in ingredient + method for spice in ['พริก', 'เผ็ด']):
                    match = True
                    reason = "รสเผ็ด"
            
            elif intent == 'easy':
                ingredient_count = len(row['ingredient'].split('\n'))
                if ingredient_count <= 6:
                    match = True
                    reason = "ทำง่าย"
            
            elif intent == 'quick':
                method = row['method'].lower()
                if any(quick_method in method for quick_method in ['ผัด', 'เจียว', 'ลวก']):
                    match = True
                    reason = "ทำได้เร็ว"
            
            elif intent == 'healthy':
                nutrition = self._get_recipe_nutrition(row)
                if (nutrition.get('calories', 999) < 300 and 
                    nutrition.get('sodium', 999) < 800 and
                    nutrition.get('fiber', 0) > 2):
                    match = True
                    reason = "เพื่อสุขภาพ"
            
            if match:
                nutrition = self._get_recipe_nutrition(row)
                suggestions.append({
                    "name": row['name'],
                    "reason": reason,
                    "nutrition": nutrition,
                    "ingredient": row.get('ingredient', ''),
                    "method": row.get('method', '')
                })
                
                if len(suggestions) >= count:
                    break
        
        return suggestions

    def _get_recipe_nutrition(self, row) -> Optional[Dict]:
        """ดึงข้อมูลโภชนาการของเมนู - ช่วยให้โค้ดสั้นลง"""
        
        # ตรวจสอบว่ามีข้อมูลโภชนาการในคอลัมน์หรือไม่
        nutrition_columns = [
            'calories', 'protein', 'carbs', 'fat', 'fiber',
            'vitamin_a', 'vitamin_c', 'vitamin_b1', 'vitamin_b2',
            'calcium', 'iron', 'potassium', 'sodium'
        ]
        
        nutrition_data = {}
        has_nutrition = False
        
        for col in nutrition_columns:
            if hasattr(row, col) and pd.notna(getattr(row, col, None)):
                nutrition_data[col] = float(getattr(row, col))
                if getattr(row, col) > 0:
                    has_nutrition = True
            else:
                nutrition_data[col] = 0.0
        
        if has_nutrition:
            return nutrition_data
        
        # หากไม่มีข้อมูลในคอลัมน์ ให้คำนวณจากวัตถุดิบ
        try:
            calculated_nutrition = self.nutrition_api.calculate_recipe_nutrition(
                row['ingredient'] if hasattr(row, 'ingredient') else '', 
                use_api=False, 
                adjust_consumption=True
            )
            return calculated_nutrition.get('total_nutrition', nutrition_data)
        except:
            return nutrition_data

    def get_balanced_nutrition_recommendations(self, max_results: int) -> List[Dict]:
        """แนะนำเมนูที่มีโภชนาการสมดุล - เวอร์ชันปรับปรุง"""
        
        recommendations = []
        
        for _, row in self.data.iterrows():
            nutrition = self._get_recipe_nutrition(row)
            if not nutrition:
                continue
            
            # คำนวณคะแนนความสมดุลแบบละเอียด
            balance_score = self._calculate_detailed_balance_score(nutrition)
            
            if balance_score >= 6:  # เกณฑ์ความสมดุลที่เข้มงวดขึ้น
                recommendations.append({
                    'name': row['name'],
                    'reason': f"โภชนาการสมดุลดีเยี่ยม (คะแนน {balance_score}/10)",
                    'nutrition': nutrition,
                    'balance_score': balance_score,
                    'ingredient': row.get('ingredient', ''),
                    'method': row.get('method', '')
                })
        
        recommendations.sort(key=lambda x: x['balance_score'], reverse=True)
        return recommendations[:max_results]

    def _calculate_detailed_balance_score(self, nutrition: Dict) -> float:
        """คำนวณคะแนนความสมดุลแบบละเอียด"""
        
        score = 0
        
        # คะแนนจากแคลอรี่ที่เหมาะสม (200-400 kcal)
        calories = nutrition.get('calories', 0)
        if 250 <= calories <= 350:
            score += 2
        elif 200 <= calories < 250 or 350 < calories <= 400:
            score += 1.5
        elif 150 <= calories < 200 or 400 < calories <= 500:
            score += 1
        
        # คะแนนจากโปรตีนที่เพียงพอ (15-30g)
        protein = nutrition.get('protein', 0)
        if 20 <= protein <= 25:
            score += 2
        elif 15 <= protein < 20 or 25 < protein <= 30:
            score += 1.5
        elif 10 <= protein < 15 or 30 < protein <= 35:
            score += 1
        
        # คะแนนจากไขมันที่เหมาะสม (8-20g)
        fat = nutrition.get('fat', 0)
        if 10 <= fat <= 15:
            score += 1.5
        elif 8 <= fat < 10 or 15 < fat <= 20:
            score += 1
        elif 5 <= fat < 8 or 20 < fat <= 25:
            score += 0.5
        
        # คะแนนจากคาร์โบไฮเดรตที่เหมาะสม (15-40g)
        carbs = nutrition.get('carbs', 0)
        if 20 <= carbs <= 30:
            score += 1.5
        elif 15 <= carbs < 20 or 30 < carbs <= 40:
            score += 1
        elif 10 <= carbs < 15 or 40 < carbs <= 50:
            score += 0.5
        
        # คะแนนจากใยอาหาร (>3g)
        fiber = nutrition.get('fiber', 0)
        if fiber >= 5:
            score += 1.5
        elif fiber >= 3:
            score += 1
        elif fiber >= 2:
            score += 0.5
        
        # คะแนนจากวิตามินซี (>15mg)
        vitamin_c = nutrition.get('vitamin_c', 0)
        if vitamin_c >= 25:
            score += 1
        elif vitamin_c >= 15:
            score += 0.5
        
        # คะแนนจากแคลเซียม (>100mg)
        calcium = nutrition.get('calcium', 0)
        if calcium >= 150:
            score += 1
        elif calcium >= 100:
            score += 0.5
        
        # คะแนนจากเหล็ก (>2mg)
        iron = nutrition.get('iron', 0)
        if iron >= 3:
            score += 1
        elif iron >= 2:
            score += 0.5
        
        # หักคะแนนจากโซเดียมสูง
        sodium = nutrition.get('sodium', 0)
        if sodium > 1500:
            score -= 2
        elif sodium > 1200:
            score -= 1.5
        elif sodium > 1000:
            score -= 1
        elif sodium > 800:
            score -= 0.5
        
        return max(score, 0)

    def get_health_conscious_recommendations(self, max_results: int) -> List[Dict]:
        """แนะนำเมนูเพื่อสุขภาพ - เวอร์ชันปรับปรุง"""
        
        recommendations = []
        
        for _, row in self.data.iterrows():
            nutrition = self._get_recipe_nutrition(row)
            if not nutrition:
                continue
            
            # คำนวณคะแนนสุขภาพแบบละเอียด
            health_score = self._calculate_detailed_health_score(nutrition)
            
            if health_score >= 5:  # เกณฑ์สุขภาพที่เข้มงวดขึ้น
                recommendations.append({
                    'name': row['name'],
                    'reason': f"เพื่อสุขภาพ (คะแนน {health_score}/8)",
                    'nutrition': nutrition,
                    'health_score': health_score,
                    'ingredient': row.get('ingredient', ''),
                    'method': row.get('method', '')
                })
        
        recommendations.sort(key=lambda x: x['health_score'], reverse=True)
        return recommendations[:max_results]

    def _calculate_detailed_health_score(self, nutrition: Dict) -> float:
        """คำนวณคะแนนสุขภาพแบบละเอียด"""
        
        score = 0
        
        # แคลอรี่ไม่สูงเกินไป
        calories = nutrition.get('calories', 0)
        if calories <= 250:
            score += 2
        elif calories <= 350:
            score += 1
        
        # ไขมันต่ำ
        fat = nutrition.get('fat', 0)
        if fat <= 8:
            score += 2
        elif fat <= 15:
            score += 1
        
        # โซเดียมต่ำ
        sodium = nutrition.get('sodium', 0)
        if sodium <= 600:
            score += 2
        elif sodium <= 1000:
            score += 1
        
        # ใยอาหารสูง
        fiber = nutrition.get('fiber', 0)
        if fiber >= 4:
            score += 1.5
        elif fiber >= 3:
            score += 1
        elif fiber >= 2:
            score += 0.5
        
        # วิตามินซีสูง
        vitamin_c = nutrition.get('vitamin_c', 0)
        if vitamin_c >= 20:
            score += 1
        elif vitamin_c >= 10:
            score += 0.5
        
        # แคลเซียมสูง
        calcium = nutrition.get('calcium', 0)
        if calcium >= 150:
            score += 1
        elif calcium >= 100:
            score += 0.5
        
        # เหล็กสูง
        iron = nutrition.get('iron', 0)
        if iron >= 3:
            score += 1
        elif iron >= 2:
            score += 0.5
        
        # โปรตีนเพียงพอ
        protein = nutrition.get('protein', 0)
        if protein >= 15:
            score += 0.5
        
        return score

    def get_popular_recipes(self, max_results: int) -> List[Dict]:
        """แนะนำเมนูยอดนิยม - เวอร์ชันปรับปรุง"""
        
        # เมนูยอดนิยมที่มีโภชนาการดี
        popular_criteria = [
            'ผัดกะเพราหมูสับ', 'ต้มยำกุ้งน้ำใส', 'ส้มตำไทย', 
            'แกงเขียวหวานไก่', 'ผัดไทยกุ้งสด', 'ไข่เจียวฟู',
            'ข้าวผัดกุ้ง', 'ยำวุ้นเส้นทะเล', 'ลาบหมูอีสาน'
        ]
        
        recommendations = []
        
        for name in popular_criteria:
            # หาเมนูที่ใกล้เคียงในฐานข้อมูล
            best_match = self.find_best_matching_recipe(name)
            if best_match is not None:
                nutrition = self._get_recipe_nutrition(best_match)
                
                recommendations.append({
                    'name': best_match['name'],
                    'reason': 'เมนูยอดนิยมที่คนไทยชื่นชอบ',
                    'nutrition': nutrition,
                    'ingredient': best_match.get('ingredient', ''),
                    'method': best_match.get('method', '')
                })
                
                if len(recommendations) >= max_results:
                    break
        
        return recommendations

    def find_similar_menu_names(self, query: str, max_results: int) -> List[str]:
        """หาชื่อเมนูที่คล้ายคลึง - เวอร์ชันปรับปรุง"""
        
        similarities = []
        
        for _, row in self.data.iterrows():
            similarity = SequenceMatcher(None, query.lower(), row['name'].lower()).ratio()
            
            # ตรวจสอบการตรงกันของคำเฉพาะ
            query_words = query.lower().split()
            name_words = row['name'].lower().split()
            
            word_match_score = 0
            for q_word in query_words:
                for n_word in name_words:
                    if q_word in n_word or n_word in q_word:
                        word_match_score += 1
                        break
            
            # รวมคะแนนความคล้ายคลึง
            final_score = (similarity * 0.7) + (word_match_score / max(len(query_words), 1) * 0.3)
            
            if final_score > 0.3:
                similarities.append((row['name'], final_score))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in similarities[:max_results]]

    def extract_comparison_menus(self, query: str) -> List[str]:
        """ดึงชื่อเมนูที่ต้องการเปรียบเทียบ - เวอร์ชันปรับปรุง"""
        
        # รูปแบบการเปรียบเทียบที่หลากหลายขึ้น
        patterns = [
            r'(\S+)\s*กับ\s*(\S+)',
            r'(\S+)\s*และ\s*(\S+)',
            r'(\S+)\s*หรือ\s*(\S+)',
            r'(\S+)\s*เปรียบเทียบ\s*(\S+)',
            r'(\S+)\s*vs\s*(\S+)',
            r'(\S+)\s*ดีกว่า\s*(\S+)',
            r'เลือก\s*(\S+)\s*หรือ\s*(\S+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                menu1, menu2 = match.groups()
                
                # ทำความสะอาดชื่อเมนู
                menu1 = self._clean_menu_name_for_comparison(menu1)
                menu2 = self._clean_menu_name_for_comparison(menu2)
                
                # ตรวจสอบว่าเป็นชื่อเมนูจริงไหม
                if (self.find_similar_menu_names(menu1, 1) and 
                    self.find_similar_menu_names(menu2, 1)):
                    return [menu1, menu2]
        
        return []

    def _clean_menu_name_for_comparison(self, menu_name: str) -> str:
        """ทำความสะอาดชื่อเมนูสำหรับการเปรียบเทียบ"""
        
        # ลบคำที่ไม่จำเป็น
        stop_words = ['อะไร', 'ไหน', 'ใด', 'อัน', 'ตัว', 'ครับ', 'ค่ะ']
        
        for word in stop_words:
            menu_name = menu_name.replace(word, '')
        
        return menu_name.strip()

    def get_nutrition_summary_for_query(self, query: str) -> Optional[str]:
        """สร้างสรุปโภชนาการสำหรับคำถาม"""
        
        # ตรวจสอบว่ามีการถามเรื่องโภชนาการเฉพาะหรือไม่
        nutrition_summaries = {
            'แคลอรี่': "💡 แคลอรี่เป็นหน่วยวัดพลังงานในอาหาร ผู้ใหญ่ควรได้รับ 1,800-2,200 แคลอรี่ต่อวัน",
            'โปรตีน': "💪 โปรตีนช่วยสร้างและซ่อมแซมกล้ามเนื้อ ควรได้รับ 0.8-1.2 กรัมต่อน้ำหนัก 1 กิโลกรัม",
            'ไขมัน': "🫒 ไขมันให้พลังงานและช่วยดูดซึมวิตามิน ควรเป็น 20-35% ของแคลอรี่รวม",
            'คาร์โบไฮเดรต': "🍞 คาร์โบไฮเดรตเป็นแหล่งพลังงานหลัก ควรเป็น 45-65% ของแคลอรี่รวม",
            'ใยอาหาร': "🌾 ใยอาหารช่วยระบบย่อยและลดคอเลสเตอรอล ควรได้รับ 25-30 กรัมต่อวัน",
            'วิตามินซี': "🍊 วิตามินซีเสริมภูมิคุ้มกันและช่วยดูดซึมเหล็ก ควรได้รับ 75-90 มก.ต่อวัน",
            'แคลเซียม': "🦴 แคลเซียมสำคัญต่อกระดูกและฟัน ควรได้รับ 1,000-1,200 มก.ต่อวัน",
            'เหล็ก': "🩸 เหล็กช่วยสร้างเม็ดเลือดแดงและขนส่งออกซิเจน ควรได้รับ 8-18 มก.ต่อวัน",
            'โซเดียม': "🧂 โซเดียมควบคุมสมดุลน้ำในร่างกาย แต่ควรจำกัดไม่เกิน 2,300 มก.ต่อวัน"
        }
        
        for nutrient, summary in nutrition_summaries.items():
            if nutrient in query:
                return summary
        
        return None
