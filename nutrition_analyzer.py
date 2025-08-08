import pandas as pd
import numpy as np
import requests
import json
import re
from typing import Dict, List, Tuple, Optional
import time
from functools import lru_cache
import pickle
import os

class ThaiNutritionAnalyzer:
    """
    คลาสสำหรับวิเคราะห์คุณค่าทางโภชนาการของวัตถุดิบอาหารไทย
    """
    
    def __init__(self, cache_file='nutrition_cache.pkl'):
        self.cache_file = cache_file
        self.nutrition_cache = self._load_cache()
        
        # ฐานข้อมูลคุณค่าทางโภชนาการพื้นฐานของวัตถุดิบไทย (ต่อ 100 กรัม)
        self.thai_ingredients_nutrition = {
            'กุ้ง': {
                'พลังงาน': 99, 'โปรตีน': 20.1, 'ไขมัน': 1.7,
                'คาร์โบไฮเดรต': 0.9, 'แคลเซียม': 54, 'ฟอสฟอรัส': 244,
                'เหล็ก': 2.1, 'วิตามินเอ': 180, 'วิตามินบี1': 0.03,
                'วิตามินบี2': 0.05, 'ไนอาซิน': 3.0
            },
            'หมู': {
                'พลังงาน': 242, 'โปรตีน': 16.9, 'ไขมัน': 19.2,
                'คาร์โบไฮเดรต': 0, 'แคลเซียม': 11, 'ฟอสฟอรัส': 198,
                'เหล็ก': 0.9, 'วิตามินเอ': 7, 'วิตามินบี1': 0.83,
                'วิตามินบี2': 0.25, 'ไนอาซิน': 4.8
            },
            'ไก่': {
                'พลังงาน': 165, 'โปรตีน': 31.0, 'ไขมัน': 3.6,
                'คาร์โบไฮเดรต': 0, 'แคลเซียม': 15, 'ฟอสฟอรัส': 228,
                'เหล็ก': 1.0, 'วิตามินเอ': 41, 'วิตามินบี1': 0.08,
                'วิตามินบี2': 0.16, 'ไนอาซิน': 8.9
            },
            'ปลา': {
                'พลังงาน': 124, 'โปรตีน': 20.4, 'ไขมัน': 4.5,
                'คาร์โบไฮเดรต': 0, 'แคลเซียม': 20, 'ฟอสฟอรัส': 200,
                'เหล็ก': 0.8, 'วิตามินเอ': 30, 'วิตามินบี1': 0.05,
                'วิตามินบี2': 0.10, 'ไนอาซิน': 2.8
            },
            'ไข่': {
                'พลังงาน': 155, 'โปรตีน': 12.6, 'ไขมัน': 10.6,
                'คาร์โบไฮเดรต': 1.1, 'แคลเซียม': 56, 'ฟอสฟอรัส': 180,
                'เหล็ก': 1.8, 'วิตามินเอ': 540, 'วิตามินบี1': 0.10,
                'วิตามินบี2': 0.47, 'ไนอาซิน': 0.1
            },
            'พริกไทย': {
                'พลังงาน': 255, 'โปรตีน': 10.4, 'ไขมัน': 3.3,
                'คาร์โบไฮเดรต': 64.8, 'แคลเซียม': 443, 'ฟอสฟอรัส': 173,
                'เหล็ก': 9.7, 'วิตามินเอ': 370, 'วิตามินบี1': 0.10,
                'วิตามินบี2': 0.18, 'ไนอาซิน': 1.1
            },
            'กระเทียม': {
                'พลังงาน': 149, 'โปรตีน': 6.4, 'ไขมัน': 0.5,
                'คาร์โบไฮเดรต': 33.1, 'แคลเซียม': 181, 'ฟอสฟอรัส': 153,
                'เหล็ก': 1.7, 'วิตามินเอ': 0, 'วิตามินบี1': 0.20,
                'วิตามินบี2': 0.11, 'ไนอาซิน': 0.7, 'วิตามินซี': 31
            },
            'หอม': {
                'พลังงาน': 40, 'โปรตีน': 1.1, 'ไขมัน': 0.1,
                'คาร์โบไฮเดรต': 9.3, 'แคลเซียม': 23, 'ฟอสฟอรัส': 29,
                'เหล็ก': 0.2, 'วิตามินเอ': 0, 'วิตามินบี1': 0.05,
                'วิตามินบี2': 0.03, 'ไนอาซิน': 0.1, 'วิตามินซี': 7
            },
            'ข่า': {
                'พลังงาน': 71, 'โปรตีน': 1.8, 'ไขมัน': 0.7,
                'คาร์โบไฮเดรต': 15.5, 'แคลเซียม': 16, 'ฟอสฟอรัส': 36,
                'เหล็ก': 1.8, 'วิตามินเอ': 0, 'วิตามินบี1': 0.03,
                'วิตามินบี2': 0.03, 'ไนอาซิน': 0.6
            },
            'ตะไคร้': {
                'พลังงาน': 99, 'โปรตีน': 1.8, 'ไขมัน': 0.5,
                'คาร์โบไฮเดรต': 25.3, 'แคลเซียม': 65, 'ฟอสฟอรัส': 101,
                'เหล็ก': 8.2, 'วิตามินเอ': 0, 'วิตามินบี1': 0.07,
                'วิตามินบี2': 0.14, 'ไนอาซิน': 1.1
            },
            'ใบมะกรูด': {
                'พลังงาน': 70, 'โปรตีน': 1.5, 'ไขมัน': 0.6,
                'คาร์โบไฮเดรต': 16.8, 'แคลเซียม': 629, 'ฟอสฟอรัส': 63,
                'เหล็ก': 0.8, 'วิตามินเอ': 0, 'วิตามินบี1': 0.01,
                'วิตามินบี2': 0.26, 'ไนอาซิน': 0.9, 'วิตามินซี': 30
            },
            'น้ำปลา': {
                'พลังงาน': 25, 'โปรตีน': 4.0, 'ไขมัน': 0,
                'คาร์โบไฮเดรต': 2.0, 'แคลเซียม': 20, 'ฟอสฟอรัส': 30,
                'เหล็ก': 0.5, 'โซเดียม': 5720
            },
            'มะพร้าว': {
                'พลังงาน': 354, 'โปรตีน': 3.3, 'ไขมัน': 33.5,
                'คาร์โบไฮเดรต': 15.2, 'แคลเซียม': 14, 'ฟอสฟอรัส': 113,
                'เหล็ก': 2.4, 'วิตามินเอ': 0, 'วิตามินบี1': 0.07,
                'วิตามินบี2': 0.02, 'ไนอาซิน': 0.5, 'วิตามินซี': 3
            },
            'ผักชี': {
                'พลังงาน': 23, 'โปรตีน': 2.1, 'ไขมัน': 0.5,
                'คาร์โบไฮเดรต': 3.7, 'แคลเซียม': 67, 'ฟอสฟอรัส': 48,
                'เหล็ก': 1.8, 'วิตามินเอ': 6748, 'วิตามินบี1': 0.07,
                'วิตามินบี2': 0.16, 'ไนอาซิน': 1.1, 'วิตามินซี': 27
            },
            'แป้ง': {
                'พลังงาน': 364, 'โปรตีน': 10.3, 'ไขมัน': 1.0,
                'คาร์โบไฮเดรต': 76.3, 'แคลเซียม': 18, 'ฟอสฟอรัส': 108,
                'เหล็ก': 1.2, 'วิตามินเอ': 0, 'วิตามินบี1': 0.12,
                'วิตามินบี2': 0.04, 'ไนอาซิน': 1.3
            },
            'น้ำมันหมู': {
                'พลังงาน': 900, 'โปรตีน': 0, 'ไขมัน': 100,
                'คาร์โบไฮเดรต': 0, 'แคลเซียม': 0, 'ฟอสฟอรัส': 0,
                'เหล็ก': 0, 'วิตามินเอ': 0, 'วิตามินบี1': 0,
                'วิตามินบี2': 0, 'ไนอาซิน': 0
            }
        }
        
        # แมปชื่อวัตถุดิบภาษาไทยกับภาษาอังกฤษ
        self.ingredient_mapping = {
            'กุ้ง': ['shrimp', 'prawn'],
            'หมู': ['pork'],
            'ไก่': ['chicken'],
            'ปลา': ['fish'],
            'ไข่': ['egg'],
            'พริกไทย': ['black pepper', 'pepper'],
            'กระเทียม': ['garlic'],
            'หอม': ['onion', 'shallot'],
            'ข่า': ['galangal'],
            'ตะไคร้': ['lemongrass'],
            'ใบมะกรูด': ['kaffir lime leaves'],
            'น้ำปลา': ['fish sauce'],
            'มะพร้าว': ['coconut'],
            'ผักชี': ['coriander', 'cilantro'],
            'แป้ง': ['flour'],
            'น้ำมันหมู': ['lard', 'pork fat']
        }
        
    def _load_cache(self) -> Dict:
        """โหลดแคชข้อมูลคุณค่าทางโภชนาการ"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """บันทึกแคชข้อมูลคุณค่าทางโภชนาการ"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.nutrition_cache, f)
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการบันทึกแคช: {e}")
    
    def extract_ingredients(self, ingredient_text: str) -> List[Tuple[str, float, str]]:
        """
        แยกวัตถุดิบและปริมาณจากข้อความ
        
        Args:
            ingredient_text: ข้อความรายการวัตถุดิบ
            
        Returns:
            List ของ (ชื่อวัตถุดิบ, ปริมาณ, หน่วย)
        """
        ingredients = []
        lines = ingredient_text.strip().split('\n')
        
        for line in lines:
            if line.strip() and line.strip().startswith('-'):
                # ลบเครื่องหมาย - ออก
                ingredient_line = line.strip()[1:].strip()
                
                # แยกชื่อวัตถุดิบและปริมาณ
                # ค้นหาตัวเลขในบรรทัด
                numbers = re.findall(r'(\d+(?:\.\d+)?)\s*(\w+)', ingredient_line)
                
                if numbers:
                    # มีปริมาณระบุ
                    quantity = float(numbers[0][0])
                    unit = numbers[0][1]
                    # ตัดส่วนปริมาณออกเพื่อเหลือแต่ชื่อ
                    name = re.sub(r'\d+(?:\.\d+)?\s*\w+', '', ingredient_line).strip()
                else:
                    # ไม่มีปริมาณระบุ
                    quantity = 100  # ค่าเริ่มต้น 100 กรัม
                    unit = 'กรัม'
                    name = ingredient_line
                
                # ทำความสะอาดชื่อวัตถุดิบ
                name = re.sub(r'\s+', ' ', name).strip()
                
                if name:
                    ingredients.append((name, quantity, unit))
        
        return ingredients
    
    def get_base_ingredient(self, ingredient_name: str) -> Optional[str]:
        """
        หาวัตถุดิบพื้นฐานจากชื่อวัตถุดิบ
        
        Args:
            ingredient_name: ชื่อวัตถุดิบ
            
        Returns:
            ชื่อวัตถุดิบพื้นฐาน หรือ None ถ้าไม่พบ
        """
        ingredient_lower = ingredient_name.lower()
        
        # ค้นหาวัตถุดิบพื้นฐาน
        for base, keywords in self.thai_ingredients_nutrition.items():
            if base in ingredient_name:
                return base
                
        # ค้นหาคำสำคัญ
        ingredient_keywords = {
            'กุ้ง': ['กุ้ง', 'กุ้งนาง', 'กุ้งแห้ง', 'กุ้งตะเข็บ'],
            'หมู': ['หมู', 'เนื้อหมู', 'หมูสับ', 'หมูบด'],
            'ไก่': ['ไก่', 'เนื้อไก่', 'อกไก่', 'น่องไก่'],
            'ปลา': ['ปลา', 'ปลาทู', 'ปลาช่อน', 'ปลากะพง'],
            'ไข่': ['ไข่', 'ไข่เป็ด', 'ไข่ไก่', 'ไข่นก'],
            'พริกไทย': ['พริกไทย', 'พริกไทยดำ', 'พริกไทยขาว'],
            'กระเทียม': ['กระเทียม', 'กระเทียมดอง'],
            'หอม': ['หอม', 'หอมแดง', 'หอมใหญ่', 'ต้นหอม'],
            'ผักชี': ['ผักชี', 'ผักชีฝรั่ง', 'รากผักชี']
        }
        
        for base, keywords in ingredient_keywords.items():
            for keyword in keywords:
                if keyword in ingredient_name:
                    return base
                    
        return None
    
    def calculate_nutrition(self, ingredient_name: str, quantity: float, unit: str) -> Dict[str, float]:
        """
        คำนวณคุณค่าทางโภชนาการของวัตถุดิบ
        
        Args:
            ingredient_name: ชื่อวัตถุดิบ
            quantity: ปริมาณ
            unit: หน่วย
            
        Returns:
            Dictionary ของคุณค่าทางโภชนาการ
        """
        # หาวัตถุดิบพื้นฐาน
        base_ingredient = self.get_base_ingredient(ingredient_name)
        
        if base_ingredient and base_ingredient in self.thai_ingredients_nutrition:
            base_nutrition = self.thai_ingredients_nutrition[base_ingredient].copy()
            
            # แปลงหน่วยเป็นกรัม
            quantity_in_grams = self.convert_to_grams(quantity, unit)
            
            # คำนวณคุณค่าทางโภชนาการตามปริมาณ
            factor = quantity_in_grams / 100  # ข้อมูลพื้นฐานต่อ 100 กรัม
            
            nutrition = {}
            for nutrient, value in base_nutrition.items():
                nutrition[nutrient] = round(value * factor, 2)
                
            return nutrition
        
        return {}
    
    def convert_to_grams(self, quantity: float, unit: str) -> float:
        """
        แปลงหน่วยเป็นกรัม
        
        Args:
            quantity: ปริมาณ
            unit: หน่วย
            
        Returns:
            ปริมาณในหน่วยกรัม
        """
        # การแปลงหน่วยพื้นฐาน
        unit_conversions = {
            'กรัม': 1,
            'กก.': 1000,
            'กิโลกรัม': 1000,
            'ขีด': 100,
            'ช้อนโต๊ะ': 15,
            'ช้อนชา': 5,
            'ถ้วย': 240,
            'ถ้วยชา': 200,
            'ฟอง': 50,  # สำหรับไข่
            'ตัว': 100,  # สำหรับกุ้ง
            'แผ่น': 50,  # ค่าเฉลี่ย
            'หัว': 30,   # สำหรับหอม/กระเทียม
            'กลีบ': 3,   # สำหรับกระเทียม
            'ต้น': 10,   # สำหรับผักชี/ต้นหอม
            'ใบ': 1,     # สำหรับใบต่างๆ
            'ผล': 100,  # ค่าเฉลี่ย
            'ฝัก': 20,   # ค่าเฉลี่ย
        }
        
        # ค้นหาหน่วยที่ตรงกัน
        for unit_key, factor in unit_conversions.items():
            if unit_key in unit:
                return quantity * factor
                
        # ถ้าไม่พบหน่วย ใช้ค่าเริ่มต้น 100 กรัม
        return quantity * 100
    
    @lru_cache(maxsize=128)
    def fetch_nutrition_from_api(self, ingredient_name: str) -> Optional[Dict]:
        """
        ดึงข้อมูลคุณค่าทางโภชนาการจาก API ภายนอก (USDA Food Data Central)
        
        Args:
            ingredient_name: ชื่อวัตถุดิบ
            
        Returns:
            Dictionary ของคุณค่าทางโภชนาการ หรือ None ถ้าไม่พบ
        """
        # ตรวจสอบแคช
        if ingredient_name in self.nutrition_cache:
            return self.nutrition_cache[ingredient_name]
            
        # สำหรับ demo จะใช้ข้อมูลจากฐานข้อมูลภายในก่อน
        # ในการใช้งานจริงสามารถเชื่อมต่อกับ API เช่น USDA Food Data Central
        # API_KEY = 'YOUR_API_KEY'
        # url = f"https://api.nal.usda.gov/fdc/v1/foods/search?query={ingredient_name}&api_key={API_KEY}"
        
        # ค้นหาชื่อภาษาอังกฤษ
        english_names = []
        for thai, eng_list in self.ingredient_mapping.items():
            if thai in ingredient_name:
                english_names.extend(eng_list)
                
        # จำลองการดึงข้อมูลจาก API
        # ในการใช้งานจริงจะต้องเรียก API จริง
        # try:
        #     response = requests.get(url, timeout=5)
        #     if response.status_code == 200:
        #         data = response.json()
        #         # ประมวลผลข้อมูลจาก API
        #         # ...
        # except:
        #     pass
        
        return None
    
    def analyze_recipe_nutrition(self, ingredients_text: str) -> Dict:
        """
        วิเคราะห์คุณค่าทางโภชนาการทั้งหมดของสูตรอาหาร
        
        Args:
            ingredients_text: ข้อความรายการวัตถุดิบ
            
        Returns:
            Dictionary ของคุณค่าทางโภชนาการรวม
        """
        # แยกวัตถุดิบ
        ingredients = self.extract_ingredients(ingredients_text)
        
        # คำนวณคุณค่าทางโภชนาการรวม
        total_nutrition = {}
        ingredient_details = []
        
        for name, quantity, unit in ingredients:
            nutrition = self.calculate_nutrition(name, quantity, unit)
            
            if nutrition:
                # เพิ่มคุณค่าทางโภชนาการรวม
                for nutrient, value in nutrition.items():
                    if nutrient not in total_nutrition:
                        total_nutrition[nutrient] = 0
                    total_nutrition[nutrient] += value
                    
                # เก็บรายละเอียดแต่ละวัตถุดิบ
                ingredient_details.append({
                    'name': name,
                    'quantity': quantity,
                    'unit': unit,
                    'nutrition': nutrition
                })
        
        # ปัดเศษค่าคุณค่าทางโภชนาการรวม
        for nutrient in total_nutrition:
            total_nutrition[nutrient] = round(total_nutrition[nutrient], 2)
            
        return {
            'total_nutrition': total_nutrition,
            'ingredients': ingredient_details
        }
    
    def get_nutrition_summary(self, nutrition_data: Dict) -> str:
        """
        สร้างสรุปคุณค่าทางโภชนาการในรูปแบบข้อความ
        
        Args:
            nutrition_data: ข้อมูลคุณค่าทางโภชนาการ
            
        Returns:
            ข้อความสรุปคุณค่าทางโภชนาการ
        """
        total = nutrition_data.get('total_nutrition', {})
        
        if not total:
            return "ไม่มีข้อมูลคุณค่าทางโภชนาการ"
            
        summary = "📊 **คุณค่าทางโภชนาการโดยประมาณ:**\n\n"
        
        # สารอาหารหลัก
        if 'พลังงาน' in total:
            summary += f"⚡ พลังงาน: {total['พลังงาน']} แคลอรี\n"
        if 'โปรตีน' in total:
            summary += f"🥩 โปรตีน: {total['โปรตีน']} กรัม\n"
        if 'ไขมัน' in total:
            summary += f"🧈 ไขมัน: {total['ไขมัน']} กรัม\n"
        if 'คาร์โบไฮเดรต' in total:
            summary += f"🌾 คาร์โบไฮเดรต: {total['คาร์โบไฮเดรต']} กรัม\n"
            
        # แร่ธาตุ
        summary += "\n**แร่ธาตุ:**\n"
        minerals = ['แคลเซียม', 'ฟอสฟอรัส', 'เหล็ก', 'โซเดียม']
        for mineral in minerals:
            if mineral in total and total[mineral] > 0:
                summary += f"• {mineral}: {total[mineral]} มก.\n"
                
        # วิตามิน
        summary += "\n**วิตามิน:**\n"
        vitamins = ['วิตามินเอ', 'วิตามินบี1', 'วิตามินบี2', 'ไนอาซิน', 'วิตามินซี']
        for vitamin in vitamins:
            if vitamin in total and total[vitamin] > 0:
                unit = 'IU' if vitamin == 'วิตามินเอ' else 'มก.'
                summary += f"• {vitamin}: {total[vitamin]} {unit}\n"
                
        return summary
    
    def add_new_ingredient_nutrition(self, ingredient_name: str, nutrition_data: Dict):
        """
        เพิ่มข้อมูลคุณค่าทางโภชนาการของวัตถุดิบใหม่
        
        Args:
            ingredient_name: ชื่อวัตถุดิบ
            nutrition_data: ข้อมูลคุณค่าทางโภชนาการ
        """
        self.thai_ingredients_nutrition[ingredient_name] = nutrition_data
        self.nutrition_cache[ingredient_name] = nutrition_data
        self._save_cache()
