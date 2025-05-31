import requests
import json
import pandas as pd
import re
from typing import Dict, List, Optional, Tuple
import streamlit as st

class NutritionAPI:
    """คลาสสำหรับจัดการข้อมูลโภชนาการจาก API และฐานข้อมูลท้องถิ่น"""
    
    def __init__(self):
        # URL ของ API ข้อมูลโภชนาการ (สามารถใช้ USDA FoodData Central API)
        self.api_url = "https://api.nal.usda.gov/fdc/v1"
        self.api_key = None  # จะตั้งค่าจากการตั้งค่าของผู้ใช้
        
        # ฐานข้อมูลโภชนาการท้องถิ่น
        self.local_nutrition_db = {
            # ข้อมูลโภชนาการสำหรับวัตถุดิบไทย (ต่อ 100 กรัม)
            "ข้าว": {
                "calories": 130, "protein": 2.7, "carbs": 28, "fat": 0.3, "fiber": 0.4,
                "vitamin_a": 0, "vitamin_c": 0, "vitamin_b1": 0.07, "vitamin_b2": 0.02,
                "calcium": 10, "iron": 0.8, "potassium": 115, "sodium": 5
            },
            "ไข่ไก่": {
                "calories": 155, "protein": 13, "carbs": 1.1, "fat": 11, "fiber": 0,
                "vitamin_a": 540, "vitamin_c": 0, "vitamin_b1": 0.04, "vitamin_b2": 0.42,
                "calcium": 56, "iron": 1.75, "potassium": 138, "sodium": 124
            },
            "กุ้ง": {
                "calories": 99, "protein": 18, "carbs": 0.2, "fat": 1.4, "fiber": 0,
                "vitamin_a": 54, "vitamin_c": 2.1, "vitamin_b1": 0.02, "vitamin_b2": 0.04,
                "calcium": 70, "iron": 0.5, "potassium": 259, "sodium": 111
            },
            "หมู": {
                "calories": 242, "protein": 27, "carbs": 0, "fat": 14, "fiber": 0,
                "vitamin_a": 2, "vitamin_c": 0.7, "vitamin_b1": 0.66, "vitamin_b2": 0.23,
                "calcium": 19, "iron": 0.87, "potassium": 423, "sodium": 62
            },
            "น้ำมันหมู": {
                "calories": 902, "protein": 0, "carbs": 0, "fat": 100, "fiber": 0,
                "vitamin_a": 0, "vitamin_c": 0, "vitamin_b1": 0, "vitamin_b2": 0,
                "calcium": 0, "iron": 0, "potassium": 0, "sodium": 0
            },
            "น้ำมันพืช": {
                "calories": 884, "protein": 0, "carbs": 0, "fat": 100, "fiber": 0,
                "vitamin_a": 0, "vitamin_c": 0, "vitamin_b1": 0, "vitamin_b2": 0,
                "calcium": 0, "iron": 0, "potassium": 0, "sodium": 0
            },
            "กระเทียม": {
                "calories": 149, "protein": 6.4, "carbs": 33, "fat": 0.5, "fiber": 2.1,
                "vitamin_a": 9, "vitamin_c": 31, "vitamin_b1": 0.2, "vitamin_b2": 0.11,
                "calcium": 181, "iron": 1.7, "potassium": 401, "sodium": 17
            },
            "หอมใหญ่": {
                "calories": 40, "protein": 1.1, "carbs": 9.3, "fat": 0.1, "fiber": 1.7,
                "vitamin_a": 2, "vitamin_c": 7.4, "vitamin_b1": 0.05, "vitamin_b2": 0.03,
                "calcium": 23, "iron": 0.21, "potassium": 146, "sodium": 4
            },
            "ผักชี": {
                "calories": 23, "protein": 2.1, "carbs": 3.7, "fat": 0.5, "fiber": 2.8,
                "vitamin_a": 3377, "vitamin_c": 27, "vitamin_b1": 0.07, "vitamin_b2": 0.16,
                "calcium": 67, "iron": 1.77, "potassium": 521, "sodium": 46
            },
            "พริกไทย": {
                "calories": 251, "protein": 10.4, "carbs": 64, "fat": 3.3, "fiber": 25,
                "vitamin_a": 547, "vitamin_c": 0, "vitamin_b1": 0.11, "vitamin_b2": 0.18,
                "calcium": 443, "iron": 9.7, "potassium": 1329, "sodium": 20
            },
            "น้ำปลา": {
                "calories": 42, "protein": 5.8, "carbs": 1.5, "fat": 0.8, "fiber": 0,
                "vitamin_a": 0, "vitamin_c": 0, "vitamin_b1": 0.03, "vitamin_b2": 0.22,
                "calcium": 85, "iron": 2.03, "potassium": 84, "sodium": 6976
            },
            "มะนาว": {
                "calories": 29, "protein": 0.7, "carbs": 9.3, "fat": 0.2, "fiber": 2.8,
                "vitamin_a": 22, "vitamin_c": 53, "vitamin_b1": 0.03, "vitamin_b2": 0.02,
                "calcium": 33, "iron": 0.6, "potassium": 138, "sodium": 2
            },
            "กะทิ": {
                "calories": 230, "protein": 2.3, "carbs": 6, "fat": 24, "fiber": 2.2,
                "vitamin_a": 0, "vitamin_c": 2.8, "vitamin_b1": 0.03, "vitamin_b2": 0,
                "calcium": 16, "iron": 1.64, "potassium": 263, "sodium": 15
            },
            "มะพร้าว": {
                "calories": 354, "protein": 3.3, "carbs": 15, "fat": 33, "fiber": 9,
                "vitamin_a": 0, "vitamin_c": 3.3, "vitamin_b1": 0.07, "vitamin_b2": 0.02,
                "calcium": 14, "iron": 2.43, "potassium": 356, "sodium": 20
            },
            "น้ำตาล": {
                "calories": 387, "protein": 0, "carbs": 100, "fat": 0, "fiber": 0,
                "vitamin_a": 0, "vitamin_c": 0, "vitamin_b1": 0, "vitamin_b2": 0,
                "calcium": 1, "iron": 0.01, "potassium": 2, "sodium": 1
            }
        }
        
        # หน่วยแปลงที่พบบ่อย
        self.unit_conversion = {
            "ช้อนโต๊ะ": 15,  # มิลลิลิตร
            "ช้อนชา": 5,
            "ถ้วย": 240,
            "ถ้วยชา": 150,
            "กิโลกรัม": 1000,  # กรัม
            "กก": 1000,
            "ขีด": 15,  # กรัม (หน่วยไทยโบราณ)
            "บาท": 15,  # กรัม
            "ตัว": 100,  # ประมาณการ เช่น กุ้งตัวกลาง
            "ฟอง": 50,  # ไข่ฟองกลาง
            "หัว": 50,  # หอมใหญ่หัวกลาง
            "กลีบ": 3,   # กระเทียม 1 กลีบ
            "ต้น": 30,   # ผักชี 1 ต้น
            "ใบ": 2,     # ใบมะกรูด
            "เม็ด": 0.5, # พริกไทย 1 เม็ด
            "แว่น": 2,   # ข่า 1 แว่น
            "ผล": 150,   # มะนาว 1 ผล
            "ราก": 5     # รากผักชี 1 ราก
        }
        
        # สัดส่วนที่บริโภคจริง (บางวัตถุดิบใช้ในการปรุงแต่ไม่ได้กินหมด)
        self.consumption_ratio = {
            "น้ำมันหมู": 0.3,    # ใช้ทอดแต่ไม่กินหมด
            "น้ำมันพืช": 0.3,     # ใช้ทอดแต่ไม่กินหมด
            "กะทิ": 0.8,          # ใช้ในแกงส่วนใหญ่จะกิน
            "น้ำปลา": 1.0,        # ใช้ปรุงรสกินหมด
            "น้ำตาล": 1.0,        # ใช้ปรุงรสกินหมด
            "เกลือ": 1.0,         # ใช้ปรุงรสกินหมด
            "พริกไทย": 1.0,       # ใช้ปรุงรสกินหมด
        }

    def set_api_key(self, api_key: str):
        """ตั้งค่า API key"""
        self.api_key = api_key

    def search_ingredient_api(self, ingredient: str) -> Optional[Dict]:
        """ค้นหาข้อมูลโภชนาการจาก API"""
        if not self.api_key:
            return None
            
        try:
            # ค้นหาในฐานข้อมูล USDA
            search_url = f"{self.api_url}/foods/search"
            params = {
                "api_key": self.api_key,
                "query": ingredient,
                "dataType": ["Foundation", "SR Legacy"],
                "pageSize": 1
            }
            
            response = requests.get(search_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("foods"):
                    food_id = data["foods"][0]["fdcId"]
                    return self.get_nutrition_details_api(food_id)
            return None
            
        except Exception as e:
            st.warning(f"ไม่สามารถเชื่อมต่อ API: {str(e)}")
            return None

    def get_nutrition_details_api(self, food_id: int) -> Optional[Dict]:
        """ดึงรายละเอียดโภชนาการจาก API"""
        try:
            detail_url = f"{self.api_url}/food/{food_id}"
            params = {"api_key": self.api_key}
            
            response = requests.get(detail_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                nutrients = {}
                
                # แปลงข้อมูลโภชนาการ
                for nutrient in data.get("foodNutrients", []):
                    nutrient_name = nutrient.get("nutrient", {}).get("name", "")
                    value = nutrient.get("amount", 0)
                    
                    # แปลงเป็นรูปแบบที่ใช้ในระบบ
                    if "Energy" in nutrient_name:
                        nutrients["calories"] = value
                    elif "Protein" in nutrient_name:
                        nutrients["protein"] = value
                    elif "Carbohydrate" in nutrient_name:
                        nutrients["carbs"] = value
                    elif "Total lipid" in nutrient_name:
                        nutrients["fat"] = value
                    elif "Fiber" in nutrient_name:
                        nutrients["fiber"] = value
                    elif "Vitamin A" in nutrient_name:
                        nutrients["vitamin_a"] = value
                    elif "Vitamin C" in nutrient_name:
                        nutrients["vitamin_c"] = value
                    elif "Thiamin" in nutrient_name:
                        nutrients["vitamin_b1"] = value
                    elif "Riboflavin" in nutrient_name:
                        nutrients["vitamin_b2"] = value
                    elif "Calcium" in nutrient_name:
                        nutrients["calcium"] = value
                    elif "Iron" in nutrient_name:
                        nutrients["iron"] = value
                    elif "Potassium" in nutrient_name:
                        nutrients["potassium"] = value
                    elif "Sodium" in nutrient_name:
                        nutrients["sodium"] = value
                
                return nutrients
            return None
            
        except Exception as e:
            st.warning(f"ไม่สามารถดึงรายละเอียดจาก API: {str(e)}")
            return None

    def normalize_ingredient_name(self, ingredient: str) -> str:
        """ปรับแต่งชื่อวัตถุดิบให้เป็นมาตรฐาน"""
        # ลบข้อความที่ไม่จำเป็น
        ingredient = re.sub(r'\d+.*', '', ingredient)  # ลบตัวเลขและข้อความที่ตามมา
        ingredient = re.sub(r'[^\u0E00-\u0E7Fa-zA-Z\s]', '', ingredient)  # เก็บเฉพาะตัวอักษรไทย-อังกฤษ
        ingredient = ingredient.strip()
        
        # แปลงคำพ้องความหมาย
        synonyms = {
            "กุ้งนาง": "กุ้ง",
            "กุ้งตะเข็บ": "กุ้ง",
            "เนื้อหมู": "หมู",
            "เนื้อโค": "เนื้อ",
            "เนื้อวัว": "เนื้อ",
            "หอมหัวใหญ่": "หอมใหญ่",
            "หอมหัวเล็ก": "หอมแดง",
            "หัวหอม": "หอมแดง",
            "น้ำมันรำ": "น้ำมันพืช",
            "น้ำมันมะพร้าว": "น้ำมันพืช"
        }
        
        for synonym, standard in synonyms.items():
            if synonym in ingredient:
                ingredient = ingredient.replace(synonym, standard)
        
        return ingredient

    def extract_quantity_and_unit(self, ingredient_text: str) -> Tuple[float, str, str]:
        """แยกปริมาณ หน่วย และชื่อวัตถุดิบ"""
        # รูปแบบการหาปริมาณและหน่วย
        patterns = [
            r'(\d+(?:\.\d+)?)\s*([ก-๙a-zA-Z]+)\s*(.+)',  # เลข + หน่วย + ชื่อวัตถุดิบ
            r'(.+?)\s+(\d+(?:\.\d+)?)\s*([ก-๙a-zA-Z]+)',  # ชื่อวัตถุดิบ + เลข + หน่วย
            r'(.+)',  # เฉพาะชื่อวัตถุดิบ
        ]
        
        for pattern in patterns:
            match = re.match(pattern, ingredient_text.strip())
            if match:
                groups = match.groups()
                if len(groups) == 3 and groups[0].replace('.', '').isdigit():
                    # รูปแบบ: เลข + หน่วย + ชื่อ
                    quantity = float(groups[0])
                    unit = groups[1]
                    ingredient = groups[2]
                elif len(groups) == 3 and groups[1].replace('.', '').isdigit():
                    # รูปแบบ: ชื่อ + เลข + หน่วย
                    ingredient = groups[0]
                    quantity = float(groups[1])
                    unit = groups[2]
                else:
                    # เฉพาะชื่อวัตถุดิบ - ประมาณค่า
                    ingredient = groups[0]
                    quantity = self.estimate_default_quantity(ingredient)
                    unit = self.estimate_default_unit(ingredient)
                break
        else:
            # หากไม่พบรูปแบบใด ใช้ค่าเริ่มต้น
            ingredient = ingredient_text
            quantity = self.estimate_default_quantity(ingredient)
            unit = self.estimate_default_unit(ingredient)
        
        return quantity, unit, self.normalize_ingredient_name(ingredient)

    def estimate_default_quantity(self, ingredient: str) -> float:
        """ประมาณปริมาณเริ่มต้นสำหรับวัตถุดิบที่ไม่ระบุปริมาณ"""
        defaults = {
            "น้ำมัน": 2,      # 2 ช้อนโต๊ะ
            "น้ำปลา": 1,      # 1 ช้อนโต๊ะ
            "น้ำตาล": 1,      # 1 ช้อนชา
            "เกลือ": 0.5,     # 1/2 ช้อนชา
            "พริกไทย": 3,     # 3 เม็ด
            "กระเทียม": 3,    # 3 กลีบ
            "หอม": 2,         # 2 หัว
            "ผักชี": 2,       # 2 ต้น
            "ไข่": 2,         # 2 ฟอง
            "เนื้อ": 200,     # 200 กรัม
            "หมู": 200,       # 200 กรัม
            "กุ้ง": 150,      # 150 กรัม
            "ปลา": 300,       # 300 กรัม
        }
        
        for key, value in defaults.items():
            if key in ingredient:
                return value
        
        return 100  # ค่าเริ่มต้น

    def estimate_default_unit(self, ingredient: str) -> str:
        """ประมาณหน่วยเริ่มต้นสำหรับวัตถุดิบ"""
        unit_map = {
            "น้ำมัน": "ช้อนโต๊ะ",
            "น้ำปลา": "ช้อนโต๊ะ",
            "น้ำตาล": "ช้อนชา",
            "เกลือ": "ช้อนชา",
            "พริกไทย": "เม็ด",
            "กระเทียม": "กลีบ",
            "หอม": "หัว",
            "ผักชี": "ต้น",
            "ไข่": "ฟอง",
            "เนื้อ": "กรัม",
            "หมู": "กรัม",
            "กุ้ง": "กรัม",
            "ปลา": "กรัม",
        }
        
        for key, unit in unit_map.items():
            if key in ingredient:
                return unit
        
        return "กรัม"

    def convert_to_grams(self, quantity: float, unit: str, ingredient: str) -> float:
        """แปลงปริมาณเป็นกรัม"""
        # หากหน่วยเป็นกรัมอยู่แล้ว
        if unit in ["กรัม", "g", "gram", "grams"]:
            return quantity
        
        # แปลงจากหน่วยอื่น
        if unit in self.unit_conversion:
            # สำหรับของเหลว (มล. -> กรัม ใช้ density ประมาณ 1)
            if unit in ["ช้อนโต๊ะ", "ช้อนชา", "ถ้วย", "ถ้วยชา"]:
                ml = quantity * self.unit_conversion[unit]
                # ความหนาแน่นต่างกันตามชนิดวัตถุดิบ
                if "น้ำมัน" in ingredient:
                    return ml * 0.92  # น้ำมันเบากว่าน้ำ
                elif "น้ำปลา" in ingredient:
                    return ml * 1.1   # น้ำปลาหนักกว่าน้ำเล็กน้อย
                else:
                    return ml  # ใช้ 1:1 สำหรับของเหลวทั่วไป
            else:
                return quantity * self.unit_conversion[unit]
        
        # หากไม่พบหน่วย ใช้ค่าตามที่ระบุ
        return quantity

    def get_nutrition_data(self, ingredient: str, use_api: bool = True) -> Optional[Dict]:
        """ดึงข้อมูลโภชนาการสำหรับวัตถุดิบ"""
        normalized_ingredient = self.normalize_ingredient_name(ingredient)
        
        # ลองค้นหาใน API ก่อน (หากเปิดใช้งาน)
        if use_api and self.api_key:
            api_data = self.search_ingredient_api(normalized_ingredient)
            if api_data:
                return api_data
        
        # ค้นหาในฐานข้อมูลท้องถิ่น
        for local_ingredient, nutrition in self.local_nutrition_db.items():
            if local_ingredient in normalized_ingredient or normalized_ingredient in local_ingredient:
                return nutrition.copy()
        
        # หากไม่พบ ให้ค่าเริ่มต้น
        return {
            "calories": 50, "protein": 2, "carbs": 10, "fat": 1, "fiber": 1,
            "vitamin_a": 10, "vitamin_c": 5, "vitamin_b1": 0.1, "vitamin_b2": 0.1,
            "calcium": 20, "iron": 1, "potassium": 100, "sodium": 10
        }

    def calculate_recipe_nutrition(self, ingredients_text: str, use_api: bool = True, 
                                 adjust_consumption: bool = True) -> Dict:
        """คำนวณค่าโภชนาการของสูตรอาหาร"""
        total_nutrition = {
            "calories": 0, "protein": 0, "carbs": 0, "fat": 0, "fiber": 0,
            "vitamin_a": 0, "vitamin_c": 0, "vitamin_b1": 0, "vitamin_b2": 0,
            "calcium": 0, "iron": 0, "potassium": 0, "sodium": 0
        }
        
        ingredient_details = []
        
        # แยกวัตถุดิบแต่ละรายการ
        ingredients = [ing.strip() for ing in ingredients_text.split('\n') if ing.strip() and ing.strip().startswith('-')]
        
        for ingredient_line in ingredients:
            # ลบเครื่องหมาย -
            ingredient_text = ingredient_line.replace('-', '').strip()
            
            # แยกปริมาณและชื่อวัตถุดิบ
            quantity, unit, ingredient_name = self.extract_quantity_and_unit(ingredient_text)
            
            # แปลงเป็นกรัม
            grams = self.convert_to_grams(quantity, unit, ingredient_name)
            
            # ปรับสัดส่วนการบริโภค (หากเปิดใช้งาน)
            if adjust_consumption:
                consumption_factor = self.consumption_ratio.get(ingredient_name, 1.0)
                effective_grams = grams * consumption_factor
            else:
                effective_grams = grams
            
            # ดึงข้อมูลโภชนาการ
            nutrition_per_100g = self.get_nutrition_data(ingredient_name, use_api)
            
            if nutrition_per_100g:
                # คำนวณค่าโภชนาการตามปริมาณจริง
                factor = effective_grams / 100
                ingredient_nutrition = {}
                
                for nutrient, value_per_100g in nutrition_per_100g.items():
                    nutrient_value = value_per_100g * factor
                    ingredient_nutrition[nutrient] = nutrient_value
                    total_nutrition[nutrient] += nutrient_value
                
                ingredient_details.append({
                    "name": ingredient_name,
                    "quantity": quantity,
                    "unit": unit,
                    "grams": grams,
                    "effective_grams": effective_grams,
                    "nutrition": ingredient_nutrition
                })
        
        return {
            "total_nutrition": total_nutrition,
            "ingredient_details": ingredient_details
        }

    def check_api_status(self) -> Tuple[bool, str]:
        """ตรวจสอบสถานะการเชื่อมต่อ API"""
        if not self.api_key:
            return False, "ไม่ได้ตั้งค่า API Key"
        
        try:
            # ทดสอบการเชื่อมต่อด้วยการค้นหาข้อมูลธรรมดา
            search_url = f"{self.api_url}/foods/search"
            params = {
                "api_key": self.api_key,
                "query": "chicken",
                "pageSize": 1
            }
            
            response = requests.get(search_url, params=params, timeout=5)
            if response.status_code == 200:
                return True, "เชื่อมต่อ API สำเร็จ"
            else:
                return False, f"API ตอบกลับ status code: {response.status_code}"
                
        except requests.exceptions.Timeout:
            return False, "การเชื่อมต่อ API หมดเวลา"
        except requests.exceptions.ConnectionError:
            return False, "ไม่สามารถเชื่อมต่อ API ได้"
        except Exception as e:
            return False, f"เกิดข้อผิดพลาด: {str(e)}"
