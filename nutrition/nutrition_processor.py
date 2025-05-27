# nutrition/nutrition_processor.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import re

class NutritionProcessor:
    def __init__(self):
        self.unit_conversions = {
            'ช้อนโต๊ะ': 15,  # กรัม
            'ช้อนชา': 5,
            'ถ้วย': 240,
            'ถ้วยชา': 200,
            'ตัว': None,  # ต้องดูตามชนิด
            'ฟอง': None,
            'หัว': None,
            'ผล': None,
            'กิโลกรัม': 1000,
            'ขีด': 600
        }
        
        # น้ำหนักเฉลี่ยของวัตถุดิบบางชนิด
        self.average_weights = {
            'ไข่': {'ฟอง': 50},
            'กุ้งนาง': {'ตัว': 30},
            'หอม': {'หัว': 40},
            'กระเทียม': {'กลีบ': 5},
            'มะนาว': {'ผล': 60}
        }
    
    def parse_ingredient_line(self, line: str) -> Tuple[str, float, str]:
        """แยกวิเคราะห์บรรทัดวัตถุดิบ"""
        # ตัวอย่าง: "- กุ้งนาง 4 ตัว" -> ('กุ้งนาง', 4, 'ตัว')
        pattern = r'-\s*(.+?)\s+(\d+(?:\.\d+)?)\s*(.+?)$'
        match = re.match(pattern, line.strip())
        
        if match:
            ingredient = match.group(1).strip()
            amount = float(match.group(2))
            unit = match.group(3).strip()
            return ingredient, amount, unit
        
        # กรณีไม่มีจำนวนชัดเจน
        return line.strip('- '), 1, 'พอควร'
    
    def convert_to_grams(self, ingredient: str, amount: float, unit: str) -> float:
        """แปลงหน่วยเป็นกรัม"""
        # ตรวจสอบหน่วยพื้นฐาน
        if unit in self.unit_conversions and self.unit_conversions[unit]:
            return amount * self.unit_conversions[unit]
        
        # ตรวจสอบน้ำหนักเฉลี่ย
        if ingredient in self.average_weights:
            if unit in self.average_weights[ingredient]:
                return amount * self.average_weights[ingredient][unit]
        
        # ค่าเริ่มต้น
        return amount * 50  # ประมาณ 50 กรัมต่อหน่วย
    
    def calculate_recipe_nutrition(self, recipe_row: pd.Series, 
                                 nutrition_fetcher) -> Dict:
        """คำนวณคุณค่าโภชนาการของสูตรอาหาร"""
        ingredients_text = recipe_row['ingredient']
        total_nutrition = {
            'calories': 0, 'protein': 0, 'fat': 0, 'carbs': 0,
            'fiber': 0, 'sugar': 0, 'sodium': 0,
            'vitamin_a': 0, 'vitamin_c': 0, 'calcium': 0, 'iron': 0
        }
        
        # แยกวิเคราะห์แต่ละวัตถุดิบ
        ingredient_lines = ingredients_text.split('\n')
        total_weight = 0
        
        for line in ingredient_lines:
            if line.strip().startswith('-'):
                ingredient, amount, unit = self.parse_ingredient_line(line)
                weight_grams = self.convert_to_grams(ingredient, amount, unit)
                total_weight += weight_grams
                
                # ดึงข้อมูลโภชนาการ
                nutrition_data = nutrition_fetcher.fetch_nutrition(ingredient)
                if nutrition_data:
                    # คำนวณตามสัดส่วน (ข้อมูลต่อ 100g)
                    ratio = weight_grams / 100
                    for nutrient in total_nutrition:
                        if nutrient in nutrition_data:
                            total_nutrition[nutrient] += nutrition_data[nutrient] * ratio
        
        # คำนวณต่อ 100 กรัมของอาหารสำเร็จรูป
        if total_weight > 0:
            for nutrient in total_nutrition:
                total_nutrition[nutrient] = round(
                    total_nutrition[nutrient] * 100 / total_weight, 2
                )
        
        return total_nutrition