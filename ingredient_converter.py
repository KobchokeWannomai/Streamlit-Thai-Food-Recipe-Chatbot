"""
ระบบแปลงหน่วยและประมาณน้ำหนัก/ปริมาตรของวัตถุดิบอาหารไทย
Thai Ingredient Unit Converter
"""

import re
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class IngredientConverter:
    """คลาสสำหรับแปลงหน่วยและประมาณน้ำหนักของวัตถุดิบ"""
    
    def __init__(self):
        # พจนานุกรมแปลงหน่วย
        self.unit_conversions = {
            # หน่วยน้ำหนัก
            'กิโลกรัม': 1000,
            'กิโล': 1000,
            'กก.': 1000,
            'kg': 1000,
            'กรัม': 1,
            'ก.': 1,
            'g': 1,
            
            # หน่วยปริมาตร (แปลงเป็น ml)
            'ลิตร': 1000,
            'ล.': 1000,
            'l': 1000,
            'มิลลิลิตร': 1,
            'มล.': 1,
            'ml': 1,
            'ซีซี': 1,
            'cc': 1,
            
            # หน่วยครัว
            'ถ้วย': 240,  # ml
            'ถ้วยตวง': 240,
            'ช้อนโต๊ะ': 15,  # ml
            'ช้อนชา': 5,  # ml
            'ช้อนกาแฟ': 2.5,  # ml
        }
        
        # น้ำหนักโดยประมาณของวัตถุดิบต่างๆ (กรัมต่อหน่วย)
        self.ingredient_weights = {
            # เนื้อสัตว์
            'หมู': {'ชิ้น': 30, 'แผ่น': 50, 'ตัว': 200},
            'ไก่': {'ชิ้น': 40, 'ตัว': 1500, 'น่อง': 150, 'สะโพก': 200, 'อก': 250},
            'เนื้อ': {'ชิ้น': 35, 'แผ่น': 60},
            'กุ้ง': {'ตัว': 20, 'ตัวกลาง': 15, 'ตัวใหญ่': 25, 'ตัวเล็ก': 10},
            'ปลา': {'ตัว': 300, 'ชิ้น': 80, 'แผ่น': 100},
            'หอย': {'ตัว': 5, 'ถ้วย': 150},
            'ปลาหมึก': {'ตัว': 200, 'วง': 20},
            
            # ไข่
            'ไข่': {'ฟอง': 60, 'ฟองเล็ก': 50, 'ฟองใหญ่': 70},
            'ไข่ไก่': {'ฟอง': 60},
            'ไข่เป็ด': {'ฟอง': 70},
            
            # ผัก
            'กะหล่ำปลี': {'หัว': 1000, 'ใบ': 30, 'ถ้วย': 70},
            'คะน้า': {'ต้น': 50, 'กำ': 100, 'ถ้วย': 60},
            'ผักบุ้ง': {'กำ': 80, 'ถ้วย': 40},
            'ผักกาด': {'ต้น': 60, 'หัว': 300, 'ใบ': 20},
            'มะเขือ': {'ผล': 30, 'ลูก': 30, 'ถ้วย': 120},
            'มะเขือเทศ': {'ผล': 100, 'ลูก': 100, 'ลูกเล็ก': 50},
            'แตงกวา': {'ผล': 200, 'ลูก': 200, 'ถ้วย': 100},
            'ถั่วฝักยาว': {'ฝัก': 10, 'กำ': 50, 'ถ้วย': 60},
            'ถั่วงอก': {'ถ้วย': 50, 'กำมือ': 30},
            'หอมใหญ่': {'หัว': 150, 'หัวเล็ก': 100, 'หัวใหญ่': 200},
            'หอมแดง': {'หัว': 10, 'ถ้วย': 80},
            
            # เครื่องปรุง
            'กระเทียม': {'กลีบ': 5, 'หัว': 40, 'ช้อนโต๊ะ': 10},
            'พริก': {'เม็ด': 2, 'ถ้วย': 40},
            'พริกขี้หนู': {'เม็ด': 1, 'ถ้วย': 30},
            'พริกแห้ง': {'เม็ด': 0.5, 'ถ้วย': 20},
            'ขิง': {'แว่น': 5, 'ชิ้น': 10, 'หัว': 100},
            'ข่า': {'แว่น': 5, 'ชิ้น': 10, 'หัว': 150},
            'ตะไคร้': {'ต้น': 20, 'ช้อนโต๊ะ': 15},
            'ใบมะกรูด': {'ใบ': 0.5, 'ถ้วย': 10},
            'ผักชี': {'ต้น': 5, 'ถ้วย': 15, 'ช้อนโต๊ะ': 5},
            'ใบโหระพา': {'ใบ': 0.3, 'ถ้วย': 20},
            'ใบกะเพรา': {'ใบ': 0.2, 'ถ้วย': 15},
            
            # ผลไม้
            'มะนาว': {'ผล': 60, 'ลูก': 60, 'ช้อนโต๊ะ': 15},
            'มะละกอ': {'ถ้วย': 140, 'ผล': 500},
            'มะม่วง': {'ผล': 200, 'ลูก': 200, 'ถ้วย': 150},
            
            # อื่นๆ
            'ถั่วลิสง': {'ถ้วย': 140, 'กำมือ': 30, 'ช้อนโต๊ะ': 15},
            'งา': {'ช้อนโต๊ะ': 10, 'ช้อนชา': 3},
            'มะพร้าวขูด': {'ถ้วย': 80, 'ช้อนโต๊ะ': 10},
        }
        
        # ความหนาแน่นของของเหลว (g/ml)
        self.liquid_densities = {
            'น้ำ': 1.0,
            'น้ำปลา': 1.2,
            'น้ำมัน': 0.92,
            'น้ำมันพืช': 0.92,
            'น้ำมันหมู': 0.9,
            'กะทิ': 0.95,
            'นม': 1.03,
            'น้ำตาล': 1.59,  # น้ำเชื่อม
            'น้ำซุป': 1.0,
            'ซอส': 1.1,
            'น้ำส้ม': 1.05,
            'น้ำมะนาว': 1.03,
        }
    
    def extract_quantity_and_unit(self, ingredient_text: str) -> Tuple[float, str, str]:
        """แยกปริมาณ หน่วย และชื่อวัตถุดิบ"""
        # ลองหาตัวเลขและหน่วย
        patterns = [
            r'(\d+\.?\d*)\s*([^\s]+)',  # ตัวเลข + หน่วย
            r'(\d+)\s+(\d+/\d+)\s*([^\s]+)',  # เลขผสมเศษส่วน
            r'(\d+/\d+)\s*([^\s]+)',  # เศษส่วน
        ]
        
        for pattern in patterns:
            match = re.search(pattern, ingredient_text)
            if match:
                if len(match.groups()) == 3:  # เลขผสมเศษส่วน
                    whole = float(match.group(1))
                    frac_parts = match.group(2).split('/')
                    fraction = float(frac_parts[0]) / float(frac_parts[1])
                    quantity = whole + fraction
                    unit = match.group(3)
                elif '/' in match.group(1):  # เศษส่วน
                    frac_parts = match.group(1).split('/')
                    quantity = float(frac_parts[0]) / float(frac_parts[1])
                    unit = match.group(2)
                else:  # ตัวเลขธรรมดา
                    quantity = float(match.group(1))
                    unit = match.group(2)
                
                # หาชื่อวัตถุดิบ
                ingredient_name = ingredient_text[match.end():].strip()
                return quantity, unit, ingredient_name
        
        # ถ้าไม่พบตัวเลข ให้ถือว่าเป็น 1 หน่วย
        return 1.0, '', ingredient_text.strip()
    
    def convert_to_grams(self, quantity: float, unit: str, ingredient_name: str) -> float:
        """แปลงปริมาณเป็นกรัม"""
        # ทำความสะอาดชื่อวัตถุดิบ
        clean_name = self._clean_ingredient_name(ingredient_name)
        
        # ตรวจสอบว่าเป็นหน่วยน้ำหนักหรือไม่
        if unit.lower() in self.unit_conversions:
            base_value = self.unit_conversions[unit.lower()]
            if unit.lower() in ['ถ้วย', 'ช้อนโต๊ะ', 'ช้อนชา', 'ช้อนกาแฟ', 'ลิตร', 'ล.', 'l', 'มิลลิลิตร', 'มล.', 'ml']:
                # หน่วยปริมาตร - ต้องคูณกับความหนาแน่น
                density = self._get_density(clean_name)
                return quantity * base_value * density
            else:
                # หน่วยน้ำหนัก
                return quantity * base_value
        
        # ตรวจสอบน้ำหนักจากพจนานุกรม
        for key, weights in self.ingredient_weights.items():
            if key in clean_name:
                if unit in weights:
                    return quantity * weights[unit]
                # ลองหาหน่วยที่ใกล้เคียง
                for weight_unit, weight in weights.items():
                    if weight_unit in unit or unit in weight_unit:
                        return quantity * weight
        
        # ถ้าไม่พบ ให้ประมาณการ
        return self._estimate_weight(quantity, unit, clean_name)
    
    def _clean_ingredient_name(self, name: str) -> str:
        """ทำความสะอาดชื่อวัตถุดิบ"""
        # ลบคำที่ไม่จำเป็น
        unwanted = ['สด', 'แห้ง', 'ดิบ', 'สุก', 'หั่น', 'ซอย', 'สับ', 'ฝอย', 'ละเอียด', 'หยาบ', 'บาง', 'หนา']
        clean = name
        for word in unwanted:
            clean = clean.replace(word, '')
        return clean.strip()
    
    def _get_density(self, ingredient: str) -> float:
        """หาความหนาแน่นของวัตถุดิบ"""
        for liquid, density in self.liquid_densities.items():
            if liquid in ingredient:
                return density
        return 1.0  # ค่าเริ่มต้น
    
    def _estimate_weight(self, quantity: float, unit: str, ingredient: str) -> float:
        """ประมาณน้ำหนักเมื่อไม่มีข้อมูล"""
        # ประมาณการตามประเภทวัตถุดิบ
        default_weights = {
            'ตัว': 100,
            'ชิ้น': 30,
            'แผ่น': 40,
            'ผล': 100,
            'ลูก': 80,
            'หัว': 150,
            'ต้น': 50,
            'ใบ': 5,
            'กลีบ': 5,
            'เม็ด': 2,
            'ฝัก': 20,
            'กำ': 80,
            'กำมือ': 50,
            'มัด': 100,
            'ห่อ': 200,
        }
        
        if unit in default_weights:
            return quantity * default_weights[unit]
        
        # ถ้าไม่มีหน่วย ให้ถือว่าเป็นกรัม
        return quantity * 100  # ประมาณ 100 กรัมต่อหน่วย
    
    def parse_and_convert_ingredient(self, ingredient_text: str) -> Dict[str, any]:
        """แยกวิเคราะห์และแปลงหน่วยวัตถุดิบ"""
        # แยกปริมาณ หน่วย และชื่อ
        quantity, unit, name = self.extract_quantity_and_unit(ingredient_text)
        
        # แปลงเป็นกรัม
        weight_grams = self.convert_to_grams(quantity, unit, name)
        
        # คำนวณตัวคูณสำหรับโภชนาการ (เทียบกับ 100g)
        multiplier = weight_grams / 100.0
        
        return {
            'original_text': ingredient_text,
            'quantity': quantity,
            'unit': unit,
            'name': name,
            'weight_grams': weight_grams,
            'nutrition_multiplier': multiplier,
            'is_liquid': any(liquid in name.lower() for liquid in self.liquid_densities.keys())
        }

# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    converter = IngredientConverter()
    
    # ทดสอบการแปลงหน่วย
    test_ingredients = [
        "กุ้ง 200 กรัม",
        "หมูสับ 1/2 กิโลกรัม",
        "ไข่ไก่ 2 ฟอง",
        "น้ำปลา 2 ช้อนโต๊ะ",
        "กะทิ 1 ถ้วย",
        "พริกขี้หนู 5 เม็ด",
        "ใบกะเพรา 1 ถ้วย",
        "กระเทียม 3 กลีบ",
        "มะนาว 2 ลูก",
        "น้ำมันพืช 3 ช้อนโต๊ะ"
    ]
    
    print("ทดสอบการแปลงหน่วยวัตถุดิบ:")
    print("=" * 60)
    
    for ingredient in test_ingredients:
        result = converter.parse_and_convert_ingredient(ingredient)
        print(f"\nวัตถุดิบ: {ingredient}")
        print(f"  ปริมาณ: {result['quantity']} {result['unit']}")
        print(f"  ชื่อ: {result['name']}")
        print(f"  น้ำหนัก: {result['weight_grams']:.1f} กรัม")
        print(f"  ตัวคูณโภชนาการ: {result['nutrition_multiplier']:.2f}")
        print(f"  เป็นของเหลว: {'ใช่' if result['is_liquid'] else 'ไม่ใช่'}")
