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
            'ขีด': 100,  # 1 ขีด = 100 กรัม
            
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
            'ถ้วยชา': 240,
            'ช้อนโต๊ะ': 15,  # ml
            'ช้อนชา': 5,  # ml
            'ช้อนกาแฟ': 2.5,  # ml
            'ช้อนหวาน': 10,  # ml
            'ทัพพี': 60,  # ml
            'ฝา': 5,  # ml (ฝาขวด)
        }
        
        # น้ำหนักโดยประมาณของวัตถุดิบต่างๆ (กรัมต่อหน่วย)
        self.ingredient_weights = {
            # เนื้อสัตว์
            'หมู': {'ชิ้น': 30, 'แผ่น': 50, 'ตัว': 200, 'ก้อน': 100, 'default': 100},
            'เนื้อหมู': {'ชิ้น': 30, 'แผ่น': 50, 'ตัว': 200, 'ถ้วย': 140, 'default': 100},
            'ไก่': {'ชิ้น': 40, 'ตัว': 1500, 'น่อง': 150, 'สะโพก': 200, 'อก': 250, 'default': 150},
            'เนื้อไก่': {'ชิ้น': 40, 'ถ้วย': 140, 'default': 100},
            'เนื้อ': {'ชิ้น': 35, 'แผ่น': 60, 'default': 100},
            'เนื้อโค': {'ชิ้น': 35, 'แผ่น': 60, 'default': 100},
            'เนื้อวัว': {'ชิ้น': 35, 'แผ่น': 60, 'default': 100},
            'กุ้ง': {'ตัว': 20, 'ตัวกลาง': 15, 'ตัวใหญ่': 25, 'ตัวเล็ก': 10, 'ถ้วย': 100, 'default': 15},
            'กุ้งนาง': {'ตัว': 25, 'default': 25},
            'กุ้งแห้ง': {'ช้อนโต๊ะ': 10, 'ถ้วย': 50, 'default': 10},
            'ปลา': {'ตัว': 300, 'ชิ้น': 80, 'แผ่น': 100, 'ริ้ว': 150, 'default': 200},
            'หอย': {'ตัว': 5, 'ถ้วย': 150, 'default': 100},
            'หอยแมลงภู่': {'ตัว': 5, 'ถ้วย': 150, 'default': 100},
            'ปลาหมึก': {'ตัว': 200, 'วง': 20, 'default': 150},
            'ปู': {'ตัว': 150, 'ตำ': 150, 'default': 150},
            
            # ไข่
            'ไข่': {'ฟอง': 50, 'ฟองเล็ก': 40, 'ฟองใหญ่': 60, 'default': 50},
            'ไข่ไก่': {'ฟอง': 50, 'default': 50},
            'ไข่เป็ด': {'ฟอง': 60, 'default': 60},
            'ไข่เค็ม': {'ฟอง': 60, 'default': 60},
            'ไข่แดง': {'ฟอง': 20, 'default': 20},
            'ไข่ขาว': {'ฟอง': 30, 'default': 30},
            
            # ผัก
            'กะหล่ำปลี': {'หัว': 1000, 'ใบ': 30, 'ถ้วย': 70, 'default': 50},
            'คะน้า': {'ต้น': 50, 'กำ': 100, 'ถ้วย': 60, 'default': 50},
            'ผักบุ้ง': {'กำ': 80, 'ถ้วย': 40, 'กำมือ': 50, 'default': 50},
            'ผักกาด': {'ต้น': 60, 'หัว': 300, 'ใบ': 20, 'default': 50},
            'ผักกาดขาว': {'ต้น': 60, 'หัว': 300, 'ใบ': 20, 'default': 50},
            'มะเขือ': {'ผล': 30, 'ลูก': 30, 'ถ้วย': 120, 'default': 30},
            'มะเขือเทศ': {'ผล': 100, 'ลูก': 100, 'ลูกเล็ก': 50, 'default': 100},
            'แตงกวา': {'ผล': 200, 'ลูก': 200, 'ถ้วย': 100, 'default': 200},
            'ถั่วฝักยาว': {'ฝัก': 10, 'กำ': 50, 'ถ้วย': 60, 'default': 30},
            'ถั่วงอก': {'ถ้วย': 50, 'กำมือ': 30, 'default': 30},
            'ถั่วพู': {'ฝัก': 15, 'default': 30},
            'หอมใหญ่': {'หัว': 150, 'หัวเล็ก': 100, 'หัวใหญ่': 200, 'default': 150},
            'หอมแดง': {'หัว': 10, 'ถ้วย': 80, 'default': 10},
            'หัวหอม': {'หัว': 10, 'กลีบ': 5, 'default': 10},
            'ต้นหอม': {'ต้น': 5, 'ถ้วย': 30, 'default': 5},
            'ผักชี': {'ต้น': 5, 'ถ้วย': 15, 'ช้อนโต๊ะ': 5, 'default': 5},
            'โหระพา': {'ใบ': 0.3, 'ถ้วย': 20, 'ช่อ': 5, 'default': 10},
            'กะเพรา': {'ใบ': 0.2, 'ถ้วย': 15, 'default': 10},
            
            # เครื่องปรุง
            'กระเทียม': {'กลีบ': 3, 'หัว': 30, 'ช้อนโต๊ะ': 10, 'default': 3},
            'พริก': {'เม็ด': 2, 'ถ้วย': 40, 'default': 2},
            'พริกขี้หนู': {'เม็ด': 1, 'ถ้วย': 30, 'default': 1},
            'พริกชี้ฟ้า': {'เม็ด': 3, 'default': 3},
            'พริกแห้ง': {'เม็ด': 0.5, 'ถ้วย': 20, 'default': 0.5},
            'พริกไทย': {'เม็ด': 0.1, 'ช้อนชา': 2, 'ช้อนโต๊ะ': 6, 'default': 2},
            'ขิง': {'แว่น': 5, 'ชิ้น': 10, 'หัว': 100, 'ช้อนโต๊ะ': 10, 'default': 10},
            'ข่า': {'แว่น': 5, 'ชิ้น': 10, 'หัว': 150, 'default': 10},
            'ตะไคร้': {'ต้น': 20, 'ช้อนโต๊ะ': 15, 'default': 20},
            'ใบมะกรูด': {'ใบ': 0.5, 'ถ้วย': 10, 'default': 0.5},
            'รากผักชี': {'ราก': 2, 'ช้อนชา': 5, 'default': 5},
            'กะปิ': {'ช้อนชา': 10, 'ช้อนโต๊ะ': 30, 'default': 10},
            
            # ผลไม้
            'มะนาว': {'ผล': 60, 'ลูก': 60, 'ช้อนโต๊ะ': 15, 'default': 60},
            'มะละกอ': {'ถ้วย': 140, 'ผล': 500, 'default': 300},
            'มะม่วง': {'ผล': 200, 'ลูก': 200, 'ถ้วย': 150, 'default': 200},
            'มะพร้าว': {'ซีก': 200, 'ผล': 400, 'ถ้วย': 80, 'default': 100},
            'มะพร้าวขูด': {'ถ้วย': 80, 'ช้อนโต๊ะ': 10, 'default': 50},
            
            # ถั่วและธัญพืช
            'ถั่วลิสง': {'ถ้วย': 140, 'กำมือ': 30, 'ช้อนโต๊ะ': 15, 'default': 30},
            'ถั่วเขียว': {'ถ้วย': 180, 'ช้อนโต๊ะ': 20, 'default': 50},
            'ถั่วเหลือง': {'ถ้วย': 170, 'default': 50},
            'งา': {'ช้อนโต๊ะ': 10, 'ช้อนชา': 3, 'default': 10},
            'ข้าวคั่ว': {'ช้อนโต๊ะ': 10, 'ถ้วย': 80, 'default': 20},
            
            # แป้งและเส้น
            'แป้ง': {'ถ้วย': 120, 'ช้อนโต๊ะ': 15, 'default': 50},
            'แป้งข้าวเจ้า': {'ถ้วย': 120, 'ช้อนโต๊ะ': 15, 'default': 50},
            'แป้งสาลี': {'ถ้วย': 120, 'ช้อนโต๊ะ': 15, 'default': 50},
            'แป้งมัน': {'ถ้วย': 150, 'ช้อนโต๊ะ': 20, 'default': 50},
            'เส้นใหญ่': {'ถ้วย': 100, 'default': 100},
            'เส้นเล็ก': {'ถ้วย': 80, 'default': 80},
            
            # อื่นๆ
            'เต้าหู้': {'แผ่น': 100, 'ก้อน': 100, 'default': 100},
            'วุ้นเส้น': {'ก้อน': 40, 'ถ้วย': 60, 'default': 40},
            'ปลาร้า': {'ช้อนโต๊ะ': 20, 'ถ้วย': 200, 'default': 20},
            'น้ำพริกเผา': {'ช้อนโต๊ะ': 20, 'ช้อนชา': 7, 'default': 20},
        }
        
        # ความหนาแน่นของของเหลว (g/ml)
        self.liquid_densities = {
            'น้ำ': 1.0,
            'น้ำปลา': 1.2,
            'น้ำมัน': 0.92,
            'น้ำมันพืช': 0.92,
            'น้ำมันหมู': 0.9,
            'น้ำมันงา': 0.92,
            'กะทิ': 0.95,
            'หางกะทิ': 0.98,
            'หัวกะทิ': 0.93,
            'นม': 1.03,
            'น้ำตาล': 1.59,  # น้ำเชื่อม
            'น้ำซุป': 1.0,
            'ซอส': 1.1,
            'ซีอิ้ว': 1.2,
            'ซีอิ๊ว': 1.2,
            'น้ำส้ม': 1.05,
            'น้ำมะนาว': 1.03,
            'น้ำจิ้ม': 1.1,
            'เหล้า': 0.95,
            'น้ำผึ้ง': 1.4,
        }
        
        # ปริมาณเริ่มต้นสำหรับวัตถุดิบที่ไม่ระบุหน่วย (กรัม)
        self.default_portions = {
            # เครื่องปรุงพื้นฐาน
            'เกลือ': 5,
            'น้ำตาล': 10,
            'น้ำปลา': 15,
            'ซีอิ้ว': 15,
            'น้ำมัน': 15,
            'กะทิ': 100,
            'น้ำ': 240,
            
            # เครื่องเทศ
            'พริกไทย': 2,
            'รากผักชี': 5,
            'ผิวมะกรูด': 2,
            'ใบมะกรูด': 2,
            
            # ผักใบ
            'ผักชี': 10,
            'ต้นหอม': 10,
            'ใบโหระพา': 10,
            'ใบกะเพรา': 10,
            
            # default สำหรับทั่วไป
            'default': 50
        }
    
    def extract_quantity_and_unit(self, ingredient_text: str) -> Tuple[float, str, str]:
        """แยกปริมาณ หน่วย และชื่อวัตถุดิบ"""
        # ทำความสะอาดข้อความ
        ingredient_text = ingredient_text.strip()
        
        # ลองหาตัวเลขและหน่วย
        patterns = [
            # รูปแบบต่างๆ ของการระบุปริมาณ
            r'(\d+\.?\d*)\s*([^\s]+?)\s+(.+)',  # ตัวเลข + หน่วย + ชื่อ
            r'(\d+)\s+(\d+/\d+)\s*([^\s]+)\s+(.+)',  # เลขผสมเศษส่วน
            r'(\d+/\d+)\s*([^\s]+)\s+(.+)',  # เศษส่วน
            r'(.+?)\s+(\d+\.?\d*)\s*([^\s]+)$',  # ชื่อ + ตัวเลข + หน่วย
            r'(.+?)\s*\(.*?(\d+\.?\d*)\s*([^\)]+)\)',  # ชื่อ (ตัวเลข หน่วย)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, ingredient_text)
            if match:
                if len(match.groups()) == 4:  # เลขผสมเศษส่วน
                    whole = float(match.group(1))
                    frac_parts = match.group(2).split('/')
                    fraction = float(frac_parts[0]) / float(frac_parts[1])
                    quantity = whole + fraction
                    unit = match.group(3)
                    ingredient_name = match.group(4)
                elif '/' in str(match.group(1)) and len(match.groups()) == 3:  # เศษส่วน
                    frac_parts = match.group(1).split('/')
                    quantity = float(frac_parts[0]) / float(frac_parts[1])
                    unit = match.group(2)
                    ingredient_name = match.group(3)
                elif len(match.groups()) == 3 and match.group(1)[0].isdigit():  # ตัวเลขธรรมดา
                    quantity = float(match.group(1))
                    unit = match.group(2)
                    ingredient_name = match.group(3)
                elif len(match.groups()) == 3:  # ชื่อ + ตัวเลข + หน่วย
                    ingredient_name = match.group(1)
                    quantity = float(match.group(2))
                    unit = match.group(3)
                else:
                    continue
                
                return quantity, unit, ingredient_name.strip()
        
        # ถ้าไม่พบตัวเลข ให้ประมาณปริมาณตามชนิดวัตถุดิบ
        return self._estimate_default_quantity(ingredient_text), '', ingredient_text.strip()
    
    def _estimate_default_quantity(self, ingredient_name: str) -> float:
        """ประมาณปริมาณเริ่มต้นเมื่อไม่ระบุ"""
        clean_name = self._clean_ingredient_name(ingredient_name).lower()
        
        # ตรวจสอบจากรายการเครื่องปรุงพื้นฐาน
        for key, default_amount in self.default_portions.items():
            if key in clean_name:
                return 1  # คืนค่า 1 เพื่อใช้กับ default weight
        
        # ตรวจสอบคำบอกปริมาณ
        quantity_words = {
            'นิดหน่อย': 0.5,
            'เล็กน้อย': 0.5,
            'พอควร': 1,
            'ตามชอบ': 1,
            'พอประมาณ': 1,
            'เยอะ': 2,
            'มาก': 2
        }
        
        for word, multiplier in quantity_words.items():
            if word in clean_name:
                return multiplier
        
        return 1  # ค่าเริ่มต้น
    
    def convert_to_grams(self, quantity: float, unit: str, ingredient_name: str) -> float:
        """แปลงปริมาณเป็นกรัม"""
        # ทำความสะอาดชื่อวัตถุดิบ
        clean_name = self._clean_ingredient_name(ingredient_name)
        
        # กรณีไม่มีหน่วย (ใช้ค่าเริ่มต้น)
        if not unit:
            # หาน้ำหนักเริ่มต้นสำหรับวัตถุดิบนี้
            default_weight = self._get_default_weight(clean_name)
            return quantity * default_weight
        
        # ตรวจสอบว่าเป็นหน่วยน้ำหนักหรือไม่
        if unit.lower() in self.unit_conversions:
            base_value = self.unit_conversions[unit.lower()]
            if unit.lower() in ['ถ้วย', 'ถ้วยชา', 'ช้อนโต๊ะ', 'ช้อนชา', 'ช้อนกาแฟ', 'ช้อนหวาน', 'ลิตร', 'ล.', 'l', 'มิลลิลิตร', 'มล.', 'ml', 'ทัพพี', 'ฝา']:
                # หน่วยปริมาตร - ต้องคูณกับความหนาแน่น
                density = self._get_density(clean_name)
                return quantity * base_value * density
            else:
                # หน่วยน้ำหนัก
                return quantity * base_value
        
        # ตรวจสอบน้ำหนักจากพจนานุกรม
        for key, weights in self.ingredient_weights.items():
            if key in clean_name or key in ingredient_name:
                if unit in weights:
                    return quantity * weights[unit]
                # ลองหาหน่วยที่ใกล้เคียง
                for weight_unit, weight in weights.items():
                    if weight_unit in unit or unit in weight_unit:
                        return quantity * weight
                # ใช้ค่า default ของวัตถุดิบนั้นๆ
                if 'default' in weights:
                    return quantity * weights['default']
        
        # ถ้าไม่พบ ให้ประมาณการ
        return self._estimate_weight(quantity, unit, clean_name)
    
    def _get_default_weight(self, ingredient_name: str) -> float:
        """หาน้ำหนักเริ่มต้นสำหรับวัตถุดิบ"""
        clean_name = ingredient_name.lower()
        
        # ตรวจสอบจาก default_portions ก่อน
        for key, weight in self.default_portions.items():
            if key != 'default' and key in clean_name:
                return weight
        
        # ตรวจสอบจาก ingredient_weights
        for key, weights in self.ingredient_weights.items():
            if key in clean_name and 'default' in weights:
                return weights['default']
        
        # ค่าเริ่มต้นทั่วไป
        return self.default_portions.get('default', 50)
    
    def _clean_ingredient_name(self, name: str) -> str:
        """ทำความสะอาดชื่อวัตถุดิบ"""
        # ลบคำที่ไม่จำเป็น
        unwanted = ['สด', 'แห้ง', 'ดิบ', 'สุก', 'หั่น', 'ซอย', 'สับ', 'ฝอย', 'ละเอียด', 'หยาบ', 'บาง', 'หนา', 
                   'ใหม่', 'เก่า', 'อ่อน', 'แก่', 'นิดหน่อย', 'เล็กน้อย', 'พอควร', 'ตามชอบ', 'แล้ว']
        clean = name
        for word in unwanted:
            clean = clean.replace(word, '')
        return clean.strip()
    
    def _get_density(self, ingredient: str) -> float:
        """หาความหนาแน่นของวัตถุดิบ"""
        ingredient_lower = ingredient.lower()
        for liquid, density in self.liquid_densities.items():
            if liquid in ingredient_lower:
                return density
        return 1.0  # ค่าเริ่มต้น
    
    def _estimate_weight(self, quantity: float, unit: str, ingredient: str) -> float:
        """ประมาณน้ำหนักเมื่อไม่มีข้อมูล"""
        # ประมาณการตามประเภทวัตถุดิบ
        default_weights = {
            'ตัว': 30,    # ลดลงสำหรับวัตถุดิบทั่วไป
            'ชิ้น': 25,
            'แผ่น': 30,
            'ผล': 80,
            'ลูก': 60,
            'หัว': 80,
            'ต้น': 20,
            'ใบ': 2,
            'กลีบ': 3,
            'เม็ด': 1,
            'ฝัก': 15,
            'กำ': 50,
            'กำมือ': 40,
            'มัด': 80,
            'ห่อ': 150,
            'ฟอง': 50,
            'ราก': 5,
            'แว่น': 5,
            'ซีก': 100,
            'คู่': 40,
        }
        
        if unit in default_weights:
            return quantity * default_weights[unit]
        
        # ถ้าไม่มีหน่วย ให้ใช้ค่าเริ่มต้น
        return quantity * self._get_default_weight(ingredient)
    
    def parse_and_convert_ingredient(self, ingredient_text: str) -> Dict[str, any]:
        """แยกวิเคราะห์และแปลงหน่วยวัตถุดิบ"""
        # แยกปริมาณ หน่วย และชื่อ
        quantity, unit, name = self.extract_quantity_and_unit(ingredient_text)
        
        # แปลงเป็นกรัม
        weight_grams = self.convert_to_grams(quantity, unit, name)
        
        # จำกัดน้ำหนักสูงสุดไม่ให้มากเกินจริง
        max_weights = {
            'พริก': 10,
            'กระเทียม': 50,
            'ใบ': 20,
            'เครื่องเทศ': 30,
            'น้ำปลา': 100,
            'ซีอิ้ว': 100,
            'น้ำมัน': 200,
        }
        
        for key, max_weight in max_weights.items():
            if key in name.lower() and weight_grams > max_weight:
                weight_grams = max_weight
        
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
        "ไข่ไก่ 1 ฟอง",      # ควรได้ 50 กรัม
        "ไข่เป็ด 1 ฟอง",     # ควรได้ 60 กรัม
        "น้ำมันหมู 1 ช้อนโต๊ะ",  # ควรได้ 13.5 กรัม (15ml * 0.9)
        "กุ้ง 200 กรัม",
        "หมูสับ 1/2 กิโลกรัม",
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
