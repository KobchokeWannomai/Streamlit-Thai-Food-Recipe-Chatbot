import requests
import json
import pandas as pd
import re
import os
from typing import Dict, List, Optional, Tuple

class NutritionAPI:
    """คลาสสำหรับจัดการข้อมูลโภชนาการจาก API และฐานข้อมูลท้องถิ่น - เวอร์ชันปรับปรุงแล้ว"""
    
    def __init__(self):
        # URL ของ API ข้อมูลโภชนาการต่างๆ
        self.api_urls = {
            "usda": "https://api.nal.usda.gov/fdc/v1",
            "nutritionix": "https://trackapi.nutritionix.com/v2",
            "edamam": "https://api.edamam.com/api/nutrition-data/v2/nutrient"
        }
        
        # API Keys และ credentials
        self.api_credentials = {
            "usda": {"api_key": None},
            "nutritionix": {"app_id": None, "app_key": None},
            "edamam": {"app_id": None, "app_key": None}
        }
        
        self.current_api_source = "local"
        
        # โหลดฐานข้อมูลโภชนาการท้องถิ่น
        self.local_nutrition_db = self.load_local_nutrition_database()
        
        # หน่วยแปลงที่พบบ่อยในอาหารไทย - ปรับปรุงให้ครอบคลุมมากขึ้น
        self.unit_conversion = {
            # หน่วยปริมาตร
            "ช้อนโต๊ะ": 15, "ชต": 15, "tbsp": 15,
            "ช้อนชา": 5, "ชช": 5, "tsp": 5,
            "ถ้วย": 240, "cup": 240,
            "ถ้วยชา": 150,
            "ถ้วยข้าว": 180,
            "แก้ว": 200,
            "ลิตร": 1000, "l": 1000,
            "มิลลิลิตร": 1, "มล": 1, "ml": 1,
            
            # หน่วยน้ำหนัก
            "กิโลกรัม": 1000, "กก": 1000, "kg": 1000,
            "กรัม": 1, "g": 1, "gram": 1,
            "ขีด": 15,  # หน่วยไทยโบราณ
            "บาท": 15,  # หน่วยไทยโบราณ
            "ออนซ์": 28.35, "oz": 28.35,
            "ปอนด์": 453.6, "lb": 453.6,
            
            # หน่วยนับ (ประมาณการสำหรับอาหารไทย) - ปรับปรุงให้แม่นยำขึ้น
            "ตัว": 100,  # กุ้งตัวกลาง, ปลาตัวกลาง
            "ฟอง": 50,   # ไข่ฟองกลาง
            "หัว": 50,    # หอมใหญ่หัวกลาง
            "กลีบ": 3,    # กระเทียม 1 กลีบ
            "ต้น": 30,    # ผักชี 1 ต้น
            "ใบ": 2,      # ใบมะกรูด
            "เม็ด": 0.5,  # พริกไทย 1 เม็ด
            "แว่น": 2,    # ข่า 1 แว่น
            "ผล": 150,    # มะนาว 1 ผล
            "ราก": 5,     # รากผักชี 1 ราก
            "ท่อน": 20,   # ตะไคร้ 1 ท่อน
            "ก้อน": 30,   # ก้อนเล็ก
            "หยิบ": 5,    # หยิบมือ
            "ข้อมือ": 100, # ผักข้อมือหนึ่ง
            "จี้": 5,     # หอมจี้หนึ่ง
            "ลูก": 80,    # ลูกกลาง เช่น มะเขือเทศ
            "แผ่น": 10,   # แผ่นเล็ก เช่น ข่าแผ่นหนึ่ง
            "ชิ้น": 50,   # ชิ้นกลาง
            "เส้น": 1     # เส้นผัก เช่น ถั่วฝักยาว
        }
        
        # สัดส่วนที่บริโภคจริง - ปรับปรุงให้ครอบคลุมและแม่นยำขึ้น
        self.consumption_ratio = {
            # น้ำมันสำหรับทอด/ผัด - ไม่กินหมด
            "น้ำมันหมู": 0.25,        # ใช้ทอดแต่ไม่กินหมด
            "น้ำมันพืช": 0.25,         # ใช้ทอดแต่ไม่กินหมด
            "น้ำมันมะพร้าว": 0.25,     # ใช้ทอดแต่ไม่กินหมด
            "น้ำมันรำข้าว": 0.25,      # ใช้ทอดแต่ไม่กินหมด
            "น้ำมันงา": 0.3,           # ใช้ปรุงรสส่วนใหญ่กิน
            
            # ของเหลวในการปรุง
            "กะทิ": 0.85,              # ใช้ในแกงส่วนใหญ่จะกิน
            "น้ำปลา": 1.0,             # ใช้ปรุงรสกินหมด
            "ซีอิ้วขาว": 1.0,          # ใช้ปรุงรสกินหมด
            "ซีอิ้วหวาน": 1.0,         # ใช้ปรุงรสกินหมด
            "ซีอิ้วดำ": 1.0,           # ใช้ปรุงรสกินหมด
            "น้ำส้มสายชู": 1.0,        # ใช้ปรุงรสกินหมด
            
            # เครื่องปรุงรส
            "น้ำตาล": 1.0,             # ใช้ปรุงรสกินหมด
            "น้ำตาลทราย": 1.0,         # ใช้ปรุงรสกินหมด
            "น้ำตาลปึก": 1.0,          # ใช้ปรุงรสกินหมด
            "น้ำตาลโตนด": 1.0,         # ใช้ปรุงรสกินหมด
            "เกลือ": 1.0,              # ใช้ปรุงรสกินหมด
            "พริกไทย": 1.0,            # ใช้ปรุงรสกินหมด
            "พริกไทยป่น": 1.0,         # ใช้ปรุงรสกินหมด
            
            # เครื่องเทศและสมุนไพร
            "กระเทียม": 0.9,           # ส่วนใหญ่กิน
            "หอมแดง": 0.9,            # ส่วนใหญ่กิน
            "หอมใหญ่": 0.9,           # ส่วนใหญ่กิน
            "หอมหัวใหญ่": 0.9,        # ส่วนใหญ่กิน
            "ผักชี": 0.95,             # เกือบทั้งหมด
            "ใบผักชี": 0.95,          # เกือบทั้งหมด
            "รากผักชี": 0.8,          # บางส่วนอาจไม่กิน
            "ต้นหอม": 0.95,           # เกือบทั้งหมด
            
            # เครื่องแกงและเครื่องปรุงพิเศษ
            "พริกแห้ง": 1.0,          # กินหมด
            "พริกสด": 1.0,            # กินหมด
            "พริกชี้ฟ้า": 1.0,         # กินหมด
            "พริกขี้หนู": 1.0,         # กินหมด
            "ข่า": 0.7,               # บางส่วนกิน บางส่วนใช้หอม
            "ตะไคร้": 0.6,            # ส่วนใหญ่ใช้หอม
            "ใบมะกรูด": 0.8,          # ส่วนใหญ่กิน
            "กะปิ": 1.0,              # กินหมด
            "ขมิ้น": 1.0,             # กินหมด
            
            # ผักและสมุนไพร
            "ดอกกะเพรา": 0.95,       # เกือบทั้งหมด
            "ใบกะเพรา": 0.95,        # เกือบทั้งหมด
            "ใบโหระพา": 0.95,        # เกือบทั้งหมด
            "ผักบุ้ง": 0.9,           # ส่วนใหญ่กิน
            "คะน้า": 0.9,             # ส่วนใหญ่กิน
            "ผักกาดขาว": 0.9,         # ส่วนใหญ่กิน
            "ผักกาดหอม": 0.9,         # ส่วนใหญ่กิน
            
            # ผลไม้และของหวาน
            "มะนาว": 0.8,             # ใช้น้ำส่วนใหญ่
            "มะขามเปียก": 0.9,        # ส่วนใหญ่กิน
            "มะพร้าว": 0.85,          # บางส่วนเป็นกาก
            "มะพร้าวขูด": 0.95,       # เกือบทั้งหมด
            
            # อื่นๆ
            "เนย": 0.8,                # บางส่วนอาจไม่กิน
            "หอมเจียว": 0.9,           # ส่วนใหญ่กิน
            "กระเทียมเจียว": 0.9,      # ส่วนใหญ่กิน
            "น้ำเชื่อม": 1.0,          # กินหมด
            "นมสด": 1.0,              # ดื่มหมด
        }
        
        # วัตถุดิบที่มักขาดหายไปตามประเภทอาหาร - ปรับปรุงให้ครอบคลุมขึ้น
        self.missing_ingredients_by_cooking_method = {
            "ทอด": {
                "required": ["น้ำมันพืช 3 ช้อนโต๊ะ"],
                "optional": []
            },
            "เจียว": {
                "required": ["น้ำมันหมู 2 ช้อนโต๊ะ"],
                "optional": []
            },
            "ผัด": {
                "required": ["น้ำมันพืช 2 ช้อนโต๊ะ"],
                "optional": ["กระเทียม 2 กลีบ", "หอมแดง 2 หัว"]
            },
            "คั่ว": {
                "required": ["น้ำมันพืช 1 ช้อนโต๊ะ"],
                "optional": []
            },
            "ต้ม": {
                "required": [],
                "optional": ["เกลือ 1 ช้อนชา"]
            },
            "แกง": {
                "required": [],
                "optional": ["กะทิ 400 มล", "น้ำปลา 2 ช้อนโต๊ะ"]
            },
            "ย่าง": {
                "required": [],
                "optional": ["น้ำมันพืช 1 ช้อนชา"]
            },
            "ปิ้ง": {
                "required": [],
                "optional": ["น้ำมันพืช 1 ช้อนชา"]
            },
            "นึ่ง": {
                "required": [],
                "optional": []
            },
            "ยำ": {
                "required": [],
                "optional": ["น้ำปลา 2 ช้อนโต๊ะ", "มะนาว 2 ผล", "น้ำตาลปึก 2 ช้อนชา", "พริกขี้หนู 3 เม็ด"]
            },
            "คั่ว": {
                "required": ["น้ำมันพืช 1 ช้อนโต๊ะ"],
                "optional": []
            },
            "อบ": {
                "required": [],
                "optional": ["น้ำมันพืช 1 ช้อนชา"]
            }
        }

        # รายการวัตถุดิบที่ขาดหายไปตามชื่อเมนู - เพิ่มเติม
        self.recipe_specific_ingredients = {
            'ไข่เจียว': ['น้ำมันหมู 2 ช้อนโต๊ะ'],
            'ไข่ดาว': ['น้ำมันหมู 2 ช้อนโต๊ะ'],
            'ไข่ทอด': ['น้ำมันพืช 3 ช้อนโต๊ะ'],
            'ปลาทอด': ['น้ำมันพืช 1 ถ้วย', 'แป้งสาลี 3 ช้อนโต๊ะ'],
            'เนื้อทอด': ['น้ำมันพืช 1 ถ้วย', 'แป้งสาลี 2 ช้อนโต๊ะ'],
            'ไก่ทอด': ['น้ำมันพืช 1 ถ้วย', 'แป้งสาลี 3 ช้อนโต๊ะ'],
            'ข้าวผัด': ['น้ำมันพืช 2 ช้อนโต๊ะ', 'ไข่ไก่ 2 ฟอง'],
            'ข้าวผัดกุ้ง': ['น้ำมันพืช 2 ช้อนโต๊ะ', 'ไข่ไก่ 2 ฟอง'],
            'ข้าวผัดหมู': ['น้ำมันพืช 2 ช้อนโต๊ะ', 'ไข่ไก่ 2 ฟอง'],
            'ผัดไทย': ['น้ำมันพืช 3 ช้อนโต๊ะ', 'ไข่ไก่ 2 ฟอง'],
            'ผัดซีอิ้ว': ['น้ำมันพืช 2 ช้อนโต๊ะ'],
            'ผัดขี้เมา': ['น้ำมันพืช 2 ช้อนโต๊ะ'],
            'ก๋วยเตี๋ยว': ['น้ำซุป 2 ถ้วย'],
            'ก๋วยเตี๋ยวเรือ': ['น้ำซุป 2 ถ้วย'],
            'ราดหน้า': ['น้ำมันพืช 2 ช้อนโต๊ะ', 'แป้งข้าวโพด 2 ช้อนโต๊ะ'],
            'ผัดกะเพรา': ['น้ำมันพืช 2 ช้อนโต๊ะ'],
            'ผัดกะเพราหมู': ['น้ำมันพืช 2 ช้อนโต๊ะ'],
            'ผัดกะเพราไก่': ['น้ำมันพืช 2 ช้อนโต๊ะ'],
            'ผัดกะเพรากุ้ง': ['น้ำมันพืช 2 ช้อนโต๊ะ']
        }

    def load_local_nutrition_database(self) -> Dict:
        """โหลดฐานข้อมูลโภชนาการท้องถิ่นจากไฟล์ CSV"""
        nutrition_db = {}
        
        # ลองโหลดจากไฟล์ CSV
        csv_file = "thai_ingredients_nutrition.csv"
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                for _, row in df.iterrows():
                    ingredient_name = row['ingredient']
                    nutrition_data = {
                        "calories": row.get('calories', 0),
                        "protein": row.get('protein', 0),
                        "carbs": row.get('carbs', 0),
                        "fat": row.get('fat', 0),
                        "fiber": row.get('fiber', 0),
                        "vitamin_a": row.get('vitamin_a', 0),
                        "vitamin_c": row.get('vitamin_c', 0),
                        "vitamin_b1": row.get('vitamin_b1', 0),
                        "vitamin_b2": row.get('vitamin_b2', 0),
                        "calcium": row.get('calcium', 0),
                        "iron": row.get('iron', 0),
                        "potassium": row.get('potassium', 0),
                        "sodium": row.get('sodium', 0)
                    }
                    nutrition_db[ingredient_name] = nutrition_data
                    
                print(f"โหลดข้อมูลโภชนาการจากไฟล์ CSV สำเร็จ: {len(nutrition_db)} รายการ")
                return nutrition_db
                
            except Exception as e:
                print(f"Error loading nutrition CSV: {e}")
                pass
        
        # หากไม่มีไฟล์ CSV ใช้ข้อมูลเริ่มต้น
        print("ใช้ข้อมูลโภชนาการเริ่มต้น")
        return self.get_default_nutrition_database()

    def get_default_nutrition_database(self) -> Dict:
        """ฐานข้อมูลโภชนาการเริ่มต้นสำหรับวัตถุดิบไทย (ต่อ 100 กรัม)"""
        return {
            "ข้าวสาร": {
                "calories": 130, "protein": 2.7, "carbs": 28, "fat": 0.3, "fiber": 0.4,
                "vitamin_a": 0, "vitamin_c": 0, "vitamin_b1": 0.07, "vitamin_b2": 0.02,
                "calcium": 10, "iron": 0.8, "potassium": 115, "sodium": 5
            },
            "ไข่ไก่": {
                "calories": 155, "protein": 13, "carbs": 1.1, "fat": 11, "fiber": 0,
                "vitamin_a": 540, "vitamin_c": 0, "vitamin_b1": 0.04, "vitamin_b2": 0.42,
                "calcium": 56, "iron": 1.75, "potassium": 138, "sodium": 124
            },
            "ไข่เป็ด": {
                "calories": 185, "protein": 13, "carbs": 1.4, "fat": 14, "fiber": 0,
                "vitamin_a": 674, "vitamin_c": 0, "vitamin_b1": 0.11, "vitamin_b2": 0.44,
                "calcium": 64, "iron": 2.7, "potassium": 222, "sodium": 146
            },
            "กุ้งนาง": {
                "calories": 99, "protein": 18, "carbs": 0.2, "fat": 1.4, "fiber": 0,
                "vitamin_a": 54, "vitamin_c": 2.1, "vitamin_b1": 0.02, "vitamin_b2": 0.04,
                "calcium": 70, "iron": 0.5, "potassium": 259, "sodium": 111
            },
            "กุ้งตะเข็บ": {
                "calories": 105, "protein": 20, "carbs": 0.5, "fat": 1.8, "fiber": 0,
                "vitamin_a": 48, "vitamin_c": 1.8, "vitamin_b1": 0.02, "vitamin_b2": 0.05,
                "calcium": 65, "iron": 0.6, "potassium": 240, "sodium": 120
            },
            "เนื้อหมู": {
                "calories": 242, "protein": 27, "carbs": 0, "fat": 14, "fiber": 0,
                "vitamin_a": 2, "vitamin_c": 0.7, "vitamin_b1": 0.66, "vitamin_b2": 0.23,
                "calcium": 19, "iron": 0.87, "potassium": 423, "sodium": 62
            },
            "เนื้อโค": {
                "calories": 250, "protein": 26, "carbs": 0, "fat": 15, "fiber": 0,
                "vitamin_a": 0, "vitamin_c": 0, "vitamin_b1": 0.06, "vitamin_b2": 0.15,
                "calcium": 18, "iron": 2.6, "potassium": 318, "sodium": 55
            },
            "เนื้อไก่": {
                "calories": 165, "protein": 31, "carbs": 0, "fat": 3.6, "fiber": 0,
                "vitamin_a": 21, "vitamin_c": 1.6, "vitamin_b1": 0.07, "vitamin_b2": 0.12,
                "calcium": 15, "iron": 1.3, "potassium": 256, "sodium": 82
            },
            "ปลาช่อน": {
                "calories": 112, "protein": 18.7, "carbs": 0, "fat": 3.6, "fiber": 0,
                "vitamin_a": 45, "vitamin_c": 0.9, "vitamin_b1": 0.02, "vitamin_b2": 0.11,
                "calcium": 89, "iron": 0.9, "potassium": 358, "sodium": 54
            },
            "ปลาทู": {
                "calories": 146, "protein": 20.1, "carbs": 0, "fat": 6.4, "fiber": 0,
                "vitamin_a": 167, "vitamin_c": 0, "vitamin_b1": 0.1, "vitamin_b2": 0.23,
                "calcium": 150, "iron": 1.2, "potassium": 240, "sodium": 76
            },
            "ปลาหมึก": {
                "calories": 92, "protein": 15.6, "carbs": 3.1, "fat": 1.4, "fiber": 0,
                "vitamin_a": 33, "vitamin_c": 4.8, "vitamin_b1": 0.02, "vitamin_b2": 0.4,
                "calcium": 32, "iron": 0.7, "potassium": 246, "sodium": 44
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
            "น้ำมันมะพร้าว": {
                "calories": 862, "protein": 0, "carbs": 0, "fat": 100, "fiber": 0,
                "vitamin_a": 0, "vitamin_c": 0, "vitamin_b1": 0, "vitamin_b2": 0,
                "calcium": 0, "iron": 0, "potassium": 0, "sodium": 0
            },
            "กระเทียม": {
                "calories": 149, "protein": 6.4, "carbs": 33, "fat": 0.5, "fiber": 2.1,
                "vitamin_a": 9, "vitamin_c": 31, "vitamin_b1": 0.2, "vitamin_b2": 0.11,
                "calcium": 181, "iron": 1.7, "potassium": 401, "sodium": 17
            },
            "หอมแดง": {
                "calories": 40, "protein": 1.1, "carbs": 9.3, "fat": 0.1, "fiber": 1.7,
                "vitamin_a": 2, "vitamin_c": 7.4, "vitamin_b1": 0.05, "vitamin_b2": 0.03,
                "calcium": 23, "iron": 0.21, "potassium": 146, "sodium": 4
            },
            "หอมใหญ่": {
                "calories": 42, "protein": 1.2, "carbs": 9.9, "fat": 0.1, "fiber": 1.9,
                "vitamin_a": 0, "vitamin_c": 8.1, "vitamin_b1": 0.05, "vitamin_b2": 0.03,
                "calcium": 25, "iron": 0.25, "potassium": 157, "sodium": 4
            },
            "หอมหัวใหญ่": {
                "calories": 42, "protein": 1.2, "carbs": 9.9, "fat": 0.1, "fiber": 1.9,
                "vitamin_a": 0, "vitamin_c": 8.1, "vitamin_b1": 0.05, "vitamin_b2": 0.03,
                "calcium": 25, "iron": 0.25, "potassium": 157, "sodium": 4
            },
            "ผักชี": {
                "calories": 23, "protein": 2.1, "carbs": 3.7, "fat": 0.5, "fiber": 2.8,
                "vitamin_a": 3377, "vitamin_c": 27, "vitamin_b1": 0.07, "vitamin_b2": 0.16,
                "calcium": 67, "iron": 1.77, "potassium": 521, "sodium": 46
            },
            "ใบผักชี": {
                "calories": 23, "protein": 2.1, "carbs": 3.7, "fat": 0.5, "fiber": 2.8,
                "vitamin_a": 3377, "vitamin_c": 27, "vitamin_b1": 0.07, "vitamin_b2": 0.16,
                "calcium": 67, "iron": 1.77, "potassium": 521, "sodium": 46
            },
            "รากผักชี": {
                "calories": 25, "protein": 2.3, "carbs": 4.1, "fat": 0.6, "fiber": 3.2,
                "vitamin_a": 3500, "vitamin_c": 30, "vitamin_b1": 0.08, "vitamin_b2": 0.18,
                "calcium": 75, "iron": 2.0, "potassium": 580, "sodium": 52
            },
            "ต้นหอม": {
                "calories": 32, "protein": 1.8, "carbs": 7.3, "fat": 0.2, "fiber": 2.6,
                "vitamin_a": 997, "vitamin_c": 18.8, "vitamin_b1": 0.05, "vitamin_b2": 0.08,
                "calcium": 72, "iron": 1.5, "potassium": 276, "sodium": 16
            },
            "พริกไทย": {
                "calories": 251, "protein": 10.4, "carbs": 64, "fat": 3.3, "fiber": 25,
                "vitamin_a": 547, "vitamin_c": 0, "vitamin_b1": 0.11, "vitamin_b2": 0.18,
                "calcium": 443, "iron": 9.7, "potassium": 1329, "sodium": 20
            },
            "พริกแห้ง": {
                "calories": 324, "protein": 12, "carbs": 56, "fat": 17, "fiber": 28.7,
                "vitamin_a": 21600, "vitamin_c": 76.4, "vitamin_b1": 0.33, "vitamin_b2": 0.92,
                "calcium": 148, "iron": 7.8, "potassium": 1870, "sodium": 91
            },
            "พริกสด": {
                "calories": 40, "protein": 1.9, "carbs": 9.5, "fat": 0.4, "fiber": 1.5,
                "vitamin_a": 952, "vitamin_c": 144, "vitamin_b1": 0.07, "vitamin_b2": 0.09,
                "calcium": 14, "iron": 1.03, "potassium": 322, "sodium": 9
            },
            "พริกชี้ฟ้า": {
                "calories": 40, "protein": 1.9, "carbs": 9.5, "fat": 0.4, "fiber": 1.5,
                "vitamin_a": 952, "vitamin_c": 144, "vitamin_b1": 0.07, "vitamin_b2": 0.09,
                "calcium": 14, "iron": 1.03, "potassium": 322, "sodium": 9
            },
            "พริกขี้หนู": {
                "calories": 318, "protein": 12, "carbs": 56, "fat": 17, "fiber": 28.7,
                "vitamin_a": 21600, "vitamin_c": 76.4, "vitamin_b1": 0.33, "vitamin_b2": 0.92,
                "calcium": 148, "iron": 7.8, "potassium": 1870, "sodium": 91
            },
            "น้ำปลา": {
                "calories": 42, "protein": 5.8, "carbs": 1.5, "fat": 0.8, "fiber": 0,
                "vitamin_a": 0, "vitamin_c": 0, "vitamin_b1": 0.03, "vitamin_b2": 0.22,
                "calcium": 85, "iron": 2.03, "potassium": 84, "sodium": 6976
            },
            "น้ำตาลทราย": {
                "calories": 387, "protein": 0, "carbs": 100, "fat": 0, "fiber": 0,
                "vitamin_a": 0, "vitamin_c": 0, "vitamin_b1": 0, "vitamin_b2": 0,
                "calcium": 1, "iron": 0.01, "potassium": 2, "sodium": 1
            },
            "น้ำตาลปึก": {
                "calories": 375, "protein": 0.1, "carbs": 97, "fat": 0.1, "fiber": 0,
                "vitamin_a": 0, "vitamin_c": 0, "vitamin_b1": 0.01, "vitamin_b2": 0.01,
                "calcium": 85, "iron": 1.9, "potassium": 133, "sodium": 39
            },
            "น้ำตาลโตนด": {
                "calories": 383, "protein": 0.4, "carbs": 98.1, "fat": 0.1, "fiber": 0,
                "vitamin_a": 0, "vitamin_c": 0, "vitamin_b1": 0.02, "vitamin_b2": 0.02,
                "calcium": 67, "iron": 2.26, "potassium": 162, "sodium": 35
            },
            "มะนาว": {
                "calories": 29, "protein": 0.7, "carbs": 9.3, "fat": 0.2, "fiber": 2.8,
                "vitamin_a": 22, "vitamin_c": 53, "vitamin_b1": 0.03, "vitamin_b2": 0.02,
                "calcium": 33, "iron": 0.6, "potassium": 138, "sodium": 2
            },
            "มะขามเปียก": {
                "calories": 239, "protein": 2.8, "carbs": 62.5, "fat": 0.6, "fiber": 5.1,
                "vitamin_a": 30, "vitamin_c": 3.5, "vitamin_b1": 0.43, "vitamin_b2": 0.15,
                "calcium": 74, "iron": 2.8, "potassium": 628, "sodium": 28
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
            "มะพร้าวขูด": {
                "calories": 354, "protein": 3.3, "carbs": 15, "fat": 33, "fiber": 9,
                "vitamin_a": 0, "vitamin_c": 3.3, "vitamin_b1": 0.07, "vitamin_b2": 0.02,
                "calcium": 14, "iron": 2.43, "potassium": 356, "sodium": 20
            },
            "เกลือ": {
                "calories": 0, "protein": 0, "carbs": 0, "fat": 0, "fiber": 0,
                "vitamin_a": 0, "vitamin_c": 0, "vitamin_b1": 0, "vitamin_b2": 0,
                "calcium": 24, "iron": 0.33, "potassium": 8, "sodium": 38758
            }
        }

    def set_api_key(self, api_key: str, source: str = "usda"):
        """ตั้งค่า API key สำหรับแหล่งข้อมูลที่เลือก"""
        if source in self.api_credentials:
            if source == "usda":
                self.api_credentials[source]["api_key"] = api_key
                self.current_api_source = source

    def set_nutritionix_credentials(self, app_id: str, app_key: str):
        """ตั้งค่า credentials สำหรับ Nutritionix API"""
        self.api_credentials["nutritionix"]["app_id"] = app_id
        self.api_credentials["nutritionix"]["app_key"] = app_key
        self.current_api_source = "nutritionix"

    def set_edamam_credentials(self, app_id: str, app_key: str):
        """ตั้งค่า credentials สำหรับ Edamam API"""
        self.api_credentials["edamam"]["app_id"] = app_id
        self.api_credentials["edamam"]["app_key"] = app_key
        self.current_api_source = "edamam"

    def normalize_ingredient_name(self, ingredient: str) -> str:
        """ปรับแต่งชื่อวัตถุดิบให้เป็นมาตรฐาน - ปรับปรุงให้ครอบคลุมมากขึ้น"""
        # ลบข้อความที่ไม่จำเป็น
        ingredient = re.sub(r'\d+.*', '', ingredient)  # ลบตัวเลขและข้อความที่ตามมา
        ingredient = re.sub(r'[^\u0E00-\u0E7Fa-zA-Z\s]', '', ingredient)  # เก็บเฉพาะตัวอักษรไทย-อังกฤษ
        ingredient = ingredient.strip()
        
        # แปลงคำพ้องความหมาย - เพิ่มเติม
        synonyms = {
            "กุ้ง": ["กุ้งนาง", "กุ้งตะเข็บ", "กุ้งฝอย", "กุ้งสด"],
            "หมู": ["เนื้อหมู", "หมูสับ", "สันในหมู", "สันคอหมู", "หมูกรอบ"],
            "เนื้อ": ["เนื้อโค", "เนื้อวัว"],
            "ไก่": ["เนื้อไก่", "ไก่สับ", "อกไก่", "น่องไก่", "ปีกไก่"],
            "หอมใหญ่": ["หอมหัวใหญ่"],
            "หอมแดง": ["หอมหัวเล็ก", "หัวหอม", "หอมสับ"],
            "น้ำมันพืช": ["น้ำมันรำ", "น้ำมันถั่วเหลือง", "น้ำมันดอกทานตะวัน"],
            "น้ำตาล": ["น้ำตาลทราย", "น้ำตาลขาว", "น้ำตาลแดง"],
            "ผักชี": ["ใบผักชี", "รากผักชี"],
            "ซีอิ้ว": ["ซีอิ้วขาว", "ซีอิ้วหวาน", "ซีอิ้วดำ"],
            "พริก": ["พริกสด", "พริกแห้ง", "พริกชี้ฟ้า", "พริกขี้หนู"],
            "ไข่": ["ไข่ไก่", "ไข่เป็ด"],
            "ปลา": ["ปลาช่อน", "ปลาทู", "ปลาหมึก", "ปลากะพง", "ปลาดุก"]
        }
        
        # หาและแทนที่คำพ้องความหมาย
        for standard, variants in synonyms.items():
            for variant in variants:
                if variant in ingredient:
                    ingredient = ingredient.replace(variant, standard)
                    break
            if ingredient in variants:
                ingredient = standard
                break
        
        return ingredient

    def extract_quantity_and_unit(self, ingredient_text: str) -> Tuple[float, str, str]:
        """แยกปริมาณ หน่วย และชื่อวัตถุดิบอย่างแม่นยำ - ปรับปรุงให้ดีขึ้น"""
        ingredient_text = ingredient_text.strip()
        
        # รูปแบบการหาปริมาณและหน่วย (รองรับเศษส่วนและทศนิยม)
        patterns = [
            # รูปแบบ: ตัวเลข + หน่วย + ชื่อวัตถุดิบ
            r'(\d+(?:\.\d+)?(?:/\d+)?)\s*([ก-๙a-zA-Z]+)\s*(.+)',
            # รูปแบบ: ชื่อวัตถุดิบ + ตัวเลข + หน่วย
            r'(.+?)\s+(\d+(?:\.\d+)?(?:/\d+)?)\s*([ก-๙a-zA-Z]+)(?:\s|$)',
            # รูปแบบ: เฉพาะชื่อวัตถุดิบ
            r'(.+)'
        ]
        
        for pattern in patterns:
            match = re.match(pattern, ingredient_text)
            if match:
                groups = match.groups()
                
                if len(groups) == 3 and self.is_number(groups[0]):
                    # รูปแบบ: เลข + หน่วย + ชื่อ
                    quantity = self.parse_number(groups[0])
                    unit = groups[1]
                    ingredient = groups[2]
                elif len(groups) == 3 and self.is_number(groups[1]):
                    # รูปแบบ: ชื่อ + เลข + หน่วย
                    ingredient = groups[0]
                    quantity = self.parse_number(groups[1])
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

    def is_number(self, text: str) -> bool:
        """ตรวจสอบว่าข้อความเป็นตัวเลขหรือไม่ (รองรับเศษส่วน)"""
        try:
            self.parse_number(text)
            return True
        except:
            return False

    def parse_number(self, text: str) -> float:
        """แปลงข้อความเป็นตัวเลข (รองรับเศษส่วน)"""
        text = text.strip()
        
        # หากเป็นเศษส่วน เช่น 1/2
        if '/' in text:
            parts = text.split('/')
            if len(parts) == 2:
                numerator = float(parts[0])
                denominator = float(parts[1])
                return numerator / denominator
        
        # หากเป็นทศนิยมหรือจำนวนเต็ม
        return float(text)

    def estimate_default_quantity(self, ingredient: str) -> float:
        """ประมาณปริมาณเริ่มต้นสำหรับวัตถุดิบที่ไม่ระบุปริมาณ - ปรับปรุงให้แม่นยำขึ้น"""
        ingredient_lower = ingredient.lower()
        
        # ปริมาณเริ่มต้นที่ปรับปรุงแล้ว
        defaults = {
            "น้ำมัน": 2,        # 2 ช้อนโต๊ะ
            "น้ำปลา": 1.5,      # 1.5 ช้อนโต๊ะ
            "ซีอิ้ว": 1,        # 1 ช้อนโต๊ะ
            "น้ำตาล": 1,        # 1 ช้อนชา
            "เกลือ": 0.5,       # 0.5 ช้อนชา
            "พริกไทย": 3,       # 3 เม็ด
            "กระเทียม": 3,      # 3 กลีบ
            "หอม": 2,           # 2 หัว
            "ผักชี": 2,         # 2 ต้น
            "ไข่": 2,           # 2 ฟอง
            "เนื้อ": 200,       # 200 กรัม
            "หมู": 200,         # 200 กรัม
            "ไก่": 250,         # 250 กรัม
            "กุ้ง": 150,        # 150 กรัม
            "ปลา": 300,         # 300 กรัม
            "ข่า": 3,           # 3 แว่น
            "ตะไคร้": 2,        # 2 ท่อน
            "ใบมะกรูด": 5,      # 5 ใบ
            "กะทิ": 200,        # 200 มล
            "มะนาว": 2,         # 2 ผล
            "ผัก": 100,         # 100 กรัม
            "เห็ด": 100,        # 100 กรัม
            "พริก": 5,          # 5 เม็ด
            "มะเขือ": 3,        # 3 ลูก
            "ถั่ว": 50,         # 50 กรัม
        }
        
        for key, value in defaults.items():
            if key in ingredient_lower:
                return value
        
        return 100  # ค่าเริ่มต้น 100 กรัม

    def estimate_default_unit(self, ingredient: str) -> str:
        """ประมาณหน่วยเริ่มต้นสำหรับวัตถุดิบ - ปรับปรุงให้ครอบคลุมขึ้น"""
        ingredient_lower = ingredient.lower()
        
        unit_map = {
            "น้ำมัน": "ช้อนโต๊ะ", "ซีอิ้ว": "ช้อนโต๊ะ", "น้ำปลา": "ช้อนโต๊ะ",
            "น้ำตาล": "ช้อนชา", "เกลือ": "ช้อนชา",
            "พริกไทย": "เม็ด", "กระเทียม": "กลีบ", "หอม": "หัว",
            "ผักชี": "ต้น", "ไข่": "ฟอง", "ข่า": "แว่น", "ตะไคร้": "ท่อน",
            "ใบมะกรูด": "ใบ", "กะทิ": "มล", "มะนาว": "ผล",
            "เนื้อ": "กรัม", "หมู": "กรัม", "ไก่": "กรัม", "กุ้ง": "กรัม",
            "ปลา": "กรัม", "ผัก": "กรัม", "เห็ด": "กรัม", "พริก": "เม็ด",
            "มะเขือ": "ลูก", "ถั่ว": "กรัม"
        }
        
        for key, unit in unit_map.items():
            if key in ingredient_lower:
                return unit
        
        return "กรัม"

    def convert_to_grams(self, quantity: float, unit: str, ingredient: str) -> float:
        """แปลงปริมาณเป็นกรัมอย่างแม่นยำ - ปรับปรุงให้ดีขึ้น"""
        # หากหน่วยเป็นกรัมอยู่แล้ว
        if unit.lower() in ["กรัม", "g", "gram", "grams"]:
            return quantity
        
        # แปลงจากหน่วยอื่น
        if unit in self.unit_conversion:
            base_amount = quantity * self.unit_conversion[unit]
            
            # สำหรับของเหลว (มล. -> กรัม ใช้ density)
            if unit in ["ช้อนโต๊ะ", "ช้อนชา", "ถ้วย", "ถ้วยชา", "มล", "ลิตร", "แก้ว"]:
                # ความหนาแน่นต่างกันตามชนิดวัตถุดิบ
                density_map = {
                    "น้ำมัน": 0.92, "กะทิ": 0.95, "น้ำปลา": 1.1,
                    "ซีอิ้ว": 1.15, "น้ำส้มสายชู": 1.05, "น้ำตาล": 1.6,
                    "เกลือ": 2.16, "นม": 1.03, "น้ำ": 1.0
                }
                
                density = 1.0  # ค่าเริ่มต้น
                ingredient_lower = ingredient.lower()
                for key, value in density_map.items():
                    if key in ingredient_lower:
                        density = value
                        break
                
                return base_amount * density
            else:
                return base_amount
        
        # หากไม่พบหน่วย ใช้ค่าตามที่ระบุ (สมมติเป็นกรัม)
        return quantity

    def enhance_missing_ingredients(self, ingredients_text: str, recipe_name: str = "", method_text: str = "") -> str:
        """เพิ่มวัตถุดิบที่ขาดหายไปตามวิธีการทำและชื่อเมนู - ปรับปรุงให้ทำงานได้ดีขึ้น"""
        enhanced_ingredients = ingredients_text
        missing_ingredients = []
        
        method_lower = method_text.lower() if method_text else ""
        ingredients_lower = ingredients_text.lower() if ingredients_text else ""
        recipe_name_lower = recipe_name.lower() if recipe_name else ""
        
        print(f"DEBUG: กำลังตรวจสอบวัตถุดิบที่ขาดหาย สำหรับ '{recipe_name}'")
        print(f"DEBUG: วิธีทำ: {method_lower[:100]}...")
        
        # ตรวจสอบวิธีการทำและเพิ่มวัตถุดิบที่ขาดหาย
        for cooking_method, ingredients_dict in self.missing_ingredients_by_cooking_method.items():
            if cooking_method in method_lower:
                print(f"DEBUG: พบวิธีการทำ '{cooking_method}' ในเมนู")
                
                # วัตถุดิบจำเป็น
                for ingredient in ingredients_dict["required"]:
                    ingredient_name = ingredient.split()[0]  # เอาชื่อวัตถุดิบ
                    if ingredient_name not in ingredients_lower:
                        missing_ingredients.append(f"- {ingredient}")
                        print(f"DEBUG: เพิ่มวัตถุดิบจำเป็น: {ingredient}")
                
                # วัตถุดิบเสริม (หากรายการวัตถุดิบน้อย)
                if len(ingredients_text.split('\n')) <= 5:
                    for ingredient in ingredients_dict["optional"]:
                        ingredient_name = ingredient.split()[0]
                        if ingredient_name not in ingredients_lower:
                            missing_ingredients.append(f"- {ingredient}")
                            print(f"DEBUG: เพิ่มวัตถุดิบเสริม: {ingredient}")
        
        # ตรวจสอบชื่อเมนูและเพิ่มวัตถุดิบที่ขาดหาย
        for recipe_pattern, required_ingredients in self.recipe_specific_ingredients.items():
            if recipe_pattern in recipe_name_lower:
                print(f"DEBUG: พบรูปแบบเมนู '{recipe_pattern}' ตรงกับ '{recipe_name}'")
                
                for ingredient in required_ingredients:
                    ingredient_name = ingredient.split()[0]
                    if ingredient_name not in ingredients_lower:
                        missing_ingredients.append(f"- {ingredient}")
                        print(f"DEBUG: เพิ่มวัตถุดิบเฉพาะเมนู: {ingredient}")
        
        # เพิ่มวัตถุดิบที่ขาดหาย
        if missing_ingredients:
            if enhanced_ingredients and not enhanced_ingredients.endswith('\n'):
                enhanced_ingredients += '\n'
            enhanced_ingredients += '\n'.join(missing_ingredients)
            print(f"DEBUG: เพิ่มวัตถุดิบทั้งหมด {len(missing_ingredients)} รายการ")
        else:
            print("DEBUG: ไม่พบวัตถุดิบที่ขาดหาย")
        
        return enhanced_ingredients

    def get_nutrition_data(self, ingredient: str, use_api: bool = True) -> Optional[Dict]:
        """ดึงข้อมูลโภชนาการสำหรับวัตถุดิบ (API หรือฐานข้อมูลท้องถิ่น)"""
        normalized_ingredient = self.normalize_ingredient_name(ingredient)
        
        # ลองค้นหาใน API ก่อน (หากเปิดใช้งาน)
        if use_api and self.current_api_source != "local":
            api_data = self.search_ingredient_api(normalized_ingredient)
            if api_data and api_data.get("calories", 0) > 0:
                return api_data
        
        # ค้นหาในฐานข้อมูลท้องถิ่น
        for local_ingredient, nutrition in self.local_nutrition_db.items():
            if (local_ingredient.lower() in normalized_ingredient.lower() or 
                normalized_ingredient.lower() in local_ingredient.lower()):
                return nutrition.copy()
        
        # หากไม่พบ ให้ค่าประมาณการตามประเภทวัตถุดิบ
        return self.estimate_nutrition_by_type(normalized_ingredient)

    def search_ingredient_api(self, ingredient: str) -> Optional[Dict]:
        """ค้นหาข้อมูลโภชนาการจาก API แหล่งต่างๆ"""
        if self.current_api_source == "usda":
            return self.search_usda_api(ingredient)
        elif self.current_api_source == "nutritionix":
            return self.search_nutritionix_api(ingredient)
        elif self.current_api_source == "edamam":
            return self.search_edamam_api(ingredient)
        return None

    def search_usda_api(self, ingredient: str) -> Optional[Dict]:
        """ค้นหาข้อมูลจาก USDA FoodData Central API"""
        api_key = self.api_credentials["usda"]["api_key"]
        if not api_key:
            return None
            
        try:
            # ค้นหาในฐานข้อมูล USDA
            search_url = f"{self.api_urls['usda']}/foods/search"
            params = {
                "api_key": api_key,
                "query": ingredient,
                "dataType": ["Foundation", "SR Legacy"],
                "pageSize": 1
            }
            
            response = requests.get(search_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("foods"):
                    food_id = data["foods"][0]["fdcId"]
                    return self.get_usda_nutrition_details(food_id)
            return None
            
        except Exception as e:
            print(f"Error with USDA API: {e}")
            return None

    def get_usda_nutrition_details(self, food_id: int) -> Optional[Dict]:
        """ดึงรายละเอียดโภชนาการจาก USDA API"""
        api_key = self.api_credentials["usda"]["api_key"]
        try:
            detail_url = f"{self.api_urls['usda']}/food/{food_id}"
            params = {"api_key": api_key}
            
            response = requests.get(detail_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return self.parse_usda_nutrition_data(data)
            return None
            
        except Exception as e:
            print(f"Error getting USDA details: {e}")
            return None

    def parse_usda_nutrition_data(self, data: Dict) -> Dict:
        """แปลงข้อมูลโภชนาการจาก USDA เป็นรูปแบบมาตรฐาน"""
        nutrients = {}
        
        # แปลงข้อมูลโภชนาการ
        for nutrient in data.get("foodNutrients", []):
            nutrient_name = nutrient.get("nutrient", {}).get("name", "")
            value = nutrient.get("amount", 0)
            
            # แปลงเป็นรูปแบบที่ใช้ในระบบ
            if "Energy" in nutrient_name and "kJ" not in nutrient_name:
                nutrients["calories"] = value
            elif "Protein" in nutrient_name:
                nutrients["protein"] = value
            elif "Carbohydrate" in nutrient_name and "by difference" in nutrient_name:
                nutrients["carbs"] = value
            elif "Total lipid" in nutrient_name or "Fat" in nutrient_name:
                nutrients["fat"] = value
            elif "Fiber" in nutrient_name:
                nutrients["fiber"] = value
            elif "Vitamin A" in nutrient_name and "IU" in nutrient_name:
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
        
        # เติมค่าที่ขาดหาย
        default_nutrients = {
            "calories": 0, "protein": 0, "carbs": 0, "fat": 0, "fiber": 0,
            "vitamin_a": 0, "vitamin_c": 0, "vitamin_b1": 0, "vitamin_b2": 0,
            "calcium": 0, "iron": 0, "potassium": 0, "sodium": 0
        }
        
        for key, default_value in default_nutrients.items():
            if key not in nutrients:
                nutrients[key] = default_value
        
        return nutrients

    def estimate_nutrition_by_type(self, ingredient: str) -> Dict:
        """ประมาณค่าโภชนาการตามประเภทวัตถุดิบ - ปรับปรุงให้แม่นยำขึ้น"""
        ingredient_lower = ingredient.lower()
        
        # ประเภทเนื้อสัตว์
        if any(keyword in ingredient_lower for keyword in ["เนื้อ", "หมู", "ไก่", "ปลา", "กุ้ง"]):
            return {
                "calories": 150, "protein": 20, "carbs": 0, "fat": 6, "fiber": 0,
                "vitamin_a": 20, "vitamin_c": 1, "vitamin_b1": 0.1, "vitamin_b2": 0.15,
                "calcium": 20, "iron": 1.5, "potassium": 250, "sodium": 50
            }
        
        # ประเภทผัก
        elif any(keyword in ingredient_lower for keyword in ["ผัก", "ใบ", "ต้น"]):
            return {
                "calories": 25, "protein": 2, "carbs": 5, "fat": 0.2, "fiber": 2,
                "vitamin_a": 1000, "vitamin_c": 30, "vitamin_b1": 0.05, "vitamin_b2": 0.08,
                "calcium": 50, "iron": 1, "potassium": 200, "sodium": 10
            }
        
        # ประเภทเครื่องปรุง
        elif any(keyword in ingredient_lower for keyword in ["น้ำปลา", "ซีอิ้ว", "เกลือ"]):
            return {
                "calories": 20, "protein": 3, "carbs": 2, "fat": 0, "fiber": 0,
                "vitamin_a": 0, "vitamin_c": 0, "vitamin_b1": 0.02, "vitamin_b2": 0.05,
                "calcium": 30, "iron": 1, "potassium": 50, "sodium": 3000
            }
        
        # ประเภทน้ำมัน
        elif any(keyword in ingredient_lower for keyword in ["น้ำมัน", "เนย"]):
            return {
                "calories": 884, "protein": 0, "carbs": 0, "fat": 100, "fiber": 0,
                "vitamin_a": 0, "vitamin_c": 0, "vitamin_b1": 0, "vitamin_b2": 0,
                "calcium": 0, "iron": 0, "potassium": 0, "sodium": 0
            }
        
        # ค่าเริ่มต้นทั่วไป
        return {
            "calories": 50, "protein": 2, "carbs": 10, "fat": 1, "fiber": 1,
            "vitamin_a": 10, "vitamin_c": 5, "vitamin_b1": 0.05, "vitamin_b2": 0.05,
            "calcium": 20, "iron": 0.5, "potassium": 100, "sodium": 10
        }

    def calculate_recipe_nutrition(self, ingredients_text: str, use_api: bool = True, 
                                 adjust_consumption: bool = True, 
                                 enhance_missing: bool = False,
                                 recipe_name: str = "",
                                 method_text: str = "") -> Dict:
        """คำนวณค่าโภชนาการของสูตรอาหารอย่างครอบคลุม - แก้ไขให้ทำงานถูกต้อง"""
        
        print(f"DEBUG: เริ่มคำนวณโภชนาการสำหรับ '{recipe_name}'")
        print(f"DEBUG: การตั้งค่า - enhance_missing: {enhance_missing}, adjust_consumption: {adjust_consumption}")
        
        # เพิ่มวัตถุดิบที่ขาดหาย (หากเปิดใช้งาน)
        original_ingredients = ingredients_text
        if enhance_missing:
            print("DEBUG: เริ่มเพิ่มวัตถุดิบที่ขาดหาย")
            ingredients_text = self.enhance_missing_ingredients(ingredients_text, recipe_name, method_text)
            if ingredients_text != original_ingredients:
                print("DEBUG: มีการเพิ่มวัตถุดิบแล้ว")
            else:
                print("DEBUG: ไม่มีวัตถุดิบเพิ่มเติม")
        
        total_nutrition = {
            "calories": 0, "protein": 0, "carbs": 0, "fat": 0, "fiber": 0,
            "vitamin_a": 0, "vitamin_c": 0, "vitamin_b1": 0, "vitamin_b2": 0,
            "calcium": 0, "iron": 0, "potassium": 0, "sodium": 0
        }
        
        ingredient_details = []
        
        # แยกวัตถุดิบแต่ละรายการ
        ingredients = [ing.strip() for ing in ingredients_text.split('\n') if ing.strip()]
        print(f"DEBUG: พบวัตถุดิบทั้งหมด {len(ingredients)} รายการ")
        
        for ingredient_line in ingredients:
            # ลบเครื่องหมาย - หรือ * ถ้ามี
            ingredient_text = re.sub(r'^[-*•]\s*', '', ingredient_line).strip()
            
            if not ingredient_text:
                continue
            
            # แยกปริมาณและชื่อวัตถุดิบ
            quantity, unit, ingredient_name = self.extract_quantity_and_unit(ingredient_text)
            
            # แปลงเป็นกรัม
            grams = self.convert_to_grams(quantity, unit, ingredient_name)
            
            # ปรับสัดส่วนการบริโภค (หากเปิดใช้งาน)
            consumption_factor = 1.0
            if adjust_consumption:
                # ตรวจสอบชื่อวัตถุดิบใน consumption_ratio
                for ratio_ingredient, ratio in self.consumption_ratio.items():
                    if ratio_ingredient.lower() in ingredient_name.lower():
                        consumption_factor = ratio
                        print(f"DEBUG: ปรับสัดส่วนการบริโภคของ '{ingredient_name}' เป็น {ratio}")
                        break
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
                
                # ตรวจสอบว่าเป็นวัตถุดิบที่เพิ่มใหม่หรือไม่
                was_enhanced = (ingredients_text != original_ingredients and 
                               ingredient_line not in original_ingredients)
                
                ingredient_details.append({
                    "name": ingredient_name,
                    "quantity": quantity,
                    "unit": unit,
                    "grams": grams,
                    "effective_grams": effective_grams,
                    "consumption_factor": consumption_factor,
                    "nutrition": ingredient_nutrition,
                    "was_enhanced": was_enhanced
                })
                
                print(f"DEBUG: คำนวณ '{ingredient_name}' - {effective_grams:.1f}g ได้ {ingredient_nutrition.get('calories', 0):.1f} kcal")
        
        print(f"DEBUG: คำนวณเสร็จ - รวม {total_nutrition['calories']:.1f} kcal")
        
        return {
            "total_nutrition": total_nutrition,
            "ingredient_details": ingredient_details,
            "enhanced_ingredients": ingredients_text if enhance_missing and ingredients_text != original_ingredients else None,
            "settings": {
                "use_api": use_api,
                "adjust_consumption": adjust_consumption,
                "enhance_missing": enhance_missing
            }
        }

    def check_api_status(self) -> Tuple[bool, str]:
        """ตรวจสอบสถานะการเชื่อมต่อ API"""
        if self.current_api_source == "local":
            return True, "ใช้ฐานข้อมูลท้องถิ่น"
        
        if self.current_api_source == "usda":
            api_key = self.api_credentials["usda"]["api_key"]
            if not api_key:
                return False, "ไม่ได้ตั้งค่า USDA API Key"
            
            try:
                # ทดสอบการเชื่อมต่อด้วยการค้นหาข้อมูลธรรมดา
                search_url = f"{self.api_urls['usda']}/foods/search"
                params = {
                    "api_key": api_key,
                    "query": "chicken",
                    "pageSize": 1
                }
                
                response = requests.get(search_url, params=params, timeout=5)
                if response.status_code == 200:
                    return True, "เชื่อมต่อ USDA API สำเร็จ"
                else:
                    return False, f"USDA API ตอบกลับ status code: {response.status_code}"
                    
            except requests.exceptions.Timeout:
                return False, "การเชื่อมต่อ USDA API หมดเวลา"
            except requests.exceptions.ConnectionError:
                return False, "ไม่สามารถเชื่อมต่อ USDA API ได้"
            except Exception as e:
                return False, f"เกิดข้อผิดพลาด USDA API: {str(e)}"
        
        elif self.current_api_source == "nutritionix":
            credentials = self.api_credentials["nutritionix"]
            if not credentials["app_id"] or not credentials["app_key"]:
                return False, "ไม่ได้ตั้งค่า Nutritionix credentials"
            
            return True, "Nutritionix credentials ตั้งค่าแล้ว (ยังไม่ได้ทดสอบ)"
        
        return False, "API source ไม่รู้จัก"

    def get_nutrition_summary(self) -> Dict:
        """สรุปข้อมูลในฐานข้อมูลโภชนาการ"""
        return {
            "total_ingredients": len(self.local_nutrition_db),
            "api_source": self.current_api_source,
            "available_apis": list(self.api_urls.keys()),
            "consumption_adjustments": len(self.consumption_ratio),
            "unit_conversions": len(self.unit_conversion),
            "cooking_methods": len(self.missing_ingredients_by_cooking_method),
            "recipe_patterns": len(self.recipe_specific_ingredients)
        }
