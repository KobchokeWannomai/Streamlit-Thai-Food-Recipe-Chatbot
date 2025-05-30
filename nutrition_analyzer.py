import requests
import json
import sqlite3
import re
import time
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from ingredient_converter import IngredientConverter

# กำหนดค่า logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NutritionInfo:
    """คลาสสำหรับเก็บข้อมูลโภชนาการ"""
    name: str
    calories: float = 0.0
    protein: float = 0.0
    carbs: float = 0.0
    fat: float = 0.0
    fiber: float = 0.0
    sugar: float = 0.0
    sodium: float = 0.0
    vitamin_a: float = 0.0
    vitamin_c: float = 0.0
    vitamin_d: float = 0.0
    vitamin_e: float = 0.0
    vitamin_k: float = 0.0
    vitamin_b1: float = 0.0
    vitamin_b2: float = 0.0
    vitamin_b6: float = 0.0
    vitamin_b12: float = 0.0
    folate: float = 0.0
    niacin: float = 0.0
    calcium: float = 0.0
    iron: float = 0.0
    magnesium: float = 0.0
    phosphorus: float = 0.0
    potassium: float = 0.0
    zinc: float = 0.0
    serving_size: str = "100g"
    last_updated: str = ""

class NutritionDatabase:
    """คลาสสำหรับจัดการฐานข้อมูลโภชนาการ"""
    
    def __init__(self, db_path: str = "nutrition_cache.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """สร้างฐานข้อมูลและตาราง"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS nutrition_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ingredient_name TEXT UNIQUE NOT NULL,
                nutrition_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                api_name TEXT NOT NULL,
                calls_count INTEGER DEFAULT 0,
                last_reset DATE DEFAULT CURRENT_DATE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recipe_adjustments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recipe_name TEXT UNIQUE NOT NULL,
                adjustments_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_cached_nutrition(self, ingredient: str) -> Optional[NutritionInfo]:
        """ดึงข้อมูลโภชนาการจากแคช"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT nutrition_data FROM nutrition_cache WHERE ingredient_name = ?",
            (ingredient.lower(),)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            try:
                data = json.loads(result[0])
                return NutritionInfo(**data)
            except json.JSONDecodeError:
                return None
        return None
    
    def cache_nutrition(self, ingredient: str, nutrition: NutritionInfo):
        """บันทึกข้อมูลโภชนาการลงแคช"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        nutrition_dict = nutrition.__dict__.copy()
        nutrition_dict['last_updated'] = datetime.now().isoformat()
        
        cursor.execute('''
            INSERT OR REPLACE INTO nutrition_cache (ingredient_name, nutrition_data, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        ''', (ingredient.lower(), json.dumps(nutrition_dict, ensure_ascii=False)))
        
        conn.commit()
        conn.close()
    
    def get_recipe_adjustments(self, recipe_name: str) -> Optional[dict]:
        """ดึงการปรับแต่งสำหรับสูตรอาหาร"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT adjustments_data FROM recipe_adjustments WHERE recipe_name = ?",
            (recipe_name.lower(),)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            try:
                return json.loads(result[0])
            except json.JSONDecodeError:
                return None
        return None
    
    def cache_recipe_adjustments(self, recipe_name: str, adjustments: dict):
        """บันทึกการปรับแต่งสำหรับสูตรอาหาร"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO recipe_adjustments (recipe_name, adjustments_data)
            VALUES (?, ?)
        ''', (recipe_name.lower(), json.dumps(adjustments, ensure_ascii=False)))
        
        conn.commit()
        conn.close()

class USDANutritionAPI:
    """คลาสสำหรับดึงข้อมูลจาก USDA FoodData Central API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.nal.usda.gov/fdc/v1"
        self.session = requests.Session()
        self.rate_limit_calls = 0
        self.rate_limit_reset = datetime.now()
    
    def _check_rate_limit(self):
        """ตรวจสอบ rate limit"""
        now = datetime.now()
        if now - self.rate_limit_reset > timedelta(minutes=1):
            self.rate_limit_calls = 0
            self.rate_limit_reset = now
        
        if self.rate_limit_calls >= 30:  # จำกัด 30 calls ต่อนาที
            sleep_time = 60 - (now - self.rate_limit_reset).seconds
            if sleep_time > 0:
                time.sleep(sleep_time)
                self.rate_limit_calls = 0
                self.rate_limit_reset = datetime.now()
    
    def search_food(self, query: str) -> Optional[Dict]:
        """ค้นหาอาหารจาก USDA database"""
        self._check_rate_limit()
        
        url = f"{self.base_url}/foods/search"
        params = {
            "api_key": self.api_key,
            "query": query,
            "pageSize": 5,
            "dataType": ["Foundation", "SR Legacy"]
        }
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            self.rate_limit_calls += 1
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error searching USDA API: {e}")
            return None
    
    def get_nutrition_info(self, ingredient: str) -> Optional[NutritionInfo]:
        """ดึงข้อมูลโภชนาการจาก USDA API"""
        search_result = self.search_food(ingredient)
        
        if not search_result or not search_result.get('foods'):
            return None
        
        # เลือกอาหารที่เหมาะสมที่สุด
        best_match = search_result['foods'][0]
        
        # ดึงข้อมูลโภชนาการแบบละเอียด
        food_id = best_match['fdcId']
        url = f"{self.base_url}/food/{food_id}"
        params = {"api_key": self.api_key}
        
        try:
            self._check_rate_limit()
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            self.rate_limit_calls += 1
            food_data = response.json()
            
            return self._parse_usda_nutrition(food_data, ingredient)
        except requests.RequestException as e:
            logger.error(f"Error getting detailed nutrition from USDA: {e}")
            return None
    
    def _parse_usda_nutrition(self, food_data: Dict, ingredient: str) -> NutritionInfo:
        """แปลงข้อมูลจาก USDA เป็น NutritionInfo"""
        nutrition = NutritionInfo(name=ingredient)
        
        # แมป nutrient ID กับ attribute
        nutrient_mapping = {
            1008: 'calories',      # Energy
            1003: 'protein',       # Protein
            1005: 'carbs',         # Carbohydrate
            1004: 'fat',           # Total lipid (fat)
            1079: 'fiber',         # Fiber
            2000: 'sugar',         # Sugars
            1093: 'sodium',        # Sodium
            1106: 'vitamin_a',     # Vitamin A, RAE
            1162: 'vitamin_c',     # Vitamin C
            1114: 'vitamin_d',     # Vitamin D
            1109: 'vitamin_e',     # Vitamin E
            1185: 'vitamin_k',     # Vitamin K
            1165: 'vitamin_b1',    # Thiamin
            1166: 'vitamin_b2',    # Riboflavin
            1175: 'vitamin_b6',    # Vitamin B-6
            1178: 'vitamin_b12',   # Vitamin B-12
            1186: 'folate',        # Folate, DFE
            1167: 'niacin',        # Niacin
            1087: 'calcium',       # Calcium
            1089: 'iron',          # Iron
            1090: 'magnesium',     # Magnesium
            1091: 'phosphorus',    # Phosphorus
            1092: 'potassium',     # Potassium
            1095: 'zinc',          # Zinc
        }
        
        # ดึงข้อมูล nutrients
        for nutrient in food_data.get('foodNutrients', []):
            nutrient_id = nutrient.get('nutrient', {}).get('id')
            amount = nutrient.get('amount', 0)
            
            if nutrient_id in nutrient_mapping:
                attr = nutrient_mapping[nutrient_id]
                setattr(nutrition, attr, float(amount))
        
        return nutrition

class NutritionixAPI:
    """คลาสสำหรับดึงข้อมูลจาก Nutritionix API"""
    
    def __init__(self, app_id: str, api_key: str):
        self.app_id = app_id
        self.api_key = api_key
        self.base_url = "https://trackapi.nutritionix.com/v2"
        self.session = requests.Session()
        self.daily_calls = 0
        self.last_reset = datetime.now().date()
    
    def _check_daily_limit(self):
        """ตรวจสอบ daily limit"""
        today = datetime.now().date()
        if today != self.last_reset:
            self.daily_calls = 0
            self.last_reset = today
        
        if self.daily_calls >= 200:  # Free plan limit
            raise Exception("Daily API limit reached")
    
    def get_nutrition_info(self, ingredient: str) -> Optional[NutritionInfo]:
        """ดึงข้อมูลโภชนาการจาก Nutritionix API"""
        self._check_daily_limit()
        
        url = f"{self.base_url}/natural/nutrients"
        headers = {
            'x-app-id': self.app_id,
            'x-app-key': self.api_key,
            'Content-Type': 'application/json'
        }
        
        data = {
            'query': f"1 serving of {ingredient}",
            'timezone': 'Asia/Bangkok'
        }
        
        try:
            response = self.session.post(url, headers=headers, json=data, timeout=10)
            response.raise_for_status()
            self.daily_calls += 1
            
            result = response.json()
            if result.get('foods'):
                food = result['foods'][0]
                return self._parse_nutritionix_data(food, ingredient)
        except requests.RequestException as e:
            logger.error(f"Error getting nutrition from Nutritionix: {e}")
        
        return None
    
    def _parse_nutritionix_data(self, food_data: dict, ingredient: str) -> NutritionInfo:
        """แปลงข้อมูลจาก Nutritionix เป็น NutritionInfo"""
        return NutritionInfo(
            name=ingredient,
            calories=food_data.get('nf_calories', 0),
            protein=food_data.get('nf_protein', 0),
            carbs=food_data.get('nf_total_carbohydrate', 0),
            fat=food_data.get('nf_total_fat', 0),
            fiber=food_data.get('nf_dietary_fiber', 0),
            sugar=food_data.get('nf_sugars', 0),
            sodium=food_data.get('nf_sodium', 0),
            calcium=food_data.get('nf_calcium', 0),
            iron=food_data.get('nf_iron', 0),
            potassium=food_data.get('nf_potassium', 0),
            serving_size=f"{food_data.get('serving_weight_grams', 100)}g"
        )

class ThaiNutritionData:
    """คลาสสำหรับข้อมูลโภชนาการอาหารไทยที่สร้างขึ้นเอง"""
    
    def __init__(self):
        self.thai_nutrition_db = self._load_thai_nutrition_data()
    
    def get_nutrition_info(self, ingredient: str) -> Optional[NutritionInfo]:
        """ดึงข้อมูลโภชนาการของวัตถุดิบจากฐานข้อมูลไทย"""
        ingredient_lower = ingredient.lower()
        
        # ค้นหาจากฐานข้อมูล
        for key, nutrition in self.thai_nutrition_db.items():
            if key.lower() == ingredient_lower or key.lower() in ingredient_lower:
                return nutrition
        
        return None
    
    def _load_thai_nutrition_data(self) -> Dict[str, NutritionInfo]:
        """โหลดข้อมูลโภชนาการอาหารไทยพื้นฐาน (ต่อ 100g)"""
        return {
            # เนื้อสัตว์ - ปรับค่าให้สมจริงมากขึ้น
            "หมู": NutritionInfo(
                name="หมู", calories=242, protein=14.5, fat=20.0, carbs=0,
                iron=0.9, zinc=2.4, vitamin_b1=0.7, vitamin_b12=0.7,
                phosphorus=180, potassium=290
            ),
            "เนื้อหมู": NutritionInfo(
                name="เนื้อหมู", calories=242, protein=14.5, fat=20.0, carbs=0,
                iron=0.9, zinc=2.4, vitamin_b1=0.7, vitamin_b12=0.7
            ),
            "หมูสับ": NutritionInfo(
                name="หมูสับ", calories=263, protein=17.0, fat=21.0, carbs=0,
                iron=1.0, zinc=2.5, vitamin_b12=0.8
            ),
            "ไก่": NutritionInfo(
                name="ไก่", calories=215, protein=18.6, fat=15.1, carbs=0,
                niacin=6.8, vitamin_b6=0.35, phosphorus=147, iron=0.9
            ),
            "เนื้อไก่": NutritionInfo(
                name="เนื้อไก่", calories=215, protein=18.6, fat=15.1, carbs=0,
                niacin=6.8, vitamin_b6=0.35, phosphorus=147
            ),
            "เนื้อ": NutritionInfo(
                name="เนื้อ", calories=250, protein=26.0, fat=15.0, carbs=0,
                iron=2.6, zinc=4.8, vitamin_b12=2.6, phosphorus=175
            ),
            "เนื้อโค": NutritionInfo(
                name="เนื้อโค", calories=250, protein=26.0, fat=15.0, carbs=0,
                iron=2.6, zinc=4.8, vitamin_b12=2.6
            ),
            "เนื้อวัว": NutritionInfo(
                name="เนื้อวัว", calories=250, protein=26.0, fat=15.0, carbs=0,
                iron=2.6, zinc=4.8, vitamin_b12=2.6
            ),
            "กุ้ง": NutritionInfo(
                name="กุ้ง", calories=85, protein=20.0, fat=0.5, carbs=0,
                calcium=52, iron=0.3, zinc=1.1, phosphorus=205
            ),
            "กุ้งนาง": NutritionInfo(
                name="กุ้งนาง", calories=85, protein=20.0, fat=0.5, carbs=0,
                calcium=52, iron=0.3, zinc=1.1
            ),
            "กุ้งแห้ง": NutritionInfo(
                name="กุ้งแห้ง", calories=300, protein=65.0, fat=2.0, carbs=0,
                calcium=800, iron=3.0, sodium=2500
            ),
            "ปลา": NutritionInfo(
                name="ปลา", calories=124, protein=22.0, fat=3.0, carbs=0,
                calcium=20, iron=0.8, vitamin_d=10.9, phosphorus=200
            ),
            "หอยแมลงภู่": NutritionInfo(
                name="หอยแมลงภู่", calories=86, protein=12.0, fat=2.2, carbs=3.7,
                iron=6.7, zinc=2.3, vitamin_b12=24.0
            ),
            "ปลาหมึก": NutritionInfo(
                name="ปลาหมึก", calories=92, protein=15.6, fat=1.4, carbs=3.1,
                iron=0.7, zinc=1.5, phosphorus=221
            ),
            "ปู": NutritionInfo(
                name="ปู", calories=97, protein=19.0, fat=1.5, carbs=0,
                calcium=89, iron=0.7, zinc=6.5
            ),
            
            # ไข่ - ค่าที่ถูกต้องต่อ 100g
            "ไข่": NutritionInfo(
                name="ไข่", calories=155, protein=13.0, fat=11.0, carbs=1.1,
                vitamin_a=140, vitamin_d=2.0, iron=1.8, phosphorus=200
            ),
            "ไข่ไก่": NutritionInfo(
                name="ไข่ไก่", calories=155, protein=13.0, fat=11.0, carbs=1.1,
                vitamin_a=140, vitamin_d=2.0, iron=1.8
            ),
            "ไข่เป็ด": NutritionInfo(
                name="ไข่เป็ด", calories=185, protein=13.0, fat=14.0, carbs=1.5,
                vitamin_a=180, iron=3.8, calcium=64
            ),
            "ไข่เค็ม": NutritionInfo(
                name="ไข่เค็ม", calories=190, protein=13.6, fat=13.0, carbs=1.8,
                sodium=1400, calcium=120, iron=3.2
            ),
            
            # ผัก - ปรับค่าให้สมจริง
            "กะหล่ำปลี": NutritionInfo(
                name="กะหล่ำปลี", calories=25, protein=1.3, carbs=5.8, fat=0.1,
                vitamin_c=36.6, vitamin_k=76, fiber=2.5
            ),
            "คะน้า": NutritionInfo(
                name="คะน้า", calories=26, protein=2.8, carbs=4.7, fat=0.4,
                vitamin_a=241, vitamin_c=120, calcium=105, iron=0.6
            ),
            "ผักบุ้ง": NutritionInfo(
                name="ผักบุ้ง", calories=19, protein=2.6, carbs=3.1, fat=0.2,
                vitamin_a=318, vitamin_c=55, iron=1.7, calcium=77
            ),
            "ผักกาด": NutritionInfo(
                name="ผักกาด", calories=16, protein=1.5, carbs=3.2, fat=0.2,
                vitamin_c=27, calcium=105, fiber=1.2
            ),
            "ผักกาดขาว": NutritionInfo(
                name="ผักกาดขาว", calories=16, protein=1.5, carbs=3.2, fat=0.2,
                vitamin_c=27, calcium=105
            ),
            "ถั่วฝักยาว": NutritionInfo(
                name="ถั่วฝักยาว", calories=47, protein=2.8, carbs=8.4, fat=0.4,
                vitamin_c=18.8, folate=62, fiber=2.6
            ),
            "ถั่วพู": NutritionInfo(
                name="ถั่วพู", calories=47, protein=2.8, carbs=8.4, fat=0.4,
                vitamin_c=18.8, fiber=2.6
            ),
            "ถั่วงอก": NutritionInfo(
                name="ถั่วงอก", calories=30, protein=3.0, carbs=6.0, fat=0.2,
                vitamin_c=13.2, folate=61
            ),
            "แตงกวา": NutritionInfo(
                name="แตงกวา", calories=16, protein=0.7, carbs=3.6, fat=0.1,
                vitamin_k=16.4, vitamin_c=2.8
            ),
            "มะเขือเทศ": NutritionInfo(
                name="มะเขือเทศ", calories=18, protein=0.9, carbs=3.9, fat=0.2,
                vitamin_c=14, vitamin_a=42, potassium=237
            ),
            "มะเขือ": NutritionInfo(
                name="มะเขือ", calories=25, protein=1.0, carbs=5.9, fat=0.2,
                fiber=3.0, potassium=229
            ),
            
            # เครื่องปรุง
            "หอมใหญ่": NutritionInfo(
                name="หอมใหญ่", calories=40, protein=1.1, carbs=9.3, fat=0.1,
                vitamin_c=7.4, fiber=1.7
            ),
            "หอมแดง": NutritionInfo(
                name="หอมแดง", calories=72, protein=2.5, carbs=16.8, fat=0.1,
                fiber=2.6
            ),
            "หัวหอม": NutritionInfo(
                name="หัวหอม", calories=72, protein=2.5, carbs=16.8, fat=0.1
            ),
            "กระเทียม": NutritionInfo(
                name="กระเทียม", calories=149, protein=6.4, carbs=33.1, fat=0.5,
                vitamin_c=31.2, calcium=181
            ),
            "พริก": NutritionInfo(
                name="พริก", calories=40, protein=1.9, carbs=8.8, fat=0.4,
                vitamin_c=144, vitamin_a=48
            ),
            "พริกขี้หนู": NutritionInfo(
                name="พริกขี้หนู", calories=40, protein=1.9, carbs=8.8, fat=0.4,
                vitamin_c=144, vitamin_a=48
            ),
            "พริกแห้ง": NutritionInfo(
                name="พริกแห้ง", calories=324, protein=10.6, carbs=69.9, fat=5.8,
                vitamin_a=1483, iron=7.8
            ),
            "พริกไทย": NutritionInfo(
                name="พริกไทย", calories=251, protein=10.4, carbs=63.9, fat=3.3,
                iron=9.7, calcium=443
            ),
            "ขิง": NutritionInfo(
                name="ขิง", calories=80, protein=1.8, carbs=17.8, fat=0.8,
                potassium=415, magnesium=43
            ),
            "ข่า": NutritionInfo(
                name="ข่า", calories=71, protein=1.8, carbs=15.0, fat=0.7,
                fiber=2.0
            ),
            "ตะไคร้": NutritionInfo(
                name="ตะไคร้", calories=99, protein=1.8, carbs=25.3, fat=0.5,
                iron=8.2, calcium=65
            ),
            "ใบมะกรูด": NutritionInfo(
                name="ใบมะกรูด", calories=70, protein=3.0, carbs=14.0, fat=0.7,
                calcium=830, vitamin_c=8.5
            ),
            "ผักชี": NutritionInfo(
                name="ผักชี", calories=23, protein=2.1, carbs=3.7, fat=0.5,
                vitamin_a=337, vitamin_c=27, vitamin_k=310
            ),
            "ต้นหอม": NutritionInfo(
                name="ต้นหอม", calories=32, protein=1.8, carbs=7.3, fat=0.2,
                vitamin_a=52, vitamin_c=18.8, vitamin_k=207
            ),
            "โหระพา": NutritionInfo(
                name="โหระพา", calories=23, protein=3.2, carbs=2.7, fat=0.6,
                vitamin_k=415, calcium=177, iron=3.2
            ),
            "ใบโหระพา": NutritionInfo(
                name="ใบโหระพา", calories=23, protein=3.2, carbs=2.7, fat=0.6,
                vitamin_k=415
            ),
            "กะเพรา": NutritionInfo(
                name="กะเพรา", calories=23, protein=3.2, carbs=2.7, fat=0.6,
                vitamin_k=415, calcium=177
            ),
            "ใบกะเพรา": NutritionInfo(
                name="ใบกะเพรา", calories=23, protein=3.2, carbs=2.7, fat=0.6,
                vitamin_k=415
            ),
            "รากผักชี": NutritionInfo(
                name="รากผักชี", calories=23, protein=2.1, carbs=3.7, fat=0.5,
                calcium=67
            ),
            "กะปิ": NutritionInfo(
                name="กะปิ", calories=174, protein=20.0, fat=10.0, carbs=3.0,
                sodium=3500, calcium=300
            ),
            
            # เครื่องปรุงของเหลว
            "น้ำปลา": NutritionInfo(
                name="น้ำปลา", calories=35, protein=6.0, carbs=3.0, fat=0,
                sodium=7720
            ),
            "ซีอิ้ว": NutritionInfo(
                name="ซีอิ้ว", calories=53, protein=8.0, carbs=4.9, fat=0.1,
                sodium=5490
            ),
            "ซีอิ๊ว": NutritionInfo(
                name="ซีอิ๊ว", calories=53, protein=8.0, carbs=4.9, fat=0.1,
                sodium=5490
            ),
            "น้ำมันพืช": NutritionInfo(
                name="น้ำมันพืช", calories=884, protein=0, fat=100, carbs=0,
                vitamin_e=14.4
            ),
            "น้ำมันหมู": NutritionInfo(
                name="น้ำมันหมู", calories=902, protein=0, fat=100, carbs=0,
                vitamin_e=0.6
            ),
            "น้ำมัน": NutritionInfo(
                name="น้ำมัน", calories=884, protein=0, fat=100, carbs=0
            ),
            "กะทิ": NutritionInfo(
                name="กะทิ", calories=230, protein=2.3, fat=23.8, carbs=5.5,
                iron=1.6, magnesium=37
            ),
            "หัวกะทิ": NutritionInfo(
                name="หัวกะทิ", calories=330, protein=3.3, fat=35.0, carbs=6.0,
                iron=1.9
            ),
            "หางกะทิ": NutritionInfo(
                name="หางกะทิ", calories=180, protein=1.8, fat=17.0, carbs=4.0,
                iron=1.3
            ),
            "น้ำตาล": NutritionInfo(
                name="น้ำตาล", calories=387, protein=0, fat=0, carbs=99.8,
                calcium=1
            ),
            "น้ำตาลทราย": NutritionInfo(
                name="น้ำตาลทราย", calories=387, protein=0, fat=0, carbs=99.8
            ),
            "น้ำตาลปึก": NutritionInfo(
                name="น้ำตาลปึก", calories=377, protein=0.4, fat=0, carbs=97.3,
                calcium=85, iron=4.6
            ),
            "เกลือ": NutritionInfo(
                name="เกลือ", calories=0, protein=0, fat=0, carbs=0,
                sodium=38758
            ),
            "น้ำ": NutritionInfo(
                name="น้ำ", calories=0, protein=0, fat=0, carbs=0
            ),
            
            # ข้าว/แป้ง
            "ข้าว": NutritionInfo(
                name="ข้าว", calories=130, protein=2.7, carbs=28.2, fat=0.3,
                niacin=1.6, magnesium=25, phosphorus=68
            ),
            "ข้าวสาร": NutritionInfo(
                name="ข้าวสาร", calories=365, protein=7.1, carbs=80.0, fat=0.7,
                iron=0.8, niacin=4.3
            ),
            "ข้าวเหนียว": NutritionInfo(
                name="ข้าวเหนียว", calories=370, protein=6.8, carbs=81.7, fat=0.6,
                iron=0.8
            ),
            "แป้ง": NutritionInfo(
                name="แป้ง", calories=364, protein=10.3, carbs=76.3, fat=1.0,
                iron=1.2, niacin=1.3
            ),
            "แป้งข้าวเจ้า": NutritionInfo(
                name="แป้งข้าวเจ้า", calories=366, protein=5.9, carbs=80.1, fat=1.4,
                iron=0.4
            ),
            "แป้งสาลี": NutritionInfo(
                name="แป้งสาลี", calories=364, protein=10.3, carbs=76.3, fat=1.0,
                iron=1.2, folate=26
            ),
            "แป้งมัน": NutritionInfo(
                name="แป้งมัน", calories=338, protein=0.2, carbs=83.1, fat=0.1,
                calcium=20
            ),
            
            # ถั่วและธัญพืช
            "ถั่วลิสง": NutritionInfo(
                name="ถั่วลิสง", calories=567, protein=25.8, carbs=16.1, fat=49.2,
                vitamin_e=8.3, niacin=12.1, magnesium=168
            ),
            "ถั่วเขียว": NutritionInfo(
                name="ถั่วเขียว", calories=347, protein=23.9, carbs=62.6, fat=1.2,
                iron=6.7, folate=625
            ),
            "ถั่วเหลือง": NutritionInfo(
                name="ถั่วเหลือง", calories=446, protein=36.5, carbs=30.2, fat=19.9,
                calcium=277, iron=15.7, folate=375
            ),
            "งา": NutritionInfo(
                name="งา", calories=573, protein=17.7, carbs=23.5, fat=49.7,
                calcium=975, iron=14.6, magnesium=351
            ),
            
            # ผลไม้
            "มะนาว": NutritionInfo(
                name="มะนาว", calories=29, protein=1.1, carbs=9.3, fat=0.3,
                vitamin_c=53, fiber=2.8
            ),
            "มะพร้าว": NutritionInfo(
                name="มะพร้าว", calories=354, protein=3.3, carbs=15.2, fat=33.5,
                fiber=9.0, potassium=356
            ),
            "มะพร้าวขูด": NutritionInfo(
                name="มะพร้าวขูด", calories=660, protein=6.9, carbs=23.7, fat=64.5,
                fiber=16.3, iron=3.3
            ),
            
            # อื่นๆ
            "เต้าหู้": NutritionInfo(
                name="เต้าหู้", calories=76, protein=8.1, carbs=1.9, fat=4.8,
                calcium=350, iron=5.4
            ),
            "เต้าหู้เหลือง": NutritionInfo(
                name="เต้าหู้เหลือง", calories=76, protein=8.1, carbs=1.9, fat=4.8,
                calcium=350
            ),
            "วุ้นเส้น": NutritionInfo(
                name="วุ้นเส้น", calories=351, protein=0.2, carbs=86.1, fat=0.1,
                iron=1.5
            ),
            "ข้าวคั่ว": NutritionInfo(
                name="ข้าวคั่ว", calories=382, protein=8.0, carbs=82.0, fat=2.0,
                iron=1.0
            ),
            "น้ำพริกเผา": NutritionInfo(
                name="น้ำพริกเผา", calories=210, protein=8.5, carbs=15.0, fat=12.0,
                sodium=1200
            ),
            "ปลาร้า": NutritionInfo(
                name="ปลาร้า", calories=133, protein=15.0, fat=8.0, carbs=2.0,
                sodium=4000, calcium=200
            ),
        }

class CookingAdjustmentHelper:
    """ตัวช่วยสำหรับปรับการคำนวณโภชนาการตามวิธีการทำอาหาร"""
    
    @staticmethod
    def get_cooking_adjustments(recipe_name: str) -> dict:
        """ดึงการปรับแต่งสำหรับการทำอาหาร"""
        cooking_adjustments = {
            "ไข่เจียว": {
                "oil_absorption": 0.1,  # ดูดซับน้ำมัน 10%
                "missing_ingredients": [
                    {"name": "น้ำมันพืช", "amount": 3, "unit": "ช้อนโต๊ะ", "consumed": 0.1}
                ]
            },
            "ไข่ดาว": {
                "oil_absorption": 0.2,  # ดูดซับน้ำมัน 20%
                "missing_ingredients": [
                    {"name": "น้ำมันพืช", "amount": 2, "unit": "ช้อนโต๊ะ", "consumed": 0.2}
                ]
            },
            "ผัดกะเพรา": {
                "oil_absorption": 0.7,  # ดูดซับน้ำมัน 70%
                "oil_adjustment": True
            },
            "ผัดไทย": {
                "oil_absorption": 0.8,
                "oil_adjustment": True
            },
            "ต้มยำกุ้ง": {
                "broth_consumption": 0.6,  # บริโภคน้ำซุป 60%
            },
            "แกงเขียวหวาน": {
                "coconut_milk_adjustment": True,
                "curry_paste_dilution": 0.8
            }
        }
        
        recipe_name_lower = recipe_name.lower()
        for key, adjustments in cooking_adjustments.items():
            if key.lower() in recipe_name_lower:
                return adjustments
        
        return {}
    
    @staticmethod
    def apply_cooking_adjustments(ingredients_data: dict, recipe_name: str) -> dict:
        """ใช้การปรับแต่งการทำอาหาร"""
        adjustments = CookingAdjustmentHelper.get_cooking_adjustments(recipe_name)
        
        if not adjustments:
            return ingredients_data
        
        adjusted_data = ingredients_data.copy()
        
        # ปรับการดูดซึมน้ำมัน
        if "oil_absorption" in adjustments:
            oil_ratio = adjustments["oil_absorption"]
            for ingredient_key, nutrition in adjusted_data.items():
                if any(oil_word in ingredient_key.lower() for oil_word in ["น้ำมัน", "oil"]):
                    # ปรับปริมาณโภชนาการตามอัตราการดูดซึม
                    nutrition.calories *= oil_ratio
                    nutrition.fat *= oil_ratio
                    nutrition.vitamin_e *= oil_ratio
        
        # เพิ่มวัตถุดิบที่ขาดหายไป
        if "missing_ingredients" in adjustments:
            for missing in adjustments["missing_ingredients"]:
                # ตรวจสอบว่ามีวัตถุดิบนี้อยู่แล้วหรือไม่
                has_ingredient = any(missing["name"] in key for key in adjusted_data.keys())
                
                if not has_ingredient:
                    # เพิ่มวัตถุดิบที่ขาดหายไป
                    from ingredient_converter import IngredientConverter
                    converter = IngredientConverter()
                    
                    # คำนวณน้ำหนัก
                    weight_grams = converter.convert_to_grams(
                        missing["amount"], missing["unit"], missing["name"]
                    )
                    
                    # ปรับตามปริมาณที่บริโภคจริง
                    consumed_ratio = missing.get("consumed", 1.0)
                    effective_weight = weight_grams * consumed_ratio
                    multiplier = effective_weight / 100.0
                    
                    # ดึงข้อมูลโภชนาการพื้นฐาน
                    thai_data = ThaiNutritionData()
                    base_nutrition = thai_data.get_nutrition_info(missing["name"])
                    
                    if base_nutrition:
                        # สร้างข้อมูลโภชนาการที่ปรับแล้ว
                        adjusted_nutrition = NutritionInfo(
                            name=missing["name"],
                            calories=base_nutrition.calories * multiplier,
                            protein=base_nutrition.protein * multiplier,
                            carbs=base_nutrition.carbs * multiplier,
                            fat=base_nutrition.fat * multiplier,
                            fiber=base_nutrition.fiber * multiplier,
                            sugar=base_nutrition.sugar * multiplier,
                            sodium=base_nutrition.sodium * multiplier,
                            vitamin_a=base_nutrition.vitamin_a * multiplier,
                            vitamin_c=base_nutrition.vitamin_c * multiplier,
                            vitamin_d=base_nutrition.vitamin_d * multiplier,
                            vitamin_e=base_nutrition.vitamin_e * multiplier,
                            vitamin_k=base_nutrition.vitamin_k * multiplier,
                            vitamin_b1=base_nutrition.vitamin_b1 * multiplier,
                            vitamin_b2=base_nutrition.vitamin_b2 * multiplier,
                            vitamin_b6=base_nutrition.vitamin_b6 * multiplier,
                            vitamin_b12=base_nutrition.vitamin_b12 * multiplier,
                            folate=base_nutrition.folate * multiplier,
                            niacin=base_nutrition.niacin * multiplier,
                            calcium=base_nutrition.calcium * multiplier,
                            iron=base_nutrition.iron * multiplier,
                            magnesium=base_nutrition.magnesium * multiplier,
                            phosphorus=base_nutrition.phosphorus * multiplier,
                            potassium=base_nutrition.potassium * multiplier,
                            zinc=base_nutrition.zinc * multiplier,
                            serving_size=f"{missing['amount']} {missing['unit']} (บริโภค {consumed_ratio*100:.0f}%)"
                        )
                        
                        key = f"{missing['name']} ({missing['amount']} {missing['unit']}) [เพิ่มเติม]"
                        adjusted_data[key] = adjusted_nutrition
        
        return adjusted_data

class NutritionAnalyzer:
    """คลาสหลักสำหรับวิเคราะห์คุณค่าทางโภชนาการ"""
    
    def __init__(self, usda_api_key: Optional[str] = None, nutritionix_app_id: Optional[str] = None, nutritionix_api_key: Optional[str] = None):
        self.db = NutritionDatabase()
        self.thai_data = ThaiNutritionData()
        self.usda_api = USDANutritionAPI(usda_api_key) if usda_api_key else None
        self.nutritionix_api = NutritionixAPI(nutritionix_app_id, nutritionix_api_key) if nutritionix_app_id and nutritionix_api_key else None
        self.converter = IngredientConverter()
        self.cooking_helper = CookingAdjustmentHelper()
    
    def analyze_ingredients(self, ingredients_text: str, recipe_name: str = "", apply_cooking_adjustments: bool = False) -> Dict[str, NutritionInfo]:
        """วิเคราะห์คุณค่าทางโภชนาการของวัตถุดิบทั้งหมด"""
        ingredients = self._parse_ingredients(ingredients_text)
        nutrition_data = {}
        
        for ingredient_line in ingredients:
            # ใช้ตัวแปลงหน่วยเพื่อหาน้ำหนักจริง
            converted = self.converter.parse_and_convert_ingredient(ingredient_line)
            ingredient_name = converted['name']
            multiplier = converted['nutrition_multiplier']
            
            # ดึงข้อมูลโภชนาการพื้นฐาน (ต่อ 100g)
            base_nutrition = self.get_ingredient_nutrition(ingredient_name)
            
            if base_nutrition:
                # ปรับค่าโภชนาการตามน้ำหนักจริง
                adjusted_nutrition = NutritionInfo(
                    name=ingredient_name,
                    calories=base_nutrition.calories * multiplier,
                    protein=base_nutrition.protein * multiplier,
                    carbs=base_nutrition.carbs * multiplier,
                    fat=base_nutrition.fat * multiplier,
                    fiber=base_nutrition.fiber * multiplier,
                    sugar=base_nutrition.sugar * multiplier,
                    sodium=base_nutrition.sodium * multiplier,
                    vitamin_a=base_nutrition.vitamin_a * multiplier,
                    vitamin_c=base_nutrition.vitamin_c * multiplier,
                    vitamin_d=base_nutrition.vitamin_d * multiplier,
                    vitamin_e=base_nutrition.vitamin_e * multiplier,
                    vitamin_k=base_nutrition.vitamin_k * multiplier,
                    vitamin_b1=base_nutrition.vitamin_b1 * multiplier,
                    vitamin_b2=base_nutrition.vitamin_b2 * multiplier,
                    vitamin_b6=base_nutrition.vitamin_b6 * multiplier,
                    vitamin_b12=base_nutrition.vitamin_b12 * multiplier,
                    folate=base_nutrition.folate * multiplier,
                    niacin=base_nutrition.niacin * multiplier,
                    calcium=base_nutrition.calcium * multiplier,
                    iron=base_nutrition.iron * multiplier,
                    magnesium=base_nutrition.magnesium * multiplier,
                    phosphorus=base_nutrition.phosphorus * multiplier,
                    potassium=base_nutrition.potassium * multiplier,
                    zinc=base_nutrition.zinc * multiplier,
                    serving_size=f"{converted['weight_grams']:.0f}g ({converted['quantity']} {converted['unit']})"
                )
                
                # เก็บข้อมูลโภชนาการที่ปรับแล้ว
                key = f"{ingredient_name} ({converted['quantity']} {converted['unit']})"
                nutrition_data[key] = adjusted_nutrition
        
        # ใช้การปรับแต่งการทำอาหารถ้าเปิดใช้งาน
        if apply_cooking_adjustments and recipe_name:
            nutrition_data = self.cooking_helper.apply_cooking_adjustments(nutrition_data, recipe_name)
        
        return nutrition_data
    
    def get_ingredient_nutrition(self, ingredient: str) -> Optional[NutritionInfo]:
        """ดึงข้อมูลโภชนาการของวัตถุดิบ"""
        # 1. ตรวจสอบแคชก่อน
        cached = self.db.get_cached_nutrition(ingredient)
        if cached:
            return cached
        
        # 2. ตรวจสอบฐานข้อมูลไทย
        thai_nutrition = self.thai_data.get_nutrition_info(ingredient)
        if thai_nutrition:
            self.db.cache_nutrition(ingredient, thai_nutrition)
            return thai_nutrition
        
        # 3. ใช้ USDA API (ถ้ามี API key)
        if self.usda_api:
            try:
                usda_nutrition = self.usda_api.get_nutrition_info(ingredient)
                if usda_nutrition:
                    self.db.cache_nutrition(ingredient, usda_nutrition)
                    return usda_nutrition
            except Exception as e:
                logger.warning(f"USDA API error for {ingredient}: {e}")
        
        # 4. ใช้ Nutritionix API (ถ้ามี API key)
        if self.nutritionix_api:
            try:
                nutritionix_nutrition = self.nutritionix_api.get_nutrition_info(ingredient)
                if nutritionix_nutrition:
                    self.db.cache_nutrition(ingredient, nutritionix_nutrition)
                    return nutritionix_nutrition
            except Exception as e:
                logger.warning(f"Nutritionix API error for {ingredient}: {e}")
        
        # 5. สร้างข้อมูลพื้นฐาน
        basic_nutrition = NutritionInfo(name=ingredient)
        self.db.cache_nutrition(ingredient, basic_nutrition)
        return basic_nutrition
    
    def _parse_ingredients(self, ingredients_text: str) -> List[str]:
        """แยกวัตถุดิบจากข้อความ (คืนค่าข้อความเต็มพร้อมปริมาณ)"""
        # แยกตามบรรทัด
        lines = ingredients_text.strip().split('\n')
        ingredients = []
        
        for line in lines:
            line = line.strip()
            if line and line.startswith('-'):
                # ลบ '-' แต่เก็บข้อความทั้งหมด
                ingredient_full = line[1:].strip()
                if ingredient_full and len(ingredient_full) > 1:
                    ingredients.append(ingredient_full)
        
        return ingredients
    
    def calculate_total_nutrition(self, nutrition_data: Dict[str, NutritionInfo]) -> NutritionInfo:
        """คำนวณคุณค่าทางโภชนาการรวม"""
        total = NutritionInfo(name="รวม")
        
        for nutrition in nutrition_data.values():
            total.calories += nutrition.calories
            total.protein += nutrition.protein
            total.carbs += nutrition.carbs
            total.fat += nutrition.fat
            total.fiber += nutrition.fiber
            total.sugar += nutrition.sugar
            total.sodium += nutrition.sodium
            total.vitamin_a += nutrition.vitamin_a
            total.vitamin_c += nutrition.vitamin_c
            total.vitamin_d += nutrition.vitamin_d
            total.vitamin_e += nutrition.vitamin_e
            total.vitamin_k += nutrition.vitamin_k
            total.vitamin_b1 += nutrition.vitamin_b1
            total.vitamin_b2 += nutrition.vitamin_b2
            total.vitamin_b6 += nutrition.vitamin_b6
            total.vitamin_b12 += nutrition.vitamin_b12
            total.folate += nutrition.folate
            total.niacin += nutrition.niacin
            total.calcium += nutrition.calcium
            total.iron += nutrition.iron
            total.magnesium += nutrition.magnesium
            total.phosphorus += nutrition.phosphorus
            total.potassium += nutrition.potassium
            total.zinc += nutrition.zinc
        
        return total
    
    def analyze_recipe(self, recipe_name: str, ingredients: str, apply_cooking_adjustments: bool = False) -> dict:
        """วิเคราะห์โภชนาการสำหรับสูตรอาหาร (สำหรับ streamlit app)"""
        # วิเคราะห์โภชนาการ
        nutrition_data = self.analyze_ingredients(ingredients, recipe_name, apply_cooking_adjustments)
        total_nutrition = self.calculate_total_nutrition(nutrition_data)
        
        # สร้างผลลัพธ์ในรูปแบบ dict
        result = {
            'recipe_name': recipe_name,
            'total_nutrition': {
                'calories': total_nutrition.calories,
                'protein': total_nutrition.protein,
                'carbs': total_nutrition.carbs,
                'fat': total_nutrition.fat,
                'fiber': total_nutrition.fiber,
                'vitamins': {
                    'วิตามิน A': total_nutrition.vitamin_a,
                    'วิตามิน C': total_nutrition.vitamin_c,
                    'วิตามิน D': total_nutrition.vitamin_d,
                    'วิตามิน E': total_nutrition.vitamin_e,
                    'วิตามิน K': total_nutrition.vitamin_k,
                    'วิตามิน B1': total_nutrition.vitamin_b1,
                    'วิตามิน B2': total_nutrition.vitamin_b2,
                    'วิตามิน B6': total_nutrition.vitamin_b6,
                    'วิตามิน B12': total_nutrition.vitamin_b12,
                },
                'minerals': {
                    'แคลเซียม': total_nutrition.calcium,
                    'เหล็ก': total_nutrition.iron,
                    'แมกนีเซียม': total_nutrition.magnesium,
                    'ฟอสฟอรัส': total_nutrition.phosphorus,
                    'โพแทสเซียม': total_nutrition.potassium,
                    'สังกะสี': total_nutrition.zinc,
                    'โซเดียม': total_nutrition.sodium,
                }
            },
            'ingredients': [],
            'ingredient_count': len(nutrition_data),
            'enhanced': apply_cooking_adjustments
        }
        
        # เพิ่มรายละเอียดแต่ละวัตถุดิบ
        for ingredient_key, nutrition in nutrition_data.items():
            # แยกชื่อวัตถุดิบออกจาก key ที่มีปริมาณ
            if '(' in ingredient_key:
                ingredient_name = ingredient_key.split('(')[0].strip()
            else:
                ingredient_name = ingredient_key
            
            result['ingredients'].append({
                'ingredient': ingredient_key,  # แสดงชื่อพร้อมปริมาณ
                'nutrition': nutrition  # NutritionInfo object
            })
        
        return result

# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    # สร้าง analyzer (ใส่ USDA API key ถ้ามี)
    analyzer = NutritionAnalyzer()
    
    # ตัวอย่างวัตถุดิบ
    ingredients_text = """
    - กุ้งนาง 4 ตัว
    - พริกไทย 5 เม็ด
    - กระเทียมกลีบใหญ่ 2 กลีบ
    - รากผักชี 5 ราก
    - น้ำปลา 2 ช้อนโต๊ะ
    - น้ำมันหมู 1 ช้อนโต๊ะ
    """
    
    # วิเคราะห์โภชนาการ
    nutrition_data = analyzer.analyze_ingredients(ingredients_text)
    
    # แสดงผล
    for ingredient, nutrition in nutrition_data.items():
        print(f"\n{ingredient}:")
        print(f"  พลังงาน: {nutrition.calories:.1f} แคลอรี่")
        print(f"  โปรตีน: {nutrition.protein:.1f} กรัม")
        print(f"  คาร์โบไฮเดรต: {nutrition.carbs:.1f} กรัม")
        print(f"  ไขมัน: {nutrition.fat:.1f} กรัม")
    
    # คำนวณรวม
    total = analyzer.calculate_total_nutrition(nutrition_data)
    print(f"\nรวมทั้งหมด:")
    print(f"  พลังงาน: {total.calories:.1f} แคลอรี่")
    print(f"  โปรตีน: {total.protein:.1f} กรัม")
    
    # ทดสอบการปรับแต่งการทำอาหาร
    print(f"\n=== ทดสอบการปรับแต่งการทำอาหาร ===")
    enhanced_data = analyzer.analyze_ingredients(ingredients_text, "ไข่เจียว", apply_cooking_adjustments=True)
    enhanced_total = analyzer.calculate_total_nutrition(enhanced_data)
    print(f"พลังงานหลังปรับแต่ง: {enhanced_total.calories:.1f} แคลอรี่")
