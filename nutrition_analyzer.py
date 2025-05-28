import requests
import json
import sqlite3
import re
import time
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

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

class USDANutritionAPI:
    """คลาสสำหรับดึงข้อมูลจาก USDA FoodData Central API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.nal.usda.gov/fdc/v1"
        self.session = requests.Session()
    
    def search_food(self, query: str) -> Optional[Dict]:
        """ค้นหาอาหารจาก USDA database"""
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
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
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

class ThaiNutritionData:
    """คลาสสำหรับข้อมูลโภชนาการอาหารไทยที่สร้างขึ้นเอง"""
    
    def __init__(self):
        self.thai_nutrition_db = self._load_thai_nutrition_data()
    
    def _load_thai_nutrition_data(self) -> Dict[str, NutritionInfo]:
        """โหลดข้อมูลโภชนาการอาหารไทยพื้นฐาน"""
        return {
            # เนื้อสัตว์
            "หมู": NutritionInfo(
                name="หมู", calories=242, protein=27.3, fat=14.0, carbs=0,
                iron=0.9, zinc=2.4, vitamin_b1=0.7, vitamin_b12=0.7
            ),
            "ไก่": NutritionInfo(
                name="ไก่", calories=239, protein=27.3, fat=13.6, carbs=0,
                niacin=8.2, vitamin_b6=0.5, phosphorus=182
            ),
            "เนื้อ": NutritionInfo(
                name="เนื้อ", calories=250, protein=26.0, fat=15.0, carbs=0,
                iron=2.6, zinc=4.8, vitamin_b12=2.6
            ),
            "กุ้ง": NutritionInfo(
                name="กุ้ง", calories=99, protein=18.0, fat=1.7, carbs=0.9,
                calcium=52, iron=0.5, zinc=1.6
            ),
            "ปลา": NutritionInfo(
                name="ปลา", calories=206, protein=22.0, fat=12.0, carbs=0,
                calcium=20, iron=1.0, vitamin_d=10.9
            ),
            
            # ผัก
            "กะหล่ำปลี": NutritionInfo(
                name="กะหล่ำปลี", calories=25, protein=1.3, carbs=5.8, fat=0.1,
                vitamin_c=36.6, vitamin_k=76, folate=43, fiber=2.5
            ),
            "คะน้า": NutritionInfo(
                name="คะน้า", calories=22, protein=2.2, carbs=4.2, fat=0.3,
                vitamin_a=241, vitamin_c=45, calcium=105, iron=1.5
            ),
            "ผักบุ้ง": NutritionInfo(
                name="ผักบุ้ง", calories=19, protein=2.6, carbs=3.1, fat=0.2,
                vitamin_a=318, vitamin_c=55, iron=2.5, calcium=77
            ),
            
            # เครื่องปรุง
            "น้ำปลา": NutritionInfo(
                name="น้ำปลา", calories=10, protein=1.5, carbs=1.0, fat=0,
                sodium=1413
            ),
            "กะทิ": NutritionInfo(
                name="กะทิ", calories=230, protein=2.3, fat=23.8, carbs=5.5,
                iron=3.9, magnesium=37
            ),
            "น้ำตาล": NutritionInfo(
                name="น้ำตาล", calories=387, protein=0, fat=0, carbs=100,
                calcium=1
            ),
            
            # ข้าว/แป้ง
            "ข้าว": NutritionInfo(
                name="ข้าว", calories=130, protein=2.7, carbs=28, fat=0.3,
                niacin=1.6, vitamin_b6=0.1, magnesium=25
            ),
            "แป้ง": NutritionInfo(
                name="แป้ง", calories=364, protein=10.3, carbs=76.3, fat=0.9,
                iron=1.2, niacin=5.9, folate=26
            )
        }
    
    def get_nutrition_info(self, ingredient: str) -> Optional[NutritionInfo]:
        """ดึงข้อมูลโภชนาการจากฐานข้อมูลไทย"""
        # ทำ fuzzy matching สำหรับวัตถุดิบไทย
        ingredient_clean = self._clean_thai_ingredient(ingredient)
        
        # ค้นหาแบบตรงไปตรงมา
        if ingredient_clean in self.thai_nutrition_db:
            return self.thai_nutrition_db[ingredient_clean]
        
        # ค้นหาแบบ partial match
        for key, nutrition in self.thai_nutrition_db.items():
            if key in ingredient_clean or ingredient_clean in key:
                return nutrition
        
        return None
    
    def _clean_thai_ingredient(self, ingredient: str) -> str:
        """ทำความสะอาดชื่อวัตถุดิบภาษาไทย"""
        # ลบข้อความที่ไม่จำเป็น
        unwanted_words = ['ขนาด', 'กลาง', 'เล็ก', 'ใหญ่', 'สด', 'แห้ง', 'ต้ม', '1', '2', '3', '4', '5']
        result = ingredient
        for word in unwanted_words:
            result = result.replace(word, '')
        
        return result.strip()

class NutritionAnalyzer:
    """คลาสหลักสำหรับวิเคราะห์คุณค่าทางโภชนาการ"""
    
    def __init__(self, usda_api_key: Optional[str] = None):
        self.db = NutritionDatabase()
        self.thai_data = ThaiNutritionData()
        self.usda_api = USDANutritionAPI(usda_api_key) if usda_api_key else None
    
    def analyze_ingredients(self, ingredients_text: str) -> Dict[str, NutritionInfo]:
        """วิเคราะห์คุณค่าทางโภชนาการของวัตถุดิบทั้งหมด"""
        ingredients = self._parse_ingredients(ingredients_text)
        nutrition_data = {}
        
        for ingredient in ingredients:
            nutrition = self.get_ingredient_nutrition(ingredient)
            if nutrition:
                nutrition_data[ingredient] = nutrition
        
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
            usda_nutrition = self.usda_api.get_nutrition_info(ingredient)
            if usda_nutrition:
                self.db.cache_nutrition(ingredient, usda_nutrition)
                return usda_nutrition
        
        # 4. สร้างข้อมูลพื้นฐาน
        basic_nutrition = NutritionInfo(name=ingredient)
        self.db.cache_nutrition(ingredient, basic_nutrition)
        return basic_nutrition
    
    def _parse_ingredients(self, ingredients_text: str) -> List[str]:
        """แยกวัตถุดิบจากข้อความ"""
        # แยกตามบรรทัด
        lines = ingredients_text.strip().split('\n')
        ingredients = []
        
        for line in lines:
            line = line.strip()
            if line and line.startswith('-'):
                # ลบ '-' และข้อความที่ไม่จำเป็น
                ingredient = line[1:].strip()
                
                # ลบจำนวนและหน่วย
                ingredient = re.sub(r'\d+[\s]*[กชฟผลถ้วยช้อนกิโลกรัมกลีบใบเม็ดตัวคู่].*', '', ingredient)
                ingredient = re.sub(r'\([^)]*\)', '', ingredient)  # ลบข้อความในวงเล็บ
                
                ingredient = ingredient.strip()
                if ingredient and len(ingredient) > 1:
                    ingredients.append(ingredient)
        
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
    
    def analyze_recipe(self, recipe_name: str, ingredients: str) -> dict:
        """วิเคราะห์โภชนาการสำหรับสูตรอาหาร (สำหรับ streamlit app)"""
        # วิเคราะห์โภชนาการ
        nutrition_data = self.analyze_ingredients(ingredients)
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
                }
            },
            'ingredients': [],
            'ingredient_count': len(nutrition_data)
        }
        
        # เพิ่มรายละเอียดแต่ละวัตถุดิบ
        for ingredient, nutrition in nutrition_data.items():
            result['ingredients'].append({
                'ingredient': ingredient,
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
