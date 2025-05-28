import requests
import json
import pandas as pd
import re
import time
from typing import Dict, List, Optional, Tuple
import sqlite3
from dataclasses import dataclass
import logging

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NutritionInfo:
    """โครงสร้างข้อมูลคุณค่าทางโภชนาการ"""
    ingredient: str
    calories: float = 0.0
    protein: float = 0.0
    carbs: float = 0.0
    fat: float = 0.0
    fiber: float = 0.0
    vitamins: Dict[str, float] = None
    minerals: Dict[str, float] = None
    
    def __post_init__(self):
        if self.vitamins is None:
            self.vitamins = {}
        if self.minerals is None:
            self.minerals = {}

class NutritionAnalyzer:
    """คลาสสำหรับวิเคราะห์และจัดการข้อมูลโภชนาการ"""
    
    def __init__(self, db_path: str = "nutrition.db"):
        self.db_path = db_path
        self.init_database()
        
        # Thai ingredient mapping สำหรับแปลงชื่อวัตถุดิบไทยเป็นภาษาอังกฤษ
        self.thai_to_english = {
            "กุ้ง": "shrimp",
            "หมู": "pork", 
            "ไก่": "chicken",
            "เนื้อ": "beef",
            "ปลา": "fish",
            "ข้าว": "rice",
            "แป้ง": "flour",
            "น้ำตาล": "sugar",
            "เกลือ": "salt",
            "พริก": "chili",
            "กระเทียม": "garlic",
            "หอม": "onion",
            "ขิง": "ginger",
            "ข่า": "galangal",
            "ตะไคร้": "lemongrass",
            "มะพร้าว": "coconut",
            "น้ำมัน": "oil",
            "ถั่ว": "bean",
            "ผัก": "vegetable",
            "ใบ": "leaf",
            "ราก": "root",
            "เห็ด": "mushroom",
            "ไข่": "egg",
            "นม": "milk",
            "เนย": "butter",
            "มะเขือเทศ": "tomato",
            "หน่อไม้": "bamboo shoot",
            "ฟัก": "gourd",
            "แตงกวา": "cucumber"
        }
        
    def init_database(self):
        """สร้างฐานข้อมูลสำหรับเก็บข้อมูลโภชนาการ"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # สร้างตาราง nutrition_data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS nutrition_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ingredient_thai TEXT UNIQUE,
                ingredient_english TEXT,
                calories REAL DEFAULT 0,
                protein REAL DEFAULT 0,
                carbs REAL DEFAULT 0,
                fat REAL DEFAULT 0,
                fiber REAL DEFAULT 0,
                vitamins TEXT,  -- JSON string
                minerals TEXT,  -- JSON string
                source TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                confidence_score REAL DEFAULT 0.5
            )
        ''')
        
        # สร้างตาราง recipe_nutrition สำหรับเก็บข้อมูลโภชนาการของสูตรอาหาร
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recipe_nutrition (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recipe_name TEXT UNIQUE,
                total_calories REAL DEFAULT 0,
                total_protein REAL DEFAULT 0,
                total_carbs REAL DEFAULT 0,
                total_fat REAL DEFAULT 0,
                total_fiber REAL DEFAULT 0,
                vitamins_total TEXT,  -- JSON string
                minerals_total TEXT,  -- JSON string
                serving_size INTEGER DEFAULT 1,
                last_calculated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def extract_ingredients(self, ingredient_text: str) -> List[str]:
        """แยกวัตถุดิบจากข้อความ"""
        if not ingredient_text:
            return []
        
        # แยกตามบรรทัดและขีด
        lines = ingredient_text.split('\n')
        ingredients = []
        
        for line in lines:
            line = line.strip()
            if not line or not line.startswith('-'):
                continue
                
            # ลบขีดออก
            line = line[1:].strip()
            
            # แยกชื่อวัตถุดิบจากปริมาณ
            # ใช้ regex หาชื่อวัตถุดิบก่อนเลขหรือหน่วยวัด
            ingredient_match = re.match(r'^([^0-9]+?)(?:\s*\d+.*)?$', line)
            if ingredient_match:
                ingredient = ingredient_match.group(1).strip()
                if ingredient:
                    ingredients.append(ingredient)
        
        return ingredients
    
    def translate_ingredient(self, thai_ingredient: str) -> str:
        """แปลชื่อวัตถุดิบจากไทยเป็นอังกฤษ"""
        thai_ingredient = thai_ingredient.lower()
        
        # ค้นหาคำที่ตรงกันในพจนานุกรม
        for thai, english in self.thai_to_english.items():
            if thai in thai_ingredient:
                return english
        
        # ถ้าไม่พบ ส่งคืนคำเดิม
        return thai_ingredient
    
    def get_nutrition_from_api(self, ingredient_english: str) -> Optional[NutritionInfo]:
        """ดึงข้อมูลโภชนาการจาก API (ตัวอย่างใช้ FDC API)"""
        try:
            # FDC API (USDA Food Database) - ฟรี
            api_key = "YOUR_FDC_API_KEY"  # ต้องสมัคร API key ฟรีจาก https://fdc.nal.usda.gov/api-guide.html
            
            # ค้นหาอาหาร
            search_url = f"https://api.nal.usda.gov/fdc/v1/foods/search"
            search_params = {
                "query": ingredient_english,
                "api_key": api_key,
                "pageSize": 1
            }
            
            response = requests.get(search_url, params=search_params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('foods') and len(data['foods']) > 0:
                    food = data['foods'][0]
                    food_id = food['fdcId']
                    
                    # ดึงข้อมูลโภชนาการรายละเอียด
                    detail_url = f"https://api.nal.usda.gov/fdc/v1/food/{food_id}"
                    detail_params = {"api_key": api_key}
                    
                    detail_response = requests.get(detail_url, params=detail_params, timeout=10)
                    
                    if detail_response.status_code == 200:
                        detail_data = detail_response.json()
                        return self._parse_fdc_nutrition(ingredient_english, detail_data)
            
            # ถ้า FDC ไม่สำเร็จ ลองใช้ Nutritionix API
            return self._get_nutrition_from_nutritionix(ingredient_english)
            
        except Exception as e:
            logger.error(f"Error getting nutrition from API for {ingredient_english}: {e}")
            return None
    
    def _parse_fdc_nutrition(self, ingredient: str, data: dict) -> NutritionInfo:
        """แปลงข้อมูลจาก FDC API เป็น NutritionInfo"""
        nutrition = NutritionInfo(ingredient=ingredient)
        
        # วิเคราะห์ข้อมูลโภชนาการ
        for nutrient in data.get('foodNutrients', []):
            nutrient_name = nutrient.get('nutrient', {}).get('name', '').lower()
            amount = nutrient.get('amount', 0)
            
            if 'energy' in nutrient_name or 'calorie' in nutrient_name:
                nutrition.calories = amount
            elif 'protein' in nutrient_name:
                nutrition.protein = amount
            elif 'carbohydrate' in nutrient_name:
                nutrition.carbs = amount
            elif 'fat' in nutrient_name and 'fatty' not in nutrient_name:
                nutrition.fat = amount
            elif 'fiber' in nutrient_name:
                nutrition.fiber = amount
            elif 'vitamin' in nutrient_name:
                nutrition.vitamins[nutrient_name] = amount
            elif any(mineral in nutrient_name for mineral in ['calcium', 'iron', 'magnesium', 'phosphorus', 'potassium', 'sodium', 'zinc']):
                nutrition.minerals[nutrient_name] = amount
        
        return nutrition
    
    def _get_nutrition_from_nutritionix(self, ingredient_english: str) -> Optional[NutritionInfo]:
        """ดึงข้อมูลจาก Nutritionix API (alternative)"""
        try:
            app_id = "YOUR_NUTRITIONIX_APP_ID"
            app_key = "YOUR_NUTRITIONIX_APP_KEY"
            
            url = "https://trackapi.nutritionix.com/v2/natural/nutrients"
            headers = {
                'x-app-id': app_id,
                'x-app-key': app_key,
                'Content-Type': 'application/json'
            }
            
            data = {"query": f"100g {ingredient_english}"}
            
            response = requests.post(url, headers=headers, json=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('foods') and len(result['foods']) > 0:
                    food = result['foods'][0]
                    
                    nutrition = NutritionInfo(ingredient=ingredient_english)
                    nutrition.calories = food.get('nf_calories', 0)
                    nutrition.protein = food.get('nf_protein', 0)
                    nutrition.carbs = food.get('nf_total_carbohydrate', 0)
                    nutrition.fat = food.get('nf_total_fat', 0)
                    nutrition.fiber = food.get('nf_dietary_fiber', 0)
                    
                    return nutrition
            
        except Exception as e:
            logger.error(f"Error getting nutrition from Nutritionix for {ingredient_english}: {e}")
        
        return None
    
    def get_fallback_nutrition(self, ingredient: str) -> NutritionInfo:
        """ข้อมูลโภชนาการพื้นฐานสำหรับวัตถุดิบทั่วไป (fallback)"""
        # ข้อมูลโภชนาการพื้นฐานสำหรับวัตถุดิบไทยทั่วไป (ต่อ 100g)
        fallback_data = {
            "ข้าว": NutritionInfo("ข้าว", 130, 2.7, 28.0, 0.3, 0.4),
            "หมู": NutritionInfo("หมู", 242, 27.0, 0.0, 14.0, 0.0),
            "ไก่": NutritionInfo("ไก่", 165, 31.0, 0.0, 3.6, 0.0),
            "เนื้อ": NutritionInfo("เนื้อ", 250, 26.0, 0.0, 15.0, 0.0),
            "กุ้ง": NutritionInfo("กุ้ง", 99, 18.0, 0.2, 1.4, 0.0),
            "ปลา": NutritionInfo("ปลา", 206, 22.0, 0.0, 12.0, 0.0),
            "ไข่": NutritionInfo("ไข่", 155, 13.0, 1.1, 11.0, 0.0),
            "มะพร้าว": NutritionInfo("มะพร้าว", 354, 3.3, 15.0, 33.0, 9.0),
            "น้ำมัน": NutritionInfo("น้ำมัน", 884, 0.0, 0.0, 100.0, 0.0),
            "น้ำตาล": NutritionInfo("น้ำตาล", 387, 0.0, 100.0, 0.0, 0.0),
            "แป้ง": NutritionInfo("แป้ง", 364, 10.0, 76.0, 1.0, 2.7),
        }
        
        for key, nutrition in fallback_data.items():
            if key in ingredient:
                return nutrition
        
        # ค่าเริ่มต้นสำหรับวัตถุดิบที่ไม่รู้จัก
        return NutritionInfo(ingredient, 50, 1.0, 10.0, 0.5, 1.0)
    
    def save_nutrition_to_db(self, thai_ingredient: str, nutrition: NutritionInfo, source: str = "api", confidence: float = 0.8):
        """บันทึกข้อมูลโภชนาการลงฐานข้อมูล"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO nutrition_data 
                (ingredient_thai, ingredient_english, calories, protein, carbs, fat, fiber, vitamins, minerals, source, confidence_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                thai_ingredient,
                nutrition.ingredient,
                nutrition.calories,
                nutrition.protein,
                nutrition.carbs,
                nutrition.fat,
                nutrition.fiber,
                json.dumps(nutrition.vitamins, ensure_ascii=False),
                json.dumps(nutrition.minerals, ensure_ascii=False),
                source,
                confidence
            ))
            
            conn.commit()
            logger.info(f"Saved nutrition data for {thai_ingredient}")
            
        except Exception as e:
            logger.error(f"Error saving nutrition data for {thai_ingredient}: {e}")
        finally:
            conn.close()
    
    def get_nutrition_from_db(self, thai_ingredient: str) -> Optional[NutritionInfo]:
        """ดึงข้อมูลโภชนาการจากฐานข้อมูล"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT ingredient_english, calories, protein, carbs, fat, fiber, vitamins, minerals
                FROM nutrition_data 
                WHERE ingredient_thai = ?
            ''', (thai_ingredient,))
            
            result = cursor.fetchone()
            if result:
                nutrition = NutritionInfo(
                    ingredient=result[0],
                    calories=result[1],
                    protein=result[2],
                    carbs=result[3],
                    fat=result[4],
                    fiber=result[5],
                    vitamins=json.loads(result[6]) if result[6] else {},
                    minerals=json.loads(result[7]) if result[7] else {}
                )
                return nutrition
                
        except Exception as e:
            logger.error(f"Error getting nutrition from DB for {thai_ingredient}: {e}")
        finally:
            conn.close()
        
        return None
    
    def analyze_ingredient(self, thai_ingredient: str) -> NutritionInfo:
        """วิเคราะห์วัตถุดิบ 1 ชนิด - ตรวจสอบ DB ก่อน แล้วค่อยไป API"""
        # ตรวจสอบในฐานข้อมูลก่อน
        nutrition = self.get_nutrition_from_db(thai_ingredient)
        if nutrition:
            logger.info(f"Found nutrition data in DB for {thai_ingredient}")
            return nutrition
        
        # ถ้าไม่มีในฐานข้อมูล ค้นหาจาก API
        english_ingredient = self.translate_ingredient(thai_ingredient)
        nutrition = self.get_nutrition_from_api(english_ingredient)
        
        if nutrition:
            # บันทึกลงฐานข้อมูล
            self.save_nutrition_to_db(thai_ingredient, nutrition, "api", 0.8)
            logger.info(f"Got nutrition data from API for {thai_ingredient}")
        else:
            # ใช้ข้อมูล fallback
            nutrition = self.get_fallback_nutrition(thai_ingredient)
            self.save_nutrition_to_db(thai_ingredient, nutrition, "fallback", 0.3)
            logger.info(f"Using fallback nutrition data for {thai_ingredient}")
        
        return nutrition
    
    def analyze_recipe(self, recipe_name: str, ingredients_text: str) -> Dict:
        """วิเคราะห์คุณค่าทางโภชนาการของสูตรอาหารทั้งหมด"""
        ingredients = self.extract_ingredients(ingredients_text)
        
        total_nutrition = {
            'calories': 0,
            'protein': 0,
            'carbs': 0,
            'fat': 0,
            'fiber': 0,
            'vitamins': {},
            'minerals': {}
        }
        
        ingredient_details = []
        
        for ingredient in ingredients:
            nutrition = self.analyze_ingredient(ingredient)
            
            # สะสมค่าโภชนาการรวม
            total_nutrition['calories'] += nutrition.calories
            total_nutrition['protein'] += nutrition.protein
            total_nutrition['carbs'] += nutrition.carbs
            total_nutrition['fat'] += nutrition.fat
            total_nutrition['fiber'] += nutrition.fiber
            
            # สะสมวิตามิน
            for vitamin, amount in nutrition.vitamins.items():
                if vitamin in total_nutrition['vitamins']:
                    total_nutrition['vitamins'][vitamin] += amount
                else:
                    total_nutrition['vitamins'][vitamin] = amount
            
            # สะสมแร่ธาตุ
            for mineral, amount in nutrition.minerals.items():
                if mineral in total_nutrition['minerals']:
                    total_nutrition['minerals'][mineral] += amount
                else:
                    total_nutrition['minerals'][mineral] = amount
            
            ingredient_details.append({
                'ingredient': ingredient,
                'nutrition': nutrition
            })
        
        # บันทึกข้อมูลโภชนาการของสูตรอาหาร
        self.save_recipe_nutrition(recipe_name, total_nutrition)
        
        return {
            'recipe_name': recipe_name,
            'total_nutrition': total_nutrition,
            'ingredients': ingredient_details,
            'ingredient_count': len(ingredients)
        }
    
    def save_recipe_nutrition(self, recipe_name: str, nutrition: Dict):
        """บันทึกข้อมูลโภชนาการของสูตรอาหารลงฐานข้อมูล"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO recipe_nutrition 
                (recipe_name, total_calories, total_protein, total_carbs, total_fat, total_fiber, vitamins_total, minerals_total)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                recipe_name,
                nutrition['calories'],
                nutrition['protein'],
                nutrition['carbs'],
                nutrition['fat'],
                nutrition['fiber'],
                json.dumps(nutrition['vitamins'], ensure_ascii=False),
                json.dumps(nutrition['minerals'], ensure_ascii=False)
            ))
            
            conn.commit()
            logger.info(f"Saved recipe nutrition data for {recipe_name}")
            
        except Exception as e:
            logger.error(f"Error saving recipe nutrition for {recipe_name}: {e}")
        finally:
            conn.close()
    
    def search_recipes_by_nutrition(self, nutrition_criteria: Dict) -> List[Dict]:
        """ค้นหาสูตรอาหารตามเกณฑ์โภชนาการ"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        conditions = []
        params = []
        
        if 'min_calories' in nutrition_criteria:
            conditions.append("total_calories >= ?")
            params.append(nutrition_criteria['min_calories'])
        
        if 'max_calories' in nutrition_criteria:
            conditions.append("total_calories <= ?")  
            params.append(nutrition_criteria['max_calories'])
        
        if 'min_protein' in nutrition_criteria:
            conditions.append("total_protein >= ?")
            params.append(nutrition_criteria['min_protein'])
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        try:
            cursor.execute(f'''
                SELECT recipe_name, total_calories, total_protein, total_carbs, total_fat, total_fiber
                FROM recipe_nutrition 
                WHERE {where_clause}
                ORDER BY total_calories
            ''', params)
            
            results = cursor.fetchall()
            
            return [
                {
                    'recipe_name': row[0],
                    'calories': row[1],
                    'protein': row[2],
                    'carbs': row[3],
                    'fat': row[4],
                    'fiber': row[5]
                }
                for row in results
            ]
            
        except Exception as e:
            logger.error(f"Error searching recipes by nutrition: {e}")
            return []
        finally:
            conn.close()


def process_all_recipes(csv_path: str = "thai_food_processed.csv"):
    """ประมวลผลข้อมูลโภชนาการสำหรับทุกสูตรอาหารในไฟล์"""
    analyzer = NutritionAnalyzer()
    
    try:
        df = pd.read_csv(csv_path)
        results = []
        
        for index, row in df.iterrows():
            recipe_name = row['name']
            ingredients = row['ingredient']
            
            logger.info(f"Processing recipe {index + 1}/{len(df)}: {recipe_name}")
            
            result = analyzer.analyze_recipe(recipe_name, ingredients)
            results.append(result)
            
            # หน่วงเวลาเพื่อไม่ให้ API rate limit
            time.sleep(1)
        
        logger.info(f"Completed processing {len(results)} recipes")
        return results
        
    except Exception as e:
        logger.error(f"Error processing recipes: {e}")
        return []


if __name__ == "__main__":
    # ตัวอย่างการใช้งาน
    analyzer = NutritionAnalyzer()
    
    # วิเคราะห์สูตรอาหารเดียว
    ingredients_text = """- กุ้งนาง 4 ตัว
- พริกไทย 5 เม็ด
- กระเทียมกลีบใหญ่ 2 กลีบ
- รากผักชี 5 ราก
- น้ำปลา 2 ช้อนโต๊ะ
- น้ำมันหมู 1 ช้อนโต๊ะ"""
    
    result = analyzer.analyze_recipe("กุ้งทาพริกไทยกระเทียม", ingredients_text)
    print(json.dumps(result, indent=2, ensure_ascii=False))
