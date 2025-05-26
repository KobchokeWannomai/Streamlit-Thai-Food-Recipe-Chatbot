import pandas as pd
import requests
import re
import json
import time
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

# ตั้งค่าการบันทึกล็อก
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class NutritionInfo:
    """คลาสสำหรับเก็บข้อมูลคุณค่าทางโภชนาการ"""
    calories: float = 0.0
    protein: float = 0.0
    carbohydrates: float = 0.0
    fat: float = 0.0
    fiber: float = 0.0
    sugar: float = 0.0
    sodium: float = 0.0
    calcium: float = 0.0
    iron: float = 0.0
    vitamin_a: float = 0.0
    vitamin_c: float = 0.0
    vitamin_d: float = 0.0
    vitamin_e: float = 0.0
    vitamin_k: float = 0.0
    thiamin: float = 0.0
    riboflavin: float = 0.0
    niacin: float = 0.0
    vitamin_b6: float = 0.0
    folate: float = 0.0
    vitamin_b12: float = 0.0
    phosphorus: float = 0.0
    magnesium: float = 0.0
    zinc: float = 0.0
    potassium: float = 0.0

class NutritionAnalyzer:
    """คลาสสำหรับวิเคราะห์และหาข้อมูลคุณค่าทางโภชนาการ"""
    
    def __init__(self, api_key: str = None):
        """
        เริ่มต้นตัววิเคราะห์คุณค่าทางโภชนาการ
        
        Args:
            api_key: คีย์ API สำหรับเข้าถึงฐานข้อมูลคุณค่าทางโภชนาการ
        """
        self.api_key = api_key
        self.usda_base_url = "https://api.nal.usda.gov/fdc/v1"
        self.nutrition_cache = {}  # แคชข้อมูลเพื่อลดการเรียก API
        
        # พจนานุกรมแปลงชื่อวัตถุดิบไทยเป็นอังกฤษ
        self.thai_to_english = {
            'กุ้ง': 'shrimp',
            'ปลา': 'fish',
            'หมู': 'pork',
            'ไก่': 'chicken',
            'เนื้อ': 'beef',
            'ข้าว': 'rice',
            'น้ำมัน': 'oil',
            'น้ำตาล': 'sugar',
            'เกลือ': 'salt',
            'พริก': 'pepper',
            'หอม': 'onion',
            'กระเทียม': 'garlic',
            'ขิง': 'ginger',
            'ตะไคร้': 'lemongrass',
            'ใบมะกรูด': 'kaffir lime leaves',
            'ผักชี': 'cilantro',
            'มะพร้าว': 'coconut',
            'มะนาว': 'lime',
            'มะขาม': 'tamarind',
            'ถั่ว': 'beans',
            'ผักบุ้ง': 'water spinach',
            'คะน้า': 'chinese broccoli',
            'ผักกาด': 'chinese cabbage',
            'มะเขือ': 'eggplant',
            'ฟัก': 'gourd',
            'แตงกวา': 'cucumber',
            'มะเขือเทศ': 'tomato',
            'หน่อไม้': 'bamboo shoots',
            'เห็ด': 'mushroom',
            'ไข่': 'egg',
            'นม': 'milk',
            'เต้าหู้': 'tofu',
            'แป้ง': 'flour',
            'น้ำปลา': 'fish sauce',
            'ซีอิ๊ว': 'soy sauce',
            'กะปิ': 'shrimp paste'
        }

    def extract_ingredients(self, ingredient_text: str) -> List[str]:
        """
        แยกวัตถุดิบจากข้อความ
        
        Args:
            ingredient_text: ข้อความที่มีรายการวัตถุดิบ
            
        Returns:
            รายการวัตถุดิบที่แยกแล้ว
        """
        if not ingredient_text or pd.isna(ingredient_text):
            return []
            
        # แยกวัตถุดิบตามเครื่องหมาย - และ \n
        ingredients = re.split(r'[-\n]+', ingredient_text)
        
        # ทำความสะอาดและกรองเฉพาะชื่อวัตถุดิบ
        cleaned_ingredients = []
        for ingredient in ingredients:
            ingredient = ingredient.strip()
            if ingredient and len(ingredient) > 2:
                # ลบตัวเลขและหน่วยออก
                ingredient_name = re.sub(r'\d+.*$', '', ingredient).strip()
                if ingredient_name:
                    cleaned_ingredients.append(ingredient_name)
        
        return cleaned_ingredients

    def translate_ingredient(self, thai_ingredient: str) -> str:
        """
        แปลชื่อวัตถุดิบจากไทยเป็นอังกฤษ
        
        Args:
            thai_ingredient: ชื่อวัตถุดิบภาษาไทย
            
        Returns:
            ชื่อวัตถุดิบภาษาอังกฤษ
        """
        # ตรวจสอบการแปลงโดยตรง
        for thai, english in self.thai_to_english.items():
            if thai in thai_ingredient:
                return english
                
        # ถ้าไม่พบการแปลงโดยตรง ให้ใช้ชื่อเดิม
        return thai_ingredient

    def search_usda_food(self, ingredient: str) -> Optional[Dict]:
        """
        ค้นหาข้อมูลอาหารจาก USDA FoodData Central
        
        Args:
            ingredient: ชื่อวัตถุดิบ
            
        Returns:
            ข้อมูลอาหารจาก USDA หรือ None ถ้าไม่พบ
        """
        if not self.api_key:
            logger.warning("ไม่มี API key สำหรับ USDA FoodData Central")
            return None
            
        # ตรวจสอบแคช
        if ingredient in self.nutrition_cache:
            return self.nutrition_cache[ingredient]
            
        try:
            # แปลงชื่อวัตถุดิบเป็นอังกฤษ
            english_ingredient = self.translate_ingredient(ingredient)
            
            # ค้นหาอาหารใน USDA
            search_url = f"{self.usda_base_url}/foods/search"
            params = {
                'api_key': self.api_key,
                'query': english_ingredient,
                'pageSize': 5
            }
            
            response = requests.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('foods'):
                # เลือกผลลัพธ์แรกที่มีข้อมูลคุณค่าทางโภชนาการ
                for food in data['foods']:
                    if food.get('foodNutrients'):
                        food_details = self.get_food_details(food['fdcId'])
                        if food_details:
                            self.nutrition_cache[ingredient] = food_details
                            return food_details
                            
            # ถ้าไม่พบข้อมูลใน USDA ให้ใช้ข้อมูลเริ่มต้น
            default_nutrition = self.get_default_nutrition(ingredient)
            self.nutrition_cache[ingredient] = default_nutrition
            return default_nutrition
            
        except requests.RequestException as e:
            logger.error(f"ข้อผิดพลาดในการเรียก USDA API: {e}")
            return self.get_default_nutrition(ingredient)
        except Exception as e:
            logger.error(f"ข้อผิดพลาดในการค้นหาข้อมูล {ingredient}: {e}")
            return self.get_default_nutrition(ingredient)

    def get_food_details(self, fdc_id: str) -> Optional[Dict]:
        """
        ดึงรายละเอียดอาหารจาก USDA FoodData Central
        
        Args:
            fdc_id: รหัสอาหารใน USDA
            
        Returns:
            รายละเอียดอาหารหรือ None ถ้าไม่พบ
        """
        try:
            detail_url = f"{self.usda_base_url}/food/{fdc_id}"
            params = {'api_key': self.api_key}
            
            response = requests.get(detail_url, params=params, timeout=10)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"ข้อผิดพลาดในการดึงรายละเอียดอาหาร {fdc_id}: {e}")
            return None

    def get_default_nutrition(self, ingredient: str) -> Dict:
        """
        ให้ข้อมูลคุณค่าทางโภชนาการเริ่มต้นสำหรับวัตถุดิบที่ไม่พบในฐานข้อมูล
        
        Args:
            ingredient: ชื่อวัตถุดิบ
            
        Returns:
            ข้อมูลคุณค่าทางโภชนาการเริ่มต้น
        """
        # ข้อมูลคุณค่าทางโภชนาการเริ่มต้นสำหรับวัตถุดิบทั่วไป
        default_values = {
            'กุ้ง': {'calories': 99, 'protein': 18.0, 'fat': 1.4, 'calcium': 52, 'iron': 0.5},
            'ปลา': {'calories': 206, 'protein': 22.0, 'fat': 12.0, 'calcium': 20, 'iron': 1.0},
            'หมู': {'calories': 242, 'protein': 27.0, 'fat': 14.0, 'iron': 0.9, 'zinc': 2.9},
            'ไก่': {'calories': 239, 'protein': 27.0, 'fat': 14.0, 'iron': 1.0, 'zinc': 1.3},
            'เนื้อ': {'calories': 250, 'protein': 26.0, 'fat': 15.0, 'iron': 2.6, 'zinc': 4.8},
            'ข้าว': {'calories': 130, 'carbohydrates': 28.0, 'protein': 2.7, 'fat': 0.3},
            'น้ำมัน': {'calories': 884, 'fat': 100.0},
            'น้ำตาล': {'calories': 387, 'carbohydrates': 100.0},
            'เกลือ': {'sodium': 38758},
            'ไข่': {'calories': 155, 'protein': 13.0, 'fat': 11.0, 'vitamin_a': 160, 'vitamin_d': 20}
        }
        
        # ค้นหาข้อมูลเริ่มต้นที่เหมาะสม
        for key, values in default_values.items():
            if key in ingredient:
                return {
                    'description': ingredient,
                    'foodNutrients': [
                        {'nutrientName': name, 'value': value, 'unitName': 'per 100g'}
                        for name, value in values.items()
                    ]
                }
        
        # ถ้าไม่พบข้อมูลเริ่มต้น ให้ใช้ค่าเริ่มต้นทั่วไป
        return {
            'description': ingredient,
            'foodNutrients': [
                {'nutrientName': 'calories', 'value': 0, 'unitName': 'per 100g'}
            ]
        }

    def extract_nutrition_values(self, food_data: Dict) -> NutritionInfo:
        """
        แยกข้อมูลคุณค่าทางโภชนาการจากข้อมูลอาหาร
        
        Args:
            food_data: ข้อมูลอาหารจาก API
            
        Returns:
            ข้อมูลคุณค่าทางโภชนาการ
        """
        nutrition = NutritionInfo()
        
        if not food_data or 'foodNutrients' not in food_data:
            return nutrition
            
        # แมปชื่อสารอาหารกับคุณสมบัติ
        nutrient_mapping = {
            'Energy': 'calories',
            'Protein': 'protein',
            'Carbohydrate, by difference': 'carbohydrates',
            'Total lipid (fat)': 'fat',
            'Fiber, total dietary': 'fiber',
            'Sugars, total including NLEA': 'sugar',
            'Sodium, Na': 'sodium',
            'Calcium, Ca': 'calcium',
            'Iron, Fe': 'iron',
            'Vitamin A, RAE': 'vitamin_a',
            'Vitamin C, total ascorbic acid': 'vitamin_c',
            'Vitamin D (D2 + D3)': 'vitamin_d',
            'Vitamin E (alpha-tocopherol)': 'vitamin_e',
            'Vitamin K (phylloquinone)': 'vitamin_k',
            'Thiamin': 'thiamin',
            'Riboflavin': 'riboflavin',
            'Niacin': 'niacin',
            'Vitamin B-6': 'vitamin_b6',
            'Folate, total': 'folate',
            'Vitamin B-12': 'vitamin_b12',
            'Phosphorus, P': 'phosphorus',
            'Magnesium, Mg': 'magnesium',
            'Zinc, Zn': 'zinc',
            'Potassium, K': 'potassium'
        }
        
        for nutrient in food_data['foodNutrients']:
            nutrient_name = nutrient.get('nutrientName', '')
            value = nutrient.get('value', 0)
            
            for api_name, attr_name in nutrient_mapping.items():
                if api_name.lower() in nutrient_name.lower():
                    setattr(nutrition, attr_name, float(value))
                    break
        
        return nutrition

    def analyze_recipe_nutrition(self, ingredients_text: str) -> Dict:
        """
        วิเคราะห์คุณค่าทางโภชนาการของสูตรอาหาร
        
        Args:
            ingredients_text: ข้อความรายการวัตถุดิบ
            
        Returns:
            ข้อมูลคุณค่าทางโภชนาการรวมของสูตรอาหาร
        """
        ingredients = self.extract_ingredients(ingredients_text)
        total_nutrition = NutritionInfo()
        ingredient_details = []
        
        for ingredient in ingredients:
            if len(ingredient.strip()) > 2:  # ตรวจสอบว่าชื่อวัตถุดิบมีความยาวเพียงพอ
                food_data = self.search_usda_food(ingredient)
                if food_data:
                    nutrition = self.extract_nutrition_values(food_data)
                    
                    # รวมคุณค่าทางโภชนาการ (ประมาณการ)
                    total_nutrition.calories += nutrition.calories * 0.1  # สมมติว่าใช้ 10% ของ 100g
                    total_nutrition.protein += nutrition.protein * 0.1
                    total_nutrition.carbohydrates += nutrition.carbohydrates * 0.1
                    total_nutrition.fat += nutrition.fat * 0.1
                    total_nutrition.fiber += nutrition.fiber * 0.1
                    total_nutrition.calcium += nutrition.calcium * 0.1
                    total_nutrition.iron += nutrition.iron * 0.1
                    total_nutrition.vitamin_a += nutrition.vitamin_a * 0.1
                    total_nutrition.vitamin_c += nutrition.vitamin_c * 0.1
                    
                    ingredient_details.append({
                        'ingredient': ingredient,
                        'nutrition': nutrition
                    })
                
                # เพิ่มหน่วงเวลาเพื่อไม่ให้เรียก API บ่อยเกินไป
                time.sleep(0.1)
        
        return {
            'total_nutrition': total_nutrition,
            'ingredient_details': ingredient_details
        }

    def process_csv_file(self, input_file: str, output_file: str = None):
        """
        ประมวลผลไฟล์ CSV และเพิ่มข้อมูลคุณค่าทางโภชนาการ
        
        Args:
            input_file: ไฟล์ CSV ต้นฉบับ
            output_file: ไฟล์ CSV ผลลัพธ์ (ถ้าไม่ระบุจะเขียนทับไฟล์เดิม)
        """
        if output_file is None:
            output_file = input_file.replace('.csv', '_with_nutrition.csv')
        
        try:
            # อ่านไฟล์ CSV
            df = pd.read_csv(input_file)
            logger.info(f"อ่านข้อมูล {len(df)} รายการจากไฟล์ {input_file}")
            
            # เพิ่มคอลัมน์ใหม่สำหรับคุณค่าทางโภชนาการ
            nutrition_columns = [
                'calories', 'protein', 'carbohydrates', 'fat', 'fiber',
                'calcium', 'iron', 'vitamin_a', 'vitamin_c', 'vitamin_d'
            ]
            
            for col in nutrition_columns:
                df[col] = 0.0
            
            df['nutrition_info'] = ''
            
            # วิเคราะห์คุณค่าทางโภชนาการสำหรับแต่ละสูตร
            for idx, row in df.iterrows():
                logger.info(f"กำลังวิเคราะห์สูตรที่ {idx + 1}: {row['name']}")
                
                try:
                    nutrition_analysis = self.analyze_recipe_nutrition(row['ingredient'])
                    total_nutrition = nutrition_analysis['total_nutrition']
                    
                    # บันทึกข้อมูลคุณค่าทางโภชนาการ
                    df.at[idx, 'calories'] = round(total_nutrition.calories, 2)
                    df.at[idx, 'protein'] = round(total_nutrition.protein, 2)
                    df.at[idx, 'carbohydrates'] = round(total_nutrition.carbohydrates, 2)
                    df.at[idx, 'fat'] = round(total_nutrition.fat, 2)
                    df.at[idx, 'fiber'] = round(total_nutrition.fiber, 2)
                    df.at[idx, 'calcium'] = round(total_nutrition.calcium, 2)
                    df.at[idx, 'iron'] = round(total_nutrition.iron, 2)
                    df.at[idx, 'vitamin_a'] = round(total_nutrition.vitamin_a, 2)
                    df.at[idx, 'vitamin_c'] = round(total_nutrition.vitamin_c, 2)
                    df.at[idx, 'vitamin_d'] = round(total_nutrition.vitamin_d, 2)
                    
                    # บันทึกข้อมูลรายละเอียด
                    ingredient_summary = []
                    for detail in nutrition_analysis['ingredient_details']:
                        ingredient_summary.append(f"{detail['ingredient']}: {detail['nutrition'].calories:.0f} cal")
                    
                    df.at[idx, 'nutrition_info'] = '; '.join(ingredient_summary)
                    
                except Exception as e:
                    logger.error(f"ข้อผิดพลาดในการวิเคราะห์สูตร {row['name']}: {e}")
                    continue
            
            # บันทึกไฟล์ผลลัพธ์
            df.to_csv(output_file, index=False)
            logger.info(f"บันทึกไฟล์ผลลัพธ์: {output_file}")
            
            # แสดงสถิติสรุป
            self.print_nutrition_summary(df)
            
        except Exception as e:
            logger.error(f"ข้อผิดพลาดในการประมวลผลไฟล์: {e}")

    def print_nutrition_summary(self, df: pd.DataFrame):
        """
        แสดงสถิติสรุปคุณค่าทางโภชนาการ
        
        Args:
            df: DataFrame ที่มีข้อมูลคุณค่าทางโภชนาการ
        """
        print("\n=== สรุปคุณค่าทางโภชนาการ ===")
        print(f"จำนวนสูตรอาหาร: {len(df)} สูตร")
        
        nutrition_cols = ['calories', 'protein', 'carbohydrates', 'fat', 'calcium', 'iron']
        for col in nutrition_cols:
            if col in df.columns:
                avg_value = df[col].mean()
                max_value = df[col].max()
                min_value = df[col].min()
                print(f"{col.capitalize()}: เฉลี่ย {avg_value:.2f}, สูงสุด {max_value:.2f}, ต่ำสุด {min_value:.2f}")

# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    # ตั้งค่า API key สำหรับ USDA FoodData Central
    # สมัครได้ฟรีที่ https://fdc.nal.usda.gov/api-guide.html
    API_KEY = "YOUR_USDA_API_KEY_HERE"  # เปลี่ยนเป็น API key ของคุณ
    
    # สร้างตัววิเคราะห์คุณค่าทางโภชนาการ
    analyzer = NutritionAnalyzer(api_key=API_KEY)
    
    # ประมวลผลไฟล์ CSV
    analyzer.process_csv_file("thai_food_processed.csv", "thai_food_with_nutrition.csv")
