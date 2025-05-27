# nutrition/nutrition_fetcher.py
import requests
import pandas as pd
from typing import Dict, Optional
import json
from datetime import datetime

class NutritionFetcher:
    def __init__(self):
        self.sources = {
            'usda': 'https://api.nal.usda.gov/fdc/v1/',
            'thai_fda': 'local_database',  # ใช้ข้อมูลจาก อย.ไทย
            'nutritionix': 'https://api.nutritionix.com/v1_1/'
        }
        self.cache_file = 'data/nutrition_cache.json'
        self.load_cache()
    
    def load_cache(self):
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                self.cache = json.load(f)
        except FileNotFoundError:
            self.cache = {}
    
    def save_cache(self):
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
    
    def fetch_nutrition(self, ingredient: str, amount: float = 100) -> Optional[Dict]:
        """ดึงข้อมูลโภชนาการต่อ 100 กรัม"""
        # ตรวจสอบ cache ก่อน
        if ingredient in self.cache:
            cache_data = self.cache[ingredient]
            # ตรวจสอบอายุข้อมูล (30 วัน)
            if self._is_cache_valid(cache_data['last_updated']):
                return cache_data['nutrition']
        
        # ลองดึงจากแหล่งข้อมูลต่างๆ
        nutrition_data = None
        
        # 1. ลองจากฐานข้อมูลท้องถิ่นก่อน
        nutrition_data = self._fetch_from_local_db(ingredient)
        
        # 2. ถ้าไม่มี ลองจาก API
        if not nutrition_data:
            nutrition_data = self._fetch_from_api(ingredient)
        
        # 3. ถ้ายังไม่มี ใช้การประมาณจากวัตถุดิบคล้ายกัน
        if not nutrition_data:
            nutrition_data = self._estimate_nutrition(ingredient)
        
        # บันทึก cache
        if nutrition_data:
            self.cache[ingredient] = {
                'nutrition': nutrition_data,
                'last_updated': datetime.now().isoformat()
            }
            self.save_cache()
        
        return nutrition_data
    
    def _fetch_from_local_db(self, ingredient: str) -> Optional[Dict]:
        """ดึงจากฐานข้อมูลท้องถิ่น"""
        try:
            df = pd.read_csv('data/ingredients_nutrition.csv')
            row = df[df['ingredient_name'] == ingredient]
            if not row.empty:
                return row.iloc[0].to_dict()
        except:
            pass
        return None
    
    def _fetch_from_api(self, ingredient: str) -> Optional[Dict]:
        """ดึงจาก API ภายนอก (ต้องมี API key)"""
        # ตัวอย่างการเรียก API
        # headers = {'x-api-key': 'YOUR_API_KEY'}
        # response = requests.get(f"{self.sources['nutritionix']}/search/{ingredient}")
        # if response.status_code == 200:
        #     return self._parse_api_response(response.json())
        return None
    
    def _estimate_nutrition(self, ingredient: str) -> Optional[Dict]:
        """ประมาณค่าจากวัตถุดิบคล้ายกัน"""
        similar_ingredients = self._find_similar_ingredients(ingredient)
        if similar_ingredients:
            # คำนวณค่าเฉลี่ย
            return self._calculate_average_nutrition(similar_ingredients)
        return None