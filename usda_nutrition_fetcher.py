#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USDA Nutrition Data Fetcher
สคริปต์สำหรับดึงข้อมูลโภชนาการจาก USDA FoodData Central API
"""

import requests
import pandas as pd
import json
import time
import os
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import csv
from tqdm import tqdm

class USDANutritionFetcher:
    """คลาสสำหรับดึงข้อมูลโภชนาการจาก USDA API"""
    
    def __init__(self, api_key: str = "DEMO_KEY"):
        """
        เริ่มต้น USDA Nutrition Fetcher
        
        Args:
            api_key (str): USDA API Key (ใช้ DEMO_KEY ถ้าไม่มี)
        """
        self.api_key = api_key
        self.base_url = "https://api.nal.usda.gov/fdc/v1"
        self.rate_limit_delay = 1.0  # วินาทีระหว่างการเรียก API
        
        # แมปชื่อสารอาหารจาก USDA เป็นคอลัมน์ในฐานข้อมูล
        self.nutrient_mapping = {
            'Energy': 'calories_per_100g',
            'Energy (Atwater General Factors)': 'calories_per_100g',
            'Energy (Atwater Specific Factors)': 'calories_per_100g',
            'Protein': 'protein_per_100g',
            'Total lipid (fat)': 'fat_per_100g',
            'Carbohydrate, by difference': 'carbohydrates_per_100g',
            'Fiber, total dietary': 'fiber_per_100g',
            'Vitamin C, total ascorbic acid': 'vitamin_c_per_100g',
            'Calcium, Ca': 'calcium_per_100g',
            'Iron, Fe': 'iron_per_100g',
            'Magnesium, Mg': 'magnesium_per_100g',
            'Phosphorus, P': 'phosphorus_per_100g',
            'Potassium, K': 'potassium_per_100g',
            'Zinc, Zn': 'zinc_per_100g',
            'Sodium, Na': 'sodium_per_100g',
            'Vitamin B-6': 'vitamin_b6_per_100g',
            'Vitamin K (phylloquinone)': 'vitamin_k_per_100g',
            'Thiamin': 'vitamin_b1_per_100g',
            'Riboflavin': 'vitamin_b2_per_100g',
            'Niacin': 'vitamin_b3_per_100g',
            'Folate, total': 'folate_per_100g',
            'Vitamin A, RAE': 'vitamin_a_per_100g',
            'Vitamin B-12': 'vitamin_b12_per_100g',
            'Vitamin E (alpha-tocopherol)': 'vitamin_e_per_100g'
        }
        
        # หน่วยมาตรฐานสำหรับแต่ละสารอาหาร
        self.nutrient_units = {
            'calories_per_100g': 'kcal',
            'protein_per_100g': 'g',
            'fat_per_100g': 'g', 
            'carbohydrates_per_100g': 'g',
            'fiber_per_100g': 'g',
            'vitamin_c_per_100g': 'mg',
            'calcium_per_100g': 'mg',
            'iron_per_100g': 'mg',
            'magnesium_per_100g': 'mg',
            'phosphorus_per_100g': 'mg',
            'potassium_per_100g': 'mg',
            'zinc_per_100g': 'mg',
            'sodium_per_100g': 'mg',
            'vitamin_b6_per_100g': 'mg',
            'vitamin_k_per_100g': 'mcg',
            'vitamin_b1_per_100g': 'mg',
            'vitamin_b2_per_100g': 'mg',
            'vitamin_b3_per_100g': 'mg',
            'folate_per_100g': 'mcg',
            'vitamin_a_per_100g': 'mcg',
            'vitamin_b12_per_100g': 'mcg',
            'vitamin_e_per_100g': 'mg'
        }

    def search_foods(self, query: str, limit: int = 10, 
                    data_types: List[str] = None) -> List[Dict]:
        """
        ค้นหาอาหารจาก USDA API
        
        Args:
            query (str): คำค้นหา
            limit (int): จำนวนผลลัพธ์สูงสุด
            data_types (List[str]): ประเภทข้อมูล (Foundation, SR Legacy, etc.)
            
        Returns:
            List[Dict]: รายการผลลัพธ์การค้นหา
        """
        if data_types is None:
            data_types = ['Foundation', 'SR Legacy']
        
        try:
            url = f"{self.base_url}/foods/search"
            params = {
                'query': query,
                'dataType': data_types,
                'pageSize': min(limit, 200),  # API limit
                'api_key': self.api_key
            }
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('foods', [])
            elif response.status_code == 429:
                print("⚠️  API rate limit exceeded. กำลังรอ...")
                time.sleep(60)  # รอ 1 นาที
                return self.search_foods(query, limit, data_types)  # ลองใหม่
            else:
                print(f"❌ API Error: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            print(f"❌ Error searching foods: {e}")
            return []

    def get_food_details(self, fdc_id: int) -> Optional[Dict]:
        """
        ดึงรายละเอียดอาหารจาก FDC ID
        
        Args:
            fdc_id (int): FDC ID ของอาหาร
            
        Returns:
            Dict: ข้อมูลรายละเอียดอาหาร หรือ None ถ้าไม่พบ
        """
        try:
            url = f"{self.base_url}/food/{fdc_id}"
            params = {
                'api_key': self.api_key,
                'format': 'abridged'
            }
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                print("⚠️  API rate limit exceeded. กำลังรอ...")
                time.sleep(60)
                return self.get_food_details(fdc_id)
            else:
                print(f"❌ API Error for FDC ID {fdc_id}: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error getting food details for FDC ID {fdc_id}: {e}")
            return None

    def extract_nutrition_data(self, food_details: Dict, 
                              thai_name: str) -> Dict:
        """
        แยกข้อมูลโภชนาการจากรายละเอียดอาหาร
        
        Args:
            food_details (Dict): ข้อมูลรายละเอียดจาก USDA
            thai_name (str): ชื่อไทยของวัตถุดิบ
            
        Returns:
            Dict: ข้อมูลโภชนาการที่จัดรูปแบบแล้ว
        """
        # ข้อมูลพื้นฐาน
        nutrition_row = {
            'thai_name': thai_name,
            'english_name': food_details.get('description', ''),
            'usda_description': food_details.get('description', ''),
            'fdc_id': food_details.get('fdcId', ''),
            'data_type': food_details.get('dataType', ''),
            'publication_date': food_details.get('publicationDate', datetime.now().strftime('%Y-%m-%d'))
        }
        
        # เริ่มต้นค่าสารอาหารทั้งหมดเป็น 0
        for nutrient_col in self.nutrient_mapping.values():
            nutrition_row[nutrient_col] = 0.0
            nutrition_row[nutrient_col.replace('_per_100g', '_unit')] = self.nutrient_units.get(nutrient_col, 'g')
        
        # แยกข้อมูลสารอาหาร
        food_nutrients = food_details.get('foodNutrients', [])
        
        for nutrient_info in food_nutrients:
            nutrient_name = nutrient_info.get('nutrient', {}).get('name', '')
            nutrient_amount = nutrient_info.get('amount', 0)
            nutrient_unit = nutrient_info.get('nutrient', {}).get('unitName', '')
            
            # หาคอลัมน์ที่ตรงกับสารอาหารนี้
            if nutrient_name in self.nutrient_mapping:
                column_name = self.nutrient_mapping[nutrient_name]
                
                # แปลงหน่วย (ถ้าจำเป็น)
                converted_amount = self.convert_nutrient_unit(
                    nutrient_amount, nutrient_unit, column_name)
                
                nutrition_row[column_name] = converted_amount
        
        return nutrition_row

    def convert_nutrient_unit(self, amount: float, from_unit: str, 
                            nutrient_column: str) -> float:
        """
        แปลงหน่วยสารอาหารให้เป็นมาตรฐาน
        
        Args:
            amount (float): ปริมาณ
            from_unit (str): หน่วยต้นทาง  
            nutrient_column (str): คอลัมน์สารอาหาร
            
        Returns:
            float: ปริมาณที่แปลงหน่วยแล้ว
        """
        target_unit = self.nutrient_units.get(nutrient_column, 'g')
        
        # แปลงหน่วยพลังงาน
        if 'calories' in nutrient_column:
            if from_unit.upper() == 'KJ':  # Kilojoule to kcal
                return amount * 0.239006
            elif from_unit.upper() in ['KCAL', 'CAL']:
                return amount
        
        # แปลงหน่วยน้ำหนัก
        weight_conversions = {
            ('G', 'MG'): 1000,      # กรัมเป็นมิลลิกรัม
            ('MG', 'G'): 0.001,     # มิลลิกรัมเป็นกรัม
            ('UG', 'MG'): 0.001,    # ไมโครกรัมเป็นมิลลิกรัม
            ('UG', 'MCG'): 1,       # ไมโครกรัม = ไมโครกรัม
            ('MCG', 'UG'): 1        # ไมโครกรัม = ไมโครกรัม
        }
        
        from_unit_upper = from_unit.upper()
        target_unit_upper = target_unit.upper()
        
        if (from_unit_upper, target_unit_upper) in weight_conversions:
            return amount * weight_conversions[(from_unit_upper, target_unit_upper)]
        elif (target_unit_upper, from_unit_upper) in weight_conversions:
            return amount / weight_conversions[(target_unit_upper, from_unit_upper)]
        
        # ถ้าหน่วยเหมือนกันหรือไม่รู้จักการแปลง ให้คืนค่าเดิม
        return amount

    def fetch_ingredients_nutrition(self, ingredients_list: List[str], 
                                   output_file: str = None) -> pd.DataFrame:
        """
        ดึงข้อมูลโภชนาการสำหรับรายการวัตถุดิบ
        
        Args:
            ingredients_list (List[str]): รายการวัตถุดิบ
            output_file (str): ไฟล์สำหรับบันทึกผล
            
        Returns:
            pd.DataFrame: ข้อมูลโภชนาการที่ดึงมา
        """
        nutrition_data = []
        failed_ingredients = []
        
        print(f"🔍 กำลังดึงข้อมูลโภชนาการสำหรับ {len(ingredients_list)} วัตถุดิบ...")
        
        for i, ingredient in enumerate(tqdm(ingredients_list, desc="ดึงข้อมูล")):
            # ค้นหาอาหารที่ตรงกับวัตถุดิบ
            search_results = self.search_foods(ingredient, limit=3)
            
            if search_results:
                # เลือกผลลัพธ์แรกที่ดีที่สุด
                best_match = search_results[0]
                fdc_id = best_match.get('fdcId')
                
                if fdc_id:
                    # ดึงรายละเอียดอาหาร
                    food_details = self.get_food_details(fdc_id)
                    
                    if food_details:
                        # แยกข้อมูลโภชนาการ
                        nutrition_row = self.extract_nutrition_data(
                            food_details, ingredient)
                        nutrition_data.append(nutrition_row)
                        
                        # แสดงความคืบหน้า
                        if (i + 1) % 10 == 0:
                            print(f"✅ ดึงข้อมูลแล้ว {i + 1}/{len(ingredients_list)} รายการ")
                    else:
                        failed_ingredients.append(ingredient)
                else:
                    failed_ingredients.append(ingredient)
            else:
                failed_ingredients.append(ingredient)
            
            # หน่วงเวลาเพื่อไม่ให้เกิน rate limit
            time.sleep(self.rate_limit_delay)
        
        # สร้าง DataFrame
        df = pd.DataFrame(nutrition_data)
        
        # บันทึกไฟล์
        if output_file and not df.empty:
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"💾 บันทึกข้อมูลไปที่: {output_file}")
        
        # แสดงสรุปผล
        print(f"\n📊 สรุปผลการดึงข้อมูล:")
        print(f"   ✅ สำเร็จ: {len(nutrition_data)} วัตถุดิบ")
        print(f"   ❌ ล้มเหลว: {len(failed_ingredients)} วัตถุดิบ")
        
        if failed_ingredients:
            print(f"\n⚠️  วัตถุดิบที่ไม่พบข้อมูล:")
            for ingredient in failed_ingredients[:10]:  # แสดงแค่ 10 รายการแรก
                print(f"   - {ingredient}")
            if len(failed_ingredients) > 10:
                print(f"   ... และอีก {len(failed_ingredients) - 10} รายการ")
        
        return df

    def extract_ingredients_from_recipes(self, recipes_file: str) -> List[str]:
        """
        แยกรายการวัตถุดิบจากไฟล์สูตรอาหาร
        
        Args:
            recipes_file (str): ไฟล์สูตรอาหาร
            
        Returns:
            List[str]: รายการวัตถุดิบที่ไม่ซ้ำ
        """
        try:
            df = pd.read_csv(recipes_file)
            
            if 'text_ingradiant' not in df.columns:
                raise ValueError("ไม่พบคอลัมน์ 'text_ingradiant' ในไฟล์")
            
            all_ingredients = set()
            
            for ingredients_text in df['text_ingradiant'].dropna():
                # แยกแต่ละบรรทัดของวัตถุดิบ
                for line in str(ingredients_text).split('\n'):
                    ingredient = line.strip()
                    if ingredient and not ingredient.startswith('-'):
                        # ทำความสะอาดชื่อวัตถุดิบ
                        clean_ingredient = self.clean_ingredient_name(ingredient)
                        if clean_ingredient:
                            all_ingredients.add(clean_ingredient)
            
            return sorted(list(all_ingredients))
            
        except Exception as e:
            print(f"❌ Error extracting ingredients: {e}")
            return []

    def clean_ingredient_name(self, ingredient: str) -> str:
        """
        ทำความสะอาดชื่อวัตถุดิบสำหรับการค้นหา
        
        Args:
            ingredient (str): ชื่อวัตถุดิบดิบ
            
        Returns:
            str: ชื่อวัตถุดิบที่ทำความสะอาดแล้ว
        """
        # ลบ prefix เช่น "- "
        ingredient = ingredient.lstrip('- •*')
        
        # ลบตัวเลขและหน่วย
        ingredient = re.sub(r'\d+(?:\.\d+)?\s*[^\s]*', '', ingredient)
        
        # ลบคำอธิบายในวงเล็บ
        ingredient = re.sub(r'\([^)]*\)', '', ingredient)
        
        # ลบคำที่ไม่จำเป็น
        unnecessary_words = [
            'สด', 'แห้ง', 'ป่น', 'หั่น', 'สับ', 'ละเอียด', 'หยาบ',
            'ขาว', 'แดง', 'เขียว', 'เหลือง', 'ดำ',
            'เล็ก', 'ใหญ่', 'กลาง', 'อ่อน', 'แก่',
            'ตามชอบ', 'ตามใจ', 'เล็กน้อย', 'พอประมาณ'
        ]
        
        for word in unnecessary_words:
            ingredient = ingredient.replace(word, '')
        
        return ingredient.strip()

    def update_existing_nutrition_data(self, existing_file: str, 
                                     new_ingredients: List[str]) -> pd.DataFrame:
        """
        อัปเดตข้อมูลโภชนาการที่มีอยู่ด้วยวัตถุดิบใหม่
        
        Args:
            existing_file (str): ไฟล์ข้อมูลเดิม
            new_ingredients (List[str]): วัตถุดิบใหม่
            
        Returns:
            pd.DataFrame: ข้อมูลที่อัปเดตแล้ว
        """
        try:
            # โหลดข้อมูลเดิม
            existing_df = pd.read_csv(existing_file)
            existing_ingredients = set(existing_df['thai_name'].tolist())
            
            # หาวัตถุดิบที่ยังไม่มี
            missing_ingredients = [ing for ing in new_ingredients 
                                 if ing not in existing_ingredients]
            
            if not missing_ingredients:
                print("✅ ข้อมูลทั้งหมดมีอยู่แล้ว ไม่ต้องดึงใหม่")
                return existing_df
            
            print(f"🆕 พบวัตถุดิบใหม่ {len(missing_ingredients)} รายการ")
            
            # ดึงข้อมูลสำหรับวัตถุดิบที่หายไป
            new_df = self.fetch_ingredients_nutrition(missing_ingredients)
            
            if not new_df.empty:
                # รวมข้อมูลเดิมและใหม่
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                
                # บันทึกข้อมูลที่อัปเดต
                backup_file = f"{existing_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                existing_df.to_csv(backup_file, index=False, encoding='utf-8-sig')
                print(f"💾 สำรองข้อมูลเดิมไปที่: {backup_file}")
                
                combined_df.to_csv(existing_file, index=False, encoding='utf-8-sig')
                print(f"✅ อัปเดตไฟล์ {existing_file} แล้ว")
                
                return combined_df
            else:
                return existing_df
                
        except Exception as e:
            print(f"❌ Error updating nutrition data: {e}")
            return pd.DataFrame()

def main():
    """ฟังก์ชันหลักสำหรับรันสคริปต์"""
    parser = argparse.ArgumentParser(
        description='🔍 USDA Nutrition Data Fetcher - ดึงข้อมูลโภชนาการจาก USDA API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ตัวอย่างการใช้งาน:
  # ดึงข้อมูลจากไฟล์สูตรอาหาร
  python usda_nutrition_fetcher.py --recipes thai_food_processed_cleaned.csv --output nutrition_data.csv
  
  # ดึงข้อมูลวัตถุดิบเฉพาะ
  python usda_nutrition_fetcher.py --ingredients "กระเทียม,หอมใหญ่,มะเขือเทศ" --output ingredients_nutrition.csv
  
  # อัปเดตข้อมูลที่มีอยู่
  python usda_nutrition_fetcher.py --recipes thai_food_processed_cleaned.csv --update existing_nutrition.csv
  
  # ใช้ API key ของตนเอง
  python usda_nutrition_fetcher.py --api-key YOUR_API_KEY --recipes data.csv --output nutrition.csv
        """
    )
    
    parser.add_argument(
        '--api-key', '-k',
        type=str,
        default='DEMO_KEY',
        help='USDA API Key (default: DEMO_KEY)'
    )
    
    parser.add_argument(
        '--recipes', '-r',
        type=str,
        help='ไฟล์สูตรอาหารสำหรับแยกวัตถุดิบ'
    )
    
    parser.add_argument(
        '--ingredients', '-i',
        type=str,
        help='รายการวัตถุดิบ (คั่นด้วยจุลภาค) เช่น "กระเทียม,หอมใหญ่,พริกไทย"'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=f'usda_nutrition_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
        help='ไฟล์เอาต์พุต (default: usda_nutrition_YYYYMMDD_HHMMSS.csv)'
    )
    
    parser.add_argument(
        '--update', '-u',
        type=str,
        help='อัปเดตไฟล์ข้อมูลที่มีอยู่'
    )
    
    parser.add_argument(
        '--rate-limit',
        type=float,
        default=1.0,
        help='หน่วงเวลาระหว่างการเรียก API (วินาที, default: 1.0)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔍 USDA Nutrition Data Fetcher")
    print("=" * 70)
    print(f"API Key: {'***' + args.api_key[-4:] if len(args.api_key) > 4 else 'DEMO_KEY'}")
    print(f"Rate Limit: {args.rate_limit} วินาที")
    print("-" * 70)
    
    # สร้าง fetcher object
    fetcher = USDANutritionFetcher(args.api_key)
    fetcher.rate_limit_delay = args.rate_limit
    
    # กำหนดรายการวัตถุดิบ
    ingredients_list = []
    
    if args.recipes:
        if not os.path.exists(args.recipes):
            print(f"❌ Error: ไม่พบไฟล์ {args.recipes}")
            return
        
        print(f"📖 กำลังแยกวัตถุดิบจาก: {args.recipes}")
        ingredients_list = fetcher.extract_ingredients_from_recipes(args.recipes)
        print(f"✅ พบวัตถุดิบทั้งหมด: {len(ingredients_list)} รายการ")
        
    elif args.ingredients:
        ingredients_list = [ing.strip() for ing in args.ingredients.split(',') if ing.strip()]
        print(f"📝 วัตถุดิบที่กำหนด: {len(ingredients_list)} รายการ")
        
    else:
        print("❌ Error: กรุณาระบุ --recipes หรือ --ingredients")
        return
    
    if not ingredients_list:
        print("❌ Error: ไม่พบวัตถุดิบที่จะดึงข้อมูล")
        return
    
    # แสดงตัวอย่างวัตถุดิบ
    print(f"\n📋 ตัวอย่างวัตถุดิบ (แสดง 10 รายการแรก):")
    for i, ingredient in enumerate(ingredients_list[:10], 1):
        print(f"   {i:2d}. {ingredient}")
    
    if len(ingredients_list) > 10:
        print(f"   ... และอีก {len(ingredients_list) - 10} รายการ")
    
    # เริ่มดึงข้อมูล
    if args.update:
        # โหมดอัปเดต
        print(f"\n🔄 อัปเดตไฟล์: {args.update}")
        result_df = fetcher.update_existing_nutrition_data(args.update, ingredients_list)
    else:
        # โหมดดึงข้อมูลใหม่
        print(f"\n🚀 เริ่มดึงข้อมูลโภชนาการ...")
        result_df = fetcher.fetch_ingredients_nutrition(ingredients_list, args.output)
    
    if not result_df.empty:
        print(f"\n✅ เสร็จสิ้น! ได้ข้อมูลโภชนาการทั้งหมด {len(result_df)} รายการ")
        print(f"📁 ไฟล์เอาต์พุต: {args.output}")
        
        # สรุปข้อมูล
        print(f"\n📊 สรุปข้อมูลที่ได้:")
        print(f"   - จำนวนคอลัมน์: {len(result_df.columns)}")
        print(f"   - ข้อมูล USDA ประเภท: {', '.join(result_df['data_type'].unique())}")
        
    else:
        print("\n❌ ไม่สามารถดึงข้อมูลได้")
    
    print("\n🎉 สำเร็จ! ข้อมูลพร้อมใช้งานกับแอปพลิเคชัน Thai Food Nutrition Analyzer")

if __name__ == "__main__":
    # Import ที่จำเป็นเพิ่มเติม
    import re
    
    main()
