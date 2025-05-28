#!/usr/bin/env python3
"""
Batch Nutrition Processor
ประมวลผลข้อมูลโภชนาการสำหรับทุกสูตรอาหารในฐานข้อมูล
"""

import pandas as pd
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
import logging
from nutrition_analyzer import NutritionAnalyzer

# ตั้งค่า logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nutrition_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BatchNutritionProcessor:
    """คลาสสำหรับประมวลผลข้อมูลโภชนาการแบบ batch"""
    
    def __init__(self, csv_file: str = "thai_food_processed.csv", batch_size: int = 5):
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.analyzer = NutritionAnalyzer()
        self.results = []
        
    def load_existing_results(self, output_file: str = "nutrition_results.json") -> dict:
        """โหลดผลลัพธ์ที่ประมวลผลไว้แล้ว"""
        if Path(output_file).exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.info(f"Loaded {len(data)} existing nutrition results")
                    return {item['recipe_name']: item for item in data}
            except Exception as e:
                logger.error(f"Error loading existing results: {e}")
        return {}
    
    def save_results(self, output_file: str = "nutrition_results.json"):
        """บันทึกผลลัพธ์"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(self.results)} nutrition results to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def analyze_recipe(self, recipe_name: str, ingredients: str) -> dict:
        """วิเคราะห์โภชนาการสำหรับสูตรอาหาร"""
        # วิเคราะห์โภชนาการ
        nutrition_data = self.analyzer.analyze_ingredients(ingredients)
        total_nutrition = self.analyzer.calculate_total_nutrition(nutrition_data)
        
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
                    'vitamin_a': total_nutrition.vitamin_a,
                    'vitamin_c': total_nutrition.vitamin_c,
                    'vitamin_d': total_nutrition.vitamin_d,
                    'vitamin_e': total_nutrition.vitamin_e,
                    'vitamin_k': total_nutrition.vitamin_k,
                    'vitamin_b1': total_nutrition.vitamin_b1,
                    'vitamin_b2': total_nutrition.vitamin_b2,
                    'vitamin_b6': total_nutrition.vitamin_b6,
                    'vitamin_b12': total_nutrition.vitamin_b12,
                },
                'minerals': {
                    'calcium': total_nutrition.calcium,
                    'iron': total_nutrition.iron,
                    'magnesium': total_nutrition.magnesium,
                    'phosphorus': total_nutrition.phosphorus,
                    'potassium': total_nutrition.potassium,
                    'zinc': total_nutrition.zinc,
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
    
    def process_recipes_incrementally(self, force_reprocess: bool = False):
        """ประมวลผลสูตรอาหารแบบ incremental (ไม่ประมวลผลซ้ำ)"""
        try:
            # โหลดข้อมูลสูตรอาหาร
            df = pd.read_csv(self.csv_file)
            logger.info(f"Loaded {len(df)} recipes from {self.csv_file}")
            
            # โหลดผลลัพธ์ที่มีอยู่แล้ว
            existing_results = self.load_existing_results() if not force_reprocess else {}
            
            # รายการสูตรที่ต้องประมวลผล
            recipes_to_process = []
            for index, row in df.iterrows():
                recipe_name = row['name']
                if recipe_name not in existing_results:
                    recipes_to_process.append((index, row))
            
            logger.info(f"Found {len(recipes_to_process)} new recipes to process")
            
            if not recipes_to_process:
                logger.info("All recipes already processed!")
                self.results = list(existing_results.values())
                return
            
            # เริ่มจากผลลัพธ์เก่า
            self.results = list(existing_results.values())
            
            # ประมวลผลเป็น batch
            total_batches = (len(recipes_to_process) + self.batch_size - 1) // self.batch_size
            
            for batch_num in range(0, len(recipes_to_process), self.batch_size):
                batch_recipes = recipes_to_process[batch_num:batch_num + self.batch_size]
                current_batch = (batch_num // self.batch_size) + 1
                
                logger.info(f"Processing batch {current_batch}/{total_batches} ({len(batch_recipes)} recipes)")
                
                for index, row in batch_recipes:
                    recipe_name = row['name']
                    ingredients = row['ingredient']
                    
                    try:
                        logger.info(f"  Processing: {recipe_name}")
                        result = self.analyze_recipe(recipe_name, ingredients)
                        
                        # แปลง NutritionInfo objects เป็น dict เพื่อบันทึก JSON
                        serializable_result = self._make_serializable(result)
                        self.results.append(serializable_result)
                        
                        logger.info(f"  ✓ Completed: {recipe_name}")
                        
                    except Exception as e:
                        logger.error(f"  ✗ Error processing {recipe_name}: {e}")
                        # เพิ่มผลลัพธ์ว่างเพื่อไม่ให้ประมวลผลซ้ำ
                        empty_result = {
                            'recipe_name': recipe_name,
                            'total_nutrition': {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0, 'fiber': 0, 'vitamins': {}, 'minerals': {}},
                            'ingredients': [],
                            'ingredient_count': 0,
                            'error': str(e),
                            'processed_at': datetime.now().isoformat()
                        }
                        self.results.append(empty_result)
                
                # บันทึกผลลัพธ์หลังแต่ละ batch
                self.save_results()
                
                # หน่วงเวลาระหว่าง batch เพื่อป้องกัน API rate limit
                if current_batch < total_batches:
                    logger.info(f"Waiting 5 seconds before next batch...")
                    time.sleep(5)
            
            logger.info(f"✓ Completed processing all recipes! Total: {len(self.results)}")
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            # บันทึกผลลัพธ์ที่มีอยู่
            if self.results:
                self.save_results()
    
    def _make_serializable(self, result: dict) -> dict:
        """แปลง NutritionInfo objects เป็น dict ที่สามารถแปลงเป็น JSON ได้"""
        serializable = {
            'recipe_name': result['recipe_name'],
            'total_nutrition': result['total_nutrition'],
            'ingredient_count': result['ingredient_count'],
            'ingredients': [],
            'processed_at': datetime.now().isoformat()
        }
        
        # แปลง ingredient details
        for ingredient_info in result['ingredients']:
            nutrition = ingredient_info['nutrition']
            ingredient_data = {
                'ingredient': ingredient_info['ingredient'],
                'nutrition': {
                    'name': nutrition.name,
                    'calories': nutrition.calories,
                    'protein': nutrition.protein,
                    'carbs': nutrition.carbs,
                    'fat': nutrition.fat,
                    'fiber': nutrition.fiber,
                    'vitamins': {
                        'vitamin_a': nutrition.vitamin_a,
                        'vitamin_c': nutrition.vitamin_c,
                        'vitamin_d': nutrition.vitamin_d,
                        'vitamin_e': nutrition.vitamin_e,
                        'vitamin_k': nutrition.vitamin_k,
                        'vitamin_b1': nutrition.vitamin_b1,
                        'vitamin_b2': nutrition.vitamin_b2,
                        'vitamin_b6': nutrition.vitamin_b6,
                        'vitamin_b12': nutrition.vitamin_b12,
                    },
                    'minerals': {
                        'calcium': nutrition.calcium,
                        'iron': nutrition.iron,
                        'magnesium': nutrition.magnesium,
                        'phosphorus': nutrition.phosphorus,
                        'potassium': nutrition.potassium,
                        'zinc': nutrition.zinc,
                    }
                }
            }
            serializable['ingredients'].append(ingredient_data)
        
        return serializable
    
    def generate_nutrition_summary(self):
        """สร้างสรุปข้อมูลโภชนาการ"""
        if not self.results:
            logger.warning("No results to summarize")
            return
        
        # คำนวณสถิติ
        total_recipes = len([r for r in self.results if 'error' not in r])
        error_recipes = len([r for r in self.results if 'error' in r])
        
        calories_list = [r['total_nutrition']['calories'] for r in self.results if 'error' not in r]
        protein_list = [r['total_nutrition']['protein'] for r in self.results if 'error' not in r]
        
        summary = {
            'total_recipes_processed': len(self.results),
            'successful_recipes': total_recipes,
            'failed_recipes': error_recipes,
            'nutrition_stats': {
                'avg_calories': sum(calories_list) / len(calories_list) if calories_list else 0,
                'max_calories': max(calories_list) if calories_list else 0,
                'min_calories': min(calories_list) if calories_list else 0,
                'avg_protein': sum(protein_list) / len(protein_list) if protein_list else 0,
                'max_protein': max(protein_list) if protein_list else 0,
                'min_protein': min(protein_list) if protein_list else 0,
            },
            'top_calorie_recipes': sorted(
                [r for r in self.results if 'error' not in r], 
                key=lambda x: x['total_nutrition']['calories'], 
                reverse=True
            )[:10],
            'top_protein_recipes': sorted(
                [r for r in self.results if 'error' not in r], 
                key=lambda x: x['total_nutrition']['protein'], 
                reverse=True
            )[:10],
            'generated_at': datetime.now().isoformat()
        }
        
        # บันทึกสรุป
        with open('nutrition_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info("✓ Generated nutrition summary")
        
        # แสดงสรุปบนหน้าจอ
        print("\n" + "="*50)
        print("NUTRITION PROCESSING SUMMARY")
        print("="*50)
        print(f"Total recipes processed: {len(self.results)}")
        print(f"Successful: {total_recipes}")
        print(f"Failed: {error_recipes}")
        print(f"Average calories per recipe: {summary['nutrition_stats']['avg_calories']:.1f}")
        print(f"Average protein per recipe: {summary['nutrition_stats']['avg_protein']:.1f}g")
        print("\nTop 5 highest calorie recipes:")
        for i, recipe in enumerate(summary['top_calorie_recipes'][:5], 1):
            print(f"  {i}. {recipe['recipe_name']}: {recipe['total_nutrition']['calories']:.0f} kcal")
        print("="*50)
    
    def create_enhanced_dataset(self, output_file: str = "thai_food_with_nutrition.csv"):
        """สร้างไฟล์ CSV ใหม่ที่มีข้อมูลโภชนาการ"""
        try:
            # โหลดข้อมูลเดิม
            df = pd.read_csv(self.csv_file)
            
            # เพิ่มคอลัมน์โภชนาการ
            nutrition_columns = {
                'calories': [],
                'protein': [],
                'carbs': [],
                'fat': [],
                'fiber': [],
                'vitamins_json': [],
                'minerals_json': [],
                'ingredient_count': []
            }
            
            # สร้าง mapping จากผลลัพธ์
            nutrition_map = {r['recipe_name']: r for r in self.results if 'error' not in r}
            
            for _, row in df.iterrows():
                recipe_name = row['name']
                
                if recipe_name in nutrition_map:
                    nutrition = nutrition_map[recipe_name]['total_nutrition']
                    nutrition_columns['calories'].append(nutrition['calories'])
                    nutrition_columns['protein'].append(nutrition['protein'])
                    nutrition_columns['carbs'].append(nutrition['carbs'])
                    nutrition_columns['fat'].append(nutrition['fat'])
                    nutrition_columns['fiber'].append(nutrition['fiber'])
                    nutrition_columns['vitamins_json'].append(json.dumps(nutrition['vitamins'], ensure_ascii=False))
                    nutrition_columns['minerals_json'].append(json.dumps(nutrition['minerals'], ensure_ascii=False))
                    nutrition_columns['ingredient_count'].append(nutrition_map[recipe_name]['ingredient_count'])
                else:
                    # ค่าเริ่มต้นสำหรับสูตรที่ไม่มีข้อมูล
                    nutrition_columns['calories'].append(0)
                    nutrition_columns['protein'].append(0)
                    nutrition_columns['carbs'].append(0)
                    nutrition_columns['fat'].append(0)
                    nutrition_columns['fiber'].append(0)
                    nutrition_columns['vitamins_json'].append('{}')
                    nutrition_columns['minerals_json'].append('{}')
                    nutrition_columns['ingredient_count'].append(0)
            
            # เพิ่มคอลัมน์ใหม่เข้าไปใน DataFrame
            for col, values in nutrition_columns.items():
                df[col] = values
            
            # บันทึกไฟล์ใหม่
            df.to_csv(output_file, index=False, encoding='utf-8')
            logger.info(f"✓ Created enhanced dataset: {output_file}")
            
        except Exception as e:
            logger.error(f"Error creating enhanced dataset: {e}")


def main():
    parser = argparse.ArgumentParser(description='Process nutrition data for Thai recipes')
    parser.add_argument('--input', type=str, default='thai_food_processed.csv', 
                        help='Input CSV file path')
    parser.add_argument('--batch-size', type=int, default=5, 
                        help='Number of recipes to process in each batch')
    parser.add_argument('--force', action='store_true', 
                        help='Force reprocess all recipes (ignore existing results)')
    parser.add_argument('--summary-only', action='store_true', 
                        help='Generate summary only (skip processing)')
    
    args = parser.parse_args()
    
    processor = BatchNutritionProcessor(args.input, args.batch_size)
    
    if not args.summary_only:
        # ประมวลผลข้อมูลโภชนาการ
        processor.process_recipes_incrementally(force_reprocess=args.force)
    else:
        # โหลดผลลัพธ์ที่มีอยู่
        existing_results = processor.load_existing_results()
        processor.results = list(existing_results.values())
    
    # สร้างสรุปและไฟล์ข้อมูลใหม่
    processor.generate_nutrition_summary()
    processor.create_enhanced_dataset()


if __name__ == "__main__":
    main()
