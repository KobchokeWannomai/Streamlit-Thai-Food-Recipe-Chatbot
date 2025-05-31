#!/usr/bin/env python3
"""
Batch Nutrition Processor - Enhanced Version
ประมวลผลข้อมูลโภชนาการสำหรับทุกสูตรอาหารในฐานข้อมูล
รองรับเมนูอาหารไทยทั้งหมดและการค้นหาที่แม่นยำ
"""

import pandas as pd
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
import logging
from nutrition_analyzer import NutritionAnalyzer
from config import Config
import threading
import queue
import concurrent.futures
from typing import List, Dict, Optional
import pickle

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
    """คลาสสำหรับประมวลผลข้อมูลโภชนาการแบบ batch - ปรับปรุงแล้ว"""
    
    def __init__(self, csv_file: str = "thai_food_processed.csv", batch_size: int = 5, use_threading: bool = False):
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.use_threading = use_threading
        self.analyzer = NutritionAnalyzer()
        self.results = []
        self.processed_count = 0
        self.error_count = 0
        self.start_time = None
        
        # สำหรับ threading
        self.result_queue = queue.Queue()
        self.max_workers = min(4, batch_size)  # จำกัดจำนวน workers
        
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
    
    def save_checkpoint(self, output_file: str = "nutrition_results.json"):
        """บันทึก checkpoint ระหว่างการประมวลผล"""
        checkpoint_file = output_file.replace('.json', '_checkpoint.json')
        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
            logger.info(f"Checkpoint saved: {len(self.results)} results")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
    
    def analyze_recipe_threaded(self, recipe_data: tuple) -> dict:
        """วิเคราะห์โภชนาการสำหรับสูตรอาหารแบบ threaded"""
        index, row = recipe_data
        recipe_name = row['name']
        ingredients = row['ingredient']
        
        try:
            logger.debug(f"Processing: {recipe_name}")
            result = self.analyze_recipe(recipe_name, ingredients)
            
            # แปลง NutritionInfo objects เป็น dict เพื่อบันทึก JSON
            serializable_result = self._make_serializable(result)
            
            logger.debug(f"Completed: {recipe_name}")
            return serializable_result
            
        except Exception as e:
            logger.error(f"Error processing {recipe_name}: {e}")
            # เพิ่มผลลัพธ์ว่างเพื่อไม่ให้ประมวลผลซ้ำ
            return {
                'recipe_name': recipe_name,
                'total_nutrition': {
                    'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0, 'fiber': 0, 
                    'vitamins': {}, 'minerals': {}
                },
                'ingredients': [],
                'ingredient_count': 0,
                'error': str(e),
                'processed_at': datetime.now().isoformat()
            }
    
    def analyze_recipe(self, recipe_name: str, ingredients: str) -> dict:
        """วิเคราะห์โภชนาการสำหรับสูตรอาหาร - ปรับปรุงแล้ว"""
        # ตรวจสอบว่าเป็นเมนูที่ต้องการการปรับแต่งพิเศษหรือไม่
        apply_cooking_adjustments = self._should_apply_cooking_adjustments(recipe_name)
        
        # วิเคราะห์โภชนาการ
        nutrition_data = self.analyzer.analyze_ingredients(
            ingredients, recipe_name, apply_cooking_adjustments
        )
        total_nutrition = self.analyzer.calculate_total_nutrition(nutrition_data)
        
        # จัดหมวดหมู่เมนู
        category = self._categorize_recipe(recipe_name)
        
        # ประเมินความยากในการทำ
        difficulty = self._estimate_difficulty(ingredients, recipe_name)
        
        # สร้างผลลัพธ์ในรูปแบบ dict
        result = {
            'recipe_name': recipe_name,
            'category': category,
            'difficulty': difficulty,
            'cooking_adjustments_applied': apply_cooking_adjustments,
            'total_nutrition': {
                'calories': total_nutrition.calories,
                'protein': total_nutrition.protein,
                'carbs': total_nutrition.carbs,
                'fat': total_nutrition.fat,
                'fiber': total_nutrition.fiber,
                'sugar': total_nutrition.sugar,
                'sodium': total_nutrition.sodium,
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
                    'folate': total_nutrition.folate,
                    'niacin': total_nutrition.niacin,
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
    
    def _should_apply_cooking_adjustments(self, recipe_name: str) -> bool:
        """ตรวจสอบว่าควรใช้การปรับแต่งการทำอาหารหรือไม่"""
        cooking_methods = ['ทอด', 'ผัด', 'ย่าง', 'ปิ้ง', 'แกง', 'ต้ม']
        recipe_lower = recipe_name.lower()
        
        # ตรวจสอบวิธีการทำอาหาร
        has_cooking_method = any(method in recipe_lower for method in cooking_methods)
        
        # ตรวจสอบเมนูเฉพาะที่ต้องการการปรับแต่ง
        special_adjustments = [
            'ไข่เจียว', 'ไข่ดาว', 'กุ้งทอด', 'ปลาทอด', 'กล้วยทอด', 'ฟักทองทอด',
            'หมูทอด', 'ไข่เค็มทอด', 'เนื้อเครื่องเทศทอด'
        ]
        
        needs_special_adjustment = any(menu in recipe_lower for menu in special_adjustments)
        
        return has_cooking_method or needs_special_adjustment
    
    def _categorize_recipe(self, recipe_name: str) -> str:
        """จัดหมวดหมู่เมนูอาหาร"""
        name_lower = recipe_name.lower()
        
        # หมวดหมู่ตามวิธีการทำ
        if any(word in name_lower for word in ['ผัด']):
            return 'อาหารผัด'
        elif any(word in name_lower for word in ['แกง']):
            return 'อาหารแกง'
        elif any(word in name_lower for word in ['ต้ม', 'ซุป']):
            return 'อาหารต้ม'
        elif any(word in name_lower for word in ['ยำ', 'ส้มตำ']):
            return 'อาหารยำ'
        elif any(word in name_lower for word in ['ทอด']):
            return 'อาหารทอด'
        elif any(word in name_lower for word in ['ไข่']):
            return 'เมนูไข่'
        elif any(word in name_lower for word in ['ขนม', 'หวาน', 'เชื่อม', 'สังขยา']):
            return 'ของหวาน'
        elif any(word in name_lower for word in ['น้ำพริก']):
            return 'น้ำพริก'
        elif any(word in name_lower for word in ['ไส้กรอก', 'กงเชียง']):
            return 'ไส้กรอก'
        elif any(word in name_lower for word in ['บะหมี่', 'ก๋วยเตี๋ยว', 'หมี่']):
            return 'เส้น'
        else:
            return 'อื่นๆ'
    
    def _estimate_difficulty(self, ingredients: str, recipe_name: str) -> str:
        """ประเมินความยากในการทำอาหาร"""
        # นับจำนวนวัตถุดิบ
        ingredient_count = len([line for line in ingredients.split('\n') if line.strip().startswith('-')])
        
        # ตรวจสอบความซับซ้อนจากชื่อเมนู
        complex_dishes = [
            'ห่อหมก', 'บรรจุไส้', 'ทรงเครื่อง', 'เครื่องเทศ', 'พุดชาจีน',
            'มักกะโรนี', 'ขนมกลีบ', 'เปียกปูน', 'ฉี่ฉู่', 'สาลี่โคโก้'
        ]
        
        name_lower = recipe_name.lower()
        is_complex = any(complex_word in name_lower for complex_word in complex_dishes)
        
        # กำหนดระดับความยาก
        if ingredient_count <= 4 and not is_complex:
            return 'ง่าย'
        elif ingredient_count > 8 or is_complex:
            return 'ยาก'
        else:
            return 'ปานกลาง'
    
    def process_recipes_incrementally(self, force_reprocess: bool = False, use_threading: bool = None):
        """ประมวลผลสูตรอาหารแบบ incremental - ปรับปรุงแล้ว"""
        if use_threading is None:
            use_threading = self.use_threading
            
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
            self.start_time = time.time()
            
            # ประมวลผลเป็น batch
            total_batches = (len(recipes_to_process) + self.batch_size - 1) // self.batch_size
            
            for batch_num in range(0, len(recipes_to_process), self.batch_size):
                batch_recipes = recipes_to_process[batch_num:batch_num + self.batch_size]
                current_batch = (batch_num // self.batch_size) + 1
                
                logger.info(f"Processing batch {current_batch}/{total_batches} ({len(batch_recipes)} recipes)")
                
                if use_threading and len(batch_recipes) > 1:
                    self._process_batch_threaded(batch_recipes)
                else:
                    self._process_batch_sequential(batch_recipes)
                
                # บันทึกผลลัพธ์หลังแต่ละ batch
                self.save_results()
                
                # บันทึก checkpoint ทุก 5 batches
                if current_batch % 5 == 0:
                    self.save_checkpoint()
                
                # แสดงสถิติ
                self._print_progress_stats(current_batch, total_batches)
                
                # หน่วงเวลาระหว่าง batch เพื่อป้องกัน API rate limit
                if current_batch < total_batches:
                    logger.info(f"Waiting 3 seconds before next batch...")
                    time.sleep(3)
            
            # สรุปผลการประมวลผล
            self._print_final_summary()
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            # บันทึกผลลัพธ์ที่มีอยู่
            if self.results:
                self.save_results()
                self.save_checkpoint()
    
    def _process_batch_sequential(self, batch_recipes: List[tuple]):
        """ประมวลผลแบบ sequential"""
        for index, row in batch_recipes:
            recipe_name = row['name']
            ingredients = row['ingredient']
            
            try:
                logger.info(f"  Processing: {recipe_name}")
                result = self.analyze_recipe(recipe_name, ingredients)
                
                # แปลง NutritionInfo objects เป็น dict เพื่อบันทึก JSON
                serializable_result = self._make_serializable(result)
                self.results.append(serializable_result)
                self.processed_count += 1
                
                logger.info(f"  ✓ Completed: {recipe_name}")
                
            except Exception as e:
                logger.error(f"  ✗ Error processing {recipe_name}: {e}")
                self.error_count += 1
                # เพิ่มผลลัพธ์ว่างเพื่อไม่ให้ประมวลผลซ้ำ
                empty_result = self._create_empty_result(recipe_name, str(e))
                self.results.append(empty_result)
    
    def _process_batch_threaded(self, batch_recipes: List[tuple]):
        """ประมวลผลแบบ threaded"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_recipe = {
                executor.submit(self.analyze_recipe_threaded, recipe_data): recipe_data[1]['name']
                for recipe_data in batch_recipes
            }
            
            for future in concurrent.futures.as_completed(future_to_recipe):
                recipe_name = future_to_recipe[future]
                try:
                    result = future.result()
                    self.results.append(result)
                    if 'error' in result:
                        self.error_count += 1
                        logger.error(f"  ✗ Error in threaded processing: {recipe_name}")
                    else:
                        self.processed_count += 1
                        logger.info(f"  ✓ Completed (threaded): {recipe_name}")
                        
                except Exception as e:
                    logger.error(f"  ✗ Future error for {recipe_name}: {e}")
                    self.error_count += 1
                    empty_result = self._create_empty_result(recipe_name, str(e))
                    self.results.append(empty_result)
    
    def _create_empty_result(self, recipe_name: str, error_msg: str) -> dict:
        """สร้างผลลัพธ์ว่างสำหรับสูตรที่ประมวลผลไม่สำเร็จ"""
        return {
            'recipe_name': recipe_name,
            'category': 'ไม่ทราบ',
            'difficulty': 'ไม่ทราบ',
            'cooking_adjustments_applied': False,
            'total_nutrition': {
                'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0, 'fiber': 0,
                'sugar': 0, 'sodium': 0, 'vitamins': {}, 'minerals': {}
            },
            'ingredients': [],
            'ingredient_count': 0,
            'error': error_msg,
            'processed_at': datetime.now().isoformat()
        }
    
    def _print_progress_stats(self, current_batch: int, total_batches: int):
        """แสดงสถิติความคืบหน้า"""
        if self.start_time:
            elapsed_time = time.time() - self.start_time
            avg_time_per_batch = elapsed_time / current_batch
            estimated_remaining = avg_time_per_batch * (total_batches - current_batch)
            
            logger.info(f"Progress: {current_batch}/{total_batches} batches "
                       f"({current_batch/total_batches*100:.1f}%)")
            logger.info(f"Processed: {self.processed_count}, Errors: {self.error_count}")
            logger.info(f"Elapsed: {elapsed_time/60:.1f}m, "
                       f"Estimated remaining: {estimated_remaining/60:.1f}m")
    
    def _print_final_summary(self):
        """แสดงสรุปผลการประมวลผลสุดท้าย"""
        if self.start_time:
            total_time = time.time() - self.start_time
            logger.info(f"✓ Completed processing all recipes!")
            logger.info(f"Total time: {total_time/60:.1f} minutes")
            logger.info(f"Total recipes: {len(self.results)}")
            logger.info(f"Successful: {self.processed_count}")
            logger.info(f"Errors: {self.error_count}")
            if self.processed_count > 0:
                logger.info(f"Average time per recipe: {total_time/self.processed_count:.2f} seconds")
    
    def _make_serializable(self, result: dict) -> dict:
        """แปลง NutritionInfo objects เป็น dict ที่สามารถแปลงเป็น JSON ได้ - ปรับปรุงแล้ว"""
        serializable = {
            'recipe_name': result['recipe_name'],
            'category': result.get('category', 'อื่นๆ'),
            'difficulty': result.get('difficulty', 'ปานกลาง'),
            'cooking_adjustments_applied': result.get('cooking_adjustments_applied', False),
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
                    'sugar': nutrition.sugar,
                    'sodium': nutrition.sodium,
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
                        'folate': nutrition.folate,
                        'niacin': nutrition.niacin,
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
        """สร้างสรุปข้อมูลโภชนาการ - ปรับปรุงแล้ว"""
        if not self.results:
            logger.warning("No results to summarize")
            return
        
        # แยกสูตรที่สำเร็จและไม่สำเร็จ
        successful_results = [r for r in self.results if 'error' not in r]
        error_results = [r for r in self.results if 'error' in r]
        
        # คำนวณสถิติ
        total_recipes = len(self.results)
        successful_count = len(successful_results)
        error_count = len(error_results)
        
        if successful_results:
            calories_list = [r['total_nutrition']['calories'] for r in successful_results]
            protein_list = [r['total_nutrition']['protein'] for r in successful_results]
            
            # สถิติพื้นฐาน
            summary = {
                'processing_info': {
                    'total_recipes_processed': total_recipes,
                    'successful_recipes': successful_count,
                    'failed_recipes': error_count,
                    'success_rate': (successful_count / total_recipes * 100) if total_recipes > 0 else 0,
                    'processing_date': datetime.now().isoformat()
                },
                'nutrition_stats': {
                    'avg_calories': sum(calories_list) / len(calories_list) if calories_list else 0,
                    'max_calories': max(calories_list) if calories_list else 0,
                    'min_calories': min(calories_list) if calories_list else 0,
                    'avg_protein': sum(protein_list) / len(protein_list) if protein_list else 0,
                    'max_protein': max(protein_list) if protein_list else 0,
                    'min_protein': min(protein_list) if protein_list else 0,
                },
                'category_distribution': {},
                'difficulty_distribution': {},
                'cooking_adjustments_stats': {},
                'top_recipes': {
                    'highest_calorie': sorted(successful_results, 
                                            key=lambda x: x['total_nutrition']['calories'], 
                                            reverse=True)[:10],
                    'highest_protein': sorted(successful_results, 
                                            key=lambda x: x['total_nutrition']['protein'], 
                                            reverse=True)[:10],
                    'lowest_calorie': sorted(successful_results, 
                                           key=lambda x: x['total_nutrition']['calories'])[:10],
                    'balanced_nutrition': []
                },
                'generated_at': datetime.now().isoformat()
            }
            
            # วิเคราะห์การกระจายตามหมวดหมู่
            categories = {}
            difficulties = {}
            cooking_adjustments = {'applied': 0, 'not_applied': 0}
            
            for result in successful_results:
                # หมวดหมู่
                category = result.get('category', 'อื่นๆ')
                categories[category] = categories.get(category, 0) + 1
                
                # ความยาก
                difficulty = result.get('difficulty', 'ปานกลาง')
                difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
                
                # การปรับแต่งการทำอาหาร
                if result.get('cooking_adjustments_applied', False):
                    cooking_adjustments['applied'] += 1
                else:
                    cooking_adjustments['not_applied'] += 1
            
            summary['category_distribution'] = categories
            summary['difficulty_distribution'] = difficulties
            summary['cooking_adjustments_stats'] = cooking_adjustments
            
            # หาเมนูที่มีโภชนาการสมดุล
            balanced_recipes = []
            for result in successful_results:
                nutrition = result['total_nutrition']
                calories = nutrition['calories']
                protein = nutrition['protein']
                
                # เกณฑ์โภชนาการสมดุล
                if (200 <= calories <= 400 and protein >= 15):
                    balanced_recipes.append(result)
            
            summary['top_recipes']['balanced_nutrition'] = sorted(
                balanced_recipes, 
                key=lambda x: x['total_nutrition']['protein'], 
                reverse=True
            )[:10]
        
        else:
            summary = {
                'processing_info': {
                    'total_recipes_processed': total_recipes,
                    'successful_recipes': 0,
                    'failed_recipes': error_count,
                    'success_rate': 0,
                    'processing_date': datetime.now().isoformat()
                },
                'message': 'No successful results to analyze',
                'generated_at': datetime.now().isoformat()
            }
        
        # บันทึกสรุป
        summary_file = 'nutrition_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info("✓ Generated nutrition summary")
        
        # แสดงสรุปบนหน้าจอ
        print("\n" + "="*60)
        print("NUTRITION PROCESSING SUMMARY")
        print("="*60)
        print(f"Total recipes processed: {total_recipes}")
        print(f"Successful: {successful_count}")
        print(f"Failed: {error_count}")
        if successful_count > 0:
            print(f"Success rate: {successful_count/total_recipes*100:.1f}%")
            print(f"Average calories per recipe: {summary['nutrition_stats']['avg_calories']:.1f}")
            print(f"Average protein per recipe: {summary['nutrition_stats']['avg_protein']:.1f}g")
            
            print(f"\nCategory distribution:")
            for category, count in summary['category_distribution'].items():
                print(f"  {category}: {count} recipes")
            
            print(f"\nDifficulty distribution:")
            for difficulty, count in summary['difficulty_distribution'].items():
                print(f"  {difficulty}: {count} recipes")
            
            print(f"\nCooking adjustments:")
            cooking_stats = summary['cooking_adjustments_stats']
            print(f"  Applied: {cooking_stats['applied']} recipes")
            print(f"  Not applied: {cooking_stats['not_applied']} recipes")
            
            print(f"\nTop 5 highest calorie recipes:")
            for i, recipe in enumerate(summary['top_recipes']['highest_calorie'][:5], 1):
                print(f"  {i}. {recipe['recipe_name']}: {recipe['total_nutrition']['calories']:.0f} kcal")
        print("="*60)
    
    def create_enhanced_dataset(self, output_file: str = "thai_food_with_nutrition.csv"):
        """สร้างไฟล์ CSV ใหม่ที่มีข้อมูลโภชนาการ - ปรับปรุงแล้ว"""
        try:
            # โหลดข้อมูลเดิม
            df = pd.read_csv(self.csv_file)
            
            # เพิ่มคอลัมน์โภชนาการ - เพิ่มคอลัมน์ใหม่
            nutrition_columns = {
                'calories': [], 'protein': [], 'carbs': [], 'fat': [], 'fiber': [],
                'sugar': [], 'sodium': [], 'vitamin_a': [], 'vitamin_c': [],
                'vitamin_d': [], 'vitamin_e': [], 'vitamin_k': [], 'vitamin_b1': [],
                'vitamin_b2': [], 'vitamin_b6': [], 'vitamin_b12': [], 'folate': [],
                'niacin': [], 'calcium': [], 'iron': [], 'magnesium': [],
                'phosphorus': [], 'potassium': [], 'zinc': [], 'ingredient_count': [],
                'category': [], 'difficulty': [], 'cooking_adjustments_applied': [],
                'vitamins_json': [], 'minerals_json': []
            }
            
            # สร้าง mapping จากผลลัพธ์
            nutrition_map = {r['recipe_name']: r for r in self.results if 'error' not in r}
            
            for _, row in df.iterrows():
                recipe_name = row['name']
                
                if recipe_name in nutrition_map:
                    nutrition_data = nutrition_map[recipe_name]
                    total_nutrition = nutrition_data['total_nutrition']
                    
                    # ข้อมูลโภชนาการหลัก
                    nutrition_columns['calories'].append(total_nutrition['calories'])
                    nutrition_columns['protein'].append(total_nutrition['protein'])
                    nutrition_columns['carbs'].append(total_nutrition['carbs'])
                    nutrition_columns['fat'].append(total_nutrition['fat'])
                    nutrition_columns['fiber'].append(total_nutrition['fiber'])
                    nutrition_columns['sugar'].append(total_nutrition['sugar'])
                    nutrition_columns['sodium'].append(total_nutrition['sodium'])
                    
                    # วิตามิน
                    vitamins = total_nutrition['vitamins']
                    nutrition_columns['vitamin_a'].append(vitamins.get('vitamin_a', 0))
                    nutrition_columns['vitamin_c'].append(vitamins.get('vitamin_c', 0))
                    nutrition_columns['vitamin_d'].append(vitamins.get('vitamin_d', 0))
                    nutrition_columns['vitamin_e'].append(vitamins.get('vitamin_e', 0))
                    nutrition_columns['vitamin_k'].append(vitamins.get('vitamin_k', 0))
                    nutrition_columns['vitamin_b1'].append(vitamins.get('vitamin_b1', 0))
                    nutrition_columns['vitamin_b2'].append(vitamins.get('vitamin_b2', 0))
                    nutrition_columns['vitamin_b6'].append(vitamins.get('vitamin_b6', 0))
                    nutrition_columns['vitamin_b12'].append(vitamins.get('vitamin_b12', 0))
                    nutrition_columns['folate'].append(vitamins.get('folate', 0))
                    nutrition_columns['niacin'].append(vitamins.get('niacin', 0))
                    
                    # แร่ธาตุ
                    minerals = total_nutrition['minerals']
                    nutrition_columns['calcium'].append(minerals.get('calcium', 0))
                    nutrition_columns['iron'].append(minerals.get('iron', 0))
                    nutrition_columns['magnesium'].append(minerals.get('magnesium', 0))
                    nutrition_columns['phosphorus'].append(minerals.get('phosphorus', 0))
                    nutrition_columns['potassium'].append(minerals.get('potassium', 0))
                    nutrition_columns['zinc'].append(minerals.get('zinc', 0))
                    
                    # ข้อมูลเพิ่มเติม
                    nutrition_columns['ingredient_count'].append(nutrition_data['ingredient_count'])
                    nutrition_columns['category'].append(nutrition_data.get('category', 'อื่นๆ'))
                    nutrition_columns['difficulty'].append(nutrition_data.get('difficulty', 'ปานกลาง'))
                    nutrition_columns['cooking_adjustments_applied'].append(
                        nutrition_data.get('cooking_adjustments_applied', False)
                    )
                    
                    # JSON สำหรับวิตามินและแร่ธาตุ
                    nutrition_columns['vitamins_json'].append(
                        json.dumps(vitamins, ensure_ascii=False)
                    )
                    nutrition_columns['minerals_json'].append(
                        json.dumps(minerals, ensure_ascii=False)
                    )
                    
                else:
                    # ค่าเริ่มต้นสำหรับสูตรที่ไม่มีข้อมูล
                    for key in nutrition_columns:
                        if key in ['category', 'difficulty']:
                            nutrition_columns[key].append('ไม่ทราบ')
                        elif key in ['cooking_adjustments_applied']:
                            nutrition_columns[key].append(False)
                        elif key in ['vitamins_json', 'minerals_json']:
                            nutrition_columns[key].append('{}')
                        else:
                            nutrition_columns[key].append(0)
            
            # เพิ่มคอลัมน์ใหม่เข้าไปใน DataFrame
            for col, values in nutrition_columns.items():
                df[col] = values
            
            # บันทึกไฟล์ใหม่
            df.to_csv(output_file, index=False, encoding='utf-8')
            logger.info(f"✓ Created enhanced dataset: {output_file}")
            print(f"✓ สร้างไฟล์ข้อมูลใหม่: {output_file}")
            print(f"  - เพิ่ม {len(nutrition_columns)} คอลัมน์โภชนาการ")
            print(f"  - รวม {len(df)} สูตรอาหาร")
            
        except Exception as e:
            logger.error(f"Error creating enhanced dataset: {e}")

def main():
    parser = argparse.ArgumentParser(description='Process nutrition data for Thai recipes - Enhanced Version')
    parser.add_argument('--input', type=str, default='thai_food_processed.csv', 
                        help='Input CSV file path')
    parser.add_argument('--batch-size', type=int, default=5, 
                        help='Number of recipes to process in each batch')
    parser.add_argument('--force', action='store_true', 
                        help='Force reprocess all recipes (ignore existing results)')
    parser.add_argument('--summary-only', action='store_true', 
                        help='Generate summary only (skip processing)')
    parser.add_argument('--threading', action='store_true',
                        help='Use threading for faster processing')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of worker threads (when using threading)')
    
    args = parser.parse_args()
    
    # สร้าง processor
    processor = BatchNutritionProcessor(
        args.input, 
        args.batch_size,
        use_threading=args.threading
    )
    
    if args.threading:
        processor.max_workers = min(args.workers, args.batch_size)
        logger.info(f"Threading enabled with {processor.max_workers} workers")
    
    if not args.summary_only:
        # ประมวลผลข้อมูลโภชนาการ
        logger.info("Starting nutrition analysis...")
        processor.process_recipes_incrementally(force_reprocess=args.force, use_threading=args.threading)
    else:
        # โหลดผลลัพธ์ที่มีอยู่
        existing_results = processor.load_existing_results()
        processor.results = list(existing_results.values())
        logger.info(f"Loaded {len(processor.results)} existing results for summary generation")
    
    # สร้างสรุปและไฟล์ข้อมูลใหม่
    processor.generate_nutrition_summary()
    processor.create_enhanced_dataset()
    
    print("\n" + "="*60)
    print("✅ BATCH PROCESSING COMPLETED!")
    print("="*60)
    print("Generated files:")
    print("  - nutrition_results.json (detailed results)")
    print("  - nutrition_summary.json (analysis summary)")
    print("  - thai_food_with_nutrition.csv (enhanced dataset)")
    print("\nNext steps:")
    print("  1. Run 'streamlit run streamlit_app.py' to start the chatbot")
    print("  2. The enhanced search will support all Thai menu items")
    print("  3. Nutrition data is now available for all processed recipes")

if __name__ == "__main__":
    main()
