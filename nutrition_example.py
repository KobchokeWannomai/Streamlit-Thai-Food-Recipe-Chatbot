#!/usr/bin/env python3
"""
ตัวอย่างการใช้งานขั้นสูงของระบบวิเคราะห์คุณค่าทางโภชนาการ
Advanced Thai Food Nutrition Analysis Examples with API Integration
"""

import pandas as pd
from nutrition_analyzer import NutritionAnalyzer, CookingAdjustmentHelper
from ingredient_converter import IngredientConverter
from config import Config
import json
import time
import asyncio
from datetime import datetime

def example_enhanced_nutrition_analysis():
    """ตัวอย่างการวิเคราะห์โภชนาการขั้นสูงพร้อม API Integration"""
    print("🔬 ตัวอย่างการวิเคราะห์โภชนาการขั้นสูง")
    print("=" * 60)
    
    # สร้าง analyzer พร้อม API integration
    analyzer = NutritionAnalyzer(
        usda_api_key=Config.USDA_API_KEY,
        nutritionix_app_id=Config.NUTRITIONIX_APP_ID,
        nutritionix_api_key=Config.NUTRITIONIX_API_KEY
    )
    
    print(f"🔌 API Status:")
    print(f"  • USDA API: {'✅ Connected' if Config.is_api_configured()['usda'] else '❌ Not configured'}")
    print(f"  • Nutritionix API: {'✅ Connected' if Config.is_api_configured()['nutritionix'] else '❌ Not configured'}")
    print()
    
    # ตัวอย่างเมนู - ไข่เจียว (มีการปรับแต่งการทำอาหาร)
    recipe_name = "ไข่เจียว"
    ingredients_text = """
    - ไข่ไก่ 2 ฟอง
    - เกลือ 1/2 ช้อนชา
    - พริกไทย 1/4 ช้อนชา
    """
    
    print(f"📋 วิเคราะห์เมนู: {recipe_name}")
    print(f"🥘 วัตถุดิบที่ระบุ:\n{ingredients_text}")
    
    # วิเคราะห์แบบปกติ
    print("\n📊 การวิเคราะห์แบบปกติ:")
    normal_data = analyzer.analyze_ingredients(ingredients_text, recipe_name, apply_cooking_adjustments=False)
    normal_total = analyzer.calculate_total_nutrition(normal_data)
    
    print(f"  ⚡ พลังงาน: {normal_total.calories:.1f} แคลอรี่")
    print(f"  🥩 โปรตีน: {normal_total.protein:.1f} กรัม")
    print(f"  🥑 ไขมัน: {normal_total.fat:.1f} กรัม")
    
    # วิเคราะห์แบบขั้นสูง (มีการปรับแต่งการทำอาหาร)
    print("\n🧪 การวิเคราะห์ขั้นสูง (มีการปรับแต่งการทำอาหาร):")
    enhanced_data = analyzer.analyze_ingredients(ingredients_text, recipe_name, apply_cooking_adjustments=True)
    enhanced_total = analyzer.calculate_total_nutrition(enhanced_data)
    
    print(f"  ⚡ พลังงาน: {enhanced_total.calories:.1f} แคลอรี่")
    print(f"  🥩 โปรตีน: {enhanced_total.protein:.1f} กรัม")
    print(f"  🥑 ไขมัน: {enhanced_total.fat:.1f} กรัม")
    
    # แสดงการเปรียบเทียบ
    calorie_diff = enhanced_total.calories - normal_total.calories
    fat_diff = enhanced_total.fat - normal_total.fat
    
    print(f"\n📈 การเปรียบเทียบ:")
    print(f"  🔥 ความแตกต่างแคลอรี่: +{calorie_diff:.1f} แคลอรี่ (จากน้ำมันทอด)")
    print(f"  🧈 ความแตกต่างไขมัน: +{fat_diff:.1f} กรัม (จากการดูดซึมน้ำมัน)")
    
    # แสดงรายละเอียดวัตถุดิบที่เพิ่มเติม
    print(f"\n🔍 วัตถุดิบที่เพิ่มเติมโดยระบบ:")
    for ingredient_key, nutrition in enhanced_data.items():
        if "[เพิ่มเติม]" in ingredient_key:
            print(f"  • {ingredient_key}: {nutrition.calories:.1f} แคลอรี่")

def example_cooking_adjustment_analysis():
    """ตัวอย่างการปรับแต่งการคำนวณตามวิธีการทำอาหาร"""
    print("\n\n🔥 ตัวอย่างการปรับแต่งตามวิธีการทำอาหาร")
    print("=" * 60)
    
    helper = CookingAdjustmentHelper()
    
    # เมนูที่มีการปรับแต่งต่างกัน
    test_recipes = {
        "ไข่เจียว": "ทอดน้ำมันมาก",
        "ไข่ดาว": "ทอดน้ำมันน้อย", 
        "ผัดกะเพรา": "ผัดด้วยน้ำมันปานกลาง",
        "ต้มยำกุ้ง": "ต้มน้ำซุป",
        "แกงเขียวหวาน": "แกงกะทิ"
    }
    
    for recipe_name, cooking_method in test_recipes.items():
        adjustments = helper.get_cooking_adjustments(recipe_name)
        print(f"\n🍽️  {recipe_name} ({cooking_method}):")
        
        if adjustments:
            if "oil_absorption" in adjustments:
                absorption_rate = adjustments["oil_absorption"] * 100
                print(f"  🛢️  การดูดซึมน้ำมัน: {absorption_rate:.0f}%")
            
            if "missing_ingredients" in adjustments:
                print(f"  ➕ วัตถุดิบที่เพิ่ม: {len(adjustments['missing_ingredients'])} รายการ")
                for missing in adjustments["missing_ingredients"]:
                    consumed_pct = missing.get("consumed", 1.0) * 100
                    print(f"     • {missing['name']}: {missing['amount']} {missing['unit']} (บริโภค {consumed_pct:.0f}%)")
            
            if "broth_consumption" in adjustments:
                broth_pct = adjustments["broth_consumption"] * 100
                print(f"  🍲 การบริโภคน้ำซุป: {broth_pct:.0f}%")
        else:
            print(f"  ℹ️  ไม่มีการปรับแต่งพิเศษ (ใช้การคำนวณมาตรฐาน)")

def example_api_integration_comparison():
    """ตัวอย่างการเปรียบเทียบข้อมูลจาก API ต่างๆ"""
    print("\n\n🌐 ตัวอย่างการเปรียบเทียบข้อมูลจาก API ต่างๆ")
    print("=" * 60)
    
    # สร้าง analyzer หลายแบบ
    analyzers = {
        "ฐานข้อมูลไทย": NutritionAnalyzer(),
        "USDA API": NutritionAnalyzer(usda_api_key=Config.USDA_API_KEY) if Config.USDA_API_KEY else None,
        "Local + USDA": NutritionAnalyzer(usda_api_key=Config.USDA_API_KEY) if Config.USDA_API_KEY else None
    }
    
    # วัตถุดิบทดสอบ
    test_ingredients = ["ไข่ไก่", "กุ้ง", "ข้าว", "น้ำมันพืช"]
    
    results = {}
    
    for ingredient in test_ingredients:
        results[ingredient] = {}
        print(f"\n🧪 ทดสอบ: {ingredient}")
        
        for source_name, analyzer in analyzers.items():
            if analyzer is None:
                print(f"  {source_name}: ❌ ไม่ได้กำหนดค่า")
                continue
                
            try:
                nutrition = analyzer.get_ingredient_nutrition(ingredient)
                if nutrition and nutrition.calories > 0:
                    results[ingredient][source_name] = {
                        'calories': nutrition.calories,
                        'protein': nutrition.protein,
                        'fat': nutrition.fat
                    }
                    print(f"  {source_name}: ✅ {nutrition.calories:.1f} kcal, {nutrition.protein:.1f}g โปรตีน")
                else:
                    print(f"  {source_name}: ⚠️  ไม่พบข้อมูล")
            except Exception as e:
                print(f"  {source_name}: ❌ Error: {str(e)[:50]}...")
    
    # สรุปผลการเปรียบเทียบ
    print(f"\n📊 สรุปการเปรียบเทียบ:")
    for ingredient, sources in results.items():
        if len(sources) > 1:
            calories_values = [data['calories'] for data in sources.values()]
            min_cal, max_cal = min(calories_values), max(calories_values)
            variance = ((max_cal - min_cal) / min_cal * 100) if min_cal > 0 else 0
            print(f"  {ingredient}: ความแตกต่าง {variance:.1f}% ({min_cal:.0f}-{max_cal:.0f} kcal)")

def example_enhanced_search_capabilities():
    """ตัวอย่างความสามารถการค้นหาขั้นสูง"""
    print("\n\n🔍 ตัวอย่างการค้นหาขั้นสูง")
    print("=" * 60)
    
    # ตัวอย่างการขยายคำค้นหา
    search_expansions = Config.SEARCH_ENHANCEMENT['query_expansions']
    
    print("🚀 การขยายคำค้นหาอัตโนมัติ:")
    test_queries = ['ไข่', 'หมู', 'ผัด', 'แกง']
    
    for query in test_queries:
        if query in search_expansions:
            expanded = search_expansions[query]
            print(f"  '{query}' → {', '.join(expanded[:3])}{'...' if len(expanded) > 3 else ''}")
        else:
            print(f"  '{query}' → ไม่มีการขยาย")
    
    # ตัวอย่างการค้นหาแบบ semantic
    print(f"\n🧠 การค้นหาแบบ Semantic:")
    semantic_examples = [
        ("อาหารง่ายๆ", "ไข่เจียว, ไข่ดาว, ข้าวผัด"),
        ("เมนูเช้า", "ข้าวต้ม, โจ๊ก, ขนมปัง"),
        ("อาหารเผ็ด", "ส้มตำ, ลาบ, น้ำพริก"),
        ("เมนูทอด", "ไข่เจียว, ปลาทอด, กุ้งทอด")
    ]
    
    for query, expected in semantic_examples:
        print(f"  '{query}' คาดว่าจะหา: {expected}")

def example_nutrition_goal_tracking():
    """ตัวอย่างการติดตามเป้าหมายโภชนาการ"""
    print("\n\n🎯 ตัวอย่างการติดตามเป้าหมายโภชนาการ")
    print("=" * 60)
    
    # เป้าหมายตัวอย่าง
    daily_goals = {
        'calories': 2000,
        'protein': 100,
        'carbs': 250,
        'fat': 65,
        'fiber': 25
    }
    
    # มื้ออาหารตัวอย่าง
    meals = {
        'เช้า': {
            'recipes': ['ข้าวต้มกุ้ง'],
            'calories': 280, 'protein': 15, 'carbs': 45, 'fat': 5, 'fiber': 2
        },
        'กลางวัน': {
            'recipes': ['ผัดกะเพรา', 'ข้าวสวย'],
            'calories': 520, 'protein': 28, 'carbs': 58, 'fat': 22, 'fiber': 3
        },
        'เย็น': {
            'recipes': ['แกงเขียวหวาน', 'ข้าวสวย'],
            'calories': 450, 'protein': 25, 'carbs': 52, 'fat': 18, 'fiber': 4
        },
        'ว่าง': {
            'recipes': ['มะม่วงข้าวเหนียว'],
            'calories': 320, 'protein': 4, 'carbs': 68, 'fat': 8, 'fiber': 3
        }
    }
    
    print("🍽️  แผนอาหารประจำวัน:")
    total_nutrition = {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0, 'fiber': 0}
    
    for meal_time, meal_data in meals.items():
        print(f"\n  {meal_time}: {', '.join(meal_data['recipes'])}")
        print(f"    ⚡ {meal_data['calories']} kcal | 🥩 {meal_data['protein']}g")
        
        for nutrient in total_nutrition:
            total_nutrition[nutrient] += meal_data[nutrient]
    
    print(f"\n📊 สรุปโภชนาการทั้งวัน vs เป้าหมาย:")
    for nutrient, total_value in total_nutrition.items():
        goal_value = daily_goals[nutrient]
        percentage = (total_value / goal_value) * 100
        status = "✅" if 90 <= percentage <= 110 else "⚠️" if percentage < 90 else "🔴"
        
        print(f"  {status} {nutrient.capitalize()}: {total_value:.0f}/{goal_value} ({percentage:.1f}%)")
    
    # คำแนะนำ
    print(f"\n💡 คำแนะนำ:")
    if total_nutrition['protein'] < daily_goals['protein'] * 0.9:
        print("  • เพิ่มโปรตีน: ลองเพิ่มไข่, เต้าหู้, หรือถั่ว")
    if total_nutrition['fiber'] < daily_goals['fiber'] * 0.9:
        print("  • เพิ่มใยอาหาร: เพิ่มผักใบเขียว หรือผลไม้")
    if total_nutrition['calories'] > daily_goals['calories'] * 1.1:
        print("  • ลดแคลอรี่: ลดปริมาณข้าว หรือเลือกวิธีทำอาหารที่ใช้น้ำมันน้อย")

def example_batch_recipe_analysis():
    """ตัวอย่างการวิเคราะห์สูตรอาหารแบบ batch"""
    print("\n\n⚡ ตัวอย่างการวิเคราะห์แบบ Batch Processing")
    print("=" * 60)
    
    # สูตรอาหารตัวอย่าง
    recipes = [
        {
            'name': 'ผัดกะเพรา',
            'ingredients': '''
            - หมูสับ 200 กรัม
            - ใบกะเพรา 1 ถ้วย
            - พริกขี้หนู 5 เม็ด
            - กระเทียม 5 กลีบ
            - น้ำปลา 2 ช้อนโต๊ะ
            - น้ำตาล 1 ช้อนชา
            '''
        },
        {
            'name': 'ส้มตำ',
            'ingredients': '''
            - มะละกอดิบขูดฝอย 2 ถ้วย
            - มะเขือเทศ 3 ผล
            - ถั่วฝักยาว 5 ฝัก
            - กุ้งแห้ง 2 ช้อนโต๊ะ
            - ถั่วลิสง 2 ช้อนโต๊ะ
            - น้ำปลา 2 ช้อนโต๊ะ
            - น้ำตาลปึก 2 ช้อนโต๊ะ
            '''
        },
        {
            'name': 'แกงเขียวหวาน',
            'ingredients': '''
            - ไก่ 300 กรัม
            - กะทิ 1 ถ้วย
            - น้ำพริกแกงเขียวหวาน 3 ช้อนโต๊ะ
            - มะเขือพวง 10 ผล
            - ใบมะกรูด 5 ใบ
            - น้ำปลา 2 ช้อนโต๊ะ
            '''
        }
    ]
    
    analyzer = NutritionAnalyzer()
    results = []
    
    print("🔄 กำลังประมวลผล...")
    start_time = time.time()
    
    for i, recipe in enumerate(recipes, 1):
        print(f"  {i}/{len(recipes)}: {recipe['name']}")
        
        # วิเคราะห์แบบปกติและแบบขั้นสูง
        normal_result = analyzer.analyze_recipe(recipe['name'], recipe['ingredients'])
        enhanced_result = analyzer.analyze_recipe(
            recipe['name'], recipe['ingredients']
        )
        
        results.append({
            'name': recipe['name'],
            'normal': normal_result['total_nutrition'],
            'enhanced': enhanced_result['total_nutrition'],
            'ingredient_count': len(recipe['ingredients'].strip().split('\n')) - 1
        })
    
    processing_time = time.time() - start_time
    print(f"✅ เสร็จสิ้นใน {processing_time:.2f} วินาที")
    
    # แสดงผลเปรียบเทียบ
    print(f"\n📊 ผลการวิเคราะห์:")
    print(f"{'เมนู':<20} {'แคลอรี่':<10} {'โปรตีน':<8} {'ไขมัน':<8} {'วัตถุดิบ':<8}")
    print("-" * 60)
    
    for result in results:
        nutrition = result['normal']
        print(f"{result['name']:<20} {nutrition['calories']:<10.0f} "
              f"{nutrition['protein']:<8.1f} {nutrition['fat']:<8.1f} "
              f"{result['ingredient_count']:<8}")
    
    # หาเมนูที่ดีที่สุดในแต่ละหมวด
    best_recipes = {
        'lowest_calorie': min(results, key=lambda x: x['normal']['calories']),
        'highest_protein': max(results, key=lambda x: x['normal']['protein']),
        'most_balanced': min(results, key=lambda x: abs(x['normal']['calories'] - 400))
    }
    
    print(f"\n🏆 เมนูแนะนำ:")
    print(f"  🔥 แคลอรี่ต่ำสุด: {best_recipes['lowest_calorie']['name']}")
    print(f"  💪 โปรตีนสูงสุด: {best_recipes['highest_protein']['name']}")
    print(f"  ⚖️  สมดุลที่สุด: {best_recipes['most_balanced']['name']}")

def example_export_and_reporting():
    """ตัวอย่างการส่งออกและสร้างรายงาน"""
    print("\n\n📋 ตัวอย่างการส่งออกและสร้างรายงาน")
    print("=" * 60)
    
    # สร้างข้อมูลตัวอย่าง
    report_data = {
        'generated_at': datetime.now().isoformat(),
        'analysis_summary': {
            'total_recipes_analyzed': 15,
            'avg_calories_per_recipe': 380.5,
            'avg_protein_per_recipe': 22.3,
            'most_common_ingredients': [
                ('น้ำปลา', 12), ('กระเทียม', 11), ('พริก', 10), 
                ('ใบกะเพรา', 8), ('กะทิ', 7)
            ]
        },
        'nutrition_categories': {
            'high_protein_recipes': ['ลาบหมู', 'แกงเขียวหวานไก่', 'ผัดกะเพรา'],
            'low_calorie_recipes': ['ส้มตำ', 'ยำวุ้นเส้น', 'แกงส้ม'],
            'high_fiber_recipes': ['ส้มตำ', 'ยำถั่วพู', 'แกงป่า']
        },
        'cooking_method_analysis': {
            'ทอด': {'count': 3, 'avg_calories': 450, 'oil_absorption_avg': 12},
            'ผัด': {'count': 5, 'avg_calories': 380, 'oil_absorption_avg': 8},
            'ต้ม': {'count': 4, 'avg_calories': 220, 'oil_absorption_avg': 0},
            'แกง': {'count': 3, 'avg_calories': 320, 'oil_absorption_avg': 2}
        }
    }
    
    # ส่งออกเป็น JSON
    output_file = f'nutrition_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 บันทึกรายงานลงไฟล์: {output_file}")
    
    # แสดงสรุปรายงาน
    print(f"\n📈 สรุปรายงาน:")
    print(f"  📊 วิเคราะห์สูตรอาหารทั้งหมด: {report_data['analysis_summary']['total_recipes_analyzed']} สูตร")
    print(f"  ⚡ แคลอรี่เฉลี่ย: {report_data['analysis_summary']['avg_calories_per_recipe']:.1f} kcal")
    print(f"  🥩 โปรตีนเฉลี่ย: {report_data['analysis_summary']['avg_protein_per_recipe']:.1f} g")
    
    print(f"\n🥇 วัตถุดิบยอดนิยม:")
    for ingredient, count in report_data['analysis_summary']['most_common_ingredients'][:5]:
        print(f"  • {ingredient}: ใช้ใน {count} สูตร")
    
    print(f"\n🍳 การวิเคราะห์ตามวิธีการทำอาหาร:")
    for method, data in report_data['cooking_method_analysis'].items():
        print(f"  {method}: {data['count']} สูตร, เฉลี่ย {data['avg_calories']} kcal")

async def example_async_batch_processing():
    """ตัวอย่างการประมวลผลแบบ asynchronous"""
    print("\n\n⚡ ตัวอย่างการประมวลผลแบบ Asynchronous")
    print("=" * 60)
    
    # จำลองการประมวลผลหลายสูตรพร้อมกัน
    async def analyze_recipe_async(recipe_name, ingredients):
        """จำลองการวิเคราะห์แบบ async"""
        await asyncio.sleep(0.5)  # จำลองเวลาในการประมวลผล
        analyzer = NutritionAnalyzer()
        return analyzer.analyze_recipe(recipe_name, ingredients)
    
    recipes = [
        ("ผัดไทย", "เส้นจันท์, กุ้ง, ไข่, ถั่วงอก"),
        ("ต้มยำกุ้ง", "กุ้ง, เห็ด, ตะไคร้, พริก"),
        ("มัสมั่นไก่", "ไก่, กะทิ, มันฝรั่ง"),
        ("ยำวุ้นเส้น", "วุ้นเส้น, กุ้ง, หมูสับ"),
        ("แกงส้ม", "ปลา, ผักบุ้ง, มะนาว")
    ]
    
    print(f"🚀 เริ่มประมวลผล {len(recipes)} สูตรแบบ async...")
    start_time = time.time()
    
    # ประมวลผลแบบ concurrent
    tasks = [analyze_recipe_async(name, ingredients) for name, ingredients in recipes]
    results = await asyncio.gather(*tasks)
    
    processing_time = time.time() - start_time
    print(f"✅ เสร็จสิ้นใน {processing_time:.2f} วินาที")
    
    # แสดงผล
    print(f"\n📊 ผลการประมวลผล:")
    for i, (recipe_name, _) in enumerate(recipes):
        result = results[i]
        nutrition = result['total_nutrition']
        print(f"  {recipe_name}: {nutrition['calories']:.0f} kcal, {nutrition['protein']:.1f}g โปรตีน")

def main():
    """ฟังก์ชันหลักสำหรับรันตัวอย่างทั้งหมด"""
    print("🍲 Enhanced Thai Food Nutrition Analysis Examples")
    print("=" * 80)
    print("🚀 เริ่มต้นตัวอย่างขั้นสูงของระบบวิเคราะห์โภชนาการ")
    print(f"⏰ เวลา: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # รันตัวอย่างต่างๆ
        example_enhanced_nutrition_analysis()
        example_cooking_adjustment_analysis()
        example_api_integration_comparison()
        example_enhanced_search_capabilities()
        example_nutrition_goal_tracking()
        example_batch_recipe_analysis()
        example_export_and_reporting()
        
        # รัน async example
        print("\n" + "=" * 80)
        print("🔄 รันตัวอย่าง Asynchronous Processing...")
        asyncio.run(example_async_batch_processing())
        
        print("\n" + "=" * 80)
        print("✅ เสร็จสิ้นการทำงานของตัวอย่างขั้นสูงทั้งหมด!")
        print("\n💡 เคล็ดลับการใช้งานขั้นสูง:")
        print("  1. 🔌 ตั้งค่า API keys สำหรับข้อมูลโภชนาการที่แม่นยำ")
        print("  2. ⚙️  ใช้แถบการตั้งค่าในแอปเพื่อเปิดใช้ฟีเจอร์ขั้นสูง")
        print("  3. 🧪 เปิดใช้การปรับแต่งการทำอาหารสำหรับการคำนวณที่แม่นยำ")
        print("  4. 🔍 ใช้การค้นหาขั้นสูงเพื่อผลลัพธ์ที่ดีกว่า")
        print("  5. 📊 ใช้ batch processing สำหรับการวิเคราะห์หลายสูตร")
        print("  6. 📋 ส่งออกรายงานเพื่อการวิเคราะห์เชิงลึก")
        
        print("\n🔗 ทรัพยากรเพิ่มเติม:")
        print("  • streamlit run streamlit_app.py - เริ่มแอปพลิเคชัน")
        print("  • python batch_nutrition_processor.py - ประมวลผลแบบ batch")
        print("  • ดู API_SETUP_GUIDE.md สำหรับการตั้งค่า API")
        print("  • อ่าน README.md สำหรับรายละเอียดเพิ่มเติม")
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        print("💡 ตรวจสอบ:")
        print("  - การติดตั้ง dependencies ครบถ้วน")
        print("  - ไฟล์ข้อมูลที่จำเป็น")
        print("  - การตั้งค่า API keys (ถ้าต้องการใช้)")

if __name__ == "__main__":
    main()
