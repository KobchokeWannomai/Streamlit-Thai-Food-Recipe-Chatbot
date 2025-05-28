#!/usr/bin/env python3
"""
ตัวอย่างการใช้งาน Thai Food Recipe Chatbot with Nutrition Analysis
Example usage of the nutrition analysis system
"""

import json
import time
from nutrition_analyzer import NutritionAnalyzer

def example_1_basic_ingredient_analysis():
    """ตัวอย่างที่ 1: วิเคราะห์วัตถุดิบพื้นฐาน"""
    print("="*60)
    print("ตัวอย่างที่ 1: วิเคราะห์วัตถุดิบแต่ละชนิด")
    print("="*60)
    
    analyzer = NutritionAnalyzer()
    
    # รายการวัตถุดิบที่จะทดสอบ
    ingredients = ["กุ้ง", "หมู", "ไก่", "ข้าว", "น้ำมัน"]
    
    for ingredient in ingredients:
        print(f"\n🔍 วิเคราะห์: {ingredient}")
        
        nutrition = analyzer.analyze_ingredient(ingredient)
        
        print(f"  แคลอรี่: {nutrition.calories:.1f} kcal")
        print(f"  โปรตีน: {nutrition.protein:.1f} g")
        print(f"  คาร์โบไฮเดรต: {nutrition.carbs:.1f} g")
        print(f"  ไขมัน: {nutrition.fat:.1f} g")
        print(f"  ใยอาหาร: {nutrition.fiber:.1f} g")
        
        if nutrition.vitamins:
            print(f"  วิตามิน: {list(nutrition.vitamins.keys())[:3]}")
        
        # หน่วงเวลาเพื่อไม่ให้ API rate limit
        time.sleep(0.5)

def example_2_recipe_analysis():
    """ตัวอย่างที่ 2: วิเคราะห์สูตรอาหารทั้งหมด"""
    print("\n" + "="*60)
    print("ตัวอย่างที่ 2: วิเคราะห์สูตรอาหารทั้งหมด")
    print("="*60)
    
    analyzer = NutritionAnalyzer()
    
    # สูตรต้มยำกุ้ง
    recipe_name = "ต้มยำกุ้ง"
    ingredients_text = """- กุ้งนาง 5 ตัว
- น้ำปลา 2 ช้อนโต๊ะ
- มะนาว 1 ผล
- พริกขี้หนู 3 เม็ด
- ใบมะกรูด 3 ใบ
- ตะไคร้ 2 ต้น
- ข่า 3 แว่น
- เห็ดฟาง 100 กรัม
- มะเขือเทศ 2 ผล"""
    
    print(f"\n🍲 วิเคราะห์สูตร: {recipe_name}")
    print("📋 วัตถุดิบ:")
    for line in ingredients_text.split('\n'):
        if line.strip():
            print(f"  {line.strip()}")
    
    result = analyzer.analyze_recipe(recipe_name, ingredients_text)
    
    print(f"\n📊 ผลการวิเคราะห์:")
    total_nutrition = result['total_nutrition']
    print(f"  🔥 แคลอรี่รวม: {total_nutrition['calories']:.1f} kcal")
    print(f"  🥩 โปรตีนรวม: {total_nutrition['protein']:.1f} g")
    print(f"  🍞 คาร์โบไฮเดรตรวม: {total_nutrition['carbs']:.1f} g")
    print(f"  🧈 ไขมันรวม: {total_nutrition['fat']:.1f} g")
    print(f"  🌾 ใยอาหารรวม: {total_nutrition['fiber']:.1f} g")
    print(f"  📦 จำนวนวัตถุดิบ: {result['ingredient_count']} ชนิด")
    
    # แสดงรายละเอียดวัตถุดิบ
    print(f"\n🔍 รายละเอียดแต่ละวัตถุดิบ:")
    for ingredient_info in result['ingredients'][:5]:  # แสดง 5 ตัวแรก
        ingredient = ingredient_info['ingredient']
        nutrition = ingredient_info['nutrition']
        print(f"  • {ingredient}: {nutrition.calories:.0f} kcal, {nutrition.protein:.1f}g protein")

def example_3_nutrition_search():
    """ตัวอย่างที่ 3: ค้นหาสูตรอาหารตามเกณฑ์โภชนาการ"""
    print("\n" + "="*60)
    print("ตัวอย่างที่ 3: ค้นหาสูตรตามเกณฑ์โภชนาการ")
    print("="*60)
    
    analyzer = NutritionAnalyzer()
    
    # สร้างข้อมูลตัวอย่างก่อน (จำลองว่ามีข้อมูลสูตรในฐานข้อมูลแล้ว)
    sample_recipes = [
        ("ต้มยำกุ้ง", "- กุ้ง 5 ตัว\n- น้ำปลา 2 ช้อนโต๊ะ"),
        ("ผัดไทย", "- เส้นหมี่ 200g\n- กุ้ง 3 ตัว\n- ไข่ 2 ฟอง"),
        ("ส้มตำ", "- มะละกอ 1 ผล\n- กุ้งแห้ง 2 ช้อนโต๊ะ")
    ]
    
    print("🔧 สร้างข้อมูลตัวอย่าง...")
    for recipe_name, ingredients in sample_recipes:
        analyzer.analyze_recipe(recipe_name, ingredients)
        time.sleep(0.5)
    
    # ค้นหาสูตรตามเกณฑ์
    criteria_sets = [
        {
            "name": "เมนูแคลอรี่ต่ำ",
            "criteria": {"max_calories": 300},
        },
        {
            "name": "เมนูโปรตีนสูง", 
            "criteria": {"min_protein": 20},
        },
        {
            "name": "เมนูสำหรับลดน้ำหนัก",
            "criteria": {"max_calories": 250, "min_protein": 15},
        }
    ]
    
    for search in criteria_sets:
        print(f"\n🔍 ค้นหา: {search['name']}")
        print(f"📋 เกณฑ์: {search['criteria']}")
        
        results = analyzer.search_recipes_by_nutrition(search['criteria'])
        
        if results:
            print(f"✅ พบ {len(results)} เมนู:")
            for recipe in results[:3]:  # แสดง 3 เมนูแรก
                print(f"  • {recipe['recipe_name']}: {recipe['calories']:.0f} kcal, {recipe['protein']:.1f}g protein")
        else:
            print("❌ ไม่พบเมนูที่ตรงเกณฑ์")

def example_4_database_operations():
    """ตัวอย่างที่ 4: การจัดการฐานข้อมูล"""
    print("\n" + "="*60)
    print("ตัวอย่างที่ 4: การจัดการฐานข้อมูลโภชนาการ")
    print("="*60)
    
    analyzer = NutritionAnalyzer()
    
    # ตรวจสอบข้อมูลในฐานข้อมูล
    print("🔍 ตรวจสอบข้อมูลที่มีอยู่ในฐานข้อมูล:")
    
    test_ingredients = ["กุ้ง", "หมู", "ข้าว", "วัตถุดิบที่ไม่มี"]
    
    for ingredient in test_ingredients:
        existing_nutrition = analyzer.get_nutrition_from_db(ingredient)
        
        if existing_nutrition:
            print(f"  ✅ {ingredient}: มีข้อมูลในฐานข้อมูล ({existing_nutrition.calories:.0f} kcal)")
        else:
            print(f"  ❌ {ingredient}: ไม่มีข้อมูลในฐานข้อมูล")
            
            # เพิ่มข้อมูลใหม่
            print(f"    🔄 กำลังเพิ่มข้อมูลใหม่...")
            new_nutrition = analyzer.analyze_ingredient(ingredient)
            print(f"    ✅ เพิ่มแล้ว: {new_nutrition.calories:.0f} kcal")

def example_5_api_testing():
    """ตัวอย่างที่ 5: ทดสอบ API"""
    print("\n" + "="*60)
    print("ตัวอย่างที่ 5: ทดสอบการเชื่อมต่อ API")
    print("="*60)
    
    analyzer = NutritionAnalyzer()
    
    test_ingredients = ["chicken", "rice", "shrimp"]
    
    for ingredient in test_ingredients:
        print(f"\n🧪 ทดสอบ API สำหรับ: {ingredient}")
        
        try:
            # ทดสอบ FDC API
            nutrition = analyzer.get_nutrition_from_api(ingredient)
            if nutrition:
                print(f"  ✅ FDC API: {nutrition.calories:.0f} kcal")
            else:
                print(f"  ❌ FDC API: ไม่ได้รับข้อมูล")
        except Exception as e:
            print(f"  ❌ FDC API Error: {str(e)[:50]}...")
        
        try:
            # ทดสอบ Nutritionix API
            nutrition = analyzer._get_nutrition_from_nutritionix(ingredient)
            if nutrition:
                print(f"  ✅ Nutritionix API: {nutrition.calories:.0f} kcal")
            else:
                print(f"  ❌ Nutritionix API: ไม่ได้รับข้อมูล")
        except Exception as e:
            print(f"  ❌ Nutritionix API Error: {str(e)[:50]}...")
        
        # ทดสอบ Fallback
        fallback_nutrition = analyzer.get_fallback_nutrition(ingredient)
        print(f"  📋 Fallback: {fallback_nutrition.calories:.0f} kcal")
        
        time.sleep(1)  # หน่วงเพื่อไม่ให้ API rate limit

def example_6_batch_processing():
    """ตัวอย่างที่ 6: การประมวลผลแบบ batch"""
    print("\n" + "="*60)
    print("ตัวอย่างที่ 6: การประมวลผลแบบ batch")
    print("="*60)
    
    # สร้างข้อมูลตัวอย่าง
    sample_data = {
        'name': ['ต้มยำกุ้ง', 'ผัดไทย', 'ส้มตำ'],
        'ingredient': [
            '- กุ้ง 5 ตัว\n- น้ำปลา 2 ช้อนโต๊ะ',
            '- เส้นหมี่ 200g\n- ไข่ 2 ฟอง',
            '- มะละกอ 1 ผล\n- กุ้งแห้ง 2 ช้อนโต๊ะ'
        ],
        'method': ['วิธีทำต้มยำ', 'วิธีทำผัดไทย', 'วิธีทำส้มตำ']
    }
    
    # สร้างไฟล์ CSV ตัวอย่าง
    import pandas as pd
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_recipes.csv', index=False, encoding='utf-8')
    
    print("📋 สร้างไฟล์ตัวอย่าง: sample_recipes.csv")
    print("🔄 เริ่มประมวลผล...")
    
    # ประมวลผลด้วย batch processor
    from batch_nutrition_process import BatchNutritionProcessor
    
    processor = BatchNutritionProcessor('sample_recipes.csv', batch_size=2)
    processor.process_recipes_incrementally()
    
    print("✅ ประมวลผลเสร็จสิ้น")
    
    # แสดงผลลัพธ์
    if processor.results:
        print(f"\n📊 ผลลัพธ์ ({len(processor.results)} เมนู):")
        for result in processor.results:
            if 'error' not in result:
                print(f"  • {result['recipe_name']}: {result['total_nutrition']['calories']:.0f} kcal")

def main():
    """ฟังก์ชันหลักสำหรับรันตัวอย่างทั้งหมด"""
    print("🍲 Thai Food Recipe Chatbot - Nutrition Analysis Examples")
    print("🚀 เริ่มต้นการทดสอบ...")
    
    examples = [
        ("วิเคราะห์วัตถุดิบพื้นฐาน", example_1_basic_ingredient_analysis),
        ("วิเคราะห์สูตรอาหาร", example_2_recipe_analysis),
        ("ค้นหาตามโภชนาการ", example_3_nutrition_search),
        ("จัดการฐานข้อมูล", example_4_database_operations),
        ("ทดสอบ API", example_5_api_testing),
        ("ประมวลผลแบบ batch", example_6_batch_processing),
    ]
    
    print("\nเลือกตัวอย่างที่ต้องการทดสอบ:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    print("  0. รันทั้งหมด")
    print("  q. ออก")
    
    while True:
        try:
            choice = input("\nเลือก (0-6, q): ").strip().lower()
            
            if choice == 'q':
                print("👋 ขอบคุณที่ใช้งาน!")
                break
            elif choice == '0':
                # รันทั้งหมด
                for name, func in examples:
                    print(f"\n🏃‍♂️ กำลังรัน: {name}")
                    try:
                        func()
                    except Exception as e:
                        print(f"❌ เกิดข้อผิดพลาด: {e}")
                    
                    input("\nกด Enter เพื่อไปต่อ...")
                break
            elif choice.isdigit() and 1 <= int(choice) <= len(examples):
                # รันตัวอย่างเดียว
                idx = int(choice) - 1
                name, func = examples[idx]
                print(f"\n🏃‍♂️ กำลังรัน: {name}")
                try:
                    func()
                except Exception as e:
                    print(f"❌ เกิดข้อผิดพลาด: {e}")
            else:
                print("❌ กรุณาเลือกหมายเลข 0-6 หรือ q")
                
        except KeyboardInterrupt:
            print("\n👋 ขอบคุณที่ใช้งาน!")
            break
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาด: {e}")

if __name__ == "__main__":
    main()
