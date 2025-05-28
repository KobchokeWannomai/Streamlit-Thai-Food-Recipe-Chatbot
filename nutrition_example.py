#!/usr/bin/env python3
"""
ตัวอย่างการใช้งานระบบวิเคราะห์คุณค่าทางโภชนาการ
Thai Food Nutrition Analysis Example
"""

import pandas as pd
from nutrition_analyzer import NutritionAnalyzer
from config import Config
import json

def example_basic_usage():
    """ตัวอย่างการใช้งานพื้นฐาน"""
    print("🔍 ตัวอย่างการใช้งานพื้นฐาน")
    print("=" * 50)
    
    # สร้าง analyzer
    analyzer = NutritionAnalyzer()
    
    # ตัวอย่างวัตถุดิบ - กุ้งทาพริกไทยกระเทียม
    ingredients_text = """
    - กุ้งนาง 4 ตัว
    - พริกไทย 5 เม็ด
    - กระเทียมกลีบใหญ่ 2 กลีบ
    - รากผักชี 5 ราก
    - น้ำปลา 2 ช้อนโต๊ะ
    - น้ำมันหมู 1 ช้อนโต๊ะ
    """
    
    print(f"📋 วิเคราะห์เมนู: กุ้งทาพริกไทยกระเทียม")
    print(f"🥘 วัตถุดิบ:\n{ingredients_text}")
    
    # วิเคราะห์โภชนาการ
    nutrition_data = analyzer.analyze_ingredients(ingredients_text)
    
    print("\n📊 ข้อมูลโภชนาการแต่ละวัตถุดิบ:")
    for ingredient, nutrition in nutrition_data.items():
        print(f"\n🔸 {ingredient}:")
        print(f"  • พลังงาน: {nutrition.calories:.1f} แคลอรี่")
        print(f"  • โปรตีน: {nutrition.protein:.1f} กรัม")
        print(f"  • คาร์โบไฮเดรต: {nutrition.carbs:.1f} กรัม")
        print(f"  • ไขมัน: {nutrition.fat:.1f} กรัม")
        if nutrition.vitamin_c > 0:
            print(f"  • วิตามิน C: {nutrition.vitamin_c:.1f} มิลลิกรัม")
        if nutrition.iron > 0:
            print(f"  • เหล็ก: {nutrition.iron:.1f} มิลลิกรัม")
    
    # คำนวณรวม
    total = analyzer.calculate_total_nutrition(nutrition_data)
    print(f"\n🍽️  สรุปโภชนาการรวม:")
    print(f"  ⚡ พลังงาน: {total.calories:.1f} แคลอรี่")
    print(f"  🥩 โปรตีน: {total.protein:.1f} กรัม")
    print(f"  🍞 คาร์โบไฮเดรต: {total.carbs:.1f} กรัม")
    print(f"  🥑 ไขมัน: {total.fat:.1f} กรัม")
    print(f"  🌊 โซเดียม: {total.sodium:.0f} มิลลิกรัม")
    
def example_batch_analysis():
    """ตัวอย่างการวิเคราะห์หลายเมนูพร้อมกัน"""
    print("\n\n🔍 ตัวอย่างการวิเคราะห์หลายเมนู")
    print("=" * 50)
    
    # ตัวอย่างเมนูหลายอย่าง
    recipes = {
        "ผัดกะเพรา": """
        - หมูสับ 200 กรัม
        - ใบกะเพรา 1 ถ้วย
        - พริกขี้หนู 5 เม็ด
        - กระเทียม 5 กลีบ
        - น้ำปลา 2 ช้อนโต๊ะ
        - น้ำตาล 1 ช้อนชา
        - น้ำมัน 2 ช้อนโต๊ะ
        """,
        
        "ส้มตำ": """
        - มะละกอดิบ 2 ถ้วย
        - มะเขือเทศ 3 ผล
        - ถั่วฝักยาว 5 ฝัก
        - กุ้งแห้ง 2 ช้อนโต๊ะ
        - ถั่วลิสง 2 ช้อนโต๊ะ
        - พริกขี้หนู 3 เม็ด
        - กระเทียม 3 กลีบ
        - น้ำปลา 2 ช้อนโต๊ะ
        - น้ำตาลปึก 2 ช้อนโต๊ะ
        - มะนาว 2 ผล
        """,
        
        "แกงเขียวหวาน": """
        - ไก่ 300 กรัม
        - กะทิ 1 ถ้วย
        - น้ำพริกแกงเขียวหวาน 3 ช้อนโต๊ะ
        - มะเขือพวง 10 ผล
        - ใบมะกรูด 5 ใบ
        - ใบโหระพา 1/2 ถ้วย
        - น้ำปลา 2 ช้อนโต๊ะ
        - น้ำตาลปึก 1 ช้อนโต๊ะ
        """
    }
    
    analyzer = NutritionAnalyzer()
    results = []
    
    for recipe_name, ingredients in recipes.items():
        print(f"\n📋 วิเคราะห์: {recipe_name}")
        
        nutrition_data = analyzer.analyze_ingredients(ingredients)
        total_nutrition = analyzer.calculate_total_nutrition(nutrition_data)
        
        result = {
            'name': recipe_name,
            'calories': total_nutrition.calories,
            'protein': total_nutrition.protein,
            'carbs': total_nutrition.carbs,
            'fat': total_nutrition.fat,
            'fiber': total_nutrition.fiber,
            'sodium': total_nutrition.sodium
        }
        results.append(result)
        
        print(f"  ⚡ {total_nutrition.calories:.0f} แคลอรี่")
        print(f"  🥩 {total_nutrition.protein:.1f}g โปรตีน")
        print(f"  🍞 {total_nutrition.carbs:.1f}g คาร์บ")
        print(f"  🥑 {total_nutrition.fat:.1f}g ไขมัน")
    
    # เปรียบเทียบเมนู
    print(f"\n📊 เปรียบเทียบเมนู:")
    df = pd.DataFrame(results)
    print(df.to_string(index=False, float_format='%.1f'))
    
    # หาเมนูที่ดีที่สุดในแต่ละหมวด
    print(f"\n🏆 เมนูแนะนำ:")
    print(f"  🔥 แคลอรี่ต่ำสุด: {df.loc[df['calories'].idxmin(), 'name']}")
    print(f"  💪 โปรตีนสูงสุด: {df.loc[df['protein'].idxmax(), 'name']}")
    print(f"  🌾 ใยอาหารสูงสุด: {df.loc[df['fiber'].idxmax(), 'name']}")

def example_nutrition_filtering():
    """ตัวอย่างการกรองเมนูตามเกณฑ์โภชนาการ"""
    print("\n\n🔍 ตัวอย่างการกรองตามเกณฑ์โภชนาการ")
    print("=" * 50)
    
    # สร้างข้อมูลตัวอย่าง
    sample_recipes = pd.DataFrame([
        {'name': 'ผัดกะเพรา', 'calories': 420, 'protein': 25.0, 'carbs': 15.0, 'fat': 28.0},
        {'name': 'ส้มตำ', 'calories': 180, 'protein': 8.0, 'carbs': 32.0, 'fat': 3.0},
        {'name': 'แกงเขียวหวาน', 'calories': 350, 'protein': 22.0, 'carbs': 12.0, 'fat': 25.0},
        {'name': 'ลาบหมู', 'calories': 280, 'protein': 28.0, 'carbs': 8.0, 'fat': 15.0},
        {'name': 'ยำวุ้นเส้น', 'calories': 150, 'protein': 5.0, 'carbs': 28.0, 'fat': 2.0}
    ])
    
    print("📋 เมนูทั้งหมด:")
    print(sample_recipes.to_string(index=False))
    
    # กรองตามเกณฑ์ต่างๆ
    filters = [
        {'name': 'เมนูโปรตีนสูง (>20g)', 'condition': sample_recipes['protein'] > 20},
        {'name': 'เมนูแคลอรี่ต่ำ (<300)', 'condition': sample_recipes['calories'] < 300},
        {'name': 'เมนูไขมันต่ำ (<10g)', 'condition': sample_recipes['fat'] < 10},
    ]
    
    for filter_info in filters:
        filtered_recipes = sample_recipes[filter_info['condition']]
        print(f"\n🔍 {filter_info['name']}:")
        if len(filtered_recipes) > 0:
            print(filtered_recipes[['name', 'calories', 'protein', 'fat']].to_string(index=False))
        else:
            print("  ไม่มีเมนูที่ตรงเกณฑ์")

def example_daily_nutrition_tracking():
    """ตัวอย่างการติดตามโภชนาการรายวัน"""
    print("\n\n🔍 ตัวอย่างการติดตามโภชนาการรายวัน")
    print("=" * 50)
    
    # เมนูในหนึ่งวัน
    daily_meals = {
        'เช้า': {'name': 'ข้าวต้มกุ้ง', 'calories': 280, 'protein': 15, 'carbs': 45, 'fat': 5},
        'กลางวัน': {'name': 'ผัดกะเพรา + ข้าว', 'calories': 520, 'protein': 28, 'carbs': 58, 'fat': 22},
        'เย็น': {'name': 'แกงเขียวหวาน + ข้าว', 'calories': 450, 'protein': 25, 'carbs': 52, 'fat': 18},
        'ว่าง': {'name': 'มะม่วงข้าวเหนียว', 'calories': 320, 'protein': 4, 'carbs': 68, 'fat': 8}
    }
    
    print("🍽️  เมนูในหนึ่งวัน:")
    total_calories = 0
    total_protein = 0
    total_carbs = 0
    total_fat = 0
    
    for meal_time, meal_data in daily_meals.items():
        print(f"  {meal_time}: {meal_data['name']}")
        print(f"    ⚡ {meal_data['calories']} แคลอรี่")
        print(f"    🥩 {meal_data['protein']}g โปรตีน")
        print(f"    🍞 {meal_data['carbs']}g คาร์บ")
        print(f"    🥑 {meal_data['fat']}g ไขมัน\n")
        
        total_calories += meal_data['calories']
        total_protein += meal_data['protein']
        total_carbs += meal_data['carbs']
        total_fat += meal_data['fat']
    
    print(f"📊 สรุปโภชนาการทั้งวัน:")
    print(f"  ⚡ รวมแคลอรี่: {total_calories} แคลอรี่")
    print(f"  🥩 รวมโปรตีน: {total_protein} กรัม")
    print(f"  🍞 รวมคาร์บ: {total_carbs} กรัม")
    print(f"  🥑 รวมไขมัน: {total_fat} กรัม")
    
    # เปรียบเทียบกับค่าแนะนำ
    recommended = Config.DAILY_RECOMMENDED
    print(f"\n🎯 เปรียบเทียบกับค่าแนะนำ (ผู้ชายผู้ใหญ่):")
    print(f"  ⚡ แคลอรี่: {total_calories}/{recommended['calories_adult_male']} ({(total_calories/recommended['calories_adult_male']*100):.1f}%)")
    print(f"  🥩 โปรตีน: {total_protein}/{recommended['protein_adult_male']} ({(total_protein/recommended['protein_adult_male']*100):.1f}%)")
    print(f"  🍞 คาร์บ: {total_carbs}/{recommended['carbs_adult']} ({(total_carbs/recommended['carbs_adult']*100):.1f}%)")
    print(f"  🥑 ไขมัน: {total_fat}/{recommended['fat_adult']} ({(total_fat/recommended['fat_adult']*100):.1f}%)")

def example_export_nutrition_data():
    """ตัวอย่างการส่งออกข้อมูลโภชนาการ"""
    print("\n\n🔍 ตัวอย่างการส่งออกข้อมูล")
    print("=" * 50)
    
    analyzer = NutritionAnalyzer()
    
    # วิเคราะห์เมนูหลายอย่าง
    recipes_data = []
    
    sample_ingredients = {
        "ต้มยำกุ้ง": "กุ้ง, เห็ด, ตะไคร้, ใบมะกรูด, พริกขี้หนู, น้ำปลา",
        "ผัดไทย": "เส้นก๋วยเตี๋ยว, กุ้ง, ไข่, ถั่วงอก, น้ำปลา, น้ำตาล",
        "มัสมั่นไก่": "ไก่, กะทิ, น้ำพริกมัสมั่น, มันฝรั่ง, หอมใหญ่"
    }
    
    for recipe_name, ingredients in sample_ingredients.items():
        # แปลงเป็นรูปแบบที่ analyzer ต้องการ
        formatted_ingredients = "\n".join([f"- {ing.strip()}" for ing in ingredients.split(",")])
        
        nutrition_data = analyzer.analyze_ingredients(formatted_ingredients)
        total_nutrition = analyzer.calculate_total_nutrition(nutrition_data)
        
        recipe_data = {
            'recipe_name': recipe_name,
            'ingredients': ingredients,
            'calories': round(total_nutrition.calories, 1),
            'protein': round(total_nutrition.protein, 1),
            'carbs': round(total_nutrition.carbs, 1),
            'fat': round(total_nutrition.fat, 1),
            'fiber': round(total_nutrition.fiber, 1),
            'sodium': round(total_nutrition.sodium, 1),
            'vitamin_c': round(total_nutrition.vitamin_c, 1),
            'calcium': round(total_nutrition.calcium, 1),
            'iron': round(total_nutrition.iron, 1),
            'analysis_date': '2024-01-15'
        }
        recipes_data.append(recipe_data)
    
    # บันทึกเป็น JSON
    output_file = 'nutrition_analysis_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(recipes_data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 บันทึกข้อมูลโภชนาการลงไฟล์: {output_file}")
    
    # แสดงตัวอย่างข้อมูล
    print(f"\n📋 ตัวอย่างข้อมูลที่ส่งออก:")
    for recipe in recipes_data:
        print(f"  🍽️  {recipe['recipe_name']}:")
        print(f"    ⚡ {recipe['calories']} แคลอรี่")
        print(f"    🥩 {recipe['protein']}g โปรตีน")
        print(f"    🍞 {recipe['carbs']}g คาร์บ")
        print(f"    🥑 {recipe['fat']}g ไขมัน")
        print()

def main():
    """ฟังก์ชันหลักสำหรับรันตัวอย่างทั้งหมด"""
    print("🍲 Thai Food Nutrition Analysis Examples")
    print("=" * 60)
    print("🚀 เริ่มต้นตัวอย่างการใช้งานระบบวิเคราะห์โภชนาการ")
    
    try:
        # รันตัวอย่างต่างๆ
        example_basic_usage()
        example_batch_analysis()
        example_nutrition_filtering()
        example_daily_nutrition_tracking()
        example_export_nutrition_data()
        
        print("\n" + "=" * 60)
        print("✅ เสร็จสิ้นการทำงานของตัวอย่างทั้งหมด!")
        print("\n💡 เคล็ดลับการใช้งาน:")
        print("  1. ใช้ streamlit run streamlit_app.py เพื่อเริ่มแอปพลิเคชัน")
        print("  2. ใช้การค้นหาตามโภชนาการใน sidebar")
        print("  3. ดูรายละเอียดโภชนาการแต่ละเมนูได้")
        print("  4. เปรียบเทียบค่าโภชนาการระหว่างเมนูต่างๆ")
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        print("💡 ตรวจสอบการติดตั้ง dependencies และไฟล์ที่จำเป็น")

if __name__ == "__main__":
    main()
