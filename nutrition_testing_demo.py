#!/usr/bin/env python3
"""
ไฟล์ทดสอบและแสดงตัวอย่างการใช้งาน ImprovedNutritionAPI
"""

import sys
import os
from improved_nutrition_api import ImprovedNutritionAPI

def test_basic_nutrition_calculation():
    """ทดสอบการคำนวณโภชนาการพื้นฐาน"""
    print("🧪 ทดสอบการคำนวณโภชนาการพื้นฐาน")
    print("=" * 50)
    
    nutrition_api = ImprovedNutritionAPI()
    
    # ตัวอย่างวัตถุดิบง่ายๆ
    ingredients = """- ไข่ไก่ 2 ฟอง
- น้ำปลา 1 ช้อนชา
- ต้นหอม 2 ต้น"""
    
    result = nutrition_api.calculate_recipe_nutrition(
        ingredients_text=ingredients,
        use_api=False,
        adjust_consumption=False,
        enhance_missing=False,
        recipe_name="ไข่เจียวง่ายๆ",
        method_text="ตอกไข่ ใส่น้ำปลา ทอดในกะทะ"
    )
    
    print("📋 วัตถุดิบ:")
    for line in ingredients.split('\n'):
        if line.strip():
            print(f"  {line}")
    
    print("\n📊 ผลการคำนวณ:")
    nutrition = result['total_nutrition']
    print(f"  🔥 แคลอรี่: {nutrition['calories']:.1f} kcal")
    print(f"  💪 โปรตีน: {nutrition['protein']:.1f} g")
    print(f"  🍞 คาร์โบไฮเดรต: {nutrition['carbs']:.1f} g")
    print(f"  🫒 ไขมัน: {nutrition['fat']:.1f} g")
    print(f"  🧂 โซเดียม: {nutrition['sodium']:.1f} mg")
    
    print("\n✅ การทดสอบพื้นฐานเสร็จสิ้น\n")
    return result

def test_consumption_adjustment():
    """ทดสอบการปรับการบริโภคตามความเป็นจริง"""
    print("🔄 ทดสอบการปรับการบริโภคตามความเป็นจริง")
    print("=" * 50)
    
    nutrition_api = ImprovedNutritionAPI()
    
    # เมนูที่มีน้ำมันทอด
    ingredients = """- ปลาช่อน 1 ตัว
- แป้งสาลี 3 ช้อนโต๊ะ
- น้ำมันพืช 1 ถ้วย"""
    
    recipe_name = "ปลาทอด"
    method = "คลุกปลาด้วยแป้ง ทอดในน้ำมันร้อนจนกรอบ"
    
    # ทดสอบแบบไม่ปรับ
    result_no_adjust = nutrition_api.calculate_recipe_nutrition(
        ingredients_text=ingredients,
        adjust_consumption=False,
        recipe_name=recipe_name,
        method_text=method
    )
    
    # ทดสอบแบบปรับ
    result_adjusted = nutrition_api.calculate_recipe_nutrition(
        ingredients_text=ingredients,
        adjust_consumption=True,
        recipe_name=recipe_name,
        method_text=method
    )
    
    print("📋 วัตถุดิบ:")
    for line in ingredients.split('\n'):
        if line.strip():
            print(f"  {line}")
    
    print("\n📊 เปรียบเทียบผลการคำนวณ:")
    print(f"{'':20} {'ไม่ปรับ':>15} {'ปรับแล้ว':>15} {'ผลต่าง':>15}")
    print("-" * 65)
    
    nutrients = ['calories', 'fat', 'sodium']
    units = ['kcal', 'g', 'mg']
    
    for nutrient, unit in zip(nutrients, units):
        no_adj = result_no_adjust['total_nutrition'][nutrient]
        adjusted = result_adjusted['total_nutrition'][nutrient]
        diff = no_adj - adjusted
        
        print(f"{nutrient:20} {no_adj:>10.1f} {unit:>4} {adjusted:>10.1f} {unit:>4} {diff:>10.1f} {unit:>4}")
    
    print("\n💡 สังเกต: น้ำมันที่ใช้ทอดจะคำนวณเพียง 25% เพราะส่วนใหญ่เหลือในกะทะ")
    print("✅ การทดสอบการปรับการบริโภคเสร็จสิ้น\n")
    
    return result_no_adjust, result_adjusted

def test_missing_ingredients_enhancement():
    """ทดสอบการเพิ่มวัตถุดิบที่ขาดหาย"""
    print("➕ ทดสอบการเพิ่มวัตถุดิบที่ขาดหาย")
    print("=" * 50)
    
    nutrition_api = ImprovedNutritionAPI()
    
    # สูตรที่ขาดวัตถุดิบ
    incomplete_ingredients = """- เนื้อหมูสับ 200 กรัม
- ใบกะเพรา 1 ถ้วย
- พริกขี้หนู 5 เม็ด"""
    
    recipe_name = "ผัดกะเพราหมู"
    method = "ผัดหมูสับให้สุก ใส่พริกและกะเพรา ปรุงรสแล้วเสิร์ฟ"
    
    # ทดสอบแบบไม่เพิ่ม
    result_original = nutrition_api.calculate_recipe_nutrition(
        ingredients_text=incomplete_ingredients,
        enhance_missing=False,
        recipe_name=recipe_name,
        method_text=method
    )
    
    # ทดสอบแบบเพิ่ม
    result_enhanced = nutrition_api.calculate_recipe_nutrition(
        ingredients_text=incomplete_ingredients,
        enhance_missing=True,
        recipe_name=recipe_name,
        method_text=method
    )
    
    print("📋 วัตถุดิบเดิม:")
    for line in incomplete_ingredients.split('\n'):
        if line.strip():
            print(f"  {line}")
    
    print("\n📋 วัตถุดิบหลังเพิ่มเติม:")
    enhanced_ingredients = result_enhanced.get('enhanced_ingredients', incomplete_ingredients)
    for line in enhanced_ingredients.split('\n'):
        if line.strip():
            print(f"  {line}")
    
    print("\n📊 เปรียบเทียบค่าโภชนาการ:")
    print(f"{'':20} {'เดิม':>15} {'เพิ่มแล้ว':>15} {'เพิ่มขึ้น':>15}")
    print("-" * 65)
    
    nutrients = ['calories', 'fat', 'sodium', 'vitamin_c']
    units = ['kcal', 'g', 'mg', 'mg']
    
    for nutrient, unit in zip(nutrients, units):
        original = result_original['total_nutrition'][nutrient]
        enhanced = result_enhanced['total_nutrition'][nutrient]
        increase = enhanced - original
        
        print(f"{nutrient:20} {original:>10.1f} {unit:>4} {enhanced:>10.1f} {unit:>4} {increase:>+10.1f} {unit:>4}")
    
    # แสดงวัตถุดิบที่เพิ่ม
    original_lines = set(incomplete_ingredients.split('\n'))
    enhanced_lines = set(enhanced_ingredients.split('\n'))
    added_lines = enhanced_lines - original_lines
    
    if added_lines:
        print("\n➕ วัตถุดิบที่เพิ่มเติม:")
        for line in added_lines:
            if line.strip():
                print(f"  {line}")
    
    print("\n✅ การทดสอบการเพิ่มวัตถุดิบเสร็จสิ้น\n")
    
    return result_original, result_enhanced

def test_unit_conversion():
    """ทดสอบการแปลงหน่วย"""
    print("🔢 ทดสอบการแปลงหน่วย")
    print("=" * 50)
    
    nutrition_api = ImprovedNutritionAPI()
    
    test_cases = [
        ("น้ำมันพืช 2 ช้อนโต๊ะ", "น้ำมันพืช"),
        ("กุ้งนาง 150 กรัม", "กุ้ง"),
        ("ไข่ไก่ 3 ฟอง", "ไข่ไก่"),
        ("กะทิ 1 ถ้วย", "กะทิ"),
        ("น้ำปลา 2 ช้อนชา", "น้ำปลา"),
        ("กระเทียม 4 กลีบ", "กระเทียม"),
        ("มะนาว 2 ผล", "มะนาว")
    ]
    
    print("📏 การแปลงหน่วยต่างๆ:")
    print(f"{'วัตถุดิบ':25} {'ปริมาณเดิม':15} {'น้ำหนัก (กรัม)':15}")
    print("-" * 55)
    
    for ingredient_text, ingredient_name in test_cases:
        quantity, unit, name = nutrition_api.extract_quantity_and_unit(ingredient_text)
        grams = nutrition_api.convert_to_grams(quantity, unit, ingredient_name)
        
        print(f"{ingredient_text:25} {quantity} {unit:13} {grams:>10.1f} g")
    
    print("\n✅ การทดสอบการแปลงหน่วยเสร็จสิ้น\n")

def test_recipe_name_detection():
    """ทดสอบการตรวจจับวัตถุดิบตามชื่อเมนู"""
    print("🍽️ ทดสอบการตรวจจับวัตถุดิบตามชื่อเมนู")
    print("=" * 50)
    
    nutrition_api = ImprovedNutritionAPI()
    
    test_recipes = [
        ("ไข่เจียว", "- ไข่ไก่ 2 ฟอง\n- ต้นหอม 1 ต้น", "ตอกไข่ ผัดในกะทะ"),
        ("ข้าวผัด", "- ข้าวสวย 2 ถ้วย\n- กุ้งสด 100 กรัม", "ผัดข้าวกับกุ้ง"),
        ("ส้มตำ", "- มะละกอดิบ 1 ถ้วย\n- มะเขือเทศ 2 ผล", "โขลกผักและปรุงรส"),
        ("ปลาทอด", "- ปลาช่อน 1 ตัว", "ทอดปลาในน้ำมันร้อน")
    ]
    
    for recipe_name, ingredients, method in test_recipes:
        print(f"\n🍽️ เมนู: {recipe_name}")
        print("📋 วัตถุดิบเดิม:")
        for line in ingredients.split('\n'):
            if line.strip():
                print(f"  {line}")
        
        enhanced = nutrition_api.enhance_missing_ingredients(ingredients, recipe_name, method)
        
        if enhanced != ingredients:
            print("📋 วัตถุดิบหลังเพิ่มเติม:")
            for line in enhanced.split('\n'):
                if line.strip():
                    print(f"  {line}")
            
            # แสดงสิ่งที่เพิ่ม
            original_lines = set(ingredients.split('\n'))
            enhanced_lines = set(enhanced.split('\n'))
            added_lines = enhanced_lines - original_lines
            
            if added_lines:
                print("➕ เพิ่มเติม:")
                for line in added_lines:
                    if line.strip():
                        print(f"  {line}")
        else:
            print("📋 ไม่มีวัตถุดิบเพิ่มเติม")
    
    print("\n✅ การทดสอบการตรวจจับตามชื่อเมนูเสร็จสิ้น\n")

def test_comprehensive_recipe():
    """ทดสอบแบบครอบคลุมทุกฟีเจอร์"""
    print("🧪 ทดสอบแบบครอบคลุม - ผัดกะเพราหมูสับ")
    print("=" * 60)
    
    nutrition_api = ImprovedNutritionAPI()
    
    # สูตรที่ไม่ครบถ้วน
    ingredients = """- เนื้อหมูสับ 200 กรัม
- ใบกะเพรา 1 ถ้วย
- พริกขี้หนู 3 เม็ด
- กระเทียม 3 กลีบ"""
    
    recipe_name = "ผัดกะเพราหมูสับ"
    method = "ผัดกระเทียมและพริกให้หอม ใส่หมูสับผัดจนสุก ใส่กะเพราและปรุงรส"
    
    # ทดสอบทุกการตั้งค่า
    test_cases = [
        ("พื้นฐาน", False, False),
        ("ปรับการบริโภค", True, False),
        ("เพิ่มวัตถุดิบ", False, True),
        ("ครบถ้วน", True, True)
    ]
    
    results = []
    
    for case_name, adjust_consumption, enhance_missing in test_cases:
        result = nutrition_api.calculate_recipe_nutrition(
            ingredients_text=ingredients,
            adjust_consumption=adjust_consumption,
            enhance_missing=enhance_missing,
            recipe_name=recipe_name,
            method_text=method
        )
        results.append((case_name, result))
    
    print("📋 วัตถุดิบเดิม:")
    for line in ingredients.split('\n'):
        if line.strip():
            print(f"  {line}")
    
    print(f"\n📊 เปรียบเทียบผลลัพธ์ทุกโหมด:")
    print(f"{'สารอาหาร':15}", end="")
    for case_name, _ in results:
        print(f"{case_name:>12}", end="")
    print()
    print("-" * (15 + 12 * len(results)))
    
    nutrients_to_show = [
        ('แคลอรี่', 'calories', 'kcal'),
        ('โปรตีน', 'protein', 'g'),
        ('ไขมัน', 'fat', 'g'),
        ('โซเดียม', 'sodium', 'mg'),
        ('วิตามิน C', 'vitamin_c', 'mg')
    ]
    
    for name, key, unit in nutrients_to_show:
        print(f"{name:15}", end="")
        for _, result in results:
            value = result['total_nutrition'][key]
            print(f"{value:>9.1f} {unit:<2}", end="")
        print()
    
    # แสดงวัตถุดิบที่เพิ่มในโหมดครบถ้วน
    comprehensive_result = results[-1][1]
    enhanced_ingredients = comprehensive_result.get('enhanced_ingredients', ingredients)
    
    if enhanced_ingredients != ingredients:
        print(f"\n📋 วัตถุดิบในโหมดครบถ้วน:")
        for line in enhanced_ingredients.split('\n'):
            if line.strip():
                print(f"  {line}")
    
    print(f"\n📈 สรุปการปรับปรุง:")
    basic_calories = results[0][1]['total_nutrition']['calories']
    comprehensive_calories = results[-1][1]['total_nutrition']['calories']
    calorie_diff = comprehensive_calories - basic_calories
    
    print(f"  🔥 แคลอรี่เพิ่มขึ้น: {calorie_diff:+.1f} kcal ({calorie_diff/basic_calories*100:+.1f}%)")
    
    basic_ingredients = len([line for line in ingredients.split('\n') if line.strip()])
    enhanced_ingredients_count = len([line for line in enhanced_ingredients.split('\n') if line.strip()])
    ingredient_diff = enhanced_ingredients_count - basic_ingredients
    
    print(f"  📋 วัตถุดิบเพิ่มขึ้น: {ingredient_diff} รายการ")
    
    print("\n✅ การทดสอบแบบครอบคลุมเสร็จสิ้น\n")

def show_consumption_ratios():
    """แสดงสัดส่วนการบริโภคที่ใช้ในระบบ"""
    print("📊 สัดส่วนการบริโภคตามความเป็นจริง")
    print("=" * 50)
    
    nutrition_api = ImprovedNutritionAPI()
    
    print("🫒 น้ำมันและไขมัน:")
    oils = [item for item in nutrition_api.consumption_ratio.items() if 'น้ำมัน' in item[0]]
    for ingredient, ratio in oils:
        print(f"  {ingredient:20} {ratio:>6.0%} (เหลือ {1-ratio:.0%} ในกะทะ)")
    
    print(f"\n🥥 ของเหลวในการปรุง:")
    liquids = [item for item in nutrition_api.consumption_ratio.items() 
               if any(keyword in item[0] for keyword in ['กะทิ', 'น้ำซุป', 'น้ำต้ม'])]
    for ingredient, ratio in liquids:
        print(f"  {ingredient:20} {ratio:>6.0%}")
    
    print(f"\n🧂 เครื่องปรุงรส:")
    seasonings = [item for item in nutrition_api.consumption_ratio.items() 
                  if any(keyword in item[0] for keyword in ['น้ำปลา', 'ซีอิ้ว', 'เกลือ', 'น้ำตาล'])]
    for ingredient, ratio in seasonings:
        print(f"  {ingredient:20} {ratio:>6.0%}")
    
    print(f"\n🌿 เครื่องเทศ:")
    spices = [item for item in nutrition_api.consumption_ratio.items() 
              if any(keyword in item[0] for keyword in ['ใบมะกรูด', 'ตะไคร้', 'ข่า'])]
    for ingredient, ratio in spices:
        print(f"  {ingredient:20} {ratio:>6.0%} (ส่วนใหญ่ใช้หอม)")
    
    print("\n💡 หมายเหตุ: วัตถุดิบที่ไม่อยู่ในรายการจะใช้ 100%")
    print("✅ แสดงสัดส่วนการบริโภคเสร็จสิ้น\n")

def main():
    """ฟังก์ชันหลักสำหรับการทดสอบ"""
    print("🍲 Thai Food Recipe Chatbot - Nutrition API Testing")
    print("=" * 60)
    print("📝 ทดสอบและแสดงตัวอย่างการใช้งาน ImprovedNutritionAPI")
    print("=" * 60)
    print()
    
    try:
        # ทดสอบทีละขั้นตอน
        test_basic_nutrition_calculation()
        
        test_unit_conversion()
        
        show_consumption_ratios()
        
        test_consumption_adjustment()
        
        test_recipe_name_detection()
        
        test_missing_ingredients_enhancement()
        
        test_comprehensive_recipe()
        
        print("🎉 การทดสอบทั้งหมดเสร็จสิ้นสมบูรณ์!")
        print("=" * 60)
        print("💡 สามารถนำ ImprovedNutritionAPI ไปใช้งานได้แล้ว")
        
    except ImportError as e:
        print(f"❌ ไม่สามารถ import ไฟล์ได้: {e}")
        print("💡 ตรวจสอบว่าไฟล์ improved_nutrition_api.py อยู่ในโฟลเดอร์เดียวกัน")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการทดสอบ: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
