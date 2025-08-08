#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ตัวอย่างการใช้งานระบบวิเคราะห์คุณค่าทางโภชนาการสำหรับสูตรอาหารไทย
"""

import os
import pandas as pd
from nutrition_analyzer import NutritionAnalyzer
import logging

def setup_logging():
    """ตั้งค่าระบบบันทึกล็อก"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('nutrition_analysis.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def example_single_ingredient_analysis():
    """ตัวอย่างการวิเคราะห์วัตถุดิบเดี่ยว"""
    print("\n=== ตัวอย่างการวิเคราะห์วัตถุดิบเดี่ยว ===")
    
    # สร้างตัววิเคราะห์ (ใช้ demo mode ถ้าไม่มี API key)
    api_key = os.getenv('USDA_API_KEY')
    analyzer = NutritionAnalyzer(api_key=api_key)
    
    # ทดสอบวัตถุดิบต่างๆ
    test_ingredients = ['กุ้ง', 'ปลา', 'หมู', 'ไก่', 'ข้าว']
    
    for ingredient in test_ingredients:
        print(f"\n🔍 กำลังวิเคราะห์: {ingredient}")
        
        # ค้นหาข้อมูลคุณค่าทางโภชนาการ
        food_data = analyzer.search_usda_food(ingredient)
        
        if food_data:
            nutrition = analyzer.extract_nutrition_values(food_data)
            print(f"   พลังงาน: {nutrition.calories:.1f} แคลอรี่")
            print(f"   โปรตีน: {nutrition.protein:.1f} กรัม")
            print(f"   ไขมัน: {nutrition.fat:.1f} กรัม")
            print(f"   แคลเซียม: {nutrition.calcium:.1f} มิลลิกรัม")
        else:
            print("   ❌ ไม่พบข้อมูล")

def example_recipe_analysis():
    """ตัวอย่างการวิเคราะห์สูตรอาหาร"""
    print("\n=== ตัวอย่างการวิเคราะห์สูตรอาหาร ===")
    
    api_key = os.getenv('USDA_API_KEY')
    analyzer = NutritionAnalyzer(api_key=api_key)
    
    # สูตรอาหารตัวอย่าง
    sample_recipe = """- กุ้งนาง 4 ตัว
- พริกไทย 5 เม็ด
- กระเทียมกลีบใหญ่ 2 กลีบ
- รากผักชี 5 ราก
- น้ำปลา 2 ช้อนโต๊ะ
- น้ำมันหมู 1 ช้อนโต๊ะ"""
    
    print("📋 สูตรที่วิเคราะห์: กุ้งทาพริกไทยกระเทียม")
    print(f"🥘 วัตถุดิบ:\n{sample_recipe}")
    
    # วิเคราะห์คุณค่าทางโภชนาการ
    nutrition_analysis = analyzer.analyze_recipe_nutrition(sample_recipe)
    total_nutrition = nutrition_analysis['total_nutrition']
    
    print(f"\n📊 คุณค่าทางโภชนาการรวม (ประมาณการ):")
    print(f"   🔥 พลังงาน: {total_nutrition.calories:.1f} แคลอรี่")
    print(f"   🥩 โปรตีน: {total_nutrition.protein:.1f} กรัม")
    print(f"   🧈 ไขมัน: {total_nutrition.fat:.1f} กรัม")
    print(f"   🦴 แคลเซียม: {total_nutrition.calcium:.1f} มิลลิกรัม")
    print(f"   🩸 เหล็ก: {total_nutrition.iron:.1f} มิลลิกรัม")
    
    # แสดงรายละเอียดวัตถุดิบแต่ละชนิด
    print(f"\n📝 รายละเอียดวัตถุดิบ:")
    for detail in nutrition_analysis['ingredient_details']:
        ingredient = detail['ingredient']
        nutrition = detail['nutrition']
        print(f"   • {ingredient}: {nutrition.calories:.0f} แคลอรี่, {nutrition.protein:.1f}g โปรตีน")

def example_csv_processing():
    """ตัวอย่างการประมวลผลไฟล์ CSV"""
    print("\n=== ตัวอย่างการประมวลผลไฟล์ CSV ===")
    
    # ตรวจสอบว่ามีไฟล์ข้อมูลหรือไม่
    input_file = "thai_food_processed.csv"
    if not os.path.exists(input_file):
        print(f"❌ ไม่พบไฟล์ {input_file}")
        return
    
    api_key = os.getenv('USDA_API_KEY')
    if not api_key:
        print("⚠️  ไม่พบ USDA API key - จะใช้ข้อมูลเริ่มต้น")
    
    analyzer = NutritionAnalyzer(api_key=api_key)
    
    # อ่านไฟล์ข้อมูลเดิมเพื่อแสดงสถิติ
    df = pd.read_csv(input_file)
    print(f"📂 ไฟล์ต้นฉบับ: {len(df)} สูตรอาหาร")
    
    # ประมวลผลไฟล์ (เฉพาะ 5 สูตรแรกเพื่อความรวดเร็ว)
    output_file = "thai_food_sample_nutrition.csv"
    
    # สร้างข้อมูลตัวอย่าง
    sample_df = df.head(5).copy()
    sample_df.to_csv("thai_food_sample.csv", index=False)
    
    print(f"🔄 กำลังประมวลผล {len(sample_df)} สูตรแรก...")
    analyzer.process_csv_file("thai_food_sample.csv", output_file)
    
    # แสดงผลลัพธ์
    if os.path.exists(output_file):
        result_df = pd.read_csv(output_file)
        print(f"✅ สร้างไฟล์ผลลัพธ์: {output_file}")
        
        # แสดงตัวอย่างผลลัพธ์
        print("\n📊 ตัวอย่างผลลัพธ์:")
        for idx, row in result_df.iterrows():
            print(f"   {idx+1}. {row['name']}")
            print(f"      พลังงาน: {row.get('calories', 0):.1f} แคลอรี่")
            print(f"      โปรตีน: {row.get('protein', 0):.1f} กรัม")
            print()

def example_nutrition_filtering():
    """ตัวอย่างการกรองข้อมูลตามคุณค่าทางโภชนาการ"""
    print("\n=== ตัวอย่างการกรองข้อมูลตามคุณค่าทางโภชนาการ ===")
    
    # ตรวจสอบว่ามีไฟล์ข้อมูลที่มีคุณค่าทางโภชนาการหรือไม่
    nutrition_file = "thai_food_with_nutrition.csv"
    sample_file = "thai_food_sample_nutrition.csv"
    
    input_file = nutrition_file if os.path.exists(nutrition_file) else sample_file
    
    if not os.path.exists(input_file):
        print(f"❌ ไม่พบไฟล์ข้อมูลคุณค่าทางโภชนาการ")
        print("   กรุณารันการประมวลผล CSV ก่อน")
        return
    
    df = pd.read_csv(input_file)
    print(f"📂 อ่านข้อมูลจากไฟล์: {input_file}")
    print(f"📊 จำนวนสูตรททั้งหมด: {len(df)}")
    
    # ตัวอย่างการกรองข้อมูล
    filters = [
        ("อาหารโปรตีนสูง (>15g)", lambda row: row.get('protein', 0) > 15),
        ("อาหารแคลอรี่ต่ำ (<200 cal)", lambda row: row.get('calories', 0) < 200),
        ("อาหารไขมันต่ำ (<10g)", lambda row: row.get('fat', 0) < 10),
        ("อาหารอุดมแคลเซียม (>100mg)", lambda row: row.get('calcium', 0) > 100)
    ]
    
    for filter_name, filter_func in filters:
        filtered_df = df[df.apply(filter_func, axis=1)]
        print(f"\n🔍 {filter_name}: {len(filtered_df)} สูตร")
        
        if len(filtered_df) > 0:
            print("   ตัวอย่าง:")
            for idx, (_, row) in enumerate(filtered_df.head(3).iterrows()):
                print(f"   {idx+1}. {row['name']}")
        else:
            print("   ไม่พบสูตรที่ตรงตามเกณฑ์")

def display_nutrition_comparison():
    """แสดงการเปรียบเทียบคุณค่าทางโภชนาการ"""
    print("\n=== การเปรียบเทียบคุณค่าทางโภชนาการ ===")
    
    # ข้อมูลตัวอย่างสำหรับเปรียบเทียบ
    comparison_data = {
        'เมนู': ['ต้มยำกุ้ง', 'แกงเขียวหวาน', 'ผัดไทย', 'ส้มตำ'],
        'แคลอรี่': [95, 185, 255, 45],
        'โปรตีน (g)': [8.5, 12.3, 15.2, 2.1],
        'ไขมัน (g)': [3.2, 8.9, 11.4, 0.8],
        'คาร์โบไฮเดรต (g)': [8.1, 15.6, 35.2, 9.8]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print("📊 ตารางเปรียบเทียบคุณค่าทางโภชนาการ:")
    print(comparison_df.to_string(index=False))
    
    # หาเมนูที่ดีที่สุดในแต่ละหมวด
    best_for_diet = comparison_df.loc[comparison_df['แคลอรี่'].idxmin(), 'เมนู']
    best_for_protein = comparison_df.loc[comparison_df['โปรตีน (g)'].idxmax(), 'เมนู']
    best_for_low_fat = comparison_df.loc[comparison_df['ไขมัน (g)'].idxmin(), 'เมนู']
    
    print(f"\n🏆 เมนูแนะนำ:")
    print(f"   🥗 สำหรับลดน้ำหนัก: {best_for_diet}")
    print(f"   💪 สำหรับเพิ่มกล้ามเนื้อ: {best_for_protein}")
    print(f"   ❤️  สำหรับดูแลหัวใจ: {best_for_low_fat}")

def main():
    """ฟังก์ชันหลักสำหรับรันตัวอย่างทั้งหมด"""
    print("🍲 ระบบแชทบอทสูตรอาหารไทยพร้อมการวิเคราะห์คุณค่าทางโภชนาการ")
    print("=" * 70)
    
    # ตั้งค่าระบบบันทึกล็อก
    setup_logging()
    
    # ตรวจสอบ API key
    api_key = os.getenv('USDA_API_KEY')
    if api_key:
        print("✅ พบ USDA API key - จะใช้ข้อมูลจริงจาก USDA")
    else:
        print("⚠️  ไม่พบ USDA API key - จะใช้ข้อมูลเริ่มต้น")
        print("   ตั้งค่า environment variable: USDA_API_KEY=your_key")
    
    try:
        # รันตัวอย่างต่างๆ
        example_single_ingredient_analysis()
        example_recipe_analysis()
        example_csv_processing()
        example_nutrition_filtering()
        display_nutrition_comparison()
        
        print("\n" + "=" * 70)
        print("✅ เสร็จสิ้นการรันตัวอย่างทั้งหมด!")
        print("\n💡 คำแนะนำ:")
        print("   1. ตั้งค่า USDA API key เพื่อได้ข้อมูลที่แม่นยำ")
        print("   2. รันคำสั่ง 'streamlit run streamlit_app.py' เพื่อใช้งานแชทบอท")
        print("   3. ตรวจสอบไฟล์ 'nutrition_analysis.log' สำหรับรายละเอียด")
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        logging.error(f"ข้อผิดพลาดในการรันตัวอย่าง: {e}")

if __name__ == "__main__":
    main()
