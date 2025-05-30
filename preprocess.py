import pandas as pd
import re
import os
import argparse
import json
import sqlite3
from nutrition_analyzer import NutritionAnalyzer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text(text):
    """ทำความสะอาดและจัดรูปแบบข้อความ"""
    if not isinstance(text, str):
        return ""
    
    # ลบช่องว่างเกิน
    text = re.sub(r'\s+', ' ', text)
    
    # ลบอักขระพิเศษ ยกเว้นอักษรไทย ตัวเลข และเครื่องหมายพื้นฐาน
    text = re.sub(r'[^\u0E00-\u0E7Fa-zA-Z0-9\s.,\-\(\)]', '', text)
    
    return text.strip()

def preprocess_ingredients(text):
    """จัดรูปแบบรายการวัตถุดิบ"""
    if not isinstance(text, str):
        return ""
    
    # ให้แน่ใจว่าแต่ละวัตถุดิบขึ้นบรรทัดใหม่และขึ้นต้นด้วยขีด
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # ให้แน่ใจว่าแต่ละบรรทัดขึ้นต้นด้วยขีด
        if not line.startswith('-'):
            line = f"- {line}"
        
        formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

def extract_main_ingredients(ingredients_text):
    """แยกวัตถุดิบหลักสำหรับการวิเคราะห์โภชนาการ"""
    if not isinstance(ingredients_text, str):
        return []
    
    lines = ingredients_text.strip().split('\n')
    main_ingredients = []
    
    for line in lines:
        line = line.strip()
        if line and line.startswith('-'):
            # ลบขีดและทำความสะอาด
            ingredient = line[1:].strip()
            
            # ลบปริมาณและหน่วย
            ingredient = re.sub(r'\d+[\s]*[กชฟผลถ้วยช้อนกิโลกรัมกลีบใบเม็ดตัวคู่ฝักซีกแว่นราก].*', '', ingredient)
            ingredient = re.sub(r'\([^)]*\)', '', ingredient)  # ลบข้อความในวงเล็บ
            ingredient = re.sub(r'\d+', '', ingredient)  # ลบตัวเลขที่เหลือ
            
            # ทำความสะอาดคำบรรยายทั่วไป
            unwanted_words = ['ขนาด', 'กลาง', 'เล็ก', 'ใหญ่', 'สด', 'แห้ง', 'ต้ม', 'ผ่า', 'หั่น', 'สับ', 'ปอก']
            for word in unwanted_words:
                ingredient = ingredient.replace(word, '')
            
            ingredient = ingredient.strip()
            if ingredient and len(ingredient) > 1:
                main_ingredients.append(ingredient)
    
    return main_ingredients

def analyze_recipe_nutrition(ingredients_text, nutrition_analyzer):
    """วิเคราะห์โภชนาการสำหรับสูตรอาหาร"""
    try:
        nutrition_data = nutrition_analyzer.analyze_ingredients(ingredients_text)
        total_nutrition = nutrition_analyzer.calculate_total_nutrition(nutrition_data)
        
        # คืนค่าตัวชี้วัดโภชนาการสำคัญ
        return {
            'calories': round(total_nutrition.calories, 1),
            'protein': round(total_nutrition.protein, 1),
            'carbs': round(total_nutrition.carbs, 1),
            'fat': round(total_nutrition.fat, 1),
            'fiber': round(total_nutrition.fiber, 1),
            'sodium': round(total_nutrition.sodium, 1),
            'vitamin_c': round(total_nutrition.vitamin_c, 1),
            'calcium': round(total_nutrition.calcium, 1),
            'iron': round(total_nutrition.iron, 1),
            'main_ingredients': extract_main_ingredients(ingredients_text)
        }
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการวิเคราะห์โภชนาการ: {e}")
        return {
            'calories': 0,
            'protein': 0,
            'carbs': 0,
            'fat': 0,
            'fiber': 0,
            'sodium': 0,
            'vitamin_c': 0,
            'calcium': 0,
            'iron': 0,
            'main_ingredients': extract_main_ingredients(ingredients_text)
        }

def create_nutrition_summary(df):
    """สร้างสถิติสรุปโภชนาการ"""
    if 'nutrition_calories' not in df.columns:
        return {}
    
    summary = {
        'total_recipes': len(df),
        'avg_calories': df['nutrition_calories'].mean(),
        'avg_protein': df['nutrition_protein'].mean(),
        'avg_carbs': df['nutrition_carbs'].mean(),
        'avg_fat': df['nutrition_fat'].mean(),
        'high_protein_recipes': len(df[df['nutrition_protein'] > 20]),
        'low_calorie_recipes': len(df[df['nutrition_calories'] < 300]),
        'high_fiber_recipes': len(df[df['nutrition_fiber'] > 5]),
        'most_common_ingredients': []
    }
    
    # หาวัตถุดิบที่ใช้บ่อยที่สุด
    all_ingredients = []
    for ingredients_list in df['nutrition_main_ingredients']:
        if isinstance(ingredients_list, list):
            all_ingredients.extend(ingredients_list)
    
    from collections import Counter
    ingredient_counts = Counter(all_ingredients)
    summary['most_common_ingredients'] = ingredient_counts.most_common(10)
    
    return summary

def preprocess_data(input_file, output_file, analyze_nutrition=True, usda_api_key=None):
    """ประมวลผลชุดข้อมูลอาหารไทยพร้อมการวิเคราะห์โภชนาการ"""
    # ตรวจสอบว่าไฟล์ input มีอยู่หรือไม่
    if not os.path.exists(input_file):
        print(f"ข้อผิดพลาด: ไม่พบไฟล์ input '{input_file}'")
        return False
    
    try:
        # อ่านไฟล์ CSV
        df = pd.read_csv(input_file)
        
        # ตรวจสอบคอลัมน์ที่จำเป็น
        required_columns = ['name', 'ingredient', 'method']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"ข้อผิดพลาด: ขาดคอลัมน์ที่จำเป็น: {', '.join(missing_columns)}")
            return False
        
        print(f"กำลังประมวลผล {len(df)} สูตรอาหาร...")
        
        # ทำความสะอาดข้อความในแต่ละคอลัมน์
        df['name'] = df['name'].apply(clean_text)
        df['method'] = df['method'].apply(clean_text)
        df['ingredient'] = df['ingredient'].apply(preprocess_ingredients)
        
        # ลบข้อมูลซ้ำ
        df = df.drop_duplicates(subset=['name'])
        
        # วิเคราะห์โภชนาการถ้าเปิดใช้งาน
        if analyze_nutrition:
            print("กำลังเริ่มต้นตัววิเคราะห์โภชนาการ...")
            nutrition_analyzer = NutritionAnalyzer(usda_api_key)
            
            print("กำลังวิเคราะห์โภชนาการสำหรับสูตรอาหาร...")
            nutrition_results = []
            
            for idx, row in df.iterrows():
                print(f"กำลังวิเคราะห์โภชนาการสำหรับสูตรที่ {idx + 1}/{len(df)}: {row['name']}")
                nutrition_data = analyze_recipe_nutrition(row['ingredient'], nutrition_analyzer)
                nutrition_results.append(nutrition_data)
            
            # เพิ่มคอลัมน์โภชนาการเข้าไปใน dataframe
            nutrition_df = pd.DataFrame(nutrition_results)
            
            # ใส่คำนำหน้าคอลัมน์โภชนาการ
            nutrition_df.columns = ['nutrition_' + col for col in nutrition_df.columns]
            
            # รวมกับ dataframe เดิม
            df = pd.concat([df, nutrition_df], axis=1)
            
            # สร้างสรุปโภชนาการ
            summary = create_nutrition_summary(df)
            
            # บันทึกสรุป
            summary_file = output_file.replace('.csv', '_nutrition_summary.json')
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            print(f"บันทึกสรุปโภชนาการไปที่: {summary_file}")
            print(f"แคลอรี่เฉลี่ยต่อสูตร: {summary['avg_calories']:.1f}")
            print(f"สูตรโปรตีนสูง (>20g): {summary['high_protein_recipes']} สูตร")
            print(f"สูตรแคลอรี่ต่ำ (<300 cal): {summary['low_calorie_recipes']} สูตร")
            
            # แสดงวัตถุดิบที่ใช้บ่อยที่สุด
            if summary['most_common_ingredients']:
                print("\nวัตถุดิบที่ใช้บ่อยที่สุด:")
                for ingredient, count in summary['most_common_ingredients'][:5]:
                    print(f"  - {ingredient}: {count} สูตร")
        
        # รีเซ็ต index
        df = df.reset_index(drop=True)
        
        # บันทึกข้อมูลที่ประมวลผลแล้ว
        df.to_csv(output_file, index=False)
        
        print(f"การประมวลผลเสร็จสิ้น บันทึกไปที่ '{output_file}'")
        print(f"จำนวนสูตรทั้งหมด: {len(df)} สูตร")
        
        if analyze_nutrition:
            print(f"การวิเคราะห์โภชนาการเสร็จสิ้นสำหรับ {len(df)} สูตร")
        
        # หากมีไฟล์ embeddings ให้ลบออกเพื่อให้สร้างใหม่
        embeddings_files = ['embeddings.pkl', 'model']
        for file_path in embeddings_files:
            if os.path.exists(file_path):
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    import shutil
                    shutil.rmtree(file_path)
                print(f"ลบ {file_path} แล้ว จะสร้างใหม่เมื่อรันแอป")
        
        return True
    
    except Exception as e:
        print(f"เกิดข้อผิดพลาดระหว่างการประมวลผล: {str(e)}")
        return False

def create_nutrition_database():
    """สร้างและเติมข้อมูลฐานข้อมูลโภชนาการ"""
    analyzer = NutritionAnalyzer()
    
    # เพิ่มวัตถุดิบไทยบางส่วนเข้าฐานข้อมูล
    thai_ingredients = [
        "หมู", "ไก่", "เนื้อ", "กุ้ง", "ปลา", "กะหล่ำปลี", "คะน้า", 
        "ผักบุ้ง", "น้ำปลา", "กะทิ", "น้ำตาล", "ข้าว", "แป้ง"
    ]
    
    print("กำลังสร้างฐานข้อมูลโภชนาการ...")
    for ingredient in thai_ingredients:
        nutrition = analyzer.get_ingredient_nutrition(ingredient)
        if nutrition:
            print(f"เพิ่มข้อมูลโภชนาการสำหรับ: {ingredient}")
    
    print("สร้างฐานข้อมูลโภชนาการเสร็จสิ้น!")

def main():
    parser = argparse.ArgumentParser(description='ประมวลผลข้อมูลสูตรอาหารไทยพร้อมการวิเคราะห์โภชนาการ')
    parser.add_argument('--input', type=str, default='thai_food_raw.csv', 
                        help='เส้นทางไฟล์ CSV input')
    parser.add_argument('--output', type=str, default='thai_food_processed.csv', 
                        help='เส้นทางไฟล์ CSV output')
    parser.add_argument('--analyze-nutrition', action='store_true', default=True,
                        help='วิเคราะห์ข้อมูลโภชนาการสำหรับสูตรอาหาร')
    parser.add_argument('--usda-api-key', type=str, 
                        help='USDA API key สำหรับข้อมูลโภชนาการ')
    parser.add_argument('--create-nutrition-db', action='store_true',
                        help='สร้างฐานข้อมูลโภชนาการพร้อมวัตถุดิบไทย')
    
    args = parser.parse_args()
    
    if args.create_nutrition_db:
        create_nutrition_database()
        return
    
    success = preprocess_data(
        args.input, 
        args.output, 
        args.analyze_nutrition,
        args.usda_api_key
    )
    
    if success:
        print("\n✅ การประมวลผลเสร็จสิ้นเรียบร้อย!")
        print("\nขั้นตอนถัดไป:")
        print("1. รันคำสั่ง 'streamlit run streamlit_app.py' เพื่อเริ่มแชทบอท")
        print("2. แอปจะรวมการวิเคราะห์โภชนาการสำหรับสูตรอาหารทั้งหมด")
        print("3. ใช้แถบด้านข้างเพื่อค้นหาสูตรตามเกณฑ์โภชนาการ")
    else:
        print("❌ การประมวลผลล้มเหลว กรุณาตรวจสอบข้อความแสดงข้อผิดพลาดข้างต้น")

if __name__ == "__main__":
    main()
