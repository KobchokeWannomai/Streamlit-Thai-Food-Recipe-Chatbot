import pandas as pd
import re
import os
import argparse
import time
from typing import Dict, List, Optional, Tuple
from nutrition_api import NutritionAPI

def clean_text(text):
    """ทำความสะอาดและจัดรูปแบบข้อความภาษาไทยและอังกฤษ"""
    if not isinstance(text, str):
        return ""
    
    # ลบช่องว่างเกิน
    text = re.sub(r'\s+', ' ', text)
    
    # ลบอักขระพิเศษ ยกเว้นตัวอักษรไทย ตัวเลข และเครื่องหมายวรรคตอนพื้นฐาน
    text = re.sub(r'[^\u0E00-\u0E7Fa-zA-Z0-9\s.,\-\(\)\/]', '', text)
    
    # ปรับแต่งข้อความไทยเฉพาะ
    text = text.replace('ๆ', 'ๆ ')  # เพิ่มช่องว่างหลัง ๆ
    text = re.sub(r'\s+', ' ', text)  # ลบช่องว่างเกินอีกครั้ง
    
    return text.strip()

def preprocess_ingredients(text):
    """จัดรูปแบบรายการวัตถุดิบให้เป็นมาตรฐาน"""
    if not isinstance(text, str):
        return ""
    
    # แยกบรรทัดและทำความสะอาด
    lines = text.strip().split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # ลบเครื่องหมายต่างๆ ที่อาจมีอยู่แล้ว
        line = re.sub(r'^[-*•·]\s*', '', line)
        
        # ตรวจสอบว่าเป็นรายการวัตถุดิบหรือไม่
        if line and not line.startswith('-'):
            line = f"- {line}"
        
        formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

def enhance_missing_ingredients(ingredients_text, recipe_name, method_text):
    """เพิ่มวัตถุดิบที่ขาดหายไปตามวิธีการทำและชื่อเมนู"""
    enhanced_ingredients = ingredients_text
    missing_ingredients = []
    
    # รายการวัตถุดิบที่มักขาดหายไปตามวิธีการทำ
    cooking_method_ingredients = {
        'ทอด': {
            'required': ['น้ำมันพืช 3 ช้อนโต๊ะ'],
            'optional': []
        },
        'เจียว': {
            'required': ['น้ำมันหมู 2 ช้อนโต๊ะ'],
            'optional': []
        },
        'ผัด': {
            'required': ['น้ำมันพืช 2 ช้อนโต๊ะ'],
            'optional': ['กระเทียม 2 กลีบ', 'หอมแดง 2 หัว']
        },
        'คั่ว': {
            'required': ['น้ำมันพืช 1 ช้อนโต๊ะ'],
            'optional': []
        },
        'ต้ม': {
            'required': [],
            'optional': ['เกลือ 1 ช้อนชา']
        },
        'แกง': {
            'required': [],
            'optional': ['กะทิ 400 มล', 'น้ำปลา 2 ช้อนโต๊ะ']
        },
        'ย่าง': {
            'required': [],
            'optional': ['น้ำมันพืช 1 ช้อนชา']
        },
        'ปิ้ง': {
            'required': [],
            'optional': ['น้ำมันพืช 1 ช้อนชา']
        },
        'นึ่ง': {
            'required': [],
            'optional': []
        },
        'ยำ': {
            'required': [],
            'optional': ['น้ำปลา 2 ช้อนโต๊ะ', 'มะนาว 2 ผล', 'น้ำตาลปึก 2 ช้อนชา', 'พริกขี้หนู 3 เม็ด']
        }
    }
    
    # รายการวัตถุดิบที่ขาดหายไปตามชื่อเมนู
    recipe_name_ingredients = {
        'ไข่เจียว': ['น้ำมันหมู 2 ช้อนโต๊ะ'],
        'ไข่ดาว': ['น้ำมันหมู 2 ช้อนโต๊ะ'],
        'ไข่ทอด': ['น้ำมันพืช 3 ช้อนโต๊ะ'],
        'ปลาทอด': ['น้ำมันพืช 1 ถ้วย', 'แป้งสาลี 3 ช้อนโต๊ะ'],
        'ข้าวผัด': ['น้ำมันพืช 2 ช้อนโต๊ะ', 'ไข่ไก่ 2 ฟอง'],
        'ผัดไทย': ['น้ำมันพืช 3 ช้อนโต๊ะ', 'ไข่ไก่ 2 ฟอง'],
        'ก๋วยเตี๋ยว': ['น้ำซุป 2 ถ้วย'],
        'ราดหน้า': ['น้ำมันพืช 2 ช้อนโต๊ะ', 'แป้งข้าวโพด 2 ช้อนโต๊ะ']
    }
    
    method_lower = method_text.lower() if method_text else ""
    ingredients_lower = ingredients_text.lower() if ingredients_text else ""
    recipe_name_lower = recipe_name.lower() if recipe_name else ""
    
    # ตรวจสอบวิธีการทำและเพิ่มวัตถุดิบที่ขาดหาย
    for cooking_method, ingredients_dict in cooking_method_ingredients.items():
        if cooking_method in method_lower:
            # วัตถุดิบจำเป็น
            for ingredient in ingredients_dict["required"]:
                ingredient_name = ingredient.split()[0]  # เอาชื่อวัตถุดิบ
                if ingredient_name not in ingredients_lower:
                    missing_ingredients.append(f"- {ingredient}")
            
            # วัตถุดิบเสริม (หากรายการวัตถุดิบน้อย)
            if len(ingredients_text.split('\n')) <= 5:
                for ingredient in ingredients_dict["optional"]:
                    ingredient_name = ingredient.split()[0]
                    if ingredient_name not in ingredients_lower:
                        missing_ingredients.append(f"- {ingredient}")
    
    # ตรวจสอบชื่อเมนูและเพิ่มวัตถุดิบที่ขาดหาย
    for recipe_pattern, required_ingredients in recipe_name_ingredients.items():
        if recipe_pattern in recipe_name_lower:
            for ingredient in required_ingredients:
                ingredient_name = ingredient.split()[0]
                if ingredient_name not in ingredients_lower:
                    missing_ingredients.append(f"- {ingredient}")
    
    # เพิ่มวัตถุดิบที่ขาดหาย
    if missing_ingredients:
        if enhanced_ingredients and not enhanced_ingredients.endswith('\n'):
            enhanced_ingredients += '\n'
        enhanced_ingredients += '\n'.join(missing_ingredients)
        print(f"✓ เพิ่มวัตถุดิบสำหรับ '{recipe_name}': {', '.join([ing.replace('- ', '') for ing in missing_ingredients])}")
    
    return enhanced_ingredients

def calculate_recipe_nutrition(ingredients_text, nutrition_api, enhance_missing=False, 
                             recipe_name="", method_text="", use_api=False, 
                             adjust_consumption=True):
    """คำนวณคุณค่าทางโภชนาการของสูตรอาหารอย่างแม่นยำ"""
    
    # เพิ่มวัตถุดิบที่ขาดหายไป (หากเปิดใช้งาน)
    if enhance_missing:
        ingredients_text = enhance_missing_ingredients(ingredients_text, recipe_name, method_text)
    
    # คำนวณโภชนาการ
    nutrition_data = nutrition_api.calculate_recipe_nutrition(
        ingredients_text, 
        use_api=use_api,
        adjust_consumption=adjust_consumption,
        enhance_missing=False  # เราจัดการแล้วข้างต้น
    )
    
    return nutrition_data['total_nutrition'], ingredients_text

def validate_data_quality(df):
    """ตรวจสอบคุณภาพข้อมูลและให้คำแนะนำ"""
    print("\n📊 การตรวจสอบคุณภาพข้อมูล:")
    
    # ตรวจสอบข้อมูลที่ขาดหาย
    missing_data = {}
    for col in ['name', 'ingredient', 'method']:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            missing_data[col] = missing_count
            print(f"❌ คอลัมน์ '{col}' มีข้อมูลขาดหาย {missing_count} รายการ")
    
    # ตรวจสอบข้อมูลที่ซ้ำกัน
    duplicate_names = df['name'].duplicated().sum()
    if duplicate_names > 0:
        print(f"⚠️  มีชื่อเมนูซ้ำกัน {duplicate_names} รายการ")
    
    # ตรวจสอบความยาวข้อมูล
    short_ingredients = (df['ingredient'].str.len() < 20).sum()
    short_methods = (df['method'].str.len() < 30).sum()
    
    if short_ingredients > 0:
        print(f"⚠️  มีวัตถุดิบที่สั้นเกินไป {short_ingredients} รายการ")
    if short_methods > 0:
        print(f"⚠️  มีวิธีทำที่สั้นเกินไป {short_methods} รายการ")
    
    # ตรวจสอบข้อมูลที่ผิดปกติ
    unusual_chars = 0
    for _, row in df.iterrows():
        text = str(row['name']) + str(row['ingredient']) + str(row['method'])
        if re.search(r'[^\u0E00-\u0E7Fa-zA-Z0-9\s.,\-\(\)\/]', text):
            unusual_chars += 1
    
    if unusual_chars > 0:
        print(f"⚠️  มีข้อมูลที่มีอักขระผิดปกติ {unusual_chars} รายการ")
    
    # สรุปคุณภาพข้อมูล
    quality_score = 100
    if missing_data:
        quality_score -= len(missing_data) * 20
    if duplicate_names > 0:
        quality_score -= min(duplicate_names * 5, 30)
    if short_ingredients > 0:
        quality_score -= min(short_ingredients * 2, 20)
    if short_methods > 0:
        quality_score -= min(short_methods * 2, 20)
    if unusual_chars > 0:
        quality_score -= min(unusual_chars * 1, 10)
    
    quality_score = max(quality_score, 0)
    
    if quality_score >= 90:
        print(f"✅ คุณภาพข้อมูลดีเยี่ยม ({quality_score}/100)")
    elif quality_score >= 70:
        print(f"🟡 คุณภาพข้อมูลดี ({quality_score}/100)")
    elif quality_score >= 50:
        print(f"🟠 คุณภาพข้อมูลปานกลาง ({quality_score}/100)")
    else:
        print(f"🔴 คุณภาพข้อมูลต้องปรับปรุง ({quality_score}/100)")
    
    return quality_score

def preprocess_data(input_file, output_file, add_nutrition=True, enhance_missing=False, 
                   use_api=False, adjust_consumption=True):
    """ประมวลผลข้อมูลอาหารไทยขั้นสูง"""
    
    print(f"🔄 เริ่มต้นการประมวลผลข้อมูลจากไฟล์: {input_file}")
    start_time = time.time()
    
    # ตรวจสอบไฟล์อินพุต
    if not os.path.exists(input_file):
        print(f"❌ ไม่พบไฟล์ '{input_file}'")
        return False
    
    try:
        # อ่านไฟล์ CSV
        print("📖 กำลังอ่านไฟล์ข้อมูล...")
        df = pd.read_csv(input_file, encoding='utf-8')
        print(f"✓ อ่านข้อมูลได้ {len(df)} รายการ")
        
        # ตรวจสอบคอลัมน์ที่จำเป็น
        required_columns = ['name', 'text_ingradiant', 'food_method']
        existing_columns = df.columns.tolist()
        missing_columns = [col for col in required_columns if col not in existing_columns]
        
        if missing_columns:
            print(f"❌ ขาดคอลัมน์ที่จำเป็น: {', '.join(missing_columns)}")
            print(f"📋 คอลัมน์ที่มีอยู่: {', '.join(existing_columns)}")
            return False
        
        # เปลี่ยนชื่อคอลัมน์ให้ถูกต้อง
        df = df.rename(columns={
            'text_ingradiant': 'ingredient',
            'food_method': 'method'
        })
        
        print("🧹 กำลังทำความสะอาดข้อมูล...")
        
        # ทำความสะอาดข้อความในแต่ละคอลัมน์
        df['name'] = df['name'].apply(clean_text)
        df['method'] = df['method'].apply(clean_text)
        df['ingredient'] = df['ingredient'].apply(preprocess_ingredients)
        
        # ลบรายการที่มีข้อมูลไม่ครบ
        initial_count = len(df)
        df = df.dropna(subset=['name', 'ingredient', 'method'])
        df = df[df['name'].str.len() > 0]
        df = df[df['ingredient'].str.len() > 5]
        df = df[df['method'].str.len() > 10]
        
        if len(df) < initial_count:
            print(f"⚠️  ลบข้อมูลที่ไม่ครบถ้วน {initial_count - len(df)} รายการ")
        
        # ตรวจสอบคุณภาพข้อมูล
        quality_score = validate_data_quality(df)
        
        # เพิ่มข้อมูลโภชนาการ (หากต้องการ)
        if add_nutrition:
            print("\n🧮 กำลังคำนวณข้อมูลโภชนาการ...")
            print(f"⚙️  การตั้งค่า: API={use_api}, ปรับการบริโภค={adjust_consumption}, เพิ่มวัตถุดิบ={enhance_missing}")
            
            nutrition_api = NutritionAPI()
            
            # เพิ่มคอลัมน์โภชนาการ
            nutrition_columns = [
                'calories', 'protein', 'carbs', 'fat', 'fiber',
                'vitamin_a', 'vitamin_c', 'vitamin_b1', 'vitamin_b2',
                'calcium', 'iron', 'potassium', 'sodium'
            ]
            
            for col in nutrition_columns:
                df[col] = 0.0
            
            # เพิ่มคอลัมน์สำหรับการติดตาม
            df['nutrition_enhanced'] = False
            df['nutrition_api_used'] = False
            
            # คำนวณโภชนาการสำหรับแต่ละสูตร
            successful_calculations = 0
            failed_calculations = 0
            
            for idx, row in df.iterrows():
                try:
                    # แสดงความคืบหน้า
                    if idx % 50 == 0:
                        progress = (idx / len(df)) * 100
                        print(f"📊 ความคืบหน้า: {progress:.1f}% ({idx+1}/{len(df)})")
                    
                    nutrition, enhanced_ingredients = calculate_recipe_nutrition(
                        row['ingredient'], 
                        nutrition_api, 
                        enhance_missing,
                        row['name'],
                        row['method'],
                        use_api,
                        adjust_consumption
                    )
                    
                    # อัปเดตวัตถุดิบที่ปรับปรุงแล้ว (หากมีการเพิ่ม)
                    if enhanced_ingredients != row['ingredient']:
                        df.at[idx, 'ingredient'] = enhanced_ingredients
                        df.at[idx, 'nutrition_enhanced'] = True
                    
                    # บันทึกข้อมูลโภชนาการ
                    for nutrient, value in nutrition.items():
                        if nutrient in nutrition_columns:
                            df.at[idx, nutrient] = round(value, 2)
                    
                    df.at[idx, 'nutrition_api_used'] = use_api
                    successful_calculations += 1
                        
                except Exception as e:
                    print(f"⚠️  ไม่สามารถคำนวณโภชนาการของ '{row['name']}': {str(e)}")
                    failed_calculations += 1
                    continue
            
            print(f"✅ คำนวณโภชนาการสำเร็จ: {successful_calculations} รายการ")
            if failed_calculations > 0:
                print(f"❌ คำนวณโภชนาการไม่สำเร็จ: {failed_calculations} รายการ")
        
        # ลบรายการที่ซ้ำกัน
        print("\n🔍 กำลังตรวจสอบข้อมูลซ้ำ...")
        initial_count = len(df)
        df = df.drop_duplicates(subset=['name'], keep='first')
        
        if len(df) < initial_count:
            print(f"🗑️  ลบข้อมูลซ้ำ {initial_count - len(df)} รายการ")
        
        # เรียงลำดับข้อมูล
        df = df.sort_values('name').reset_index(drop=True)
        
        # บันทึกข้อมูลที่ประมวลผลแล้ว
        print(f"\n💾 กำลังบันทึกไฟล์: {output_file}")
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        # สรุปผลการประมวลผล
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n🎉 การประมวลผลเสร็จสิ้น!")
        print(f"📁 ไฟล์เอาต์พุต: {output_file}")
        print(f"📊 จำนวนสูตรทั้งหมด: {len(df)} รายการ")
        print(f"⏱️  เวลาที่ใช้: {processing_time:.1f} วินาที")
        
        if add_nutrition:
            # แสดงสถิติโภชนาการ
            print(f"\n📈 สถิติโภชนาการเฉลี่ย:")
            nutrition_summary = {
                'แคลอรี่ (kcal)': df['calories'].mean(),
                'โปรตีน (g)': df['protein'].mean(),
                'คาร์โบไฮเดรต (g)': df['carbs'].mean(),
                'ไขมัน (g)': df['fat'].mean(),
                'ใยอาหาร (g)': df['fiber'].mean(),
                'แคลเซียม (mg)': df['calcium'].mean(),
                'เหล็ก (mg)': df['iron'].mean(),
                'โซเดียม (mg)': df['sodium'].mean()
            }
            
            for nutrient, avg_value in nutrition_summary.items():
                print(f"  {nutrient}: {avg_value:.1f}")
            
            # สถิติการปรับปรุง
            if enhance_missing:
                enhanced_count = df['nutrition_enhanced'].sum()
                print(f"\n🔧 เมนูที่เพิ่มวัตถุดิบ: {enhanced_count} รายการ ({enhanced_count/len(df)*100:.1f}%)")
            
            if use_api:
                api_used_count = df['nutrition_api_used'].sum()
                print(f"🌐 เมนูที่ใช้ข้อมูลจาก API: {api_used_count} รายการ ({api_used_count/len(df)*100:.1f}%)")
        
        # ลบไฟล์ embeddings เก่า เพื่อให้สร้างใหม่
        embeddings_files = ['embeddings.pkl', 'recipe_embeddings.pkl']
        for emb_file in embeddings_files:
            if os.path.exists(emb_file):
                os.remove(emb_file)
                print(f"🗑️  ลบไฟล์ embeddings เก่า: {emb_file}")
        
        print("✨ แอปพลิเคชันจะสร้าง embeddings ใหม่เมื่อเริ่มต้น")
        
        return True
    
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการประมวลผล: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def create_sample_enhanced_data():
    """สร้างข้อมูลตัวอย่างที่มีโภชนาการครอบคลุม"""
    print("🎨 กำลังสร้างข้อมูลตัวอย่างขั้นสูง...")
    
    sample_data = {
        'name': [
            'ไข่เจียว',
            'ผัดกะเพราหมูสับ',
            'ต้มยำกุ้งน้ำใส',
            'แกงเขียวหวานไก่',
            'ยำวุ้นเส้นกุ้งสด',
            'ข้าวผัดกุ้ง',
            'ส้มตำไทย',
            'ลาบหมู',
            'ปลาทอดน้ำปลา',
            'แกงจืดเต้าหู้หมูสับ'
        ],
        'text_ingradiant': [
            'ไข่ไก่ 3 ฟอง\nน้ำปลา 1 ช้อนชา\nต้นหอม 2 ต้น\nผักชี 1 ต้น',
            
            'เนื้อหมูสับ 200 กรัม\nใบกะเพรา 1 ถ้วย\nพริกขี้หนู 5 เม็ด\nกระเทียม 4 กลีบ\nน้ำปลา 2 ช้อนโต๊ะ\nน้ำตาลทราย 1 ช้อนชา\nซีอิ้วขาว 1 ช้อนโต๊ะ',
            
            'กุ้งนาง 300 กรัม\nเห็ดฟาง 100 กรัม\nมะนาว 3 ผล\nใบมะกรูด 5 ใบ\nตะไคร้ 3 ท่อน\nข่า 4 แว่น\nพริกขี้หนู 7 เม็ด\nน้ำปลา 3 ช้อนโต๊ะ\nน้ำตาลปึก 2 ช้อนชา',
            
            'เนื้อไก่ 400 กรัม\nกะทิ 2 ถ้วย\nน้ำพริกแกงเขียวหวาน 3 ช้อนโต๊ะ\nมะเขือเปราะ 8 ผล\nใบโหระพา 1 ถ้วย\nพริกชี้ฟ้าแดง 3 เม็ด\nน้ำปลา 2 ช้อนโต๊ะ\nน้ำตาลปึก 2 ช้อนโต๊ะ',
            
            'วุ้นเส้น 150 กรัม\nกุ้งสด 200 กรัม\nหมูสับ 100 กรัม\nมะนาว 3 ผล\nน้ำปลา 3 ช้อนโต๊ะ\nน้ำตาลปึก 2 ช้อนโต๊ะ\nพริกขี้หนู 5 เม็ด\nกระเทียม 3 กลีบ\nหอมแดง 3 หัว\nผักชี 3 ต้น',
            
            'ข้าวสวย 3 ถ้วย\nกุ้งสด 250 กรัม\nไข่ไก่ 2 ฟอง\nหอมใหญ่ 1 หัว\nกระเทียม 3 กลีบ\nซีอิ้วขาว 2 ช้อนโต๊ะ\nซีอิ้วหวาน 1 ช้อนโต๊ะ\nน้ำตาลทราย 1 ช้อนชา',
            
            'มะละกอดิบ 2 ถ้วย\nมะเขือเทศ 3 ผล\nถั่วฝักยาว 10 เส้น\nกุ้งแห้ง 2 ช้อนโต๊ะ\nถั่วลิสงคั่ว 3 ช้อนโต๊ะ\nพริกขี้หนู 5 เม็ด\nกระเทียม 3 กลีบ\nมะนาว 2 ผล\nน้ำปลา 2 ช้อนโต๊ะ\nน้ำตาลปึก 3 ช้อนโต๊ะ',
            
            'เนื้อหมูสับ 300 กรัม\nข้าวคั่ว 3 ช้อนโต๊ะ\nพริกแห้ง 8 เม็ด\nหอมแดง 5 หัว\nผักชี 5 ต้น\nใบสะระแหน่ 1 ถ้วย\nมะนาว 4 ผล\nน้ำปลา 4 ช้อนโต๊ะ',
            
            'ปลาช่อน 1 ตัว 500 กรัม\nแป้งสาลี 1 ถ้วย\nน้ำปลา 2 ช้อนโต๊ะ\nพริกไทยป่น 1 ช้อนชา\nกระเทียมป่น 1 ช้อนชา',
            
            'เต้าหู้อ่อน 200 กรัม\nหมูสับ 150 กรัม\nต้นหอม 3 ต้น\nผักชี 2 ต้น\nซีอิ้วขาว 1 ช้อนโต๊ะ\nน้ำซุปกระดูกหมู 4 ถ้วย\nพริกไทยป่น 1 ช้อนชา'
        ],
        'food_method': [
            'ตอกไข่ใส่ชาม ใส่น้ำปลา ตีให้เข้ากัน ใส่ต้นหอมและผักชีซอย ตั้งกะทะใส่น้ำมัน พอร้อนเทไข่ลงทอดจนเหลืองทั้งสองด้าน',
            
            'โขลกกระเทียมและพริกขี้หนูให้ละเอียด ตั้งกะทะใส่น้ำมัน ผัดกระเทียมพริกให้หอม ใส่หมูสับผัดจนสุก ใส่น้ำปลา น้ำตาล ซีอิ้วขาว ชิมรส ใส่ใบกะเพราผัดให้เข้ากัน',
            
            'ต้มน้ำให้เดือด ใส่ตะไคร้ ข่า ใบมะกรูด พริกขี้หนูโขลกหยาบ ต้มให้เดือดอีกครั้ง ใส่กุ้งและเห็ดฟาง ปรุงรสด้วยน้ำปลาและน้ำตาลปึก ยกลงจากเตา ใส่น้ำมะนาว',
            
            'คั่วน้ำพริกแกงเขียวหวานกับหัวกะทิให้หอม ใส่เนื้อไก่ผัดให้เข้ากัน เติมกะทิที่เหลือ ต้มให้เดือด ใส่มะเขือเปราะ ปรุงรสด้วยน้ำปลาและน้ำตาลปึก ใส่ใบโหระพาและพริกชี้ฟ้า',
            
            'แช่วุ้นเส้นให้นุ่ม ลวกกุ้งในน้ำเดือดจนสุก ลวกหมูสับจนสุก ผสมน้ำยำจากมะนาว น้ำปลา น้ำตาลปึก พริกโขลก กระเทียม หอมแดง คลุกทุกอย่างให้เข้ากัน โรยผักชี',
            
            'ตั้งกะทะใส่น้ำมัน ผัดกระเทียมและหอมใหญ่ให้หอม ใส่กุ้งผัดจนสุก ใส่ไข่คนให้เข้ากัน ใส่ข้าวผัดให้เข้ากัน ปรุงรสด้วยซีอิ้วขาว ซีอิ้วหวาน น้ำตาล',
            
            'โขลกพริก กระเทียม ถั่วลิสง กุ้งแห้งให้หยาบ ใส่มะละกอ มะเขือเทศ ถั่วฝักยาว ตำให้เข้ากัน ปรุงรสด้วยน้ำปลา น้ำตาลปึก น้ำมะนาว ชิมรสให้เปรีย้วหวานเค็ม',
            
            'คั่วข้าวให้เหลืองหอม โขลกให้หยาบ ย่างพริกแห้งให้หอม โขลกกับหอมแดงให้ละเอียด ผสมเนื้อหมูสับกับข้าวคั่ว พริกโขลก ปรุงรสด้วยน้ำปลา น้ำมะนาว โรยผักชีและใบสะระแหน่',
            
            'ล้างปลาให้สะอาด ผสมแป้งสาลีกับน้ำปลา พริกไทยป่น กระเทียมป่น คลุกเคล้าให้เข้ากัน เคลือบปลาด้วยส่วนผสม ทอดในน้ำมันร้อนจนเหลืองกรอบ',
            
            'ตั้งหม้อใส่น้ำซุป ใส่หมูสับต้มจนสุก ใส่เต้าหู้ต้มให้นุ่ม ปรุงรสด้วยซีอิ้วขาว ใส่ต้นหอมและผักชี โรยพริกไทยป่น'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    output_file = 'thai_food_sample_enhanced.csv'
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"✅ สร้างไฟล์ตัวอย่างขั้นสูง '{output_file}' เรียบร้อย")
    print(f"📊 จำนวนเมนู: {len(df)} รายการ")
    
    # แสดงตัวอย่างข้อมูล
    print("\n🍽️  ตัวอย่างเมนูที่สร้าง:")
    for i, name in enumerate(df['name'][:5], 1):
        print(f"  {i}. {name}")
    
    return output_file

def main():
    """ฟังก์ชันหลักสำหรับการประมวลผลข้อมูล"""
    parser = argparse.ArgumentParser(
        description='ระบบประมวลผลข้อมูลสูตรอาหารไทยขั้นสูง',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ตัวอย่างการใช้งาน:
  python enhanced_preprocess.py --input thai_food_raw.csv --output thai_food_processed.csv
  python enhanced_preprocess.py --input data.csv --nutrition --enhance --api
  python enhanced_preprocess.py --sample
        """
    )
    
    parser.add_argument('--input', type=str, default='thai_food_raw.csv', 
                        help='ไฟล์ CSV อินพุต (default: thai_food_raw.csv)')
    parser.add_argument('--output', type=str, default='thai_food_processed.csv', 
                        help='ไฟล์ CSV เอาต์พุต (default: thai_food_processed.csv)')
    parser.add_argument('--nutrition', action='store_true', 
                        help='เพิ่มข้อมูลโภชนาการ')
    parser.add_argument('--enhance', action='store_true', 
                        help='เพิ่มวัตถุดิบที่ขาดหาย')
    parser.add_argument('--api', action='store_true', 
                        help='ใช้ข้อมูลจาก API ภายนอก (ต้องตั้งค่า API key)')
    parser.add_argument('--no-adjust', action='store_true', 
                        help='ไม่ปรับสัดส่วนการบริโภคจริง')
    parser.add_argument('--sample', action='store_true', 
                        help='สร้างข้อมูลตัวอย่างขั้นสูง')
    
    args = parser.parse_args()
    
    print("🍲 ระบบประมวลผลข้อมูลสูตรอาหารไทยขั้นสูง")
    print("=" * 60)
    
    if args.sample:
        create_sample_enhanced_data()
        return
    
    # แสดงการตั้งค่า
    print(f"📁 ไฟล์อินพุต: {args.input}")
    print(f"📁 ไฟล์เอาต์พุต: {args.output}")
    print(f"🧮 เพิ่มข้อมูลโภชนาการ: {'✅' if args.nutrition else '❌'}")
    print(f"🔧 เพิ่มวัตถุดิบที่ขาดหาย: {'✅' if args.enhance else '❌'}")
    print(f"🌐 ใช้ข้อมูลจาก API: {'✅' if args.api else '❌'}")
    print(f"⚖️  ปรับสัดส่วนการบริโภค: {'❌' if args.no_adjust else '✅'}")
    print("-" * 60)
    
    # ยืนยันการดำเนินการ
    if args.nutrition or args.enhance or args.api:
        confirm = input("🤔 ต้องการดำเนินการต่อหรือไม่? (y/N): ").lower().strip()
        if confirm not in ['y', 'yes', 'ใช่']:
            print("❌ ยกเลิกการดำเนินการ")
            return
    
    # เริ่มการประมวลผล
    success = preprocess_data(
        args.input, 
        args.output, 
        add_nutrition=args.nutrition,
        enhance_missing=args.enhance,
        use_api=args.api,
        adjust_consumption=not args.no_adjust
    )
    
    if success:
        print("\n🎉 การประมวลผลเสร็จสิ้นสมบูรณ์!")
        print(f"📄 สามารถใช้ไฟล์ '{args.output}' กับแอปพลิเคชันได้เลย")
        print("\n💡 คำแนะนำ: รันคำสั่ง 'streamlit run app.py' เพื่อเริ่มใช้งานแอป")
    else:
        print("\n❌ การประมวลผลล้มเหลว")
        print("💡 กรุณาตรวจสอบข้อผิดพลาดข้างต้นและลองใหม่")

# Import NutritionAPI จากไฟล์ที่ปรับปรุงแล้ว
try:
    from nutrition_api import NutritionAPI
except ImportError:
    print("⚠️  ไม่พบไฟล์ nutrition_api.py หรือมีปัญหาการ import")
    print("💡 กรุณาตรวจสอบว่าไฟล์ nutrition_api.py อยู่ในโฟลเดอร์เดียวกัน")
    
    # สร้าง NutritionAPI แบบง่าย
    class NutritionAPI:
        def __init__(self):
            pass
        def calculate_recipe_nutrition(self, *args, **kwargs):
            return {"total_nutrition": {"calories": 0, "protein": 0, "carbs": 0, "fat": 0, "fiber": 0,
                                      "vitamin_a": 0, "vitamin_c": 0, "vitamin_b1": 0, "vitamin_b2": 0,
                                      "calcium": 0, "iron": 0, "potassium": 0, "sodium": 0}}
    print("✅ ใช้ NutritionAPI แบบพื้นฐานแทน")

if __name__ == "__main__":
    main()
