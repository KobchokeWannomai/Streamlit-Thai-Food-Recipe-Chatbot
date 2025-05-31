import pandas as pd
import re
import os
import argparse
from nutrition_api import NutritionAPI

def clean_text(text):
    """ทำความสะอาดและจัดรูปแบบข้อความ"""
    if not isinstance(text, str):
        return ""
    
    # ลบช่องว่างเกิน
    text = re.sub(r'\s+', ' ', text)
    
    # ลบอักขระพิเศษ ยกเว้นตัวอักษรไทย ตัวเลข และเครื่องหมายวรรคตอนพื้นฐาน
    text = re.sub(r'[^\u0E00-\u0E7Fa-zA-Z0-9\s.,\-\(\)]', '', text)
    
    return text.strip()

def preprocess_ingredients(text):
    """จัดรูปแบบรายการวัตถุดิบ"""
    if not isinstance(text, str):
        return ""
    
    # แยกบรรทัด
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # ตรวจสอบว่าขึ้นต้นด้วยเครื่องหมาย -
        if not line.startswith('-'):
            line = f"- {line}"
        
        formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

def enhance_missing_ingredients(ingredients_text, recipe_name, method_text):
    """เพิ่มวัตถุดิบที่ขาดหายไปตามวิธีการทำ"""
    enhanced_ingredients = ingredients_text
    missing_ingredients = []
    
    # รายการวัตถุดิบที่มักขาดหายไป
    common_missing = {
        'ทอด': ['น้ำมันพืช 2 ช้อนโต๊ะ'],
        'เจียว': ['น้ำมันหมู 1 ช้อนโต๊ะ'],
        'ผัด': ['น้ำมันพืช 1 ช้อนโต๊ะ'],
        'ต้ม': [],  # ปกติจะมีน้ำอยู่แล้ว
        'แกง': [],  # ปกติจะมีกะทิหรือน้ำอยู่แล้ว
        'ย่าง': [],
        'ปิ้ง': [],
        'นึ่ง': []
    }
    
    # ตรวจสอบวิธีการทำและเพิ่มวัตถุดิบที่ขาดหาย
    method_lower = method_text.lower()
    ingredients_lower = ingredients_text.lower()
    
    for cooking_method, required_ingredients in common_missing.items():
        if cooking_method in method_lower:
            for ingredient in required_ingredients:
                ingredient_name = ingredient.split()[0]  # เอาชื่อวัตถุดิบ
                if ingredient_name not in ingredients_lower:
                    missing_ingredients.append(f"- {ingredient}")
    
    # เพิ่มวัตถุดิบที่ขาดหาย
    if missing_ingredients:
        enhanced_ingredients += '\n' + '\n'.join(missing_ingredients)
        print(f"เพิ่มวัตถุดิบสำหรับ '{recipe_name}': {', '.join(missing_ingredients)}")
    
    return enhanced_ingredients

def calculate_recipe_nutrition(ingredients_text, nutrition_api, enhance_missing=False, recipe_name="", method_text=""):
    """คำนวณคุณค่าทางโภชนาการของสูตรอาหาร"""
    if enhance_missing:
        ingredients_text = enhance_missing_ingredients(ingredients_text, recipe_name, method_text)
    
    nutrition_data = nutrition_api.calculate_recipe_nutrition(
        ingredients_text, 
        use_api=False,  # ใช้ข้อมูลท้องถิ่นในการประมวลผล
        adjust_consumption=True
    )
    
    return nutrition_data['total_nutrition'], ingredients_text

def preprocess_data(input_file, output_file, add_nutrition=True, enhance_missing=False):
    """ประมวลผลข้อมูลอาหารไทย"""
    # ตรวจสอบไฟล์อินพุต
    if not os.path.exists(input_file):
        print(f"Error: ไม่พบไฟล์ '{input_file}'")
        return False
    
    try:
        # อ่านไฟล์ CSV
        df = pd.read_csv(input_file)
        
        # ตรวจสอบคอลัมน์ที่จำเป็น
        required_columns = ['name', 'text_ingradiant', 'food_method']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Error: ขาดคอลัมน์ที่จำเป็น: {', '.join(missing_columns)}")
            return False
        
        # เปลี่ยนชื่อคอลัมน์ให้ถูกต้อง
        df = df.rename(columns={
            'text_ingradiant': 'ingredient',
            'food_method': 'method'
        })
        
        # ทำความสะอาดข้อความในแต่ละคอลัมน์
        df['name'] = df['name'].apply(clean_text)
        df['method'] = df['method'].apply(clean_text)
        df['ingredient'] = df['ingredient'].apply(preprocess_ingredients)
        
        # เพิ่มข้อมูลโภชนาการ (หากต้องการ)
        if add_nutrition:
            print("กำลังคำนวณข้อมูลโภชนาการ...")
            nutrition_api = NutritionAPI()
            
            # เพิ่มคอลัมน์โภชนาการ
            nutrition_columns = [
                'calories', 'protein', 'carbs', 'fat', 'fiber',
                'vitamin_a', 'vitamin_c', 'vitamin_b1', 'vitamin_b2',
                'calcium', 'iron', 'potassium', 'sodium'
            ]
            
            for col in nutrition_columns:
                df[col] = 0.0
            
            # คำนวณโภชนาการสำหรับแต่ละสูตร
            for idx, row in df.iterrows():
                try:
                    nutrition, enhanced_ingredients = calculate_recipe_nutrition(
                        row['ingredient'], 
                        nutrition_api, 
                        enhance_missing,
                        row['name'],
                        row['method']
                    )
                    
                    # อัปเดตวัตถุดิบที่ปรับปรุงแล้ว
                    if enhance_missing:
                        df.at[idx, 'ingredient'] = enhanced_ingredients
                    
                    # บันทึกข้อมูลโภชนาการ
                    for nutrient, value in nutrition.items():
                        if nutrient in nutrition_columns:
                            df.at[idx, nutrient] = round(value, 2)
                    
                    if idx % 10 == 0:
                        print(f"ประมวลผลแล้ว {idx+1}/{len(df)} สูตร")
                        
                except Exception as e:
                    print(f"เกิดข้อผิดพลาดในการคำนวณโภชนาการของ '{row['name']}': {str(e)}")
                    continue
        
        # ลบรายการที่ซ้ำกัน
        df = df.drop_duplicates(subset=['name'])
        
        # รีเซ็ตดัชนี
        df = df.reset_index(drop=True)
        
        # บันทึกข้อมูลที่ประมวลผลแล้ว
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"การประมวลผลเสร็จสิ้น บันทึกไปยัง '{output_file}'")
        print(f"จำนวนสูตรทั้งหมด: {len(df)}")
        
        if add_nutrition:
            # แสดงสถิติโภชนาการ
            print("\n📊 สถิติโภชนาการเฉลี่ย:")
            nutrition_summary = {
                'แคลอรี่': df['calories'].mean(),
                'โปรตีน (g)': df['protein'].mean(),
                'คาร์โบไฮเดรต (g)': df['carbs'].mean(),
                'ไขมัน (g)': df['fat'].mean(),
                'ใยอาหาร (g)': df['fiber'].mean()
            }
            
            for nutrient, avg_value in nutrition_summary.items():
                print(f"  {nutrient}: {avg_value:.1f}")
        
        # ลบไฟล์ embeddings เก่า เพื่อให้สร้างใหม่
        if os.path.exists('embeddings.pkl'):
            os.remove('embeddings.pkl')
            print("ลบไฟล์ embeddings เก่า จะสร้างใหม่เมื่อรันแอป")
        
        return True
    
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการประมวลผล: {str(e)}")
        return False

def create_sample_enhanced_data():
    """สร้างข้อมูลตัวอย่างที่มีโภชนาการ"""
    sample_data = {
        'name': [
            'ไข่เจียว',
            'ผัดกะเพรา',
            'ต้มยำกุ้ง',
            'แกงเขียวหวาน',
            'ยำวุ้นเส้น'
        ],
        'ingredient': [
            '- ไข่ไก่ 2 ฟอง\n- น้ำปลา 1 ช้อนชา\n- ต้นหอม 2 ต้น',
            '- เนื้อหมูสับ 200 กรัม\n- ใบกะเพรา 1 ถ้วย\n- พริกขี้หนู 5 เม็ด\n- กระเทียม 3 กลีบ\n- น้ำปลา 2 ช้อนโต๊ะ',
            '- กุ้งนาง 200 กรัม\n- เห็ดฟาง 100 กรัม\n- มะนาว 2 ผล\n- ใบมะกรูด 3 ใบ\n- น้ำปลา 2 ช้อนโต๊ะ',
            '- เนื้อไก่ 300 กรัม\n- กะทิ 1 ถ้วย\n- มะเขือเปราะ 5 ผล\n- ใบโหระพา 1/2 ถ้วย',
            '- วุ้นเส้น 100 กรัม\n- กุ้งสด 100 กรัม\n- หมูสับ 50 กรัม\n- มะนาว 2 ผล\n- น้ำปลา 2 ช้อนโต๊ะ'
        ],
        'method': [
            'ต่อยไข่ใส่ชาม ใส่น้ำปลาคนให้เข้ากัน ตั้งกะทะใส่น้ำมัน เมื่อร้อนเทไข่ลงทอดจนเหลือง',
            'โขลกกระเทียมและพริกให้ละเอียด ผัดหมูสับจนสุก ใส่กะเพราและปรุงรส',
            'ต้มน้ำให้เดือด ใส่เครื่องต้มยำ เมื่อเดือดใส่กุ้งและเห็ด ปรุงรสและใส่มะนาว',
            'คั่วน้ำพริกแกงเขียวหวานกับหัวกะทิ ใส่ไก่และกะทิ ปรุงรสและใส่ผัก',
            'แช่วุ้นเส้นให้นุ่ม ลวกกุ้งและหมู คลุกทุกอย่างกับน้ำยำ'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('thai_food_sample.csv', index=False, encoding='utf-8')
    print("สร้างไฟล์ตัวอย่าง 'thai_food_sample.csv' เรียบร้อย")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ประมวลผลข้อมูลสูตรอาหารไทย')
    parser.add_argument('--input', type=str, default='thai_food_raw.csv', 
                        help='ไฟล์ CSV อินพุต')
    parser.add_argument('--output', type=str, default='thai_food_processed.csv', 
                        help='ไฟล์ CSV เอาต์พุต')
    parser.add_argument('--nutrition', action='store_true', 
                        help='เพิ่มข้อมูลโภชนาการ')
    parser.add_argument('--enhance', action='store_true', 
                        help='เพิ่มวัตถุดิบที่ขาดหาย')
    parser.add_argument('--sample', action='store_true', 
                        help='สร้างข้อมูลตัวอย่าง')
    
    args = parser.parse_args()
    
    if args.sample:
        create_sample_enhanced_data()
    else:
        success = preprocess_data(
            args.input, 
            args.output, 
            add_nutrition=args.nutrition,
            enhance_missing=args.enhance
        )
        
        if success:
            print("✅ การประมวลผลเสร็จสิ้น")
        else:
            print("❌ การประมวลผลล้มเหลว")
