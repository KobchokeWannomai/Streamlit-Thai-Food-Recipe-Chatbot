import pandas as pd
import re
import os
import argparse

def clean_text(text):
    """ทำความสะอาดและจัดรูปแบบข้อความ"""
    if not isinstance(text, str):
        return ""
    
    # ลบช่องว่างเกิน
    text = re.sub(r'\s+', ' ', text)
    
    # ลบอักขระพิเศษยกเว้นตัวอักษรไทย ภาษาอังกฤษ ตัวเลข และเครื่องหมายวรรคตอนพื้นฐาน
    text = re.sub(r'[^\u0E00-\u0E7Fa-zA-Z0-9\s.,\-\(\)/]', '', text)
    
    return text.strip()

def standardize_ingredients(text):
    """มาตรฐานการเขียนรายการวัตถุดิบ"""
    if not isinstance(text, str):
        return ""
    
    # แยกแต่ละบรรทัด
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # ตรวจสอบและปรับปรุงหน่วยวัด
        line = standardize_units(line)
        
        # ตรวจสอบให้แต่ละบรรทัดขึ้นต้นด้วยเครื่องหมาย -
        if not line.startswith('-'):
            line = f"- {line}"
        
        formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

def standardize_units(text):
    """มาตรฐานหน่วยวัดในวัตถุดิบ"""
    # แผนผังการแทนที่หน่วยวัด
    unit_replacements = {
        # หน่วยน้ำหนัก
        r'กิโลกรัม|กก\.|kg': 'กิโลกรัม',
        r'กรัม|ก\.|g(?!\w)': 'กรัม',
        r'ขีด|บาท': 'ขีด',
        
        # หน่วยปริมาตร  
        r'ลิตร|ล\.|L': 'ลิตร',
        r'มิลลิลิตร|มล\.|ml': 'มิลลิลิตร',
        r'ถ้วยชา': 'ถ้วยชา',
        r'ช้อนโต๊ะ': 'ช้อนโต๊ะ',
        r'ช้อนชา': 'ช้อนชา',
        
        # หน่วยนับ
        r'ตัว': 'ตัว',
        r'ผล': 'ผล', 
        r'หัว': 'หัว',
        r'กลีบ': 'กลีบ',
        r'เม็ด': 'เม็ด',
        r'ฟอง': 'ฟอง',
        r'ต้น': 'ต้น',
        r'ราก': 'ราก',
        r'ใบ': 'ใบ'
    }
    
    # ทำการแทนที่
    for pattern, replacement in unit_replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text

def extract_ingredient_info(ingredient_text):
    """แยกข้อมูลปริมาณ หน่วย และชื่อวัตถุดิบ"""
    # ลบเครื่องหมาย - ที่ด้านหน้า
    text = ingredient_text.strip().lstrip('-').strip()
    
    # ค้นหารูปแบบตัวเลขและหน่วย
    patterns = [
        r'(\d+(?:[./]\d+)?)\s*([ก-๙a-zA-Z]+)',  # ตัวเลขตามด้วยหน่วย
        r'(\d+(?:[./]\d+)?)',  # เฉพาะตัวเลข
    ]
    
    amount = None
    unit = ""
    name = text
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            amount_str = match.group(1)
            unit = match.group(2) if len(match.groups()) > 1 else ""
            
            # แปลงเศษส่วนเป็นทศนิยม
            if '/' in amount_str:
                parts = amount_str.split('/')
                try:
                    amount = float(parts[0]) / float(parts[1])
                except:
                    amount = 1
            else:
                try:
                    amount = float(amount_str)
                except:
                    amount = 1
            
            # ลบส่วนที่เป็นตัวเลขและหน่วยออก
            name = text.replace(match.group(0), '').strip()
            break
    
    return {
        'original': ingredient_text,
        'amount': amount if amount is not None else 1,
        'unit': unit,
        'name': name,
        'standardized': f"- {name} {amount if amount else ''} {unit}".strip()
    }

def validate_ingredient_amounts(ingredients_text):
    """ตรวจสอบและปรับปรุงปริมาณวัตถุดิบให้สมเหตุสมผล"""
    lines = ingredients_text.split('\n')
    validated_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        info = extract_ingredient_info(line)
        
        # ปรับปรุงปริมาณที่ไม่สมเหตุสมผล
        if info['amount'] and info['unit']:
            # ตรวจสอบปริมาณที่มากผิดปกติ
            if info['unit'] in ['กิโลกรัม', 'กก.'] and info['amount'] > 5:
                # ถ้าเกิน 5 กิโลกรัม อาจเป็นความผิดพลาด
                new_amount = info['amount'] / 1000  # แปลงเป็นกรัม
                validated_lines.append(f"- {info['name']} {new_amount} กรัม")
            elif info['unit'] in ['ถ้วย'] and info['amount'] > 10:
                # ถ้าเกิน 10 ถ้วย อาจมากเกินไป
                new_amount = min(info['amount'], 3)
                validated_lines.append(f"- {info['name']} {new_amount} {info['unit']}")
            else:
                validated_lines.append(f"- {info['name']} {info['amount']} {info['unit']}")
        elif info['amount'] and not info['unit']:
            # ถ้ามีตัวเลขแต่ไม่มีหน่วย ให้เดาหน่วยจากชื่อวัตถุดิบ
            suggested_unit = suggest_unit_from_name(info['name'])
            validated_lines.append(f"- {info['name']} {info['amount']} {suggested_unit}")
        else:
            # ถ้าไม่มีตัวเลขเลย ให้ใส่ปริมาณประมาณ
            suggested_amount, suggested_unit = suggest_amount_from_name(info['name'])
            validated_lines.append(f"- {info['name']} {suggested_amount} {suggested_unit}")
    
    return '\n'.join(validated_lines)

def suggest_unit_from_name(name):
    """เสนอแนะหน่วยวัดจากชื่อวัตถุดิบ"""
    name_lower = name.lower()
    
    # หน่วยนับ
    if any(word in name_lower for word in ['ไก่', 'หมู', 'ปลา', 'กบ']):
        return 'ตัว'
    elif any(word in name_lower for word in ['มะเขือเทศ', 'มะนาว', 'แตงกวา', 'มะละกอ']):
        return 'ผล'
    elif any(word in name_lower for word in ['หอม', 'กระเทียม']):
        return 'หัว'
    elif 'กลีบ' in name_lower:
        return 'กลีบ'
    elif 'ไข่' in name_lower:
        return 'ฟอง'
    elif any(word in name_lower for word in ['ผักชี', 'ต้นหอม']):
        return 'ต้น'
    elif any(word in name_lower for word in ['พริก']):
        return 'เม็ด'
    
    # หน่วยน้ำหนัก
    elif any(word in name_lower for word in ['น้ำมัน', 'น้ำปลา', 'น้ำส้ม']):
        return 'ช้อนโต๊ะ'
    elif any(word in name_lower for word in ['เกลือ', 'น้ำตาล', 'พริกไทย']):
        return 'ช้อนชา'
    else:
        return 'กรัม'

def suggest_amount_from_name(name):
    """เสนอแนะปริมาณจากชื่อวัตถุดิบ"""
    name_lower = name.lower()
    
    # ปริมาณสำหรับเครื่องปรุง
    if any(word in name_lower for word in ['เกลือ', 'พริกไทย']):
        return 1, 'ช้อนชา'
    elif any(word in name_lower for word in ['น้ำปลา', 'น้ำตาล']):
        return 2, 'ช้อนโต๊ะ'
    elif 'น้ำมัน' in name_lower:
        return 3, 'ช้อนโต๊ะ'
    
    # ปริมาณสำหรับผัก/เครื่องเทศ
    elif any(word in name_lower for word in ['ผักชี', 'ต้นหอม']):
        return 2, 'ต้น'
    elif 'พริก' in name_lower:
        return 3, 'เม็ด'
    elif 'กระเทียม' in name_lower:
        return 3, 'กลีบ'
    elif 'หอม' in name_lower:
        return 1, 'หัว'
    
    # ปริมาณสำหรับเนื้อสัตว์
    elif any(word in name_lower for word in ['ไก่', 'หมู', 'เนื้อ']):
        return 300, 'กรัม'
    elif any(word in name_lower for word in ['ปลา', 'กุ้ง']):
        return 200, 'กรัม'
    elif 'ไข่' in name_lower:
        return 2, 'ฟอง'
    
    # ค่าเริ่มต้น
    else:
        return 100, 'กรัม'

def preprocess_data(input_file, output_file):
    """ประมวลผลล่วงหน้าข้อมูลสูตรอาหารไทย"""
    # ตรวจสอบว่ามีไฟล์อินพุตหรือไม่
    if not os.path.exists(input_file):
        print(f"ข้อผิดพลาด: ไม่พบไฟล์อินพุต '{input_file}'")
        return False
    
    try:
        # อ่านไฟล์ CSV
        print(f"กำลังอ่านไฟล์ {input_file}...")
        df = pd.read_csv(input_file)
        
        # ตรวจสอบคอลัมน์ที่จำเป็น
        required_columns = ['name', 'text_ingradiant', 'food_method']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"ข้อผิดพลาด: ไม่พบคอลัมน์ที่จำเป็น: {', '.join(missing_columns)}")
            return False
        
        print("กำลังทำความสะอาดและปรับปรุงข้อมูล...")
        
        # ทำความสะอาดข้อความในแต่ละคอลัมน์
        df['name'] = df['name'].apply(clean_text)
        df['food_method'] = df['food_method'].apply(clean_text)
        
        # ปรับปรุงรายการวัตถุดิบ
        print("กำลังปรับปรุงรายการวัตถุดิบ...")
        df['ingredient'] = df['text_ingradiant'].apply(standardize_ingredients)
        df['ingredient'] = df['ingredient'].apply(validate_ingredient_amounts)
        
        # ลบคอลัมน์เก่า
        df = df.drop('text_ingradiant', axis=1)
        
        # เปลี่ยนชื่อคอลัมน์
        df = df.rename(columns={'food_method': 'method'})
        
        # ลบรายการที่ซ้ำ
        df = df.drop_duplicates(subset=['name'])
        
        # จัดเรียงข้อมูลใหม่
        df = df.reset_index(drop=True)
        
        # บันทึกข้อมูลที่ประมวลผลแล้ว
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"การประมวลผลเสร็จสิ้น บันทึกไปยัง '{output_file}'")
        print(f"จำนวนสูตรอาหารทั้งหมด: {len(df)}")
        
        # ลบไฟล์ embeddings ถ้ามี เพื่อให้สร้างใหม่
        if os.path.exists('embeddings.pkl'):
            os.remove('embeddings.pkl')
            print("ลบไฟล์ embeddings เดิม จะสร้างใหม่เมื่อรันแอป")
        
        # แสดงตัวอย่างข้อมูลที่ปรับปรุงแล้ว
        print("\nตัวอย่างข้อมูลที่ปรับปรุงแล้ว:")
        print("=" * 60)
        for i in range(min(3, len(df))):
            print(f"\nสูตรที่ {i+1}: {df.iloc[i]['name']}")
            print("วัตถุดิบ:")
            ingredients = df.iloc[i]['ingredient'].split('\n')
            for ing in ingredients[:5]:  # แสดง 5 รายการแรก
                if ing.strip():
                    print(f"  {ing}")
            if len(ingredients) > 5:
                print(f"  ... และอื่นๆ อีก {len(ingredients)-5} รายการ")
        
        return True
    
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการประมวลผล: {str(e)}")
        return False

def analyze_ingredients(csv_file):
    """วิเคราะห์และสรุปข้อมูลวัตถุดิบ"""
    try:
        df = pd.read_csv(csv_file)
        
        print("\nการวิเคราะห์ข้อมูลวัตถุดิบ")
        print("=" * 50)
        
        # นับจำนวนวัตถุดิบที่ใช้บ่อย
        all_ingredients = []
        for ingredients in df['ingredient']:
            for ingredient in ingredients.split('\n'):
                ingredient = ingredient.strip().lstrip('-').strip()
                if ingredient:
                    # แยกเฉพาะชื่อวัตถุดิบ ไม่เอาปริมาณ
                    name = re.sub(r'^\d+(?:[./]\d+)?\s*[ก-๙a-zA-Z]*\s*', '', ingredient)
                    if name:
                        all_ingredients.append(name.lower())
        
        # นับความถี่
        from collections import Counter
        ingredient_counts = Counter(all_ingredients)
        
        print(f"วัตถุดิบที่พบบ่อยที่สุด 10 อันดับแรก:")
        for ingredient, count in ingredient_counts.most_common(10):
            print(f"  {ingredient}: {count} ครั้ง")
        
        print(f"\nสถิติทั่วไป:")
        print(f"  จำนวนสูตรอาหารทั้งหมด: {len(df)}")
        print(f"  จำนวนวัตถุดิบที่แตกต่างกัน: {len(ingredient_counts)}")
        print(f"  วัตถุดิบเฉลี่ยต่อสูตร: {len(all_ingredients)/len(df):.1f}")
        
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการวิเคราะห์: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ประมวลผลข้อมูลสูตรอาหารไทย')
    parser.add_argument('--input', type=str, default='thai_food_raw.csv', 
                        help='เส้นทางไฟล์ CSV อินพุต')
    parser.add_argument('--output', type=str, default='thai_food_processed.csv', 
                        help='เส้นทางไฟล์ CSV เอาท์พุต')
    parser.add_argument('--analyze', action='store_true',
                        help='วิเคราะห์ข้อมูลหลังประมวลผล')
    
    args = parser.parse_args()
    
    # ประมวลผลข้อมูล
    success = preprocess_data(args.input, args.output)
    
    if success and args.analyze:
        analyze_ingredients(args.output)
