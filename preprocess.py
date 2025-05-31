#!/usr/bin/env python3
"""
ระบบประมวลผลข้อมูลสูตรอาหารไทย - ปรับปรุงใหม่
Advanced Thai Food Data Preprocessing System
รองรับการทำความสะอาดข้อมูล การวิเคราะห์โภชนาการ และการสร้างฐานข้อมูลขั้นสูง
"""

import pandas as pd
import re
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json
import sqlite3
from typing import Dict, List, Optional, Tuple
import unicodedata
from nutrition_analyzer import NutritionAnalyzer, NutritionDatabase
from ingredient_converter import IngredientConverter

# ตั้งค่า logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ThaiTextProcessor:
    """คลาสสำหรับประมวลผลข้อความภาษาไทย"""
    
    def __init__(self):
        # คำที่ไม่จำเป็นในสูตรอาหาร
        self.stopwords = {
            'และ', 'หรือ', 'ที่', 'ใน', 'บน', 'กับ', 'ด้วย', 'จาก', 'ไป', 'มา',
            'แล้ว', 'ได้', 'เป็น', 'ให้', 'ของ', 'เพื่อ', 'จะ', 'ไม่', 'ก็', 'ยัง',
            'เอา', 'นำ', 'ใส่', 'เติม', 'ลง', 'ขึ้น', 'ออก', 'เข้า', 'ผ่าน', 'ผสม'
        }
        
        # รูปแบบการเขียนหน่วยที่หลากหลาย
        self.unit_patterns = {
            r'กิโลกรัม|กิโล|กก\.': 'กิโลกรัม',
            r'กรัม|ก\.': 'กรัม',
            r'ช้อนโต๊ะ|ช้อนใหญ่|ชต\.': 'ช้อนโต๊ะ',
            r'ช้อนชา|ช้อนเล็ก|ชช\.': 'ช้อนชา',
            r'ถ้วย|ถ้วยตวง': 'ถ้วย',
            r'ลิตร|ล\.': 'ลิตร',
            r'มิลลิลิตร|มล\.': 'มิลลิลิตร',
            r'ฟอง': 'ฟอง',
            r'ตัว': 'ตัว',
            r'หัว': 'หัว',
            r'แผ่น': 'แผ่น',
            r'กลีบ': 'กลีบ',
            r'ใบ': 'ใบ',
            r'ผล|ลูก': 'ผล',
            r'เม็ด': 'เม็ด'
        }
        
        # รูปแบบการเขียนตัวเลขไทย
        self.thai_numbers = {
            'หนึ่ง': '1', 'สอง': '2', 'สาม': '3', 'สี่': '4', 'ห้า': '5',
            'หก': '6', 'เจ็ด': '7', 'แปด': '8', 'เก้า': '9', 'สิบ': '10',
            'ยี่สิบ': '20', 'สามสิบ': '30', 'สี่สิบ': '40', 'ห้าสิบ': '50',
            'หกสิบ': '60', 'เจ็ดสิบ': '70', 'แปดสิบ': '80', 'เก้าสิบ': '90',
            'ร้อย': '100', 'พัน': '1000'
        }
    
    def normalize_thai_text(self, text: str) -> str:
        """ปรับให้ข้อความภาษาไทยเป็นมาตรฐาน"""
        if not text:
            return ""
        
        # แปลงเป็น unicode ปกติ
        text = unicodedata.normalize('NFC', text)
        
        # แทนที่ตัวเลขไทยด้วยตัวเลขอารบิก
        for thai_num, arabic_num in self.thai_numbers.items():
            text = text.replace(thai_num, arabic_num)
        
        # ปรับหน่วยให้เป็นมาตรฐาน
        for pattern, standard_unit in self.unit_patterns.items():
            text = re.sub(pattern, standard_unit, text, flags=re.IGNORECASE)
        
        # ลบช่องว่างซ้ำ
        text = re.sub(r'\s+', ' ', text)
        
        # ลบช่องว่างที่จุดเริ่มต้นและจุดสิ้นสุด
        text = text.strip()
        
        return text
    
    def clean_ingredient_text(self, text: str) -> str:
        """ทำความสะอาดข้อความวัตถุดิบ"""
        text = self.normalize_thai_text(text)
        
        # ลบอักขระพิเศษที่ไม่จำเป็น
        text = re.sub(r'[^\w\s\-\.\,\(\)\/]', '', text)
        
        # แก้ไขการพิมพ์ผิดทั่วไป
        common_typos = {
            'กะเพรา': 'กะเพรา',
            'กระเพรา': 'กะเพรา',
            'ต้มยำ': 'ต้มยำ',
            'ต้มยํา': 'ต้มยำ',
            'มัสมั่น': 'มัสมั่น',
            'มัสมัน': 'มัสมั่น',
            'ส้มตำ': 'ส้มตำ',
            'ส้มตํา': 'ส้มตำ'
        }
        
        for typo, correct in common_typos.items():
            text = text.replace(typo, correct)
        
        return text
    
    def clean_method_text(self, text: str) -> str:
        """ทำความสะอาดข้อความวิธีทำ"""
        text = self.normalize_thai_text(text)
        
        # แยกขั้นตอนและจัดรูปแบบ
        steps = text.split('\n')
        cleaned_steps = []
        
        for i, step in enumerate(steps, 1):
            step = step.strip()
            if not step:
                continue
            
            # ลบเลขขั้นตอนเก่า
            step = re.sub(r'^\d+\.\s*', '', step)
            
            # เพิ่มเลขขั้นตอนใหม่
            if not step.startswith(f'{i}.'):
                step = f'{i}. {step}'
            
            cleaned_steps.append(step)
        
        return '\n'.join(cleaned_steps)

class ThaiMenuEnhancer:
    """คลาสสำหรับปรับปรุงชื่อเมนูอาหารไทยให้ครบถ้วน"""
    
    def __init__(self):
        # รายการเมนูอาหารไทยที่ครบถ้วนจากชุดข้อมูล
        self.enhanced_menu_variations = {
            # เมนูหลัก
            'กุ้งทาพริกไทยกระเทียม': [
                'กุ้งทาพริกไทย', 'กุ้งผัดพริกไทย', 'กุ้งกระเทียม', 'กุ้งพริกไทย',
                'กุ้งทาเครื่องเทศ', 'กุ้งผัดกระเทียม', 'กุ้งใส่พริกไทย'
            ],
            'ข้าวเม่าทอด': [
                'ข้าวเหม่าทอด', 'ข้าวเม่า', 'ข้าวหม้อทอด', 'ข้าวเม่าผัด',
                'ข้าวเหม่า', 'ข้าวเม่าคลุก'
            ],
            'เปรี้ยวหวานไข่ม้วน': [
                'เปรี้ยวหวาน', 'ไข่ม้วนเปรี้ยวหวาน', 'ไข่ม้วน', 'เปรี้ยวหวานไข่',
                'ไข่ม้วนหวาน', 'เปรี้ยวหวานไข่เจียว'
            ],
            'ไข่จ่อม': [
                'ไข่จ๋อม', 'ไข่ซ่อม', 'ไข่ดิบ', 'ไข่จ่อมน้ำ', 'ไข่จุ่ม'
            ],
            'งบปลาทู': [
                'งบปลา', 'ปลาทูแกง', 'แกงปลาทู', 'ปลาทูต้ม', 'งบปลาทูแกง'
            ],
            
            # เมนูพิเศษ
            'น้ำพริกจิ้มผักดิบ': [
                'น้ำพริกผักดิบ', 'น้ำพริกจิ้ม', 'น้ำพริกผัก', 'น้ำพริกสด',
                'น้ำพริกใส่ผัก', 'น้ำพริกกินกับผัก'
            ],
            'ลอยน้ำดอกไม้สด': [
                'ลอยน้ำดอกไม้', 'ลอยน้ำ', 'ขนมลอยน้ำ', 'ดอกไม้ลอยน้ำ',
                'ลอยน้ำหวาน', 'ขนมไทยลอยน้ำ'
            ],
            'ยำไข่ปลาดุก': [
                'ยำไข่ปลา', 'ไข่ปลาดุกยำ', 'ยำไข่ดุก', 'ไข่ปลายำ',
                'ยำไข่ปลาสด', 'ไข่ปลาดุกปรุงรส'
            ],
            'ปลาทูทอดปรุง': [
                'ปลาทูทอด', 'ปลาทูปรุง', 'ปลาทูผัด', 'ปลาทูทอดหวาน',
                'ปลาทูทอดน้ำปลา', 'ปลาทูทอดซอส'
            ],
            'ต้มยำกะทิ': [
                'ต้มยำน้ำกะทิ', 'ต้มยำใส่กะทิ', 'ต้มยำขาว', 'ต้มยำนม',
                'ต้มยำครีม', 'ต้มยำกะทิสด'
            ],
            
            # เมนูเนื้อ/ไก่
            'ไก่ยำ': [
                'ยำไก่', 'ไก่ลาบ', 'ยำไก่สด', 'ลาบไก่', 'ไก่ยำใส',
                'ยำไก่ต้ม', 'ไก่ยำปลา'
            ],
            'ไก่หยอง': [
                'ไก่หยองใต้', 'ไก่ผัดพริกแกง', 'ไก่ใส่พริกแกง', 'ไก่แกงใต้',
                'ไก่หยองแกง', 'ไก่พริกแกงแห้ง'
            ],
            'ไก่ทันสมัย': [
                'ไก่สมัยใหม่', 'ไก่ผัดทันสมัย', 'ไก่ปรุงใหม่', 'ไก่แฟชั่น',
                'ไก่สไตล์ใหม่', 'ไก่โมเดิร์น'
            ],
            
            # ขนมและของหวาน
            'กล้วยบวชชี': [
                'กล้วยบุชชี', 'กล้วยชุบแป้ง', 'กล้วยทอด', 'กล้วยบวชชีกะทิ',
                'กล้วยแป้ง', 'กล้วยนึ่ง'
            ],
            'มะตูมเชื่อม': [
                'มะตูม', 'มะตูมหวาน', 'มะตูมแช่อิ่ม', 'มะตูมน้ำตาล',
                'มะตูมต้ม', 'มะตูมกะทิ'
            ],
            'สังขยา': [
                'สังขยาใบเตย', 'สังขยาฟักทอง', 'ขนมสังขยา', 'สังขยาหวาน',
                'สังขยานึ่ง', 'สังขยากะทิ'
            ],
            'สาคูเปียก': [
                'ขนมสาคู', 'สาคูหวาน', 'สาคูน้ำกะทิ', 'สาคูต้ม',
                'สาคูใส', 'สาคูเปียกน้ำกะทิ'
            ],
            
            # เมนูแกง
            'แกงคั่วฟักทองกับกุ้งตะเข็บ': [
                'แกงคั่วฟักทอง', 'แกงคั่วกุ้ง', 'ฟักทองแกงคั่ว', 'แกงคั่วตะเข็บ',
                'แกงคั่วฟักกุ้ง', 'แกงคั่วฟักทองกุ้ง'
            ],
            'แกงยา': [
                'แกงยาใต้', 'แกงยาปลา', 'แกงยาผัก', 'แกงยาแท้',
                'แกงยาปักษ์ใต้', 'แกงยาเผ็ด'
            ],
            'แกงเลียง': [
                'แกงเลียงผัก', 'แกงเลียงกุ้ง', 'แกงเลียงใต้', 'แกงเลียงปลา',
                'แกงเลียงหวาน', 'แกงเลียงใส'
            ],
            
            # เมนูทอด
            'ฟักทองทอด': [
                'ฟักทองทอดกรอบ', 'ฟักทองชุบแป้ง', 'ฟักทองผัด', 'ฟักทองทอดแป้ง',
                'ฟักทองทอดน้ำปลา', 'ฟักทองทอดหวาน'
            ],
            'เนื้อเครื่องเทศทอด': [
                'เนื้อทอดเครื่องเทศ', 'เนื้อผัดเครื่องเทศ', 'เนื้อปรุงรส',
                'เนื้อทอดครื่องเทศ', 'เนื้อเทศทอด', 'เนื้อเครื่องเทศ'
            ],
            'หมูทอดเค็ม': [
                'หมูทอดกรอบ', 'หมูทอดแห้ง', 'หมูเค็มทอด', 'หมูทอดน้ำปลา',
                'หมูกรอบทอด', 'หมูทอดเกลือ'
            ],
            
            # เมนูไข่
            'ไข่กระจัง': [
                'ไข่กระจัด', 'ไข่ผัด', 'ไข่กระจังผัด', 'ไข่คน',
                'ไข่ผัดไทย', 'ไข่กระจังดาว'
            ],
            'ไข่สามชั้น': [
                'ไข่สามชั้นผัด', 'หมูสามชั้นไข่', 'ไข่ผัดสามชั้น', 'ไข่หมูสามชั้น',
                'สามชั้นไข่', 'ไข่ผัดหมูสามชั้น'
            ],
            'ไข่ในรัง': [
                'ไข่ซ่อนรัง', 'ไข่รังนก', 'ไข่ทำรัง', 'ไข่ห่อ',
                'ไข่ในแป้ง', 'ไข่รังแป้ง'
            ],
            'ไข่สวรรค์': [
                'ไข่ฟ้า', 'ไข่สวรรค์ทอง', 'ไข่แสงสวรรค์', 'ไข่เทวดา',
                'ไข่สวรรค์หวาน', 'ไข่ทองสวรรค์'
            ],
            'ไข่หวานฝอย': [
                'ไข่ฝอย', 'ไข่หวาน', 'ฝอยทอง', 'ไข่ฝอยหวาน',
                'ไข่ดาวฝอย', 'ไข่เส้นหวาน'
            ],
            'ไข่น้อค': [
                'ไข่น้อคใต้', 'ไข่ย่าง', 'ไข่เผา', 'ไข่น้อคย่าง',
                'ไข่ใต้ย่าง', 'ไข่น้อคผัด'
            ],
            'ไข่ช่อนรูป': [
                'ไข่ช่อน', 'ไข่รูปช่อน', 'ไข่ทำรูป', 'ไข่ช่อนดาว',
                'ไข่รูปพิเศษ', 'ไข่แต่งรูป'
            ],
            'ไข่ตุ๋น': [
                'ไข่ตุ๋นกะทิ', 'ไข่ตุ๋นหวาน', 'ไข่ตุ๋นนึ่ง', 'ไข่ตุ๋นเค็ม',
                'ไข่ตุ๋นน้ำ', 'ไข่ตุ๋นใส'
            ],
            
            # เมนูพิเศษอื่นๆ
            'บี๊ฟที': [
                'บีฟสเต็ก', 'เนื้อทีโบน', 'เนื้อย่าง', 'บีฟสเต็กไทย',
                'เนื้อทีบี', 'บีฟทีย่าง'
            ],
            'มักกะโรนีรังแตน': [
                'มักกะโรนี', 'รังแตนมักกะโรนี', 'พาสต้ารังแตน', 'มักกะโรนีไทย',
                'เส้นมักกะโรนี', 'พาสต้าไทย'
            ],
            'ฉี่ฉู่เมืองปราณ': [
                'ฉี่ฉู่', 'เมืองปราณ', 'ขนมฉี่ฉู่', 'ฉี่ฉู่ไทย',
                'ขนมเมืองปราณ', 'ฉี่ฉู่หวาน'
            ],
            'ทองม้วนเค็ม': [
                'ทองม้วน', 'ไข่ม้วนเค็ม', 'ทองม้วนคาว', 'ไข่ทองม้วน',
                'ทองม้วนไข่', 'ไข่ม้วนทอง'
            ],
            'เปลือกส้มโอแช่อิ่ม': [
                'เปลือกส้มโอ', 'ส้มโอแช่อิ่ม', 'เปลือกส้มโอหวาน', 'เปลือกส้มโอเชื่อม',
                'ส้มโอดอง', 'เปลือกส้มโอแช่'
            ],
            'ยำทวาย': [
                'ทวายยำ', 'ยำทวายใต้', 'ยำผลไม้', 'ทวายปรุงรส',
                'ยำทวายสด', 'ทวายผัด'
            ],
            'เมี่ยงฝัน': [
                'เมี่ยงหวาน', 'ฝันเมี่ยง', 'เมี่ยงขนม', 'เมี่ยงไทย',
                'เมี่ยงหวานใต้', 'เมี่ยงของหวาน'
            ],
            'แป้งจี่': [
                'ขนมแป้งจี่', 'แป้งจี่หวาน', 'แป้งย่าง', 'ขนมแป้งย่าง',
                'แป้งจี่ไทย', 'แป้งจี่กรอบ'
            ]
        }
    
    def expand_menu_search_terms(self, menu_name: str) -> List[str]:
        """ขยายคำค้นหาสำหรับชื่อเมนู"""
        menu_lower = menu_name.lower()
        expanded_terms = [menu_name]
        
        # ค้นหาในรายการที่มี
        for main_menu, variations in self.enhanced_menu_variations.items():
            if main_menu.lower() == menu_lower:
                expanded_terms.extend(variations)
            elif menu_lower in [v.lower() for v in variations]:
                expanded_terms.append(main_menu)
                expanded_terms.extend(variations)
        
        return list(set(expanded_terms))
    
    def standardize_menu_name(self, menu_name: str) -> str:
        """แปลงชื่อเมนูให้เป็นมาตรฐาน"""
        menu_lower = menu_name.lower().strip()
        
        # ค้นหาชื่อมาตรฐาน
        for main_menu, variations in self.enhanced_menu_variations.items():
            if menu_lower == main_menu.lower():
                return main_menu
            elif menu_lower in [v.lower() for v in variations]:
                return main_menu
        
        return menu_name

class AdvancedDataProcessor:
    """คลาสสำหรับประมวลผลข้อมูลขั้นสูง"""
    
    def __init__(self):
        self.text_processor = ThaiTextProcessor()
        self.menu_enhancer = ThaiMenuEnhancer()
        self.nutrition_analyzer = NutritionAnalyzer()
        self.nutrition_db = NutritionDatabase()
        self.ingredient_converter = IngredientConverter()
        
    def process_thai_food_data(self, input_file: str, output_file: str, 
                              analyze_nutrition: bool = False, 
                              create_enhanced_search: bool = False) -> pd.DataFrame:
        """ประมวลผลข้อมูลอาหารไทยแบบครบถ้วน"""
        logger.info(f"เริ่มประมวลผลไฟล์: {input_file}")
        
        # โหลดข้อมูล
        try:
            df = pd.read_csv(input_file, encoding='utf-8')
            logger.info(f"โหลดข้อมูลสำเร็จ: {len(df)} แถว")
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการโหลดไฟล์: {e}")
            raise
        
        # ตรวจสอบคอลัมน์ที่จำเป็น
        required_columns = ['name', 'ingredient', 'method']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"ไม่พบคอลัมน์ที่จำเป็น: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # ทำความสะอาดข้อมูล
        logger.info("กำลังทำความสะอาดข้อมูล...")
        processed_df = self._clean_data(df)
        
        # เพิ่มคำค้นหาขั้นสูง
        if create_enhanced_search:
            logger.info("กำลังสร้างคำค้นหาขั้นสูง...")
            processed_df = self._add_enhanced_search_terms(processed_df)
        
        # วิเคราะห์โภชนาการ
        if analyze_nutrition:
            logger.info("กำลังวิเคราะห์ข้อมูลโภชนาการ...")
            processed_df = self._add_nutrition_analysis(processed_df)
        
        # บันทึกผลลัพธ์
        try:
            processed_df.to_csv(output_file, index=False, encoding='utf-8')
            logger.info(f"บันทึกผลลัพธ์สำเร็จ: {output_file}")
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการบันทึกไฟล์: {e}")
            raise
        
        return processed_df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """ทำความสะอาดข้อมูล"""
        df_clean = df.copy()
        
        # ทำความสะอาดชื่อเมนู
        df_clean['name'] = df_clean['name'].apply(
            lambda x: self.menu_enhancer.standardize_menu_name(str(x)) if pd.notna(x) else x
        )
        
        # ทำความสะอาดวัตถุดิบ
        df_clean['ingredient'] = df_clean['ingredient'].apply(
            lambda x: self.text_processor.clean_ingredient_text(str(x)) if pd.notna(x) else x
        )
        
        # ทำความสะอาดวิธีทำ
        df_clean['method'] = df_clean['method'].apply(
            lambda x: self.text_processor.clean_method_text(str(x)) if pd.notna(x) else x
        )
        
        # ลบแถวที่มีข้อมูลสำคัญหายไป
        df_clean = df_clean.dropna(subset=['name', 'ingredient'])
        
        # ลบแถวที่ซ้ำ
        df_clean = df_clean.drop_duplicates(subset=['name'], keep='first')
        
        logger.info(f"ข้อมูลหลังทำความสะอาด: {len(df_clean)} แถว")
        return df_clean
    
    def _add_enhanced_search_terms(self, df: pd.DataFrame) -> pd.DataFrame:
        """เพิ่มคำค้นหาขั้นสูง"""
        df_enhanced = df.copy()
        
        search_terms = []
        menu_variations = []
        
        for _, row in df_enhanced.iterrows():
            menu_name = row['name']
            
            # สร้างคำค้นหาเพิ่มเติม
            expanded_terms = self.menu_enhancer.expand_menu_search_terms(menu_name)
            search_terms.append('|'.join(expanded_terms))
            
            # เก็บรูปแบบการเขียนที่หลากหลาย
            variations = [term for term in expanded_terms if term != menu_name]
            menu_variations.append('|'.join(variations) if variations else '')
        
        df_enhanced['search_terms'] = search_terms
        df_enhanced['menu_variations'] = menu_variations
        
        return df_enhanced
    
    def _add_nutrition_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """เพิ่มการวิเคราะห์โภชนาการ"""
        df_nutrition = df.copy()
        
        # คอลัมน์โภชนาการ
        nutrition_columns = [
            'calories', 'protein', 'carbs', 'fat', 'fiber', 'sugar', 'sodium',
            'vitamin_a', 'vitamin_c', 'vitamin_d', 'vitamin_e', 'vitamin_k',
            'vitamin_b1', 'vitamin_b2', 'vitamin_b6', 'vitamin_b12', 'folate', 'niacin',
            'calcium', 'iron', 'magnesium', 'phosphorus', 'potassium', 'zinc',
            'ingredient_count', 'cooking_method', 'difficulty_level'
        ]
        
        # เริ่มต้นคอลัมน์
        for col in nutrition_columns:
            df_nutrition[col] = 0.0
        
        # วิเคราะห์แต่ละเมนู
        for idx, row in df_nutrition.iterrows():
            try:
                recipe_name = row['name']
                ingredients = row['ingredient']
                
                # วิเคราะห์โภชนาการ
                nutrition_data = self.nutrition_analyzer.analyze_ingredients(
                    ingredients, recipe_name, apply_cooking_adjustments=True
                )
                total_nutrition = self.nutrition_analyzer.calculate_total_nutrition(nutrition_data)
                
                # เก็บข้อมูลโภชนาการ
                df_nutrition.at[idx, 'calories'] = total_nutrition.calories
                df_nutrition.at[idx, 'protein'] = total_nutrition.protein
                df_nutrition.at[idx, 'carbs'] = total_nutrition.carbs
                df_nutrition.at[idx, 'fat'] = total_nutrition.fat
                df_nutrition.at[idx, 'fiber'] = total_nutrition.fiber
                df_nutrition.at[idx, 'sugar'] = total_nutrition.sugar
                df_nutrition.at[idx, 'sodium'] = total_nutrition.sodium
                
                # วิตามิน
                df_nutrition.at[idx, 'vitamin_a'] = total_nutrition.vitamin_a
                df_nutrition.at[idx, 'vitamin_c'] = total_nutrition.vitamin_c
                df_nutrition.at[idx, 'vitamin_d'] = total_nutrition.vitamin_d
                df_nutrition.at[idx, 'vitamin_e'] = total_nutrition.vitamin_e
                df_nutrition.at[idx, 'vitamin_k'] = total_nutrition.vitamin_k
                df_nutrition.at[idx, 'vitamin_b1'] = total_nutrition.vitamin_b1
                df_nutrition.at[idx, 'vitamin_b2'] = total_nutrition.vitamin_b2
                df_nutrition.at[idx, 'vitamin_b6'] = total_nutrition.vitamin_b6
                df_nutrition.at[idx, 'vitamin_b12'] = total_nutrition.vitamin_b12
                df_nutrition.at[idx, 'folate'] = total_nutrition.folate
                df_nutrition.at[idx, 'niacin'] = total_nutrition.niacin
                
                # แร่ธาตุ
                df_nutrition.at[idx, 'calcium'] = total_nutrition.calcium
                df_nutrition.at[idx, 'iron'] = total_nutrition.iron
                df_nutrition.at[idx, 'magnesium'] = total_nutrition.magnesium
                df_nutrition.at[idx, 'phosphorus'] = total_nutrition.phosphorus
                df_nutrition.at[idx, 'potassium'] = total_nutrition.potassium
                df_nutrition.at[idx, 'zinc'] = total_nutrition.zinc
                
                # ข้อมูลเพิ่มเติม
                df_nutrition.at[idx, 'ingredient_count'] = len(nutrition_data)
                df_nutrition.at[idx, 'cooking_method'] = self._detect_cooking_method(recipe_name)
                df_nutrition.at[idx, 'difficulty_level'] = self._assess_difficulty(ingredients, recipe_name)
                
                if idx % 10 == 0:
                    logger.info(f"ประมวลผลโภชนาการแล้ว: {idx + 1}/{len(df_nutrition)} เมนู")
                    
            except Exception as e:
                logger.warning(f"ไม่สามารถวิเคราะห์โภชนาการสำหรับ {row['name']}: {e}")
                continue
        
        return df_nutrition
    
    def _detect_cooking_method(self, recipe_name: str) -> str:
        """ตรวจจับวิธีการทำอาหาร"""
        name_lower = recipe_name.lower()
        
        method_keywords = {
            'ทอด': ['ทอด'],
            'ผัด': ['ผัด'],
            'ต้ม': ['ต้ม', 'งบ'],
            'แกง': ['แกง'],
            'ยำ': ['ยำ', 'ส้มตำ'],
            'ย่าง': ['ย่าง', 'ปิ้ง'],
            'นึ่ง': ['นึ่ง'],
            'อบ': ['อบ'],
            'ตุ๋น': ['ตุ๋น'],
            'น้ำพริก': ['น้ำพริก'],
            'ห่อหมก': ['ห่อหมก']
        }
        
        for method, keywords in method_keywords.items():
            if any(keyword in name_lower for keyword in keywords):
                return method
        
        return 'อื่นๆ'
    
    def _assess_difficulty(self, ingredients: str, recipe_name: str) -> str:
        """ประเมินความยากในการทำอาหาร"""
        # นับจำนวนวัตถุดิบ
        ingredient_lines = [line.strip() for line in ingredients.split('\n') 
                           if line.strip() and line.strip().startswith('-')]
        ingredient_count = len(ingredient_lines)
        
        # เมนูที่ซับซ้อน
        complex_dishes = [
            'ห่อหมก', 'บรรจุไส้', 'ทรงเครื่อง', 'เครื่องเทศ', 'พุดชาจีน',
            'มักกะโรนี', 'ขนมกลีบ', 'เปียกปูน', 'ฉี่ฉู่', 'สาลี่โคโก้'
        ]
        
        name_lower = recipe_name.lower()
        is_complex = any(complex_word in name_lower for complex_word in complex_dishes)
        
        if ingredient_count <= 4 and not is_complex:
            return 'ง่าย'
        elif ingredient_count > 8 or is_complex:
            return 'ยาก'
        else:
            return 'ปานกลาง'
    
    def create_nutrition_database(self, df: pd.DataFrame = None):
        """สร้างฐานข้อมูลโภชนาการ"""
        logger.info("กำลังสร้างฐานข้อมูลโภชนาการ...")
        
        # เริ่มต้นฐานข้อมูล
        self.nutrition_db.init_database()
        
        if df is not None:
            # เพิ่มข้อมูลจาก DataFrame
            for _, row in df.iterrows():
                if 'calories' in row and pd.notna(row['calories']):
                    try:
                        # สร้าง NutritionInfo object
                        from nutrition_analyzer import NutritionInfo
                        nutrition_info = NutritionInfo(
                            name=row['name'],
                            calories=float(row.get('calories', 0)),
                            protein=float(row.get('protein', 0)),
                            carbs=float(row.get('carbs', 0)),
                            fat=float(row.get('fat', 0)),
                            fiber=float(row.get('fiber', 0)),
                            sugar=float(row.get('sugar', 0)),
                            sodium=float(row.get('sodium', 0)),
                            vitamin_a=float(row.get('vitamin_a', 0)),
                            vitamin_c=float(row.get('vitamin_c', 0)),
                            vitamin_d=float(row.get('vitamin_d', 0)),
                            vitamin_e=float(row.get('vitamin_e', 0)),
                            vitamin_k=float(row.get('vitamin_k', 0)),
                            vitamin_b1=float(row.get('vitamin_b1', 0)),
                            vitamin_b2=float(row.get('vitamin_b2', 0)),
                            vitamin_b6=float(row.get('vitamin_b6', 0)),
                            vitamin_b12=float(row.get('vitamin_b12', 0)),
                            folate=float(row.get('folate', 0)),
                            niacin=float(row.get('niacin', 0)),
                            calcium=float(row.get('calcium', 0)),
                            iron=float(row.get('iron', 0)),
                            magnesium=float(row.get('magnesium', 0)),
                            phosphorus=float(row.get('phosphorus', 0)),
                            potassium=float(row.get('potassium', 0)),
                            zinc=float(row.get('zinc', 0))
                        )
                        
                        # บันทึกลงฐานข้อมูล
                        self.nutrition_db.cache_nutrition(row['name'], nutrition_info)
                        
                    except Exception as e:
                        logger.warning(f"ไม่สามารถบันทึกข้อมูลโภชนาการสำหรับ {row['name']}: {e}")
        
        logger.info("สร้างฐานข้อมูลโภชนาการเสร็จสิ้น")
    
    def generate_processing_report(self, original_df: pd.DataFrame, 
                                 processed_df: pd.DataFrame) -> dict:
        """สร้างรายงานการประมวลผล"""
        report = {
            'processing_info': {
                'original_records': len(original_df),
                'processed_records': len(processed_df),
                'removed_records': len(original_df) - len(processed_df),
                'processing_date': datetime.now().isoformat()
            },
            'data_quality': {
                'duplicate_removed': len(original_df) - len(original_df.drop_duplicates(subset=['name'])),
                'missing_ingredients': len(original_df[original_df['ingredient'].isna()]),
                'missing_methods': len(original_df[original_df['method'].isna()]),
            },
            'nutrition_analysis': {
                'recipes_with_nutrition': 0,
                'avg_calories': 0,
                'avg_protein': 0,
                'cooking_methods': {},
                'difficulty_levels': {}
            }
        }
        
        # วิเคราะห์ข้อมูลโภชนาการ
        if 'calories' in processed_df.columns:
            nutrition_df = processed_df[processed_df['calories'] > 0]
            report['nutrition_analysis']['recipes_with_nutrition'] = len(nutrition_df)
            
            if len(nutrition_df) > 0:
                report['nutrition_analysis']['avg_calories'] = nutrition_df['calories'].mean()
                report['nutrition_analysis']['avg_protein'] = nutrition_df['protein'].mean()
        
        # วิเคราะห์วิธีการทำอาหาร
        if 'cooking_method' in processed_df.columns:
            report['nutrition_analysis']['cooking_methods'] = processed_df['cooking_method'].value_counts().to_dict()
        
        # วิเคราะห์ระดับความยาก
        if 'difficulty_level' in processed_df.columns:
            report['nutrition_analysis']['difficulty_levels'] = processed_df['difficulty_level'].value_counts().to_dict()
        
        return report

def main():
    """ฟังก์ชันหลัก"""
    parser = argparse.ArgumentParser(description='ประมวลผลข้อมูลสูตรอาหารไทยขั้นสูง')
    parser.add_argument('--input', type=str, default='thai_food_sample.csv',
                        help='ไฟล์ข้อมูลเข้า')
    parser.add_argument('--output', type=str, default='thai_food_processed.csv',
                        help='ไฟล์ข้อมูลออก')
    parser.add_argument('--analyze-nutrition', action='store_true',
                        help='เปิดการวิเคราะห์โภชนาการ')
    parser.add_argument('--create-nutrition-db', action='store_true',
                        help='สร้างฐานข้อมูลโภชนาการ')
    parser.add_argument('--enhanced-search', action='store_true',
                        help='เพิ่มคำค้นหาขั้นสูง')
    parser.add_argument('--report', type=str, default='processing_report.json',
                        help='ไฟล์รายงานการประมวลผล')
    
    args = parser.parse_args()
    
    # ตรวจสอบไฟล์เข้า
    if not Path(args.input).exists():
        logger.error(f"ไม่พบไฟล์เข้า: {args.input}")
        return
    
    # สร้าง processor
    processor = AdvancedDataProcessor()
    
    try:
        # โหลดข้อมูลต้นฉบับ
        original_df = pd.read_csv(args.input, encoding='utf-8')
        
        # ประมวลผลข้อมูล
        processed_df = processor.process_thai_food_data(
            args.input,
            args.output,
            analyze_nutrition=args.analyze_nutrition,
            create_enhanced_search=args.enhanced_search
        )
        
        # สร้างฐานข้อมูลโภชนาการ
        if args.create_nutrition_db:
            processor.create_nutrition_database(processed_df)
        
        # สร้างรายงาน
        report = processor.generate_processing_report(original_df, processed_df)
        
        # บันทึกรายงาน
        with open(args.report, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # แสดงสรุป
        print("\n" + "="*60)
        print("🍲 การประมวลผลข้อมูลอาหารไทยเสร็จสิ้น!")
        print("="*60)
        print(f"📊 ข้อมูลต้นฉบับ: {report['processing_info']['original_records']} เมนู")
        print(f"✅ ข้อมูลที่ประมวลผลแล้ว: {report['processing_info']['processed_records']} เมนู")
        print(f"🗑️  ข้อมูลที่ลบออก: {report['processing_info']['removed_records']} เมนู")
        
        if args.analyze_nutrition:
            nutrition_count = report['nutrition_analysis']['recipes_with_nutrition']
            print(f"🧪 เมนูที่วิเคราะห์โภชนาการ: {nutrition_count} เมนู")
            if nutrition_count > 0:
                avg_cal = report['nutrition_analysis']['avg_calories']
                avg_pro = report['nutrition_analysis']['avg_protein']
                print(f"⚡ แคลอรี่เฉลี่ย: {avg_cal:.1f} kcal")
                print(f"🥩 โปรตีนเฉลี่ย: {avg_pro:.1f} g")
        
        if args.enhanced_search:
            print("🔍 เพิ่มคำค้นหาขั้นสูงแล้ว")
        
        if args.create_nutrition_db:
            print("💾 สร้างฐานข้อมูลโภชนาการแล้ว")
        
        print(f"📋 รายงาน: {args.report}")
        print(f"💽 ไฟล์ผลลัพธ์: {args.output}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาด: {e}")
        raise

if __name__ == "__main__":
    main()
