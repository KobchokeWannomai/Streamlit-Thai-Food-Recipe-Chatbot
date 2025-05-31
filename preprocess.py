import os
from typing import Optional, Dict, List
import streamlit as st

class Config:
    """คลาสสำหรับจัดการการตั้งค่าระบบ - ปรับปรุงแล้ว"""
    
    # ไฟล์ข้อมูล
    DATA_PATH = "thai_food_processed.csv"
    EMBEDDINGS_PATH = "embeddings.pkl"
    MODEL_PATH = "model"
    NUTRITION_CACHE_PATH = "nutrition_cache.db"
    
    # การตั้งค่าโมเดล
    SENTENCE_TRANSFORMER_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'
    SIMILARITY_THRESHOLD = 0.3
    ENHANCED_SIMILARITY_THRESHOLD = 0.25
    MAX_SEARCH_RESULTS = 5
    
    # คีย์ API และการตั้งค่าภายนอก
    USDA_API_KEY: Optional[str] = os.getenv("USDA_API_KEY")
    NUTRITIONIX_API_KEY: Optional[str] = os.getenv("NUTRITIONIX_API_KEY")
    NUTRITIONIX_APP_ID: Optional[str] = os.getenv("NUTRITIONIX_APP_ID")
    
    # การตั้งค่าฐานข้อมูลโภชนาการ
    NUTRITION_CACHE_EXPIRE_DAYS = 30  # วัน
    MAX_API_CALLS_PER_DAY = 1000
    API_RATE_LIMIT_PER_MINUTE = 30
    
    # การตั้งค่า UI
    PAGE_TITLE = "แชทบอทสูตรอาหารไทยพร้อมการวิเคราะห์โภชนาการขั้นสูง"
    PAGE_ICON = "🍲"
    
    # เกณฑ์โภชนาการเริ่มต้น
    DEFAULT_MAX_CALORIES = 500
    DEFAULT_MIN_PROTEIN = 0.0
    DEFAULT_MAX_CARBS = 100.0
    DEFAULT_MAX_FAT = 50.0
    
    # ข้อมูลโภชนาการเปรียบเทียบ (ค่าแนะนำต่อวัน)
    DAILY_RECOMMENDED = {
        'calories_adult_male': 2500,
        'calories_adult_female': 2000,
        'protein_adult_male': 56,  # กรัม
        'protein_adult_female': 46,  # กรัม
        'carbs_adult': 300,  # กรัม
        'fat_adult': 70,  # กรัม
        'fiber_adult': 25,  # กรัม
        'sodium_adult': 2300,  # มิลลิกรัม
        'vitamin_c_adult': 90,  # มิลลิกรัม
        'calcium_adult': 1000,  # มิลลิกรัม
        'iron_adult_male': 8,  # มิลลิกรัม
        'iron_adult_female': 18,  # มิลลิกรัม
    }
    
    # รายการวัตถุดิบไทยพื้นฐาน
    THAI_BASIC_INGREDIENTS = [
        "หมู", "ไก่", "เนื้อ", "กุ้ง", "ปลา", "ไข่", 
        "กะหล่ำปลี", "คะน้า", "ผักบุ้ง", "ผักกาด",
        "น้ำปลา", "กะทิ", "น้ำตาล", "เกลือ", "พริก",
        "ข้าว", "แป้ง", "น้ำมัน", "กระเทียม", "หอม"
    ]
    
    # การจัดหมวดหมู่อาหาร
    FOOD_CATEGORIES = {
        'protein': ['หมู', 'ไก่', 'เนื้อ', 'กุ้ง', 'ปลา', 'ไข่', 'ถั่ว'],
        'vegetables': ['กะหล่ำปลี', 'คะน้า', 'ผักบุ้ง', 'ผักกาด', 'แตงกวา', 'มะเขือ'],
        'carbs': ['ข้าว', 'แป้ง', 'ขนมจีน', 'บะหมี่'],
        'seasonings': ['น้ำปลา', 'กะทิ', 'น้ำตาล', 'เกลือ', 'พริก', 'กระเทียม', 'หอม'],
        'oils_fats': ['น้ำมัน', 'เนย', 'มัน']
    }
    
    # การตั้งค่าขั้นสูงสำหรับการค้นหา - ปรับปรุงใหม่
    SEARCH_ENHANCEMENT = {
        'query_expansions': {
            'ไข่': ['ไข่ไก่', 'ไข่เป็ด', 'ไข่ดาว', 'ไข่เจียว', 'ไข่ต้ม', 'ไข่กระจัง', 'ไข่จ่อม', 'ไข่ม้วน', 'ไข่สามชั้น', 'ไข่ในรัง', 'ไข่เค็ม', 'ไข่สวรรค์', 'ไข่หวานฝอย', 'ไข่น้อค', 'ไข่ช่อนรูป', 'ไข่ตุ๋น'],
            'หมู': ['เนื้อหมู', 'หมูสับ', 'หมูย่าง', 'หมูทอด', 'หมูแนมสด', 'หมูทอดเค็ม', 'สลัดหมูกรอบ', 'ไส้กรอกหมู', 'หมูยอ'],
            'ไก่': ['เนื้อไก่', 'ไก่ย่าง', 'ไก่ทอด', 'ไก่ต้ม', 'ไก่ยำ', 'ไก่หยอง', 'งบไก่', 'ไก่ทันสมัย', 'กงเชียงไก่นา'],
            'กุ้ง': ['กุ้งนาง', 'กุ้งฝอย', 'กุ้งแห้ง', 'กุ้งทาพริกไทยกระเทียม', 'กุ้งเผา', 'กุ้งทอด', 'กุ้งแฝง', 'กุ้งแห้งปรุงขิง', 'เกี๊ยวกุ้ง', 'กุ้งทอดปรุงรส'],
            'ปลา': ['ปลาทู', 'ปลาดุก', 'ปลาช่อน', 'ปลาอบ', 'ปลาแนม', 'ปลาทูทอดปรุง', 'เมี่ยงปลาทู', 'ปลากุเลาทอดปรุงหน้า', 'ปลาทูชุบแป้งทอด', 'ปลาทูแนม', 'ปลาทูร่องสวน', 'ปลาแห้งปรุงกระเทียมดอง', 'ปลานึ่งกับมะเขือเทศ', 'ปลาช่อนต้มเค็มกับก๋งฉ่าย', 'ยำไข่ปลาดุก', 'ผัดไข่ปลาตะเพียน', 'ปลาโฉมตรู', 'งบปลาทู', 'ยำปลาหมึกสด'],
            'ผัด': ['ผัดไทย', 'ผัดกะเพรา', 'ผัดซีอิ๊ว', 'ผัดคะน้า', 'ผัดผักกาดขาว', 'ผัดหัวผักกาดเค็ม', 'ยอดแคผัดกรอบ', 'ผัดห่วงอาลัย', 'ผัดต้นผักกาดดอง', 'ผัดคะน้ากับซีเซ็กฉ่าย', 'ผัดเต้าหู้เหลือง', 'เนื้อผัดเทียมแหนม', 'ก๋วยเตี๋ยวผัด'],
            'แกง': ['แกงเขียวหวาน', 'แกงเผ็ด', 'แกงส้ม', 'แกงมัสมั่น', 'แกงคั่วฟักทองกับกุ้งตะเข็บ', 'แกงยา', 'แกงเลียง', 'แกงเลียงขี้เหล็ก', 'แกงเปลือกแตงโม', 'แกงต้มกะทิฟักทอง', 'แกงต้มกะทิฟันเขียว', 'แกงเผ็ดน้ำมันหมู', 'แกงเผ็ดหมู', 'แกงไส้กรอกหมูแห้ง', 'แกงเห็ดฟางกับมะเขือเทศ', 'แกงจืดลูกชิ้นกับจีฉ่าย', 'แกงจืดต้นคะน้า', 'แกงต้มเค็ม', 'แกงต้มส้ม', 'แกงจืดชนิดตีน้ำมัน', 'แกงส้มถั่วฝักยาว', 'แกงต้มหมูกับสัปรส'],
            'ต้ม': ['ต้มยำ', 'ต้มข่า', 'ต้มจืด', 'ต้มยำกะทิ', 'ต้มยำหอยแมลงภู่', 'ต้มยำปลา', 'ต้มโคล้ง', 'ต้มโคล้งกุ้ง', 'ข้าวต้มน้ำวุ้น', 'ข้าวต้มไข่', 'ไข่ต้มปรุงจับฉ่าย', 'ตับตุ๋น', 'นกพิราบตุ๋น', 'ฟักตุ๋น', 'ไข่ตุ๋น', 'ต้มหน่อไม้ไผ่ตงกับหมู'],
            'ยำ': ['ยำวุ้นเส้น', 'ยำถั่วพู', 'ยำมะม่วง', 'ยำไข่ดาว', 'ยำไข่เจียวเครื่องหมี่', 'ยำไข่แมงดา', 'ยำส้มโอ', 'ยำพริก', 'ยำทวาย', 'ยำทวายสมัยใหม่', 'ยำขมิ้นขาวกับกุ้งเค็ม'],
            'ส้ม': ['ส้มตำ', 'ส้มตำไทย', 'ส้มตำปู', 'ส้มตำแตงร้าน'],
            'ลาบ': ['ลาบหมู', 'ลาบไก่', 'ลาบเนื้อ'],
            'ข้าว': ['ข้าวผัด', 'ข้าวต้ม', 'ข้าวเหนียว', 'ข้าวเม่าทอด', 'ข้าวชวา', 'ข้าวเม่าคลุก'],
            'ทอด': ['ทอด', 'กล้วยทอด', 'กล้วยบวชชี', 'ฟักทองทอด', 'ไข่เค็มทอดกรอบ', 'เนื้อเครื่องเทศทอด'],
            'น้ำพริก': ['น้ำพริกเผา', 'น้ำพริกจิ้มผักดิบ', 'น้ำพริกพะม่า', 'น้ำพริกเครื่องสด', 'น้ำพริกปลาเค็ม', 'น้ำพริกปูเค็ม', 'น้ำพริกก้อย', 'น้ำพริกไข่เค็ม'],
            'ขนม': ['ขนมต้มแดง', 'ขนมกลีบลำดวน', 'ขนมสาลี่โคโก้', 'ขนมเปียกปูน', 'ขนมจีบหมูสับ'],
            'ไส้กรอก': ['ไส้กรอกหมู', 'ไส้กรอกข้าว', 'กงเชียงสด'],
            'เส้น': ['บะหมี่', 'ก๋วยเตี๋ยว', 'หมี่หน้าเนื้อ', 'บะหมี่ทรงเครื่อง', 'บะหมี่สำเร็จ', 'ก๋วยเตี๋ยวไส้ไข่'],
            'หอย': ['หอยแมลงภู่', 'หอยนางรม', 'ห่อหมกหอยแมลงภู่'],
            'เครื่องดื่ม': ['สาเกเชื่อม', 'ลอยน้ำดอกไม้สด'],
            'ของหวาน': ['สังขยา', 'มะตูมเชื่อม', 'สาคูเปียก', 'เปลือกส้มโอแช่อิ่ม', 'เมี่ยงฝัน', 'แป้งจี่', 'ฉี่ฉู่เมืองปราณ', 'ทองม้วนเค็ม'],
            'ผลไม้': ['มะละกอโถบรรจุใส้', 'ละมุดมีใส้'],
            'ผัก': ['ยอดแคผัดกรอบ', 'ถั่วแนม'],
            'เนื้อ': ['เนื้อเครื่องเทศทอด', 'เนื้อผัดเทียมแหนม', 'บี๊ฟที', 'นกปากซ่อมสับ'],
            'ซอส': ['ซ้อสมะเขือเทศซุป', 'เต้าเจี้ยวปรุงรส'],
            'เต้าหู้': ['เต้าหู้ยี้ปรุงรส', 'ผัดเต้าหู้เหลือง'],
            'มะเขือ': ['มะเขือเทศกุ้งเผา', 'มะเขือเทศหน้านวล', 'มะเขือยาวเครื่องเทศ']
        },
        'cooking_methods': [
            'ผัด', 'ต้ม', 'ทอด', 'ย่าง', 'นึ่ง', 'ต้น', 'แกง', 'ยำ', 'ลาบ', 'น้ำพริก',
            'คั่ว', 'ปิ้ง', 'เผา', 'อบ', 'ตุ๋น', 'ห่อหมก', 'งบ', 'คลุก', 'จิ้ม', 'ปรุง'
        ]
    }
    
    # การปรับแต่งการทำอาหาร - เพิ่มเมนูใหม่
    COOKING_ADJUSTMENTS = {
        'oil_absorption_rates': {
            'ทอด': 0.1,          # ทอดทั่วไป ดูดซับน้ำมัน 10%
            'ทอดแปลง': 0.05,     # ทอดแปลงน้อย
            'ทอดกรอบ': 0.15,     # ทอดกรอบมาก
            'ผัด': 0.7,          # ผัดดูดซับน้ำมันมาก
            'ผัดแห้ง': 0.8,      # ผัดแห้งดูดซับมากที่สุด
        },
        'missing_ingredients_common': {
            'ไข่เจียว': ['น้ำมันพืช'],
            'ไข่ดาว': ['น้ำมันพืช'],
            'ผัดกะเพรา': [],  # มีน้ำมันอยู่แล้วส่วนใหญ่
            'ต้มยำ': ['น้ำ'],
            'แกงเขียวหวาน': ['น้ำ'],
            'กุ้งทาพริกไทยกระเทียม': ['น้ำมันพืช'],
            'ข้าวเม่าทอด': ['น้ำมันพืช'],
            'ปลาทูทอดปรุง': ['น้ำมันพืช'],
            'งบปลาทู': ['น้ำ'],
            'กล้วยบวชชี': ['น้ำมันพืช'],
            'กล้วยทอด': ['น้ำมันพืช'],
            'ฟักทองทอด': ['น้ำมันพืช'],
            'เนื้อเครื่องเทศทอด': ['น้ำมันพืช'],
            'หมูทอดเค็ม': ['น้ำมันพืช'],
            'กุ้งทอดปรุงรส': ['น้ำมันพืช'],
            'ไข่เค็มทอดกรอบ': ['น้ำมันพืช']
        }
    }
    
    # ข้อความแสดงผล
    MESSAGES = {
        'th': {
            'welcome': 'ยินดีต้อนรับสู่ระบบแชทบอทสูตรอาหารไทยพร้อมข้อมูลโภชนาการขั้นสูง',
            'search_placeholder': 'ถามเกี่ยวกับอาหารไทย หรือค้นหาตามโภชนาการ...',
            'analyzing_nutrition': 'กำลังวิเคราะห์ข้อมูลโภชนาการ...',
            'analyzing_enhanced': 'กำลังวิเคราะห์ข้อมูลโภชนาการขั้นสูง...',
            'connecting_api': 'กำลังเชื่อมต่อ API...',
            'no_results': 'ไม่พบสูตรอาหารที่ตรงกับคำค้นหา',
            'nutrition_analysis_failed': 'ไม่สามารถวิเคราะห์ข้อมูลโภชนาการได้',
            'recipe_found': 'พบสูตรอาหารที่คุณต้องการพร้อมข้อมูลโภชนาการ',
            'enhanced_analysis': 'ข้อมูลที่ปรับปรุงแล้วด้วยการคำนวณขั้นสูง',
            'api_connection_success': 'เชื่อมต่อ API สำเร็จ',
            'api_connection_failed': 'การเชื่อมต่อ API ล้มเหลว',
            'using_local_data': 'ใช้ข้อมูลภายในระบบ',
        },
        'en': {
            'welcome': 'ยินดีต้อนรับสู่แชทบอทสูตรอาหารไทยขั้นสูงพร้อมการวิเคราะห์โภชนาการขั้นสูง',
            'search_placeholder': 'ถามเกี่ยวกับอาหารไทยหรือค้นหาตามโภชนาการ...',
            'analyzing_nutrition': 'กำลังวิเคราะห์ข้อมูลโภชนาการ...',
            'analyzing_enhanced': 'กำลังทำการวิเคราะห์โภชนาการขั้นสูง...',
            'connecting_api': 'กำลังเชื่อมต่อกับ API...',
            'no_results': 'ไม่พบสูตรอาหารที่ตรงกับการค้นหาของคุณ',
            'nutrition_analysis_failed': 'ไม่สามารถวิเคราะห์ข้อมูลโภชนาการได้',
            'recipe_found': 'พบสูตรอาหารที่คุณกำลังมองหาพร้อมข้อมูลโภชนาการ',
            'enhanced_analysis': 'ข้อมูลขั้นสูงพร้อมการคำนวณขั้นสูง',
            'api_connection_success': 'การเชื่อมต่อ API สำเร็จ',
            'api_connection_failed': 'การเชื่อมต่อ API ล้มเหลว',
            'using_local_data': 'ใช้ฐานข้อมูลภายใน',
        }
    }
    
    # รายการเมนูอาหารไทยทั้งหมดจากชุดข้อมูล - เพิ่มเติม
    COMPLETE_THAI_MENU_LIST = [
        'ผัดกะเพรา', 'ต้มยำกุ้ง', 'ส้มตำ', 'แกงเขียวหวาน', 'ผัดไทย', 'ไข่เจียว', 'ไข่ดาว',
        'กุ้งทาพริกไทยกระเทียม', 'ข้าวเม่าทอด', 'เปรี้ยวหวานไข่ม้วน', 'ไข่จ่อม', 'งบปลาทู',
        'น้ำพริกจิ้มผักดิบ', 'ลอยน้ำดอกไม้สด', 'ยำไข่ปลาดุก', 'ปลาทูทอดปรุง', 'ต้มยำกะทิ',
        'ไก่ยำ', 'กล้วยบวชชี', 'แกงคั่วฟักทองกับกุ้งตะเข็บ', 'ไส้กรอกหมู', 'เมี่ยงปลาทู',
        'นกพิราบตุ๋น', 'ปลากุเลาทอดปรุงหน้า', 'ห่อหมกหอยแมลงภู่', 'งบไก่', 'ไข่กระจัง',
        'หน้าตั้งแขก', 'ยำปลาหมึกสด', 'ฟักตุ๋น', 'ยำถั่วพู', 'ซ้อสมะเขือเทศซุป',
        'ปลาทูชุบแป้งทอด', 'บะหมี่ทรงเครื่อง', 'มะตูมเชื่อม', 'สังขยา', 'กุ้งแห้งปรุงขิง',
        'แกงเผ็ดน้ำมันหมู', 'ผัดต้นผักกาดดอง', 'ยำขมิ้นขาวกับกุ้งเค็ม', 'เกี๊ยวกุ้ง',
        'ข้าวต้มน้ำวุ้น', 'หมี่หน้าเนื้อ', 'ขนมสาลี่โคโก้', 'สาคูเปียก', 'ห่อหมกไข่',
        'แกงยา', 'เนื้อเครื่องเทศทอด', 'ขนมต้มแดง', 'ปลาอบ', 'ตับตุ๋น', 'ไก่หยอง',
        'สลัดหมูกรอบ', 'ยอดแคผัดกรอบ', 'ข้าวชวา', 'มะเขือเทศกุ้งเผา', 'ไข่ต้มปรุงจับฉ่าย',
        'ยำไข่ดาว', 'นกปากซ่อมสับ', 'แกงจืดชนิดตีน้ำมัน', 'กะหรี่พัฟฟ์', 'ปลาทูแนม',
        'ขนมกลีบลำดวน', 'แกงเผ็ดหมู', 'หมูแนมสด', 'มะเขือยาวเครื่องเทศ', 'ไส้กรอกข้าว',
        'น้ำเมี่ยง', 'ปลานึ่งกับมะเขือเทศ', 'น้ำพริกพะม่า', 'ไก่ต้มขนมจีน', 'ข้าวเม่าคลุก',
        'กุ้งเผากับมะเขือเปราะ', 'มะละกอโถบรรจุใส้', 'พุดชาจีนเชื่อมไส้เกาลัด', 'ข้าวต้มไข่',
        'ปลาแนม', 'แกงต้มกะทิฟักทอง', 'ละมุดมีใส้', 'บะหมี่สำเร็จ', 'ก๋วยเตี๋ยวไส้ไข่',
        'ต้มหน่อไม้ไผ่ตงกับหมู', 'ผัดห่วงอาลัย', 'ยำพริก', 'น้ำพริกเผา', 'หมูทอดเค็ม',
        'เต้าหู้ยี้ปรุงรส', 'กล้วยทอด', 'แกงต้มส้ม', 'ต้มยำปลา', 'แกงเห็ดฟางกับมะเขือเทศ',
        'ต้มโคล้งกุ้ง', 'แกงจืดลูกชิ้นกับจีฉ่าย', 'ไข่สามชั้น', 'มันผรั่งบดใส่ไส้', 'ไข่ในรัง',
        'ปลาโฉมตรู', 'ไข่เค็มชั้น', 'แกงเลียงขี้เหล็ก', 'ผัดคะน้า', 'ปลาช่อนต้มเค็มกับก๋งฉ่าย',
        'ก๋วยเตี๋ยวผัด', 'ไข่เค็มทอดกรอบ', 'แกงจืดต้นคะน้า', 'สาเกเชื่อม', 'ไข่สวรรค์',
        'มักกะโรนีรังแตน', 'น้ำพริกเครื่องสด', 'ปลาทูร่องสวน', 'แกงเปลือกแตงโม',
        'ต้มยำหอยแมลงภู่', 'แกงเลียง', 'แกงต้มกะทิฟันเขียว', 'ถั่วแนม', 'ผัดไข่ปลาตะเพียน',
        'ส้มตำแตงร้าน', 'ผัดคะน้ากับซีเซ็กฉ่าย', 'ต้มโคล้ง', 'ยำทวายสมัยใหม่', 'ผัดผักกาดขาว',
        'ผัดหัวผักกาดเค็ม', 'ไข่ช่อนรูป', 'ยำไข่เจียวเครื่องหมี่', 'กุ้งแฝง', 'บี๊ฟที',
        'กงเชียงสด', 'ไข่ตุ๋น', 'แกงต้มเค็ม', 'กุ้งทอดปรุงรส', 'ยำไข่แมงดา', 'ยำส้มโอ',
        'ไข่หวานฝอย', 'ฟักทองทอด', 'แกงไส้กรอกหมูแห้ง', 'มะเขือเทศหน้านวล',
        'ปลาแห้งปรุงกระเทียมดอง', 'ฉี่ฉู่เมืองปราณ', 'ทองม้วนเค็ม', 'เปลือกส้มโอแช่อิ่ม',
        'ยำทวาย', 'ไข่น้อค', 'เมี่ยงฝัน', 'ไข่ม้วน', 'แป้งจี่', 'น้ำพริกปลาเค็ม',
        'กงเชียงไก่นา', 'ไข่ดาวหน้ากุ้ง', 'น้ำพริกปูเค็ม', 'เต้าเจี้ยวปรุงรส', 'ขนมจีบหมูสับ',
        'แกงส้มถั่วฝักยาว', 'แกงต้มหมูกับสัปรส', 'ผัดเต้าหู้เหลือง', 'ไก่ทันสมัย',
        'เนื้อผัดเทียมแหนม', 'ไข่น้อคอีกอย่างหนึ่ง', 'น้ำพริกก้อย', 'ขนมเปียกปูน',
        'น้ำเต้าบรรจุไส้', 'น้ำพริกไข่เค็ม'
    ]
    
    @classmethod
    def get_message(cls, key: str, lang: str = 'th') -> str:
        """ดึงข้อความแสดงผล"""
        return cls.MESSAGES.get(lang, cls.MESSAGES['th']).get(key, key)
    
    @classmethod
    def is_api_configured(cls) -> dict:
        """ตรวจสอบการตั้งค่า API"""
        return {
            'usda': bool(cls.USDA_API_KEY and cls.USDA_API_KEY != "your_usda_api_key_here"),
            'nutritionix': bool(cls.NUTRITIONIX_API_KEY and cls.NUTRITIONIX_APP_ID and 
                              cls.NUTRITIONIX_API_KEY != "your_nutritionix_api_key_here"),
        }
    
    @classmethod
    def get_nutrition_source_priority(cls) -> list:
        """ลำดับความสำคัญของแหล่งข้อมูลโภชนาการ"""
        sources = ['thai_database']  # เริ่มจากฐานข้อมูลไทยเสมอ
        
        if cls.is_api_configured()['usda']:
            sources.append('usda_api')
        
        if cls.is_api_configured()['nutritionix']:
            sources.append('nutritionix_api')
        
        return sources
    
    @classmethod
    def get_streamlit_settings(cls) -> dict:
        """ดึงการตั้งค่าจาก Streamlit session state"""
        if 'settings' not in st.session_state:
            st.session_state.settings = {
                'usda_enabled': False,
                'nutritionix_enabled': False,
                'use_external_recipe_data': False,
                'accurate_cooking_calculation': False,
                'enhanced_search': True,
                'auto_scroll': True,
                'api_timeout': 10,
                'cache_duration': 24  # ชั่วโมง
            }
        return st.session_state.settings
    
    @classmethod
    def update_streamlit_settings(cls, new_settings: dict):
        """อัปเดตการตั้งค่าใน Streamlit session state"""
        if 'settings' in st.session_state:
            st.session_state.settings.update(new_settings)
        else:
            st.session_state.settings = new_settings
    
    @classmethod
    def get_enhanced_search_terms(cls, base_query: str) -> list:
        """ขยายคำค้นหาให้ครอบคลุมมากขึ้น"""
        base_lower = base_query.lower()
        expanded_terms = [base_query]
        
        # ขยายจาก query_expansions
        for key, expansions in cls.SEARCH_ENHANCEMENT['query_expansions'].items():
            if key in base_lower:
                expanded_terms.extend(expansions)
        
        # เพิ่มวิธีการทำอาหาร
        for method in cls.SEARCH_ENHANCEMENT['cooking_methods']:
            if method in base_lower:
                expanded_terms.append(method)
        
        return list(set(expanded_terms))  # ลบซ้ำ

class APIConfig:
    """การตั้งค่าเฉพาะสำหรับ API"""
    
    # การกำหนดค่า USDA API
    USDA_BASE_URL = "https://api.nal.usda.gov/fdc/v1"
    USDA_SEARCH_PARAMS = {
        'pageSize': 5,
        'dataType': ['Foundation', 'SR Legacy']
    }
    USDA_RATE_LIMIT = 30  # calls per minute
    
    # การกำหนดค่า Nutritionix API
    NUTRITIONIX_BASE_URL = "https://trackapi.nutritionix.com/v2"
    NUTRITIONIX_DAILY_LIMIT = 200  # แผนฟรี
    NUTRITIONIX_TIMEOUT = 10
    
    # การตั้งค่า API ทั่วไป
    API_TIMEOUT = 10
    API_RETRY_ATTEMPTS = 3
    API_RETRY_DELAY = 1  # วินาที
    
    # การแคช API Response
    CACHE_DURATION = {
        'nutrition_data': 86400,  # 24 ชั่วโมง
        'recipe_data': 3600,      # 1 ชั่วโมง
        'search_results': 1800    # 30 นาที
    }

class NutritionConfig:
    """การตั้งค่าเฉพาะสำหรับการวิเคราะห์โภชนาการ"""
    
    # หน่วยแสดงผล
    DISPLAY_UNITS = {
        'calories': 'แคลอรี่',
        'protein': 'กรัม',
        'carbs': 'กรัม', 
        'fat': 'กรัม',
        'fiber': 'กรัม',
        'sugar': 'กรัม',
        'sodium': 'มิลลิกรัม',
        'vitamins': 'มิลลิกรัม',
        'minerals': 'มิลลิกรัม'
    }
    
    # สีสำหรับแผนภูมิ
    CHART_COLORS = {
        'protein': '#ff6b6b',      # แดง
        'carbs': '#4ecdc4',        # เขียวฟ้า
        'fat': '#45b7d1',          # น้ำเงิน
        'fiber': '#96ceb4',        # เขียวอ่อน
        'vitamins': '#ffeaa7',     # เหลือง
        'minerals': '#dda0dd',     # ม่วงอ่อน
        'calories': '#ff7675'      # แดงอ่อน
    }
    
    # เกณฑ์การจัดประเภทอาหาร
    CLASSIFICATION_THRESHOLDS = {
        'high_protein': 20,        # กรัม
        'low_carb': 10,           # กรัม
        'low_calorie': 200,       # แคลอรี่
        'high_fiber': 5,          # กรัม
        'low_sodium': 140,        # มิลลิกรัม
        'high_calcium': 200,      # มิลลิกรัม
        'high_iron': 3,           # มิลลิกรัม
        'very_high_protein': 30,  # กรัม
        'very_low_calorie': 150,  # แคลอรี่
        'high_fat': 20,           # กรัม
        'very_high_sodium': 1000  # มิลลิกรัม
    }
    
    # การปรับแต่งการคำนวณโภชนาการ
    CALCULATION_ADJUSTMENTS = {
        'cooking_oil_absorption': {
            'deep_fry': 0.1,      # ทอดลึก
            'pan_fry': 0.05,      # ทอดกะทะ
            'stir_fry': 0.7,      # ผัด
            'sauteé': 0.8,        # ผัดแห้ง
        },
        'water_content_loss': {
            'boiling': 0.0,       # ต้ม (ไม่สูญเสีย)
            'grilling': 0.15,     # ย่าง
            'roasting': 0.1,      # อบ
            'steaming': 0.02      # นึ่ง
        },
        'nutrient_retention': {
            'vitamin_c_cooking_loss': 0.25,  # สูญเสีย 25% จากการปรุง
            'vitamin_b_cooking_loss': 0.15,  # สูญเสีย 15%
            'mineral_retention': 0.95       # เก็บไว้ได้ 95%
        }
    }

class DatabaseConfig:
    """การตั้งค่าฐานข้อมูล"""
    
    # ตารางฐานข้อมูล
    NUTRITION_CACHE_TABLE = 'nutrition_cache'
    API_USAGE_TABLE = 'api_usage'
    RECIPE_ANALYSIS_TABLE = 'recipe_analysis'
    RECIPE_ADJUSTMENTS_TABLE = 'recipe_adjustments'
    USER_PREFERENCES_TABLE = 'user_preferences'
    
    # คำสั่ง SQL
    CREATE_TABLES_SQL = {
        'nutrition_cache': '''
            CREATE TABLE IF NOT EXISTS nutrition_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ingredient_name TEXT UNIQUE NOT NULL,
                nutrition_data TEXT NOT NULL,
                source TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''',
        'api_usage': '''
            CREATE TABLE IF NOT EXISTS api_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                api_name TEXT NOT NULL,
                calls_count INTEGER DEFAULT 0,
                last_reset DATE DEFAULT CURRENT_DATE,
                daily_limit INTEGER DEFAULT 200
            )
        ''',
        'recipe_analysis': '''
            CREATE TABLE IF NOT EXISTS recipe_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recipe_name TEXT UNIQUE NOT NULL,
                nutrition_summary TEXT NOT NULL,
                enhancement_level TEXT DEFAULT 'basic',
                analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''',
        'recipe_adjustments': '''
            CREATE TABLE IF NOT EXISTS recipe_adjustments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recipe_name TEXT UNIQUE NOT NULL,
                adjustments_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''',
        'user_preferences': '''
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                preferences_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        '''
    }
    
    # การตั้งค่าประสิทธิภาพ
    PERFORMANCE_SETTINGS = {
        'cache_size': 1000,           # จำนวน entries ในแคช
        'batch_size': 50,             # ขนาด batch สำหรับการประมวลผล
        'connection_pool_size': 5,    # ขนาด connection pool
        'query_timeout': 30           # timeout สำหรับ query (วินาที)
    }

class UIConfig:
    """การตั้งค่าส่วน UI"""
    
    # ธีม
    THEME_COLORS = {
        'primary': '#4CAF50',
        'secondary': '#2196F3',
        'success': '#4CAF50',
        'warning': '#FF9800',
        'error': '#F44336',
        'info': '#2196F3'
    }
    
    # การแสดงผล
    DISPLAY_SETTINGS = {
        'max_recipes_per_search': 10,
        'max_nutrition_details': 15,
        'default_page_size': 5,
        'auto_scroll_delay': 500,      # มิลลิวินาที
        'animation_duration': 300      # มิลลิวินาที
    }
    
    # ข้อความช่วยเหลือ
    HELP_TEXTS = {
        'api_keys': {
            'usda': 'ใส่ API Key จาก USDA FoodData Central (ฟรี)',
            'nutritionix': 'ใส่ App ID และ API Key จาก Nutritionix'
        },
        'features': {
            'enhanced_search': 'ขยายการค้นหาให้ครอบคลุมมากขึ้น',
            'cooking_adjustments': 'ปรับการคำนวณตามวิธีการทำอาหาร',
            'external_data': 'ใช้ข้อมูลจาก API ภายนอกเพื่อความแม่นยำ'
        }
    }

class LoggingConfig:
    """การตั้งค่าการบันทึกล็อก"""
    
    # ระดับการบันทึก
    LOG_LEVELS = {
        'DEBUG': 10,
        'INFO': 20,
        'WARNING': 30,
        'ERROR': 40,
        'CRITICAL': 50
    }
    
    # การตั้งค่าไฟล์ล็อก
    LOG_FILES = {
        'main': 'app.log',
        'nutrition': 'nutrition.log',
        'api': 'api.log',
        'search': 'search.log'
    }
    
    # รูปแบบการบันทึก
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    
    # การหมุนไฟล์ล็อก
    LOG_ROTATION = {
        'max_bytes': 10 * 1024 * 1024,  # 10MB
        'backup_count': 5
    }

# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    print("การกำหนดค่าแชทบอทอาหารไทยขั้นสูง - ปรับปรุงแล้ว")
    print("=" * 50)
    print(f"เส้นทางข้อมูล: {Config.DATA_PATH}")
    print(f"โมเดล: {Config.SENTENCE_TRANSFORMER_MODEL}")
    print(f"สถานะ API: {Config.is_api_configured()}")
    print(f"แหล่งข้อมูลโภชนาการ: {Config.get_nutrition_source_priority()}")
    print(f"ข้อความต้อนรับ: {Config.get_message('welcome')}")
    print(f"เกณฑ์การค้นหาขั้นสูง: {Config.ENHANCED_SIMILARITY_THRESHOLD}")
    print(f"จำนวนเมนูอาหารไทยทั้งหมด: {len(Config.COMPLETE_THAI_MENU_LIST)}")
    print(f"การปรับแต่งการทำอาหารมีให้: {len(Config.COOKING_ADJUSTMENTS['oil_absorption_rates'])} ประเภท")
    print(f"การขยายการค้นหา: {len(Config.SEARCH_ENHANCEMENT['query_expansions'])} หมวดหมู่")
    
    # ทดสอบการขยายคำค้นหา
    test_query = "ไข่เจียว"
    expanded = Config.get_enhanced_search_terms(test_query)
    print(f"การขยายคำค้นหา '{test_query}': {expanded[:5]}...")
