import os
from typing import Optional, Dict, List
import streamlit as st

class Config:
    """คลาสสำหรับจัดการการตั้งค่าระบบ"""
    
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
    
    # API Keys และการตั้งค่าภายนอก
    USDA_API_KEY: Optional[str] = os.getenv("USDA_API_KEY")
    NUTRITIONIX_API_KEY: Optional[str] = os.getenv("NUTRITIONIX_API_KEY")
    NUTRITIONIX_APP_ID: Optional[str] = os.getenv("NUTRITIONIX_APP_ID")
    
    # การตั้งค่าฐานข้อมูลโภชนาการ
    NUTRITION_CACHE_EXPIRE_DAYS = 30  # วัน
    MAX_API_CALLS_PER_DAY = 1000
    API_RATE_LIMIT_PER_MINUTE = 30
    
    # การตั้งค่า UI
    PAGE_TITLE = "Thai Food Recipe Chatbot with Enhanced Nutrition"
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
    
    # การตั้งค่าขั้นสูงสำหรับการค้นหา
    SEARCH_ENHANCEMENT = {
        'query_expansions': {
            'ไข่': ['ไข่ไก่', 'ไข่เป็ด', 'ไข่ดาว', 'ไข่เจียว', 'ไข่ต้ม'],
            'หมู': ['เนื้อหมู', 'หมูสับ', 'หมูย่าง', 'หมูทอด'],
            'ไก่': ['เนื้อไก่', 'ไก่ย่าง', 'ไก่ทอด', 'ไก่ต้ม'],
            'กุ้ง': ['กุ้งนาง', 'กุ้งฝอย', 'กุ้งแห้ง'],
            'ผัด': ['ผัดไทย', 'ผัดกะเพรา', 'ผัดซีอิ๊ว'],
            'แกง': ['แกงเขียวหวาน', 'แกงเผ็ด', 'แกงส้ม', 'แกงมัสมั่น'],
            'ต้ม': ['ต้มยำ', 'ต้มข่า', 'ต้มจืด'],
            'ยำ': ['ยำวุ้นเส้น', 'ยำถั่วพู', 'ยำมะม่วง'],
            'ส้ม': ['ส้มตำ', 'ส้มตำไทย', 'ส้มตำปู'],
            'ลาบ': ['ลาบหมู', 'ลาบไก่', 'ลาบเนื้อ']
        },
        'cooking_methods': [
            'ผัด', 'ต้ม', 'ทอด', 'ย่าง', 'นึ่ง', 'ต้น', 'แกง', 'ยำ', 'ลาบ', 'น้ำพริก'
        ]
    }
    
    # การปรับแต่งการทำอาหาร
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
            'แกงเขียวหวาน': ['น้ำ']
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
            'welcome': 'Welcome to Enhanced Thai Food Recipe Chatbot with Advanced Nutrition Analysis',
            'search_placeholder': 'Ask about Thai food or search by nutrition...',
            'analyzing_nutrition': 'Analyzing nutrition data...',
            'analyzing_enhanced': 'Performing advanced nutrition analysis...',
            'connecting_api': 'Connecting to API...',
            'no_results': 'No recipes found matching your search',
            'nutrition_analysis_failed': 'Unable to analyze nutrition data',
            'recipe_found': 'Found the recipe you\'re looking for with nutrition data',
            'enhanced_analysis': 'Enhanced data with advanced calculations',
            'api_connection_success': 'API connection successful',
            'api_connection_failed': 'API connection failed',
            'using_local_data': 'Using local database',
        }
    }
    
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

class APIConfig:
    """การตั้งค่าเฉพาะสำหรับ API"""
    
    # USDA API Configuration
    USDA_BASE_URL = "https://api.nal.usda.gov/fdc/v1"
    USDA_SEARCH_PARAMS = {
        'pageSize': 5,
        'dataType': ['Foundation', 'SR Legacy']
    }
    USDA_RATE_LIMIT = 30  # calls per minute
    
    # Nutritionix API Configuration
    NUTRITIONIX_BASE_URL = "https://trackapi.nutritionix.com/v2"
    NUTRITIONIX_DAILY_LIMIT = 200  # free plan
    NUTRITIONIX_TIMEOUT = 10
    
    # General API Settings
    API_TIMEOUT = 10
    API_RETRY_ATTEMPTS = 3
    API_RETRY_DELAY = 1  # seconds
    
    # API Response Caching
    CACHE_DURATION = {
        'nutrition_data': 86400,  # 24 hours
        'recipe_data': 3600,      # 1 hour
        'search_results': 1800    # 30 minutes
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
    
    # SQL Commands
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
    print("Enhanced Thai Food Chatbot Configuration")
    print("=" * 50)
    print(f"Data path: {Config.DATA_PATH}")
    print(f"Model: {Config.SENTENCE_TRANSFORMER_MODEL}")
    print(f"API Status: {Config.is_api_configured()}")
    print(f"Nutrition sources: {Config.get_nutrition_source_priority()}")
    print(f"Welcome message: {Config.get_message('welcome')}")
    print(f"Enhanced search threshold: {Config.ENHANCED_SIMILARITY_THRESHOLD}")
    print(f"Cooking adjustments available: {len(Config.COOKING_ADJUSTMENTS['oil_absorption_rates'])} types")
    print(f"Search expansions: {len(Config.SEARCH_ENHANCEMENT['query_expansions'])} categories")
