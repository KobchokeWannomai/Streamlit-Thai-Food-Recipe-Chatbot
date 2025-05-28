import os
from typing import Optional

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
    MAX_SEARCH_RESULTS = 3
    
    # API Keys และการตั้งค่าภายนอก
    USDA_API_KEY: Optional[str] = os.getenv("USDA_API_KEY")
    NUTRITIONIX_API_KEY: Optional[str] = os.getenv("NUTRITIONIX_API_KEY")
    NUTRITIONIX_APP_ID: Optional[str] = os.getenv("NUTRITIONIX_APP_ID")
    
    # การตั้งค่าฐานข้อมูลโภชนาการ
    NUTRITION_CACHE_EXPIRE_DAYS = 30  # วัน
    MAX_API_CALLS_PER_DAY = 1000
    
    # การตั้งค่า UI
    PAGE_TITLE = "Thai Food Recipe Chatbot with Nutrition"
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
    
    # ข้อความแสดงผล
    MESSAGES = {
        'th': {
            'welcome': 'ยินดีต้อนรับสู่ระบบแชทบอทสูตรอาหารไทยพร้อมข้อมูลโภชนาการ',
            'search_placeholder': 'ถามเกี่ยวกับอาหารไทย หรือค้นหาตามโภชนาการ...',
            'analyzing_nutrition': 'กำลังวิเคราะห์ข้อมูลโภชนาการ...',
            'no_results': 'ไม่พบสูตรอาหารที่ตรงกับคำค้นหา',
            'nutrition_analysis_failed': 'ไม่สามารถวิเคราะห์ข้อมูลโภชนาการได้',
            'recipe_found': 'พบสูตรอาหารที่คุณต้องการพร้อมข้อมูลโภชนาการ',
        },
        'en': {
            'welcome': 'Welcome to Thai Food Recipe Chatbot with Nutrition Analysis',
            'search_placeholder': 'Ask about Thai food or search by nutrition...',
            'analyzing_nutrition': 'Analyzing nutrition data...',
            'no_results': 'No recipes found matching your search',
            'nutrition_analysis_failed': 'Unable to analyze nutrition data',
            'recipe_found': 'Found the recipe you\'re looking for with nutrition data',
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
            'usda': bool(cls.USDA_API_KEY),
            'nutritionix': bool(cls.NUTRITIONIX_API_KEY and cls.NUTRITIONIX_APP_ID),
        }
    
    @classmethod
    def get_nutrition_source_priority(cls) -> list:
        """ลำดับความสำคัญของแหล่งข้อมูลโภชนาการ"""
        sources = ['thai_database']  # เริ่มจากฐานข้อมูลไทยเสมอ
        
        if cls.USDA_API_KEY:
            sources.append('usda_api')
        
        if cls.NUTRITIONIX_API_KEY and cls.NUTRITIONIX_APP_ID:
            sources.append('nutritionix_api')
        
        return sources

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
        'high_iron': 3            # มิลลิกรัม
    }

class DatabaseConfig:
    """การตั้งค่าฐานข้อมูล"""
    
    # ตารางฐานข้อมูล
    NUTRITION_CACHE_TABLE = 'nutrition_cache'
    API_USAGE_TABLE = 'api_usage'
    RECIPE_ANALYSIS_TABLE = 'recipe_analysis'
    
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
                last_reset DATE DEFAULT CURRENT_DATE
            )
        ''',
        'recipe_analysis': '''
            CREATE TABLE IF NOT EXISTS recipe_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recipe_name TEXT UNIQUE NOT NULL,
                nutrition_summary TEXT NOT NULL,
                analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        '''
    }

# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    print("Thai Food Chatbot Configuration")
    print("=" * 40)
    print(f"Data path: {Config.DATA_PATH}")
    print(f"Model: {Config.SENTENCE_TRANSFORMER_MODEL}")
    print(f"API Status: {Config.is_api_configured()}")
    print(f"Nutrition sources: {Config.get_nutrition_source_priority()}")
    print(f"Welcome message: {Config.get_message('welcome')}")
