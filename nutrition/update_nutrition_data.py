# scripts/update_nutrition_data.py
import schedule
import time
from nutrition.nutrition_fetcher import NutritionFetcher
from nutrition.nutrition_processor import NutritionProcessor

def update_nutrition_database():
    """อัปเดตฐานข้อมูลโภชนาการอัตโนมัติ"""
    fetcher = NutritionFetcher()
    processor = NutritionProcessor()
    
    # โหลดข้อมูลสูตรอาหาร
    recipes_df = pd.read_csv('data/thai_food_processed.csv')
    
    # ดึงวัตถุดิบทั้งหมด
    all_ingredients = set()
    for ingredients_text in recipes_df['ingredient']:
        lines = ingredients_text.split('\n')
        for line in lines:
            if line.strip().startswith('-'):
                ingredient, _, _ = processor.parse_ingredient_line(line)
                all_ingredients.add(ingredient)
    
    # อัปเดตข้อมูลโภชนาการสำหรับวัตถุดิบใหม่
    updated_count = 0
    for ingredient in all_ingredients:
        if ingredient not in fetcher.cache:
            nutrition_data = fetcher.fetch_nutrition(ingredient)
            if nutrition_data:
                updated_count += 1
                print(f"Updated: {ingredient}")
    
    print(f"Total updated: {updated_count} ingredients")

# ตั้งเวลาอัปเดตอัตโนมัติ
schedule.every().sunday.at("02:00").do(update_nutrition_database)

if __name__ == "__main__":
    while True:
        schedule.run_pending()
        time.sleep(3600)  # ตรวจสอบทุกชั่วโมง
