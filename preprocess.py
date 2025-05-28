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
    """Clean and format text data"""
    if not isinstance(text, str):
        return ""
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters except Thai characters, numbers and basic punctuation
    text = re.sub(r'[^\u0E00-\u0E7Fa-zA-Z0-9\s.,\-\(\)]', '', text)
    
    return text.strip()

def preprocess_ingredients(text):
    """Format ingredient lists"""
    if not isinstance(text, str):
        return ""
    
    # Make sure each ingredient is on a new line and starts with a dash
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Ensure each line starts with a dash
        if not line.startswith('-'):
            line = f"- {line}"
        
        formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

def extract_main_ingredients(ingredients_text):
    """Extract main ingredients for nutrition analysis"""
    if not isinstance(ingredients_text, str):
        return []
    
    lines = ingredients_text.strip().split('\n')
    main_ingredients = []
    
    for line in lines:
        line = line.strip()
        if line and line.startswith('-'):
            # Remove dash and clean up
            ingredient = line[1:].strip()
            
            # Remove quantities and units
            ingredient = re.sub(r'\d+[\s]*[กชฟผลถ้วยช้อนกิโลกรัมกลีบใบเม็ดตัวคู่ฝักซีกแว่นราก].*', '', ingredient)
            ingredient = re.sub(r'\([^)]*\)', '', ingredient)  # Remove text in parentheses
            ingredient = re.sub(r'\d+', '', ingredient)  # Remove remaining numbers
            
            # Clean up common descriptive words
            unwanted_words = ['ขนาด', 'กลาง', 'เล็ก', 'ใหญ่', 'สด', 'แห้ง', 'ต้ม', 'ผ่า', 'หั่น', 'สับ', 'ปอก']
            for word in unwanted_words:
                ingredient = ingredient.replace(word, '')
            
            ingredient = ingredient.strip()
            if ingredient and len(ingredient) > 1:
                main_ingredients.append(ingredient)
    
    return main_ingredients

def analyze_recipe_nutrition(ingredients_text, nutrition_analyzer):
    """Analyze nutrition for a recipe"""
    try:
        nutrition_data = nutrition_analyzer.analyze_ingredients(ingredients_text)
        total_nutrition = nutrition_analyzer.calculate_total_nutrition(nutrition_data)
        
        # Return key nutrition metrics
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
        logger.error(f"Error analyzing nutrition: {e}")
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
    """Create nutrition summary statistics"""
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
    
    # Find most common ingredients
    all_ingredients = []
    for ingredients_list in df['nutrition_main_ingredients']:
        if isinstance(ingredients_list, list):
            all_ingredients.extend(ingredients_list)
    
    from collections import Counter
    ingredient_counts = Counter(all_ingredients)
    summary['most_common_ingredients'] = ingredient_counts.most_common(10)
    
    return summary

def preprocess_data(input_file, output_file, analyze_nutrition=True, usda_api_key=None):
    """Preprocess the Thai food dataset with nutrition analysis"""
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return False
    
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        # Check required columns
        required_columns = ['name', 'ingredient', 'method']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Error: Missing required columns: {', '.join(missing_columns)}")
            return False
        
        print(f"Processing {len(df)} recipes...")
        
        # Clean text in each column
        df['name'] = df['name'].apply(clean_text)
        df['method'] = df['method'].apply(clean_text)
        df['ingredient'] = df['ingredient'].apply(preprocess_ingredients)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['name'])
        
        # Analyze nutrition if requested
        if analyze_nutrition:
            print("Initializing nutrition analyzer...")
            nutrition_analyzer = NutritionAnalyzer(usda_api_key)
            
            print("Analyzing nutrition for recipes...")
            nutrition_results = []
            
            for idx, row in df.iterrows():
                print(f"Analyzing nutrition for recipe {idx + 1}/{len(df)}: {row['name']}")
                nutrition_data = analyze_recipe_nutrition(row['ingredient'], nutrition_analyzer)
                nutrition_results.append(nutrition_data)
            
            # Add nutrition columns to dataframe
            nutrition_df = pd.DataFrame(nutrition_results)
            
            # Prefix nutrition columns
            nutrition_df.columns = ['nutrition_' + col for col in nutrition_df.columns]
            
            # Combine with original dataframe
            df = pd.concat([df, nutrition_df], axis=1)
            
            # Create nutrition summary
            summary = create_nutrition_summary(df)
            
            # Save summary
            summary_file = output_file.replace('.csv', '_nutrition_summary.json')
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            print(f"Nutrition summary saved to: {summary_file}")
            print(f"Average calories per recipe: {summary['avg_calories']:.1f}")
            print(f"High protein recipes (>20g): {summary['high_protein_recipes']}")
            print(f"Low calorie recipes (<300 cal): {summary['low_calorie_recipes']}")
            
            # Display most common ingredients
            if summary['most_common_ingredients']:
                print("\nMost common ingredients:")
                for ingredient, count in summary['most_common_ingredients'][:5]:
                    print(f"  - {ingredient}: {count} recipes")
        
        # Reset index
        df = df.reset_index(drop=True)
        
        # Save preprocessed data
        df.to_csv(output_file, index=False)
        
        print(f"Preprocessing completed. Saved to '{output_file}'")
        print(f"Total recipes: {len(df)}")
        
        if analyze_nutrition:
            print(f"Nutrition analysis completed for {len(df)} recipes")
        
        # If embeddings file exists, remove it so it will be regenerated
        embeddings_files = ['embeddings.pkl', 'model']
        for file_path in embeddings_files:
            if os.path.exists(file_path):
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    import shutil
                    shutil.rmtree(file_path)
                print(f"Removed {file_path}. It will be regenerated when the app runs.")
        
        return True
    
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        return False

def create_nutrition_database():
    """Create and populate nutrition database"""
    analyzer = NutritionAnalyzer()
    
    # Add some Thai ingredients to the database
    thai_ingredients = [
        "หมู", "ไก่", "เนื้อ", "กุ้ง", "ปลา", "กะหล่ำปลี", "คะน้า", 
        "ผักบุ้ง", "น้ำปลา", "กะทิ", "น้ำตาล", "ข้าว", "แป้ง"
    ]
    
    print("Creating nutrition database...")
    for ingredient in thai_ingredients:
        nutrition = analyzer.get_ingredient_nutrition(ingredient)
        if nutrition:
            print(f"Added nutrition data for: {ingredient}")
    
    print("Nutrition database created successfully!")

def main():
    parser = argparse.ArgumentParser(description='Preprocess Thai food recipe data with nutrition analysis')
    parser.add_argument('--input', type=str, default='thai_food_raw.csv', 
                        help='Input CSV file path')
    parser.add_argument('--output', type=str, default='thai_food_processed.csv', 
                        help='Output CSV file path')
    parser.add_argument('--analyze-nutrition', action='store_true', default=True,
                        help='Analyze nutrition data for recipes')
    parser.add_argument('--usda-api-key', type=str, 
                        help='USDA API key for nutrition data')
    parser.add_argument('--create-nutrition-db', action='store_true',
                        help='Create nutrition database with Thai ingredients')
    
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
        print("\n✅ Preprocessing completed successfully!")
        print("\nNext steps:")
        print("1. Run 'streamlit run streamlit_app.py' to start the chatbot")
        print("2. The app will now include nutrition analysis for all recipes")
        print("3. Use the sidebar to search recipes by nutrition criteria")
    else:
        print("❌ Preprocessing failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
