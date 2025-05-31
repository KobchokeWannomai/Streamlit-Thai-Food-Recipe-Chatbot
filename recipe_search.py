import re
import pandas as pd
from typing import List, Dict, Tuple, Optional
from difflib import SequenceMatcher
import streamlit as st
from nutrition_api import NutritionAPI

class RecipeSearchEngine:
    """เครื่องมือค้นหาและแนะนำสูตรอาหารแบบชาญฉลาดขั้นสูง"""
    
    def __init__(self, data: pd.DataFrame, nutrition_api: NutritionAPI):
        self.data = data
        self.nutrition_api = nutrition_api
        
        # แคชข้อมูลโภชนาการของแต่ละเมนู (เพื่อประสิทธิภาพ)
        self.recipe_nutrition_cache = {}
        
        # คำหลักสำหรับการแนะนำตามโภชนาการ (ขยายเพิ่มเติม)
        self.nutrition_keywords = {
            # แคลอรี่
            "แคลอรี่ต่ำ": {"max_calories": 200},
            "แคลอรี่ปานกลาง": {"min_calories": 200, "max_calories": 400},
            "แคลอรี่สูง": {"min_calories": 400},
            
            # ไขมัน
            "ไขมันต่ำ": {"max_fat": 10},
            "ไขมันปานกลาง": {"min_fat": 10, "max_fat": 25},
            "ไขมันสูง": {"min_fat": 25},
            
            # โปรตีน
            "โปรตีนต่ำ": {"max_protein": 10},
            "โปรตีนปานกลาง": {"min_protein": 10, "max_protein": 25},
            "โปรตีนสูง": {"min_protein": 20},
            
            # คาร์โบไฮเดรต
            "คาร์โบต่ำ": {"max_carbs": 15},
            "คาร์โบไฮเดรตต่ำ": {"max_carbs": 15},
            "คาร์โบปานกลาง": {"min_carbs": 15, "max_carbs": 40},
            "คาร์โบสูง": {"min_carbs": 40},
            "คาร์โบไฮเดรตสูง": {"min_carbs": 40},
            
            # ใยอาหาร
            "ใยอาหารสูง": {"min_fiber": 5},
            "ใยอาหารต่ำ": {"max_fiber": 2},
            
            # วิตามิน
            "วิตามินเอสูง": {"min_vitamin_a": 100},
            "วิตามินซีสูง": {"min_vitamin_c": 20},
            "วิตามินบีสูง": {"min_vitamin_b1": 0.3},
            
            # แร่ธาตุ
            "แคลเซียมสูง": {"min_calcium": 100},
            "เหล็กสูง": {"min_iron": 3},
            "โปแตสเซียมสูง": {"min_potassium": 300},
            
            # โซเดียม
            "โซเดียมต่ำ": {"max_sodium": 500},
            "เค็มน้อย": {"max_sodium": 500},
            "โซเดียมสูง": {"min_sodium": 1000},
            
            # สำหรับผู้ป่วยเฉพาะ
            "เบาหวาน": {"max_carbs": 20, "max_sodium": 600},
            "ความดันสูง": {"max_sodium": 400, "min_potassium": 300},
            "ลดน้ำหนัก": {"max_calories": 250, "min_protein": 15, "min_fiber": 3},
            "เพิ่มน้ำหนัก": {"min_calories": 400, "min_protein": 20},
            "เด็ก": {"max_sodium": 300, "min_calcium": 100, "min_iron": 2},
            "ผู้สูงอายุ": {"min_protein": 20, "min_calcium": 150, "max_sodium": 500}
        }
        
        # คำหลักสำหรับประเภทอาหาร (ขยายเพิ่มเติม)
        self.food_type_keywords = {
            "ทอด": ["ทอด", "กรอบ", "เหลือง", "เจียว"],
            "ต้ม": ["ต้ม", "แกง", "น้ำ", "ซุป"],
            "ผัด": ["ผัด", "คั่ว", "xo"],
            "ย่าง": ["ย่าง", "ปิ้ง", "เผา", "บาร์บีคิว"],
            "นึ่ง": ["นึ่ง", "อบ", "ไอน้ำ"],
            "ยำ": ["ยำ", "ตำ", "สลัด", "ซ่า"],
            "ลาบ": ["ลาบ", "น้ำตก"],
            "ของหวาน": ["หวาน", "ขนม", "เชื่อม", "เค็ก", "ทองยิบ", "ทองยอด"],
            "เครื่องดื่ม": ["น้ำ", "ชา", "กาแฟ", "เครื่องดื่ม", "สมูทธี่"],
            "อาหารเช้า": ["ข้าวต้ม", "โจ๊ก", "ขนมปัง"],
            "อาหารกลางวัน": ["ข้าวผัด", "ก๋วยเตี๋ยว"],
            "อาหารเย็น": ["แกง", "ต้ม"],
            "ขนมขบเคี้ยว": ["ทอด", "กรอบ", "ขนม"]
        }
        
        # คำหลักสำหรับวัตถุดิบหลัก
        self.main_ingredient_keywords = {
            "หมู": ["หมู", "สันใน", "สันคอ", "สามชั้น"],
            "ไก่": ["ไก่", "ปีก", "น่อง", "อก"],
            "เนื้อ": ["เนื้อ", "วัว"],
            "ปลา": ["ปลา", "แซลมอน", "ทูน่า", "ดุก", "ช่อน"],
            "กุ้ง": ["กุ้ง", "กุ้งนาง", "กุ้งตะเข็บ"],
            "หมึก": ["หมึก", "ปลาหมึก"],
            "ไข่": ["ไข่", "ไข่ไก่", "ไข่เป็ด"],
            "ผัก": ["ผัก", "คะน้า", "ผักบุ้ง", "กะหล่ำ"],
            "เห็ด": ["เห็ด", "เห็ดหอม", "เห็ดฟาง"],
            "ถั่ว": ["ถั่ว", "ถั่วงอก", "ถั่วฝักยาว"],
            "มะเขือ": ["มะเขือ", "มะเขือเทศ", "มะเขือเปราะ"]
        }

    def fuzzy_search(self, query: str, threshold: float = 0.6) -> List[Tuple[str, float, int]]:
        """ค้นหาแบบ fuzzy matching รองรับการพิมพ์ผิดและคำไม่ครบ"""
        query = query.lower().strip()
        matches = []
        
        for idx, recipe_name in enumerate(self.data['name']):
            recipe_name_lower = recipe_name.lower()
            
            # คำนวณความคล้ายคลึงแบบเบื้องต้น
            similarity = SequenceMatcher(None, query, recipe_name_lower).ratio()
            
            # ตรวจสอบการมีคำคีย์เวิร์ดบางส่วน
            if query in recipe_name_lower:
                similarity = max(similarity, 0.9)
            
            # ตรวจสอบคำต่างๆ ในชื่อ (รองรับการค้นหาแบบบางส่วน)
            query_words = query.split()
            recipe_words = recipe_name_lower.split()
            
            word_matches = 0
            partial_matches = 0
            
            for q_word in query_words:
                for r_word in recipe_words:
                    # ตรวจสอบการจับคู่แบบเต็ม
                    if SequenceMatcher(None, q_word, r_word).ratio() > 0.8:
                        word_matches += 1
                        break
                    # ตรวจสอบการจับคู่แบบบางส่วน
                    elif (len(q_word) > 2 and q_word in r_word) or (len(r_word) > 2 and r_word in q_word):
                        partial_matches += 0.5
                        break
            
            # คำนวณคะแนนรวม
            if word_matches > 0 or partial_matches > 0:
                total_matches = word_matches + partial_matches
                word_similarity = total_matches / len(query_words)
                similarity = max(similarity, word_similarity * 0.85)
            
            # ตรวจสอบในส่วนผสมและวิธีทำ (คะแนนน้อยกว่า)
            if similarity < threshold:
                ingredient_text = str(self.data.iloc[idx].get('ingredient', '')).lower()
                method_text = str(self.data.iloc[idx].get('method', '')).lower()
                
                for q_word in query_words:
                    if len(q_word) > 2:
                        if q_word in ingredient_text or q_word in method_text:
                            similarity = max(similarity, 0.4)
                            break
            
            if similarity >= threshold:
                matches.append((recipe_name, similarity, idx))
        
        # เรียงลำดับตามความคล้ายคลึง
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def get_recipe_nutrition(self, recipe_index: int, use_api: bool = True, 
                           adjust_consumption: bool = True, 
                           enhance_missing: bool = False) -> Dict:
        """ดึงข้อมูลโภชนาการของสูตรอาหารพร้อมแคช"""
        cache_key = f"{recipe_index}_{use_api}_{adjust_consumption}_{enhance_missing}"
        
        if cache_key in self.recipe_nutrition_cache:
            return self.recipe_nutrition_cache[cache_key]
        
        recipe = self.data.iloc[recipe_index]
        nutrition_data = self.nutrition_api.calculate_recipe_nutrition(
            recipe['ingredient'], use_api, adjust_consumption, enhance_missing
        )
        
        self.recipe_nutrition_cache[cache_key] = nutrition_data
        return nutrition_data

    def find_recipes_by_nutrition(self, nutrition_criteria: Dict, use_api: bool = True, 
                                adjust_consumption: bool = True,
                                enhance_missing: bool = False) -> List[Tuple[str, Dict, int, float]]:
        """ค้นหาสูตรอาหารตามเกณฑ์โภชนาการ"""
        matching_recipes = []
        
        for idx in range(len(self.data)):
            recipe_name = self.data.iloc[idx]['name']
            nutrition_data = self.get_recipe_nutrition(idx, use_api, adjust_consumption, enhance_missing)
            total_nutrition = nutrition_data['total_nutrition']
            
            # ตรวจสอบเกณฑ์และคำนวณคะแนนความเหมาะสม
            meets_criteria = True
            fitness_score = 0
            criteria_count = 0
            
            for criterion, value in nutrition_criteria.items():
                criteria_count += 1
                
                if criterion.startswith('min_'):
                    nutrient = criterion[4:]  # ลบ 'min_'
                    actual_value = total_nutrition.get(nutrient, 0)
                    if actual_value < value:
                        meets_criteria = False
                        break
                    else:
                        # คำนวณคะแนนความเหมาะสม (ยิ่งเกินเกณฑ์มาก = คะแนนสูง)
                        fitness_score += min(actual_value / value, 2.0)
                        
                elif criterion.startswith('max_'):
                    nutrient = criterion[4:]  # ลบ 'max_'
                    actual_value = total_nutrition.get(nutrient, 0)
                    if actual_value > value:
                        meets_criteria = False
                        break
                    else:
                        # คำนวณคะแนนความเหมาะสม (ยิ่งต่ำกว่าเกณฑ์ = คะแนนสูง)
                        fitness_score += (value - actual_value) / value + 1.0
            
            if meets_criteria and criteria_count > 0:
                avg_fitness = fitness_score / criteria_count
                matching_recipes.append((recipe_name, nutrition_data, idx, avg_fitness))
        
        # เรียงลำดับตามคะแนนความเหมาะสม
        matching_recipes.sort(key=lambda x: x[3], reverse=True)
        
        return matching_recipes

    def analyze_query(self, query: str) -> Dict:
        """วิเคราะห์คำถามเพื่อหาเจตนาและข้อมูลที่ต้องการ"""
        query_lower = query.lower()
        analysis = {
            "intent": "general_search",  # general_search, nutrition_search, food_type_search, ingredient_search
            "nutrition_criteria": {},
            "food_types": [],
            "main_ingredients": [],
            "search_terms": [],
            "confidence": 0.0
        }
        
        # ตรวจสอบคำหลักโภชนาการ
        nutrition_matches = []
        for keyword, criteria in self.nutrition_keywords.items():
            if keyword in query_lower:
                nutrition_matches.append((keyword, criteria, len(keyword)))
                analysis["nutrition_criteria"].update(criteria)
        
        if nutrition_matches:
            analysis["intent"] = "nutrition_search"
            # ให้น้ำหนักกับคำที่ยาวกว่า (เฉพาะเจาะจงกว่า)
            analysis["confidence"] = max([length for _, _, length in nutrition_matches]) / 20.0
        
        # ตรวจสอบประเภทอาหาร
        food_type_matches = []
        for food_type, keywords in self.food_type_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    food_type_matches.append((food_type, keyword))
                    if food_type not in analysis["food_types"]:
                        analysis["food_types"].append(food_type)
        
        if food_type_matches:
            if analysis["intent"] == "general_search":
                analysis["intent"] = "food_type_search"
                analysis["confidence"] = 0.7
            elif analysis["intent"] == "nutrition_search":
                analysis["intent"] = "combined_search"  # ค้นหาแบบรวม
        
        # ตรวจสอบวัตถุดิบหลัก
        ingredient_matches = []
        for ingredient, keywords in self.main_ingredient_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    ingredient_matches.append((ingredient, keyword))
                    if ingredient not in analysis["main_ingredients"]:
                        analysis["main_ingredients"].append(ingredient)
        
        if ingredient_matches:
            if analysis["intent"] == "general_search":
                analysis["intent"] = "ingredient_search"
                analysis["confidence"] = 0.6
        
        # สกัดคำค้นหา (ลบคำที่ไม่เกี่ยวข้อง)
        filter_words = [
            "อาหาร", "เมนู", "สูตร", "วิธีทำ", "ที่มี", "สูง", "ต่ำ", "แนะนำ", "หา", "ค้นหา",
            "ปรุง", "ทำ", "กิน", "อร่อย", "ง่าย", "เร็ว", "ใส่", "กับ", "และ", "หรือ"
        ]
        words = query_lower.split()
        search_terms = [word for word in words if word not in filter_words and len(word) > 1]
        analysis["search_terms"] = search_terms
        
        return analysis

    def smart_search(self, query: str, use_api: bool = True, adjust_consumption: bool = True,
                    enhance_missing: bool = False, fuzzy_threshold: float = 0.6,
                    limit: int = 5) -> List[Dict]:
        """ระบบค้นหาอัจฉริยะขั้นสูง"""
        analysis = self.analyze_query(query)
        results = []
        
        # ค้นหาตามโภชนาการ
        if analysis["intent"] in ["nutrition_search", "combined_search"] and analysis["nutrition_criteria"]:
            nutrition_results = self.find_recipes_by_nutrition(
                analysis["nutrition_criteria"], use_api, adjust_consumption, enhance_missing
            )
            
            for recipe_name, nutrition_data, idx, fitness_score in nutrition_results[:limit]:
                results.append({
                    "name": recipe_name,
                    "similarity": min(fitness_score / 2.0, 1.0),  # ปรับให้อยู่ในช่วง 0-1
                    "index": idx,
                    "nutrition": nutrition_data,
                    "match_reason": f"ตรงกับเกณฑ์โภชนาการ (คะแนน: {fitness_score:.1f})",
                    "match_type": "nutrition"
                })
        
        # ค้นหาตามประเภทอาหาร
        if analysis["intent"] in ["food_type_search", "combined_search"] and analysis["food_types"]:
            for food_type in analysis["food_types"]:
                type_keywords = self.food_type_keywords[food_type]
                
                for idx, recipe in self.data.iterrows():
                    recipe_name_lower = recipe['name'].lower()
                    method_lower = str(recipe['method']).lower()
                    ingredient_lower = str(recipe['ingredient']).lower()
                    
                    # ตรวจสอบในชื่อ วิธีทำ และวัตถุดิบ
                    match_score = 0
                    for keyword in type_keywords:
                        if keyword in recipe_name_lower:
                            match_score += 0.8
                        elif keyword in method_lower:
                            match_score += 0.6
                        elif keyword in ingredient_lower:
                            match_score += 0.4
                    
                    if match_score > 0:
                        nutrition_data = self.get_recipe_nutrition(idx, use_api, adjust_consumption, enhance_missing)
                        
                        # ตรวจสอบว่าเมนูนี้มีอยู่ในผลลัพธ์แล้วหรือไม่
                        existing = next((r for r in results if r["index"] == idx), None)
                        if existing:
                            # อัพเดตคะแนนรวม
                            existing["similarity"] = max(existing["similarity"], match_score)
                            existing["match_reason"] += f" + ประเภท: {food_type}"
                        else:
                            results.append({
                                "name": recipe['name'],
                                "similarity": match_score,
                                "index": idx,
                                "nutrition": nutrition_data,
                                "match_reason": f"ประเภท: {food_type}",
                                "match_type": "food_type"
                            })
                        
                        if len(results) >= limit * 2:  # จำกัดการค้นหา
                            break
        
        # ค้นหาตามวัตถุดิบหลัก
        if analysis["intent"] in ["ingredient_search"] and analysis["main_ingredients"]:
            for main_ingredient in analysis["main_ingredients"]:
                ingredient_keywords = self.main_ingredient_keywords[main_ingredient]
                
                for idx, recipe in self.data.iterrows():
                    ingredient_text = str(recipe['ingredient']).lower()
                    recipe_name_lower = recipe['name'].lower()
                    
                    match_score = 0
                    for keyword in ingredient_keywords:
                        if keyword in ingredient_text:
                            match_score += 0.7
                        elif keyword in recipe_name_lower:
                            match_score += 0.5
                    
                    if match_score > 0:
                        nutrition_data = self.get_recipe_nutrition(idx, use_api, adjust_consumption, enhance_missing)
                        
                        existing = next((r for r in results if r["index"] == idx), None)
                        if existing:
                            existing["similarity"] = max(existing["similarity"], match_score)
                            existing["match_reason"] += f" + วัตถุดิบ: {main_ingredient}"
                        else:
                            results.append({
                                "name": recipe['name'],
                                "similarity": match_score,
                                "index": idx,
                                "nutrition": nutrition_data,
                                "match_reason": f"วัตถุดิบหลัก: {main_ingredient}",
                                "match_type": "ingredient"
                            })
        
        # ค้นหาแบบทั่วไป + fuzzy matching
        if analysis["intent"] == "general_search" or len(results) < limit:
            search_query = " ".join(analysis["search_terms"]) if analysis["search_terms"] else query
            fuzzy_results = self.fuzzy_search(search_query, threshold=fuzzy_threshold)
            
            for recipe_name, similarity, idx in fuzzy_results:
                # ตรวจสอบว่าเมนูนี้มีอยู่ในผลลัพธ์แล้วหรือไม่
                existing = next((r for r in results if r["index"] == idx), None)
                if not existing:
                    nutrition_data = self.get_recipe_nutrition(idx, use_api, adjust_consumption, enhance_missing)
                    results.append({
                        "name": recipe_name,
                        "similarity": similarity,
                        "index": idx,
                        "nutrition": nutrition_data,
                        "match_reason": f"ความคล้ายคลึงชื่อ: {similarity:.0%}",
                        "match_type": "fuzzy"
                    })
                
                if len(results) >= limit * 3:  # จำกัดการค้นหา
                    break
        
        # เรียงลำดับและคืนผลลัพธ์
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]

    def get_nutrition_recommendations(self, target_nutrition: str, use_api: bool = True,
                                    adjust_consumption: bool = True, enhance_missing: bool = False,
                                    limit: int = 3) -> List[Dict]:
        """แนะนำอาหารตามโภชนาการที่ต้องการ"""
        if target_nutrition in self.nutrition_keywords:
            criteria = self.nutrition_keywords[target_nutrition]
            results = self.find_recipes_by_nutrition(criteria, use_api, adjust_consumption, enhance_missing)
            
            recommendations = []
            for recipe_name, nutrition_data, idx, fitness_score in results[:limit]:
                recommendations.append({
                    "name": recipe_name,
                    "index": idx,
                    "nutrition": nutrition_data,
                    "fitness_score": fitness_score,
                    "reason": f"แนะนำสำหรับผู้ต้องการ{target_nutrition} (คะแนน: {fitness_score:.1f})"
                })
            
            return recommendations
        
        return []

    def compare_recipes_nutrition(self, recipe_indices: List[int], use_api: bool = True,
                                adjust_consumption: bool = True, enhance_missing: bool = False) -> Dict:
        """เปรียบเทียบค่าโภชนาการของหลายสูตร"""
        comparison = {
            "recipes": [],
            "nutrients_comparison": {},
            "recommendations": []
        }
        
        # รวบรวมข้อมูลโภชนาการของแต่ละสูตร
        for idx in recipe_indices:
            recipe_name = self.data.iloc[idx]['name']
            nutrition_data = self.get_recipe_nutrition(idx, use_api, adjust_consumption, enhance_missing)
            
            comparison["recipes"].append({
                "name": recipe_name,
                "index": idx,
                "nutrition": nutrition_data['total_nutrition']
            })
        
        # สร้างการเปรียบเทียบแต่ละสารอาหาร
        nutrients = ["calories", "protein", "carbs", "fat", "fiber", 
                    "vitamin_a", "vitamin_c", "calcium", "iron", "sodium"]
        
        for nutrient in nutrients:
            values = [recipe["nutrition"][nutrient] for recipe in comparison["recipes"]]
            comparison["nutrients_comparison"][nutrient] = {
                "values": values,
                "max": max(values) if values else 0,
                "min": min(values) if values else 0,
                "avg": sum(values) / len(values) if values else 0,
                "range": max(values) - min(values) if values else 0
            }
        
        # สร้างคำแนะนำ
        for i, recipe in enumerate(comparison["recipes"]):
            recommendations = []
            nutrition = recipe["nutrition"]
            
            if nutrition["calories"] == comparison["nutrients_comparison"]["calories"]["min"]:
                recommendations.append("แคลอรี่ต่ำสุด")
            elif nutrition["calories"] == comparison["nutrients_comparison"]["calories"]["max"]:
                recommendations.append("แคลอรี่สูงสุด")
            
            if nutrition["protein"] == comparison["nutrients_comparison"]["protein"]["max"]:
                recommendations.append("โปรตีนสูงสุด")
            
            if nutrition["sodium"] == comparison["nutrients_comparison"]["sodium"]["min"]:
                recommendations.append("โซเดียมต่ำสุด")
            
            comparison["recommendations"].append({
                "recipe_index": i,
                "recommendations": recommendations
            })
        
        return comparison

    def get_ingredient_alternatives(self, target_ingredient: str, 
                                  nutrition_focus: str = None, limit: int = 5) -> List[Dict]:
        """แนะนำวัตถุดิบทดแทนตามโภชนาการ"""
        alternatives = []
        target_nutrition = self.nutrition_api.get_nutrition_data(target_ingredient)
        
        if not target_nutrition:
            return alternatives
        
        # หาวัตถุดิบที่มีโภชนาการคล้ายกัน
        for ingredient, nutrition in self.nutrition_api.local_nutrition_db.items():
            if ingredient.lower() != target_ingredient.lower():
                # คำนวณความคล้ายคลึงทางโภชนาการ
                similarity_score = 0
                total_nutrients = 0
                
                nutrients_to_compare = ["protein", "fat", "carbs", "fiber"]
                if nutrition_focus:
                    if nutrition_focus == "protein":
                        nutrients_to_compare = ["protein", "calories"]
                    elif nutrition_focus == "low_fat":
                        nutrients_to_compare = ["fat", "calories"]
                    elif nutrition_focus == "vitamins":
                        nutrients_to_compare = ["vitamin_a", "vitamin_c", "vitamin_b1", "vitamin_b2"]
                
                for nutrient in nutrients_to_compare:
                    if nutrient in target_nutrition and nutrient in nutrition:
                        target_val = target_nutrition[nutrient]
                        alt_val = nutrition[nutrient]
                        
                        if target_val > 0 or alt_val > 0:
                            if target_val == 0:
                                ratio = 0 if alt_val == 0 else 0.1
                            elif alt_val == 0:
                                ratio = 0.1
                            else:
                                ratio = min(alt_val, target_val) / max(alt_val, target_val)
                            
                            similarity_score += ratio
                            total_nutrients += 1
                
                if total_nutrients > 0:
                    avg_similarity = similarity_score / total_nutrients
                    if avg_similarity > 0.3:  # ความคล้ายคลึง > 30%
                        alternatives.append({
                            "ingredient": ingredient,
                            "similarity": avg_similarity,
                            "nutrition": nutrition,
                            "comparison": self.compare_ingredients(target_nutrition, nutrition)
                        })
        
        # เรียงลำดับตามความคล้ายคลึง
        alternatives.sort(key=lambda x: x["similarity"], reverse=True)
        return alternatives[:limit]

    def compare_ingredients(self, ingredient1_nutrition: Dict, ingredient2_nutrition: Dict) -> Dict:
        """เปรียบเทียบโภชนาการของวัตถุดิบ 2 ชนิด"""
        comparison = {}
        
        for nutrient in ingredient1_nutrition:
            if nutrient in ingredient2_nutrition:
                val1 = ingredient1_nutrition[nutrient]
                val2 = ingredient2_nutrition[nutrient]
                
                if val1 == val2:
                    comparison[nutrient] = "เท่ากัน"
                elif val2 > val1:
                    if val1 == 0:
                        comparison[nutrient] = f"สูงกว่า ({val2:.1f})"
                    else:
                        percentage = ((val2 - val1) / val1) * 100
                        comparison[nutrient] = f"สูงกว่า {percentage:.0f}%"
                else:
                    if val2 == 0:
                        comparison[nutrient] = f"ต่ำกว่า ({val1:.1f})"
                    else:
                        percentage = ((val1 - val2) / val2) * 100
                        comparison[nutrient] = f"ต่ำกว่า {percentage:.0f}%"
        
        return comparison

    def get_search_suggestions(self, partial_query: str, limit: int = 5) -> List[str]:
        """แนะนำคำค้นหาตามข้อความที่ป้อนบางส่วน"""
        suggestions = []
        query_lower = partial_query.lower().strip()
        
        if len(query_lower) < 2:
            return suggestions
        
        # ค้นหาในชื่อเมนู
        for recipe_name in self.data['name']:
            if query_lower in recipe_name.lower():
                suggestions.append(recipe_name)
        
        # ค้นหาในคำหลักโภชนาการ
        for nutrition_keyword in self.nutrition_keywords:
            if query_lower in nutrition_keyword:
                suggestions.append(f"อาหาร{nutrition_keyword}")
        
        # ค้นหาในประเภทอาหาร
        for food_type in self.food_type_keywords:
            if query_lower in food_type:
                suggestions.append(f"อาหาร{food_type}")
        
        # ลบรายการซ้ำและจำกัดจำนวน
        unique_suggestions = list(dict.fromkeys(suggestions))
        return unique_suggestions[:limit]

    def clear_cache(self):
        """ล้างแคชข้อมูลโภชนาการ"""
        self.recipe_nutrition_cache.clear()
        st.success("ล้างแคชข้อมูลโภชนาการเรียบร้อย")

    def get_cache_info(self) -> Dict:
        """ข้อมูลสถิติแคช"""
        return {
            "cached_recipes": len(self.recipe_nutrition_cache),
            "available_recipes": len(self.data),
            "cache_hit_rate": len(self.recipe_nutrition_cache) / len(self.data) if len(self.data) > 0 else 0
        }
