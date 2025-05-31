import re
import pandas as pd
from typing import List, Dict, Tuple, Optional
from difflib import SequenceMatcher
import streamlit as st
from nutrition_api import NutritionAPI

class RecipeSearchEngine:
    """เครื่องมือค้นหาและแนะนำสูตรอาหารแบบชาญฉลาด"""
    
    def __init__(self, data: pd.DataFrame, nutrition_api: NutritionAPI):
        self.data = data
        self.nutrition_api = nutrition_api
        
        # ข้อมูลโภชนาการของแต่ละเมนู (จะคำนวณเมื่อใช้งาน)
        self.recipe_nutrition_cache = {}
        
        # คำหลักสำหรับการแนะนำตามโภชนาการ
        self.nutrition_keywords = {
            "แคลอรี่ต่ำ": {"max_calories": 200},
            "แคลอรี่สูง": {"min_calories": 400},
            "ไขมันต่ำ": {"max_fat": 10},
            "ไขมันสูง": {"min_fat": 20},
            "โปรตีนสูง": {"min_protein": 20},
            "โปรตีนต่ำ": {"max_protein": 10},
            "คาร์โบไฮเดรตต่ำ": {"max_carbs": 15},
            "คาร์โบไฮเดรตสูง": {"min_carbs": 30},
            "ใยอาหารสูง": {"min_fiber": 5},
            "วิตามินเอสูง": {"min_vitamin_a": 100},
            "วิตามินซีสูง": {"min_vitamin_c": 20},
            "แคลเซียมสูง": {"min_calcium": 100},
            "เหล็กสูง": {"min_iron": 3},
            "โซเดียมต่ำ": {"max_sodium": 500},
            "โซเดียมสูง": {"min_sodium": 1000},
            "โปแตสเซียมสูง": {"min_potassium": 300}
        }
        
        # คำหลักสำหรับประเภทอาหาร
        self.food_type_keywords = {
            "ทอด": ["ทอด", "กรอบ", "เหลือง"],
            "ต้ม": ["ต้ม", "แกง", "น้ำ"],
            "ผัด": ["ผัด", "คั่ว"],
            "ย่าง": ["ย่าง", "ปิ้ง", "เผา"],
            "นึ่ง": ["นึ่ง", "อบ"],
            "ยำ": ["ยำ", "ตำ", "สลัด"],
            "ของหวาน": ["หวาน", "ขนม", "เชื่อม", "เค็ก"],
            "เครื่องดื่ม": ["น้ำ", "ชา", "กาแฟ", "เครื่องดื่ม"]
        }

    def fuzzy_search(self, query: str, threshold: float = 0.6) -> List[Tuple[str, float, int]]:
        """ค้นหาแบบ fuzzy matching รองรับการพิมพ์ผิด"""
        query = query.lower().strip()
        matches = []
        
        for idx, recipe_name in enumerate(self.data['name']):
            recipe_name_lower = recipe_name.lower()
            
            # คำนวณความคล้ายคลึง
            similarity = SequenceMatcher(None, query, recipe_name_lower).ratio()
            
            # ตรวจสอบการมีคำคีย์เวิร์ดบางส่วน
            if query in recipe_name_lower:
                similarity = max(similarity, 0.8)
            
            # ตรวจสอบคำต่างๆ ในชื่อ
            query_words = query.split()
            recipe_words = recipe_name_lower.split()
            
            word_matches = 0
            for q_word in query_words:
                for r_word in recipe_words:
                    if SequenceMatcher(None, q_word, r_word).ratio() > 0.7:
                        word_matches += 1
                        break
            
            if word_matches > 0:
                word_similarity = word_matches / len(query_words)
                similarity = max(similarity, word_similarity * 0.8)
            
            if similarity >= threshold:
                matches.append((recipe_name, similarity, idx))
        
        # เรียงลำดับตามความคล้ายคลึง
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def get_recipe_nutrition(self, recipe_index: int, use_api: bool = True, 
                           adjust_consumption: bool = True) -> Dict:
        """ดึงข้อมูลโภชนาการของสูตรอาหาร"""
        cache_key = f"{recipe_index}_{use_api}_{adjust_consumption}"
        
        if cache_key in self.recipe_nutrition_cache:
            return self.recipe_nutrition_cache[cache_key]
        
        recipe = self.data.iloc[recipe_index]
        nutrition_data = self.nutrition_api.calculate_recipe_nutrition(
            recipe['ingredient'], use_api, adjust_consumption
        )
        
        self.recipe_nutrition_cache[cache_key] = nutrition_data
        return nutrition_data

    def find_recipes_by_nutrition(self, nutrition_criteria: Dict, use_api: bool = True, 
                                adjust_consumption: bool = True) -> List[Tuple[str, Dict, int]]:
        """ค้นหาสูตรอาหารตามเกณฑ์โภชนาการ"""
        matching_recipes = []
        
        for idx in range(len(self.data)):
            recipe_name = self.data.iloc[idx]['name']
            nutrition_data = self.get_recipe_nutrition(idx, use_api, adjust_consumption)
            total_nutrition = nutrition_data['total_nutrition']
            
            # ตรวจสอบเกณฑ์
            meets_criteria = True
            for criterion, value in nutrition_criteria.items():
                if criterion.startswith('min_'):
                    nutrient = criterion[4:]  # ลบ 'min_'
                    if total_nutrition.get(nutrient, 0) < value:
                        meets_criteria = False
                        break
                elif criterion.startswith('max_'):
                    nutrient = criterion[4:]  # ลบ 'max_'
                    if total_nutrition.get(nutrient, 0) > value:
                        meets_criteria = False
                        break
            
            if meets_criteria:
                matching_recipes.append((recipe_name, nutrition_data, idx))
        
        # เรียงลำดับตามความเหมาะสม
        if nutrition_criteria:
            first_criterion = list(nutrition_criteria.keys())[0]
            if first_criterion.startswith('min_'):
                nutrient = first_criterion[4:]
                matching_recipes.sort(
                    key=lambda x: x[1]['total_nutrition'].get(nutrient, 0), 
                    reverse=True
                )
            elif first_criterion.startswith('max_'):
                nutrient = first_criterion[4:]
                matching_recipes.sort(
                    key=lambda x: x[1]['total_nutrition'].get(nutrient, 0)
                )
        
        return matching_recipes

    def analyze_query(self, query: str) -> Dict:
        """วิเคราะห์คำถามเพื่อหาเจตนา"""
        query_lower = query.lower()
        analysis = {
            "intent": "general_search",  # general_search, nutrition_search, food_type_search
            "nutrition_criteria": {},
            "food_type": None,
            "search_terms": []
        }
        
        # ตรวจสอบคำหลักโภชนาการ
        for keyword, criteria in self.nutrition_keywords.items():
            if keyword in query_lower:
                analysis["intent"] = "nutrition_search"
                analysis["nutrition_criteria"].update(criteria)
                break
        
        # ตรวจสอบประเภทอาหาร
        for food_type, keywords in self.food_type_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    analysis["food_type"] = food_type
                    if analysis["intent"] == "general_search":
                        analysis["intent"] = "food_type_search"
                    break
        
        # สกัดคำค้นหา
        # ลบคำหลักที่ไม่ใช่ชื่ออาหาร
        filter_words = ["อาหาร", "เมนู", "สูตร", "วิธีทำ", "ที่มี", "สูง", "ต่ำ", "แนะนำ", "หา"]
        words = query_lower.split()
        search_terms = [word for word in words if word not in filter_words and len(word) > 1]
        analysis["search_terms"] = search_terms
        
        return analysis

    def smart_search(self, query: str, use_api: bool = True, adjust_consumption: bool = True,
                    limit: int = 5) -> List[Dict]:
        """ระบบค้นหาอัจฉริยะ"""
        analysis = self.analyze_query(query)
        results = []
        
        if analysis["intent"] == "nutrition_search":
            # ค้นหาตามเกณฑ์โภชนาการ
            nutrition_results = self.find_recipes_by_nutrition(
                analysis["nutrition_criteria"], use_api, adjust_consumption
            )
            
            for recipe_name, nutrition_data, idx in nutrition_results[:limit]:
                results.append({
                    "name": recipe_name,
                    "similarity": 1.0,  # ความเกี่ยวข้องสูง
                    "index": idx,
                    "nutrition": nutrition_data,
                    "match_reason": f"ตรงกับเกณฑ์โภชนาการ: {', '.join(analysis['nutrition_criteria'].keys())}"
                })
        
        elif analysis["intent"] == "food_type_search":
            # ค้นหาตามประเภทอาหาร
            type_keywords = self.food_type_keywords[analysis["food_type"]]
            
            for idx, recipe in self.data.iterrows():
                recipe_name_lower = recipe['name'].lower()
                method_lower = recipe['method'].lower()
                
                # ตรวจสอบในชื่อหรือวิธีทำ
                found_match = False
                for keyword in type_keywords:
                    if keyword in recipe_name_lower or keyword in method_lower:
                        found_match = True
                        break
                
                if found_match:
                    nutrition_data = self.get_recipe_nutrition(idx, use_api, adjust_consumption)
                    results.append({
                        "name": recipe['name'],
                        "similarity": 0.9,
                        "index": idx,
                        "nutrition": nutrition_data,
                        "match_reason": f"ประเภท: {analysis['food_type']}"
                    })
                    
                    if len(results) >= limit:
                        break
        
        else:
            # ค้นหาแบบทั่วไป + fuzzy matching
            if analysis["search_terms"]:
                search_query = " ".join(analysis["search_terms"])
            else:
                search_query = query
            
            fuzzy_results = self.fuzzy_search(search_query, threshold=0.4)
            
            for recipe_name, similarity, idx in fuzzy_results[:limit]:
                nutrition_data = self.get_recipe_nutrition(idx, use_api, adjust_consumption)
                results.append({
                    "name": recipe_name,
                    "similarity": similarity,
                    "index": idx,
                    "nutrition": nutrition_data,
                    "match_reason": f"ความคล้ายคลึงชื่อ: {similarity:.2f}"
                })
        
        return results

    def get_nutrition_recommendations(self, target_nutrition: str, use_api: bool = True,
                                    adjust_consumption: bool = True, limit: int = 3) -> List[Dict]:
        """แนะนำอาหารตามโภชนาการที่ต้องการ"""
        if target_nutrition in self.nutrition_keywords:
            criteria = self.nutrition_keywords[target_nutrition]
            results = self.find_recipes_by_nutrition(criteria, use_api, adjust_consumption)
            
            recommendations = []
            for recipe_name, nutrition_data, idx in results[:limit]:
                recommendations.append({
                    "name": recipe_name,
                    "index": idx,
                    "nutrition": nutrition_data,
                    "reason": f"แนะนำสำหรับผู้ต้องการ{target_nutrition}"
                })
            
            return recommendations
        
        return []

    def compare_recipes_nutrition(self, recipe_indices: List[int], use_api: bool = True,
                                adjust_consumption: bool = True) -> Dict:
        """เปรียบเทียบค่าโภชนาการของหลายสูตร"""
        comparison = {
            "recipes": [],
            "nutrients_comparison": {}
        }
        
        # รวบรวมข้อมูลโภชนาการของแต่ละสูตร
        for idx in recipe_indices:
            recipe_name = self.data.iloc[idx]['name']
            nutrition_data = self.get_recipe_nutrition(idx, use_api, adjust_consumption)
            
            comparison["recipes"].append({
                "name": recipe_name,
                "index": idx,
                "nutrition": nutrition_data['total_nutrition']
            })
        
        # สร้างการเปรียบเทียบแต่ละสารอาหาร
        nutrients = ["calories", "protein", "carbs", "fat", "fiber", 
                    "vitamin_a", "vitamin_c", "calcium", "iron"]
        
        for nutrient in nutrients:
            values = [recipe["nutrition"][nutrient] for recipe in comparison["recipes"]]
            comparison["nutrients_comparison"][nutrient] = {
                "values": values,
                "max": max(values),
                "min": min(values),
                "avg": sum(values) / len(values) if values else 0
            }
        
        return comparison

    def get_ingredient_alternatives(self, target_ingredient: str, 
                                  nutrition_focus: str = None) -> List[Dict]:
        """แนะนำวัตถุดิบทดแทนตามโภชนาการ"""
        alternatives = []
        target_nutrition = self.nutrition_api.get_nutrition_data(target_ingredient)
        
        # หาวัตถุดิบที่มีโภชนาการคล้ายกัน
        for ingredient, nutrition in self.nutrition_api.local_nutrition_db.items():
            if ingredient != target_ingredient:
                # คำนวณความคล้ายคลึงทางโภชนาการ
                similarity_score = 0
                total_nutrients = 0
                
                for nutrient in ["protein", "fat", "carbs"]:
                    if target_nutrition[nutrient] > 0:
                        ratio = min(nutrition[nutrient], target_nutrition[nutrient]) / max(nutrition[nutrient], target_nutrition[nutrient])
                        similarity_score += ratio
                        total_nutrients += 1
                
                if total_nutrients > 0:
                    avg_similarity = similarity_score / total_nutrients
                    if avg_similarity > 0.5:  # ความคล้ายคลึง > 50%
                        alternatives.append({
                            "ingredient": ingredient,
                            "similarity": avg_similarity,
                            "nutrition": nutrition
                        })
        
        # เรียงลำดับตามความคล้ายคลึง
        alternatives.sort(key=lambda x: x["similarity"], reverse=True)
        return alternatives[:5]
