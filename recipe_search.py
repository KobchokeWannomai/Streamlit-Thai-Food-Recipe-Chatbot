import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Import NutritionAPI from the main nutrition module
try:
    from nutrition_api import NutritionAPI
except ImportError:
    # Fallback import if run from different directory
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from nutrition_api import NutritionAPI
    except ImportError:
        # Simple fallback if nutrition_api is not available
        class NutritionAPI:
            def __init__(self):
                pass
            def calculate_recipe_nutrition(self, *args, **kwargs):
                return {
                    "total_nutrition": {
                        "calories": 0, "protein": 0, "carbs": 0, "fat": 0, "fiber": 0,
                        "vitamin_a": 0, "vitamin_c": 0, "vitamin_b1": 0, "vitamin_b2": 0,
                        "calcium": 0, "iron": 0, "potassium": 0, "sodium": 0
                    },
                    "ingredient_details": [],
                    "enhanced_ingredients": None,
                    "settings": {}
                }

class RecipeSearchEngine:
    """เครื่องมือค้นหาสูตรอาหารขั้นสูงพร้อมระบบ AI และ Fuzzy Matching"""
    
    def __init__(self, data: pd.DataFrame, nutrition_api: NutritionAPI = None):
        self.data = data
        self.nutrition_api = nutrition_api or NutritionAPI()
        self.recipe_nutrition_cache = {}
        self.embeddings = None
        self.model = None
        
        # คำสำคัญสำหรับการจำแนกประเภทการค้นหา
        self.search_type_keywords = {
            'nutrition_based': [
                'แคลอรี่', 'โปรตีน', 'ไขมัน', 'คาร์โบ', 'วิตามิน', 
                'แคลเซียม', 'เหล็ก', 'โซเดียม', 'ใยอาหาร',
                'สุขภาพ', 'ลดน้ำหนัก', 'เบาหวาน', 'ความดัน'
            ],
            'cooking_method': [
                'ทอด', 'ต้ม', 'ผัด', 'ย่าง', 'นึ่ง', 'แกง', 'ยำ', 'คั่ว'
            ],
            'ingredient_based': [
                'หมู', 'ไก่', 'เนื้อ', 'กุ้ง', 'ปลา', 'ไข่', 'ผัก', 'เห็ด'
            ]
        }

    def set_model_and_embeddings(self, model: SentenceTransformer, embeddings: np.ndarray):
        """ตั้งค่าโมเดล AI และ embeddings สำหรับการค้นหา"""
        self.model = model
        self.embeddings = embeddings

    def classify_search_intent(self, query: str) -> str:
        """จำแนกประเภทของการค้นหา"""
        query_lower = query.lower()
        
        scores = {}
        for category, keywords in self.search_type_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            scores[category] = score
        
        if scores['nutrition_based'] > 0:
            return 'nutrition_based'
        elif scores['cooking_method'] > 0:
            return 'cooking_method'
        elif scores['ingredient_based'] > 0:
            return 'ingredient_based'
        else:
            return 'general'

    def preprocess_query(self, query: str) -> str:
        """ประมวลผลคำค้นหาก่อนการใช้งาน"""
        # ลบคำที่ไม่จำเป็น
        stop_words = ["อาหาร", "เมนู", "สูตร", "วิธีทำ", "ทำ", "ปรุง", "อร่อย", "ง่าย", "แนะนำ"]
        
        # แยกคำและกรองคำที่ไม่จำเป็น
        words = query.lower().split()
        filtered_words = [word for word in words if word not in stop_words and len(word) > 1]
        
        return " ".join(filtered_words) if filtered_words else query.lower()

    def advanced_fuzzy_search(self, query: str, threshold: float = 0.4) -> List[Tuple[str, float, int]]:
        """การค้นหาแบบ fuzzy matching ที่ปรับปรุงแล้ว"""
        query = self.preprocess_query(query)
        matches = []
        
        for idx, recipe_name in enumerate(self.data['name']):
            recipe_name_lower = recipe_name.lower()
            
            # 1. ความคล้ายคลึงแบบ sequence matching
            sequence_similarity = SequenceMatcher(None, query, recipe_name_lower).ratio()
            
            # 2. การตรวจสอบคำที่ตรงกันทั้งหมด
            exact_match_score = 0
            if query in recipe_name_lower:
                exact_match_score = min(len(query) / len(recipe_name_lower), 1.0)
            
            # 3. การตรวจสอบคำต่างๆ แยกกัน
            query_words = query.split()
            word_scores = []
            partial_scores = []
            
            for q_word in query_words:
                if len(q_word) <= 1:
                    continue
                    
                best_match_score = 0
                best_partial_score = 0
                
                recipe_words = recipe_name_lower.split()
                for r_word in recipe_words:
                    word_similarity = SequenceMatcher(None, q_word, r_word).ratio()
                    if word_similarity >= 0.8:
                        best_match_score = max(best_match_score, word_similarity)
                    elif len(q_word) >= 3:
                        if q_word in r_word:
                            best_partial_score = max(best_partial_score, 0.7)
                        elif r_word in q_word and len(r_word) >= 3:
                            best_partial_score = max(best_partial_score, 0.6)
                
                if best_match_score > 0:
                    word_scores.append(best_match_score)
                elif best_partial_score > 0:
                    partial_scores.append(best_partial_score)
            
            # คำนวณคะแนนจากการจับคู่คำ
            word_match_score = 0
            if word_scores:
                word_match_score = sum(word_scores) / len(query_words)
            elif partial_scores:
                word_match_score = sum(partial_scores) / len(query_words) * 0.8
            
            # 4. ตรวจสอบในส่วนผสมและวิธีทำ
            content_match_score = 0
            if sequence_similarity < threshold and word_match_score < threshold:
                ingredient_text = str(self.data.iloc[idx].get('ingredient', '')).lower()
                method_text = str(self.data.iloc[idx].get('method', '')).lower()
                
                content_matches = 0
                for q_word in query_words:
                    if len(q_word) >= 3:
                        if q_word in ingredient_text:
                            content_matches += 0.3
                        elif q_word in method_text:
                            content_matches += 0.2
                
                if content_matches > 0:
                    content_match_score = min(content_matches / len(query_words), 0.5)
            
            # 5. คำนวณคะแนนรวม
            final_scores = [
                sequence_similarity * 0.3,
                exact_match_score * 0.9,
                word_match_score * 0.7,
                content_match_score * 0.4
            ]
            
            final_score = max(final_scores)
            
            # ปรับคะแนนตามความยาวของชื่อเมนู
            if final_score > 0:
                length_factor = 1.0
                if len(recipe_name_lower) <= 10 and exact_match_score > 0:
                    length_factor = 1.2
                elif len(recipe_name_lower) > 20:
                    length_factor = 0.9
                
                final_score = min(final_score * length_factor, 1.0)
            
            if final_score >= threshold:
                matches.append((recipe_name, final_score, idx))
        
        # เรียงลำดับและกรองผลลัพธ์
        matches.sort(key=lambda x: x[1], reverse=True)
        
        # กรองผลลัพธ์ที่ซ้ำกัน
        filtered_matches = []
        seen_scores = set()
        
        for match in matches:
            score_rounded = round(match[1], 2)
            if score_rounded not in seen_scores or len(filtered_matches) < 3:
                filtered_matches.append(match)
                seen_scores.add(score_rounded)
                
                if len(filtered_matches) >= 10:
                    break
        
        return filtered_matches

    def semantic_search(self, query: str, limit: int = 5) -> List[Tuple[str, float, int]]:
        """การค้นหาแบบ semantic search ด้วย AI"""
        if self.model is None or self.embeddings is None:
            return []
        
        try:
            # สร้าง embedding สำหรับคำค้นหา
            query_embedding = self.model.encode([query])
            
            # คำนวณความคล้ายคลึง
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # หาผลลัพธ์ที่ดีที่สุด
            top_indices = np.argsort(similarities)[::-1][:limit]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.3:  # threshold สำหรับ semantic search
                    recipe_name = self.data.iloc[idx]['name']
                    results.append((recipe_name, similarities[idx], idx))
            
            return results
            
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []

    def hybrid_search(self, query: str, limit: int = 5, 
                     fuzzy_weight: float = 0.7, semantic_weight: float = 0.3) -> List[Tuple[str, float, int]]:
        """การค้นหาแบบผสมผสานระหว่าง fuzzy matching และ semantic search"""
        
        # ค้นหาด้วย fuzzy matching
        fuzzy_results = self.advanced_fuzzy_search(query, threshold=0.3)
        
        # ค้นหาด้วย semantic search
        semantic_results = self.semantic_search(query, limit=limit*2)
        
        # รวมและคำนวณคะแนนผสม
        combined_results = {}
        
        # เพิ่มผลลัพธ์จาก fuzzy search
        for name, score, idx in fuzzy_results:
            combined_results[idx] = {
                'name': name,
                'fuzzy_score': score,
                'semantic_score': 0,
                'combined_score': score * fuzzy_weight
            }
        
        # เพิ่มผลลัพธ์จาก semantic search
        for name, score, idx in semantic_results:
            if idx in combined_results:
                combined_results[idx]['semantic_score'] = score
                combined_results[idx]['combined_score'] = (
                    combined_results[idx]['fuzzy_score'] * fuzzy_weight + 
                    score * semantic_weight
                )
            else:
                combined_results[idx] = {
                    'name': name,
                    'fuzzy_score': 0,
                    'semantic_score': score,
                    'combined_score': score * semantic_weight
                }
        
        # เรียงลำดับตามคะแนนรวม
        sorted_results = sorted(
            combined_results.items(), 
            key=lambda x: x[1]['combined_score'], 
            reverse=True
        )
        
        # แปลงเป็นรูปแบบผลลัพธ์
        final_results = []
        for idx, result_data in sorted_results[:limit]:
            final_results.append((
                result_data['name'], 
                result_data['combined_score'], 
                idx
            ))
        
        return final_results

    def get_recipe_nutrition(self, recipe_index: int, use_api: bool = True, 
                           adjust_consumption: bool = True, 
                           enhance_missing: bool = False) -> Dict:
        """ดึงข้อมูลโภชนาการของสูตรอาหารพร้อมแคช - แก้ไขให้ทำงานถูกต้อง"""
        cache_key = f"{recipe_index}_{use_api}_{adjust_consumption}_{enhance_missing}"
        
        if cache_key in self.recipe_nutrition_cache:
            return self.recipe_nutrition_cache[cache_key]
        
        try:
            recipe = self.data.iloc[recipe_index]
            
            # เรียกใช้ calculate_recipe_nutrition พร้อมพารามิเตอร์ครบถ้วน
            nutrition_data = self.nutrition_api.calculate_recipe_nutrition(
                ingredients_text=recipe['ingredient'], 
                use_api=use_api,
                adjust_consumption=adjust_consumption,
                enhance_missing=enhance_missing,
                recipe_name=recipe['name'],          # เพิ่มการส่งผ่าน recipe_name
                method_text=recipe['method']         # เพิ่มการส่งผ่าน method_text
            )
            
            self.recipe_nutrition_cache[cache_key] = nutrition_data
            return nutrition_data
            
        except Exception as e:
            print(f"Error calculating nutrition for recipe {recipe_index}: {str(e)}")
            # ส่งคืนข้อมูลโภชนาการเริ่มต้นหากเกิดข้อผิดพลาด
            default_nutrition = {
                "total_nutrition": {
                    "calories": 0, "protein": 0, "carbs": 0, "fat": 0, "fiber": 0,
                    "vitamin_a": 0, "vitamin_c": 0, "vitamin_b1": 0, "vitamin_b2": 0,
                    "calcium": 0, "iron": 0, "potassium": 0, "sodium": 0
                },
                "ingredient_details": [],
                "enhanced_ingredients": None,
                "settings": {
                    "use_api": use_api, 
                    "adjust_consumption": adjust_consumption, 
                    "enhance_missing": enhance_missing
                }
            }
            return default_nutrition

    def smart_search(self, query: str, use_api: bool = True, adjust_consumption: bool = True,
                    enhance_missing: bool = False, fuzzy_threshold: float = 0.4,
                    limit: int = 5, search_method: str = "hybrid") -> List[Dict]:
        """ระบบค้นหาอัจฉริยะที่รวมทุกวิธีการ - ปรับปรุงให้ทำงานถูกต้อง"""
        
        # จำแนกประเภทการค้นหา
        search_intent = self.classify_search_intent(query)
        
        # เลือกวิธีการค้นหาตามความต้องการ
        if search_method == "fuzzy":
            search_results = self.advanced_fuzzy_search(query, threshold=fuzzy_threshold)
        elif search_method == "semantic" and self.model is not None:
            search_results = self.semantic_search(query, limit=limit)
        elif search_method == "hybrid" and self.model is not None:
            search_results = self.hybrid_search(query, limit=limit)
        else:
            # ใช้ fuzzy search เป็นค่าเริ่มต้น
            search_results = self.advanced_fuzzy_search(query, threshold=fuzzy_threshold)
        
        results = []
        
        for recipe_name, similarity, recipe_idx in search_results[:limit]:
            # คำนวณข้อมูลโภชนาการ
            nutrition_data = self.get_recipe_nutrition(
                recipe_idx, use_api, adjust_consumption, enhance_missing
            )
            
            # กำหนดเหตุผลการจับคู่
            match_reason = f"ความคล้ายคลึง: {similarity:.0%}"
            if search_method == "hybrid":
                match_reason += " (AI + Fuzzy)"
            elif search_method == "semantic":
                match_reason += " (AI Semantic)"
            else:
                match_reason += " (Fuzzy Match)"
            
            # สร้างข้อมูลสำหรับผลลัพธ์
            result_data = {
                "name": recipe_name,
                "similarity": similarity,
                "index": recipe_idx,
                "nutrition": nutrition_data,
                "match_reason": match_reason,
                "match_type": search_method,
                "search_intent": search_intent
            }
            
            # เพิ่มข้อมูลการปรับปรุง (หากมี)
            if nutrition_data.get('enhanced_ingredients'):
                result_data["enhanced"] = True
                result_data["original_ingredients"] = self.data.iloc[recipe_idx]['ingredient']
                result_data["enhanced_ingredients"] = nutrition_data['enhanced_ingredients']
            
            results.append(result_data)
        
        return results

    def get_recipe_suggestions(self, limit: int = 10) -> List[str]:
        """ให้คำแนะนำสูตรอาหารแบบสุ่ม"""
        if len(self.data) == 0:
            return []
        
        # สุ่มเลือกสูตรอาหาร
        sample_size = min(limit, len(self.data))
        random_indices = np.random.choice(len(self.data), size=sample_size, replace=False)
        
        suggestions = []
        for idx in random_indices:
            recipe_name = self.data.iloc[idx]['name']
            suggestions.append(recipe_name)
        
        return suggestions

    def find_similar_recipes(self, recipe_index: int, limit: int = 5) -> List[Tuple[str, float, int]]:
        """หาสูตรอาหารที่คล้ายกัน"""
        if self.embeddings is None or recipe_index >= len(self.embeddings):
            return []
        
        try:
            # ใช้ embedding ของสูตรอาหารเป้าหมาย
            target_embedding = self.embeddings[recipe_index].reshape(1, -1)
            
            # คำนวณความคล้ายคลึงกับสูตรอื่นๆ
            similarities = cosine_similarity(target_embedding, self.embeddings)[0]
            
            # หาสูตรที่คล้ายที่สุด (ไม่รวมตัวเอง)
            similar_indices = np.argsort(similarities)[::-1][1:limit+1]
            
            results = []
            for idx in similar_indices:
                if similarities[idx] > 0.5:  # threshold สำหรับความคล้าย
                    recipe_name = self.data.iloc[idx]['name']
                    results.append((recipe_name, similarities[idx], idx))
            
            return results
            
        except Exception as e:
            print(f"Error finding similar recipes: {e}")
            return []

    def analyze_recipe_nutrition_trends(self, recipe_indices: List[int]) -> Dict:
        """วิเคราะห์แนวโน้มโภชนาการของสูตรอาหารที่เลือก"""
        if not recipe_indices:
            return {}
        
        nutrition_data = []
        for idx in recipe_indices:
            try:
                nutrition = self.get_recipe_nutrition(idx, adjust_consumption=True, enhance_missing=False)
                if nutrition and nutrition.get('total_nutrition'):
                    nutrition_data.append(nutrition['total_nutrition'])
            except:
                continue
        
        if not nutrition_data:
            return {}
        
        # คำนวณค่าสถิติ
        analysis = {}
        nutrients = ['calories', 'protein', 'carbs', 'fat', 'fiber', 'sodium', 'calcium', 'iron']
        
        for nutrient in nutrients:
            values = [data.get(nutrient, 0) for data in nutrition_data]
            if values:
                analysis[nutrient] = {
                    'average': np.mean(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'std': np.std(values),
                    'median': np.median(values)
                }
        
        return analysis

    def get_search_statistics(self) -> Dict:
        """สถิติการค้นหาและข้อมูลระบบ"""
        return {
            "total_recipes": len(self.data),
            "has_ai_model": self.model is not None,
            "has_embeddings": self.embeddings is not None,
            "cache_size": len(self.recipe_nutrition_cache),
            "nutrition_api_available": hasattr(self.nutrition_api, 'calculate_recipe_nutrition'),
            "available_search_methods": [
                "fuzzy", 
                "semantic" if self.model is not None else None,
                "hybrid" if self.model is not None else None
            ]
        }

    def clear_cache(self):
        """ล้างแคชข้อมูลโภชนาการ"""
        self.recipe_nutrition_cache.clear()
        print("Cache cleared successfully")

    def validate_recipe_data(self, recipe_index: int) -> Dict:
        """ตรวจสอบความถูกต้องของข้อมูลสูตรอาหาร"""
        if recipe_index >= len(self.data):
            return {"valid": False, "error": "Recipe index out of range"}
        
        recipe = self.data.iloc[recipe_index]
        validation = {
            "valid": True,
            "warnings": [],
            "info": {}
        }
        
        # ตรวจสอบชื่อเมนู
        if not recipe.get('name') or len(str(recipe['name']).strip()) == 0:
            validation["warnings"].append("ไม่มีชื่อเมนู")
            validation["valid"] = False
        
        # ตรวจสอบวัตถุดิบ
        ingredients = str(recipe.get('ingredient', ''))
        if len(ingredients.strip()) < 10:
            validation["warnings"].append("รายการวัตถุดิบสั้นเกินไป")
        
        ingredient_lines = [line.strip() for line in ingredients.split('\n') if line.strip()]
        validation["info"]["ingredient_count"] = len(ingredient_lines)
        
        # ตรวจสอบวิธีทำ
        method = str(recipe.get('method', ''))
        if len(method.strip()) < 20:
            validation["warnings"].append("วิธีทำสั้นเกินไป")
        
        validation["info"]["method_length"] = len(method)
        
        return validation

    def export_search_results(self, results: List[Dict], format: str = "dict") -> any:
        """ส่งออกผลลัพธ์การค้นหาในรูปแบบต่างๆ"""
        if format == "dataframe":
            # แปลงเป็น DataFrame
            df_data = []
            for result in results:
                nutrition = result.get('nutrition', {}).get('total_nutrition', {})
                df_data.append({
                    'name': result['name'],
                    'similarity': result['similarity'],
                    'calories': nutrition.get('calories', 0),
                    'protein': nutrition.get('protein', 0),
                    'fat': nutrition.get('fat', 0),
                    'carbs': nutrition.get('carbs', 0),
                    'sodium': nutrition.get('sodium', 0)
                })
            return pd.DataFrame(df_data)
        
        elif format == "summary":
            # สรุปผลลัพธ์
            total_results = len(results)
            avg_similarity = np.mean([r['similarity'] for r in results]) if results else 0
            
            return {
                "total_results": total_results,
                "average_similarity": avg_similarity,
                "best_match": results[0]['name'] if results else None,
                "search_types": list(set([r.get('search_intent', 'general') for r in results]))
            }
        
        else:
            return results  # ส่งคืนเป็น dict ปกติ
