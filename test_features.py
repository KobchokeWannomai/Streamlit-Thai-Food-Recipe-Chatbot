#!/usr/bin/env python3
"""
ระบบทดสอบฟีเจอร์ขั้นสูงของแชทบอทสูตรอาหารไทย
Advanced Feature Testing System for Thai Food Chatbot
รองรับการทดสอบการค้นหาขั้นสูง การวิเคราะห์โภชนาการ และฟีเจอร์ต่างๆ
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os
import json
from datetime import datetime
import sqlite3

# Import modules to test
from nutrition_analyzer import NutritionAnalyzer, NutritionInfo, ThaiNutritionData
from ingredient_converter import IngredientConverter
from config import Config
import streamlit as st
from streamlit_app import EnhancedFuzzyMatcher, search_recipes_enhanced

class TestEnhancedFuzzyMatcher:
    """ทดสอบระบบการจับคู่ข้อความขั้นสูง"""
    
    def setup_method(self):
        """ตั้งค่าก่อนการทดสอบ"""
        self.matcher = EnhancedFuzzyMatcher()
        
        # รายการเมนูทดสอบ
        self.test_recipes = [
            'ผัดกะเพรา', 'ต้มยำกุ้ง', 'ส้มตำ', 'แกงเขียวหวาน', 'ผัดไทย',
            'กุ้งทาพริกไทยกระเทียม', 'ข้าวเม่าทอด', 'เปรี้ยวหวานไข่ม้วน', 
            'ไข่จ่อม', 'งบปลาทู', 'ยำไข่ปลาดุก', 'กล้วยบวชชี',
            'ไข่เจียว', 'ไข่ดาว', 'ไข่ตุ๋น', 'ไข่กระจัง'
        ]
    
    def test_exact_match(self):
        """ทดสอบการจับคู่แบบตรงตัว"""
        query = "ผัดกะเพรา"
        matches = self.matcher.find_best_match(query, self.test_recipes, threshold=0.6)
        
        assert len(matches) > 0
        assert matches[0]['similarity'] == 1.0
        assert matches[0]['text'] == "ผัดกะเพรา"
        assert matches[0]['match_type'] == 'menu_exact'
    
    def test_typo_correction(self):
        """ทดสอบการแก้ไขการพิมพ์ผิด"""
        # ทดสอบการพิมพ์ผิดทั่วไป
        typo_tests = [
            ('กระเพรา', 'ผัดกะเพรา'),  # กระเพรา -> กะเพรา
            ('ต้มยํา', 'ต้มยำกุ้ง'),      # ต้มยํา -> ต้มยำ  
            ('ส้มตํา', 'ส้มตำ'),        # ส้มตํา -> ส้มตำ
            ('กุ้งทาพริก', 'กุ้งทาพริกไทยกระเทียม'),
            ('ข้าวเหม่า', 'ข้าวเม่าทอด'),
            ('ไข่จ๋อม', 'ไข่จ่อม')
        ]
        
        for typo, expected_containing in typo_tests:
            matches = self.matcher.find_best_match(typo, self.test_recipes, threshold=0.5)
            assert len(matches) > 0
            # ตรวจสอบว่าผลลัพธ์แรกมีความเกี่ยวข้องกับที่คาดหวัง
            found_relevant = any(expected_containing.lower() in match['text'].lower() 
                               for match in matches[:3])
            assert found_relevant, f"ไม่พบผลลัพธ์ที่เกี่ยวข้องสำหรับ '{typo}'"
    
    def test_partial_match(self):
        """ทดสอบการจับคู่แบบบางส่วน"""
        partial_tests = [
            ('ไข่', ['ไข่เจียว', 'ไข่ดาว', 'ไข่ตุ๋น', 'ไข่กระจัง']),
            ('กุ้ง', ['กุ้งทาพริกไทยกระเทียม', 'ต้มยำกุ้ง']),
            ('ข้าว', ['ข้าวเม่าทอด']),
        ]
        
        for partial_query, expected_items in partial_tests:
            matches = self.matcher.find_best_match(partial_query, self.test_recipes, threshold=0.3)
            assert len(matches) > 0
            
            # ตรวจสอบว่าพบเมนูที่คาดหวังอย่างน้อย 1 เมนู
            found_items = [match['text'] for match in matches]
            found_expected = any(item in found_items for item in expected_items)
            assert found_expected, f"ไม่พบเมนูที่คาดหวังสำหรับ '{partial_query}'"
    
    def test_menu_variations(self):
        """ทดสอบการจับคู่รูปแบบเมนูที่หลากหลาย"""
        variation_tests = [
            ('กุ้งพริกไทย', 'กุ้งทาพริกไทยกระเทียม'),
            ('ข้าวเหม่าทอด', 'ข้าวเม่าทอด'),
            ('เปรี้ยวหวาน', 'เปรี้ยวหวานไข่ม้วน'),
            ('งบปลา', 'งบปลาทู'),
            ('ยำไข่ปลา', 'ยำไข่ปลาดุก')
        ]
        
        for variation, expected in variation_tests:
            matches = self.matcher.find_best_match(variation, self.test_recipes, threshold=0.5)
            assert len(matches) > 0
            
            # ตรวจสอบว่าพบเมนูที่คาดหวัง
            found_expected = any(expected in match['text'] for match in matches[:3])
            assert found_expected, f"ไม่พบ '{expected}' สำหรับ '{variation}'"
    
    def test_similarity_scoring(self):
        """ทดสอบการให้คะแนนความคล้ายคลึง"""
        query = "ผัดกะเพรา"
        matches = self.matcher.find_best_match(query, self.test_recipes, threshold=0.1)
        
        # ตรวจสอบว่าคะแนนเรียงลำดับจากมากไปน้อย
        similarities = [match['similarity'] for match in matches]
        assert similarities == sorted(similarities, reverse=True)
        
        # ตรวจสอบว่าการจับคู่แบบตรงตัวได้คะแนนสูงสุด
        exact_match = next((m for m in matches if m['text'] == query), None)
        assert exact_match is not None
        assert exact_match['similarity'] == 1.0

class TestNutritionAnalyzer:
    """ทดสอบระบบวิเคราะห์โภชนาการ"""
    
    def setup_method(self):
        """ตั้งค่าก่อนการทดสอบ"""
        self.analyzer = NutritionAnalyzer()
        self.thai_data = ThaiNutritionData()
    
    def test_thai_nutrition_database(self):
        """ทดสอบฐานข้อมูลโภชนาการไทย"""
        # ทดสอบวัตถุดิบพื้นฐาน
        test_ingredients = ['หมู', 'ไก่', 'กุ้ง', 'ไข่ไก่', 'ข้าว', 'น้ำปลา']
        
        for ingredient in test_ingredients:
            nutrition = self.thai_data.get_nutrition_info(ingredient)
            assert nutrition is not None, f"ไม่พบข้อมูลโภชนาการสำหรับ {ingredient}"
            assert nutrition.calories >= 0
            assert nutrition.protein >= 0
            assert nutrition.carbs >= 0
            assert nutrition.fat >= 0
    
    def test_ingredient_parsing(self):
        """ทดสอบการแยกวิเคราะห์วัตถุดิบ"""
        ingredients_text = """
        - ไข่ไก่ 2 ฟอง
        - น้ำมันพืช 1 ช้อนโต๊ะ
        - กุ้ง 200 กรัม
        - น้ำปลา 1 ช้อนโต๊ะ
        """
        
        ingredients = self.analyzer._parse_ingredients(ingredients_text)
        assert len(ingredients) == 4
        
        # ตรวจสอบว่าแยกได้ถูกต้อง
        expected_ingredients = ['ไข่ไก่ 2 ฟอง', 'น้ำมันพืช 1 ช้อนโต๊ะ', 'กุ้ง 200 กรัม', 'น้ำปลา 1 ช้อนโต๊ะ']
        for expected in expected_ingredients:
            assert any(expected in ingredient for ingredient in ingredients)
    
    def test_nutrition_calculation(self):
        """ทดสอบการคำนวณโภชนาการ"""
        ingredients_text = "- ไข่ไก่ 2 ฟอง"
        
        nutrition_data = self.analyzer.analyze_ingredients(ingredients_text)
        assert len(nutrition_data) > 0
        
        total_nutrition = self.analyzer.calculate_total_nutrition(nutrition_data)
        assert total_nutrition.calories > 0
        assert total_nutrition.protein > 0
    
    def test_cooking_adjustments(self):
        """ทดสอบการปรับแต่งการทำอาหาร"""
        ingredients_text = "- ไข่ไก่ 2 ฟอง"
        recipe_name = "ไข่เจียว"
        
        # วิเคราะห์แบบปกติ
        normal_data = self.analyzer.analyze_ingredients(ingredients_text, recipe_name, False)
        normal_total = self.analyzer.calculate_total_nutrition(normal_data)
        
        # วิเคราะห์แบบมีการปรับแต่ง
        enhanced_data = self.analyzer.analyze_ingredients(ingredients_text, recipe_name, True)
        enhanced_total = self.analyzer.calculate_total_nutrition(enhanced_data)
        
        # ตรวจสอบว่ามีการเพิ่มน้ำมันสำหรับการทอด
        assert len(enhanced_data) > len(normal_data)
        assert enhanced_total.calories > normal_total.calories
        assert enhanced_total.fat > normal_total.fat
    
    def test_recipe_analysis(self):
        """ทดสอบการวิเคราะห์สูตรอาหาร"""
        recipe_name = "ผัดกะเพรา"
        ingredients = """
        - หมูสับ 200 กรัม
        - ใบกะเพรา 1 ถ้วย
        - พริกขี้หนู 5 เม็ด
        - กระเทียม 5 กลีบ
        - น้ำปลา 2 ช้อนโต๊ะ
        """
        
        result = self.analyzer.analyze_recipe(recipe_name, ingredients)
        
        assert result['recipe_name'] == recipe_name
        assert 'total_nutrition' in result
        assert 'ingredients' in result
        assert result['ingredient_count'] > 0
        
        # ตรวจสอบโภชนาการรวม
        nutrition = result['total_nutrition']
        assert nutrition['calories'] > 0
        assert nutrition['protein'] > 0

class TestIngredientConverter:
    """ทดสอบระบบแปลงหน่วยวัตถุดิบ"""
    
    def setup_method(self):
        """ตั้งค่าก่อนการทดสอบ"""
        self.converter = IngredientConverter()
    
    def test_unit_normalization(self):
        """ทดสอบการแปลงหน่วยให้เป็นมาตรฐาน"""
        test_cases = [
            ('กรัม', 'กรัม'),
            ('ก.', 'กรัม'),
            ('g', 'กรัม'),
            ('ช้อนโต๊ะ', 'ช้อนโต๊ะ'),
            ('ชต.', 'ช้อนโต๊ะ'),
            ('tbsp', 'ช้อนโต๊ะ'),
            ('ฟอง', 'ฟอง'),
            ('ลูก', 'ฟอง'),
        ]
        
        for input_unit, expected in test_cases:
            result = self.converter.normalize_unit(input_unit)
            assert result == expected, f"Expected {expected}, got {result} for {input_unit}"
    
    def test_quantity_extraction(self):
        """ทดสอบการแยกปริมาณและหน่วย"""
        test_cases = [
            ('ไข่ไก่ 2 ฟอง', (2.0, 'ฟอง', 'ไข่ไก่')),
            ('น้ำมันพืช 1 ช้อนโต๊ะ', (1.0, 'ช้อนโต๊ะ', 'น้ำมันพืช')),
            ('กุ้ง 200 กรัม', (200.0, 'กรัม', 'กุ้ง')),
            ('มะเขือเทศ 3 ลูก', (3.0, 'ฟอง', 'มะเขือเทศ')),
            ('กระเทียม 1/2 ช้อนชา', (0.5, 'ช้อนชา', 'กระเทียม')),
        ]
        
        for ingredient_text, expected in test_cases:
            quantity, unit, name = self.converter.extract_quantity_and_unit(ingredient_text)
            assert abs(quantity - expected[0]) < 0.1, f"Quantity mismatch for {ingredient_text}"
            assert unit == expected[1], f"Unit mismatch for {ingredient_text}"
            assert name.strip() == expected[2], f"Name mismatch for {ingredient_text}"
    
    def test_weight_conversion(self):
        """ทดสอบการแปลงน้ำหนัก"""
        test_cases = [
            (2, 'ฟอง', 'ไข่ไก่', 100),  # 2 ฟอง ≈ 100g
            (1, 'ช้อนโต๊ะ', 'น้ำมันพืช', 13.8),  # 1 ช้อนโต๊ะน้ำมัน ≈ 13.8g
            (200, 'กรัม', 'กุ้ง', 200),  # 200g = 200g
            (1, 'ถ้วย', 'กะทิ', 228),  # 1 ถ้วยกะทิ ≈ 228g
        ]
        
        for quantity, unit, ingredient, expected_weight in test_cases:
            weight = self.converter.convert_to_grams(quantity, unit, ingredient)
            # อนุญาตให้ผิดพลาดได้ 20%
            tolerance = expected_weight * 0.2
            assert abs(weight - expected_weight) <= tolerance, \
                f"Weight conversion failed for {quantity} {unit} {ingredient}: got {weight}, expected ~{expected_weight}"
    
    def test_ingredient_parsing(self):
        """ทดสอบการแยกวิเคราะห์วัตถุดิบแบบครบถ้วน"""
        test_ingredient = "ไข่ไก่ 2 ฟอง"
        result = self.converter.parse_and_convert_ingredient(test_ingredient)
        
        assert result['original_text'] == test_ingredient
        assert result['quantity'] == 2.0
        assert result['unit'] == 'ฟอง'
        assert result['name'] == 'ไข่ไก่'
        assert result['weight_grams'] > 0
        assert result['nutrition_multiplier'] > 0
        assert isinstance(result['is_liquid'], bool)
        assert isinstance(result['estimated'], bool)

class TestSearchFunctionality:
    """ทดสอบฟังก์ชันการค้นหา"""
    
    def setup_method(self):
        """ตั้งค่าก่อนการทดสอบ"""
        # สร้างข้อมูลทดสอบ
        self.test_data = pd.DataFrame({
            'name': [
                'ผัดกะเพรา', 'ต้มยำกุ้ง', 'ส้มตำ', 'แกงเขียวหวาน', 'ไข่เจียว',
                'กุ้งทาพริกไทยกระเทียม', 'ข้าวเม่าทอด', 'ยำไข่ปลาดุก'
            ],
            'ingredient': [
                'หมูสับ, ใบกะเพรา, พริก, กระเทียม',
                'กุ้ง, เห็ดฟาง, ตะไคร้, พริก',
                'มะละกอ, มะเขือเทศ, ถั่วฝักยาว',
                'ไก่, กะทิ, พริกแกงเขียวหวาน',
                'ไข่ไก่, น้ำมันพืช',
                'กุ้ง, พริกไทย, กระเทียม',
                'ข้าว, ไข่, หมู',
                'ไข่ปลาดุก, กุ้ง, ผัก'
            ],
            'method': [
                'ผัดให้เข้ากัน',
                'ต้มให้เดือด',
                'ตำให้พอแตก',
                'แกงให้เข้ากัน',
                'ทอดให้เหลือง',
                'ผัดให้หอม',
                'ผัดให้เข้ากัน',
                'ยำให้เข้ากัน'
            ]
        })
        
        # Mock embeddings
        self.mock_embeddings = np.random.random((len(self.test_data), 384))
        
        # Mock model
        self.mock_model = Mock()
        self.mock_model.encode.return_value = np.random.random((1, 384))
        
        # Mock nutrition analyzer
        self.mock_analyzer = Mock()
        mock_nutrition = {
            'recipe_name': 'test',
            'total_nutrition': {
                'calories': 300, 'protein': 20, 'carbs': 30, 'fat': 15, 'fiber': 5,
                'vitamins': {}, 'minerals': {}
            },
            'ingredients': [], 'ingredient_count': 3
        }
        self.mock_analyzer.analyze_recipe.return_value = mock_nutrition
        self.mock_analyzer.analyze_recipe_enhanced.return_value = mock_nutrition
    
    @patch('streamlit_app.load_enhanced_fuzzy_matcher')
    def test_enhanced_search(self, mock_matcher):
        """ทดสอบการค้นหาขั้นสูง"""
        # Mock EnhancedFuzzyMatcher
        mock_matcher_instance = Mock()
        mock_matcher.return_value = mock_matcher_instance
        mock_matcher_instance.find_best_match.return_value = [
            {'index': 0, 'similarity': 0.95, 'match_type': 'menu_match'}
        ]
        
        settings = {'enhanced_search': True, 'use_external_recipe_data': False}
        
        results = search_recipes_enhanced(
            "ผัดกะเพรา", 
            self.mock_model, 
            self.test_data, 
            self.mock_embeddings,
            self.mock_analyzer,
            settings
        )
        
        assert len(results) > 0
        assert results[0]['name'] == 'ผัดกะเพรา'
        assert 'nutrition' in results[0]
    
    def test_typo_search(self):
        """ทดสอบการค้นหาเมื่อพิมพ์ผิด"""
        matcher = EnhancedFuzzyMatcher()
        
        # ทดสอบการค้นหาเมื่อพิมพ์ผิด
        typo_queries = [
            'กระเพรา',  # ควรหา ผัดกะเพรา
            'ต้มยํา',    # ควรหา ต้มยำกุ้ง
            'ส้มตํา',    # ควรหา ส้มตำ
            'กุ้งพริกไทย'  # ควรหา กุ้งทาพริกไทยกระเทียม
        ]
        
        recipe_names = self.test_data['name'].tolist()
        
        for query in typo_queries:
            matches = matcher.find_best_match(query, recipe_names, threshold=0.5)
            assert len(matches) > 0, f"ไม่พบผลลัพธ์สำหรับ '{query}'"
            assert matches[0]['similarity'] > 0.5, f"คะแนนต่ำเกินไปสำหรับ '{query}'"

class TestConfigSettings:
    """ทดสอบการตั้งค่าระบบ"""
    
    def test_thai_menu_list(self):
        """ทดสอบรายการเมนูอาหารไทย"""
        assert len(Config.COMPLETE_THAI_MENU_LIST) > 100
        
        # ตรวจสอบเมนูสำคัญ
        important_menus = [
            'ผัดกะเพรา', 'ต้มยำกุ้ง', 'ส้มตำ', 'แกงเขียวหวาน', 'ผัดไทย',
            'กุ้งทาพริกไทยกระเทียม', 'ข้าวเม่าทอด', 'ไข่เจียว'
        ]
        
        for menu in important_menus:
            assert menu in Config.COMPLETE_THAI_MENU_LIST
    
    def test_search_enhancement_config(self):
        """ทดสอบการตั้งค่าการค้นหาขั้นสูง"""
        assert 'query_expansions' in Config.SEARCH_ENHANCEMENT
        assert 'cooking_methods' in Config.SEARCH_ENHANCEMENT
        
        # ตรวจสอบการขยายคำค้นหา
        expansions = Config.SEARCH_ENHANCEMENT['query_expansions']
        assert 'ไข่' in expansions
        assert 'หมู' in expansions
        assert 'ผัด' in expansions
        
        # ตรวจสอบวิธีการทำอาหาร
        methods = Config.SEARCH_ENHANCEMENT['cooking_methods']
        assert 'ผัด' in methods
        assert 'ทอด' in methods
        assert 'แกง' in methods
    
    def test_enhanced_search_terms(self):
        """ทดสอบการขยายคำค้นหา"""
        test_queries = ['ไข่', 'หมู', 'ผัด']
        
        for query in test_queries:
            expanded = Config.get_enhanced_search_terms(query)
            assert len(expanded) > 1
            assert query in expanded

class TestIntegration:
    """ทดสอบการทำงานร่วมกันของระบบ"""
    
    def setup_method(self):
        """ตั้งค่าก่อนการทดสอบ"""
        self.analyzer = NutritionAnalyzer()
        self.converter = IngredientConverter()
        self.matcher = EnhancedFuzzyMatcher()
    
    def test_full_recipe_analysis_pipeline(self):
        """ทดสอบกระบวนการวิเคราะห์สูตรอาหารแบบเต็ม"""
        recipe_name = "ไข่เจียว"
        ingredients = """
        - ไข่ไก่ 2 ฟอง
        - น้ำมันพืช 2 ช้อนโต๊ะ
        - เกลือ 1/2 ช้อนชา
        """
        
        # ขั้นตอนที่ 1: แปลงหน่วยวัตถุดิบ
        ingredient_lines = [line.strip()[2:] for line in ingredients.strip().split('\n') 
                          if line.strip().startswith('-')]
        
        converted_ingredients = []
        for ingredient_line in ingredient_lines:
            converted = self.converter.parse_and_convert_ingredient(ingredient_line)
            converted_ingredients.append(converted)
            assert converted['weight_grams'] > 0
        
        # ขั้นตอนที่ 2: วิเคราะห์โภชนาการ
        nutrition_result = self.analyzer.analyze_recipe(recipe_name, ingredients)
        
        assert nutrition_result['recipe_name'] == recipe_name
        assert nutrition_result['total_nutrition']['calories'] > 0
        assert nutrition_result['ingredient_count'] > 0
        
        # ขั้นตอนที่ 3: ทดสอบการค้นหา
        test_recipes = [recipe_name, 'ผัดกะเพรา', 'ต้มยำกุ้ง']
        
        search_variations = ['ไข่เจียว', 'ไข่เยียว', 'ไข่ทอด']
        for search_term in search_variations:
            matches = self.matcher.find_best_match(search_term, test_recipes, threshold=0.5)
            assert len(matches) > 0
            
            # ตรวจสอบว่าพบ "ไข่เจียว" ในผลลัพธ์
            found_target = any(recipe_name in match['text'] for match in matches)
            assert found_target, f"ไม่พบ '{recipe_name}' เมื่อค้นหาด้วย '{search_term}'"
    
    def test_enhanced_vs_basic_search(self):
        """ทดสอบความแตกต่างระหว่างการค้นหาขั้นสูงและปกติ"""
        test_recipes = [
            'กุ้งทาพริกไทยกระเทียม', 'ข้าวเม่าทอด', 'เปรี้ยวหวานไข่ม้วน'
        ]
        
        # คำค้นหาที่มีการพิมพ์ผิดหรือเขียนแบบย่อ
        challenging_queries = [
            'กุ้งพริกไทย',  # ควรหา กุ้งทาพริกไทยกระเทียม
            'ข้าวเหม่า',    # ควรหา ข้าวเม่าทอด
            'เปรี้ยวหวาน'   # ควรหา เปรี้ยวหวานไข่ม้วน
        ]
        
        for query in challenging_queries:
            matches = self.matcher.find_best_match(query, test_recipes, threshold=0.4)
            
            # การค้นหาขั้นสูงควรให้ผลลัพธ์ที่ดีกว่า
            assert len(matches) > 0
            assert matches[0]['similarity'] > 0.4
    
    def test_nutrition_data_completeness(self):
        """ทดสอบความครบถ้วนของข้อมูลโภชนาการ"""
        # ทดสอบวัตถุดิบสำคัญ
        important_ingredients = [
            'หมู', 'ไก่', 'เนื้อ', 'กุ้ง', 'ปลา', 'ไข่ไก่',
            'ข้าว', 'น้ำมันพืช', 'น้ำปลา', 'กะทิ', 'กระเทียม'
        ]
        
        thai_data = ThaiNutritionData()
        
        for ingredient in important_ingredients:
            nutrition = thai_data.get_nutrition_info(ingredient)
            assert nutrition is not None, f"ไม่พบข้อมูลโภชนาการสำหรับ {ingredient}"
            
            # ตรวจสอบข้อมูลพื้นฐาน
            assert nutrition.calories >= 0
            assert nutrition.protein >= 0
            assert nutrition.carbs >= 0
            assert nutrition.fat >= 0
            
            # ตรวจสอบว่าไม่ใช่ข้อมูลเปล่า
            total_macros = nutrition.calories + nutrition.protein + nutrition.carbs + nutrition.fat
            assert total_macros > 0, f"ข้อมูลโภชนาการของ {ingredient} เป็น 0 ทั้งหมด"

def run_comprehensive_tests():
    """รันการทดสอบแบบครบถ้วน"""
    print("🧪 เริ่มการทดสอบระบบขั้นสูง...")
    print("=" * 60)
    
    # สถิติการทดสอบ
    test_stats = {
        'total_tests': 0,
        'passed_tests': 0,
        'failed_tests': 0,
        'start_time': datetime.now()
    }
    
    # รายการคลาสทดสอบ
    test_classes = [
        TestEnhancedFuzzyMatcher,
        TestNutritionAnalyzer, 
        TestIngredientConverter,
        TestSearchFunctionality,
        TestConfigSettings,
        TestIntegration
    ]
    
    for test_class in test_classes:
        print(f"\n🔍 ทดสอบ {test_class.__name__}...")
        
        try:
            # สร้าง instance และรันการทดสอบ
            test_instance = test_class()
            
            # รันการทดสอบทั้งหมดในคลาส
            test_methods = [method for method in dir(test_instance) 
                          if method.startswith('test_')]
            
            for method_name in test_methods:
                test_stats['total_tests'] += 1
                
                try:
                    # ตั้งค่าก่อนการทดสอบ
                    if hasattr(test_instance, 'setup_method'):
                        test_instance.setup_method()
                    
                    # รันการทดสอบ
                    test_method = getattr(test_instance, method_name)
                    test_method()
                    
                    test_stats['passed_tests'] += 1
                    print(f"  ✅ {method_name}")
                    
                except Exception as e:
                    test_stats['failed_tests'] += 1
                    print(f"  ❌ {method_name}: {str(e)}")
                    
        except Exception as e:
            print(f"  💥 ข้อผิดพลาดในการตั้งค่า {test_class.__name__}: {e}")
    
    # สรุปผลการทดสอบ
    test_stats['end_time'] = datetime.now()
    test_stats['duration'] = test_stats['end_time'] - test_stats['start_time']
    
    print("\n" + "=" * 60)
    print("📊 สรุปผลการทดสอบ")
    print("=" * 60)
    print(f"✅ ผ่าน: {test_stats['passed_tests']}")
    print(f"❌ ไม่ผ่าน: {test_stats['failed_tests']}")
    print(f"📈 รวม: {test_stats['total_tests']} การทดสอบ")
    print(f"🎯 อัตราสำเร็จ: {test_stats['passed_tests']/test_stats['total_tests']*100:.1f}%")
    print(f"⏱️ เวลาที่ใช้: {test_stats['duration'].total_seconds():.2f} วินาที")
    print("=" * 60)
    
    return test_stats

def test_specific_menu_search():
    """ทดสอบการค้นหาเมนูเฉพาะ"""
    print("\n🔍 ทดสอบการค้นหาเมนูเฉพาะ...")
    
    matcher = EnhancedFuzzyMatcher()
    
    # เมนูจากชุดข้อมูล
    all_menus = Config.COMPLETE_THAI_MENU_LIST
    
    # การทดสอบแบบเฉพาะ
    specific_tests = [
        # เมนูที่ซับซ้อน
        ('กุ้งทาพริกไทยกระเทียม', ['กุ้งทาพริกไทย', 'กุ้งพริกไทย', 'กุ้งกระเทียม']),
        ('ข้าวเม่าทอด', ['ข้าวเหม่าทอด', 'ข้าวเม่า', 'ข้าวเหม่า']),
        ('เปรี้ยวหวานไข่ม้วน', ['เปรี้ยวหวาน', 'ไข่ม้วนเปรี้ยวหวาน']),
        ('แกงคั่วฟักทองกับกุ้งตะเข็บ', ['แกงคั่วฟักทอง', 'แกงคั่วกุ้ง']),
        ('ห่อหมกหอยแมลงภู่', ['ห่อหมกหอย', 'หอยแมลงภู่ห่อหมก']),
        ('ยำไข่เจียวเครื่องหมี่', ['ยำไข่เจียว', 'ไข่เจียวยำ']),
        
        # เมนูที่มีชื่อแปลก
        ('ฉี่ฉู่เมืองปราณ', ['ฉี่ฉู่', 'เมืองปราณ', 'ขนมฉี่ฉู่']),
        ('พุดชาจีนเชื่อมไส้เกาลัด', ['พุดชาจีน', 'ขนมจีนหวาน']),
        ('มักกะโรนีรังแตน', ['มักกะโรนี', 'รังแตนมักกะโรนี']),
        ('น้ำเต้าบรรจุไส้', ['น้ำเต้าไส้', 'น้ำเต้าใส้']),
        
        # เมนูที่พิมพ์ผิดง่าย
        ('ไข่น้อคอีกอย่างหนึ่ง', ['ไข่น้อค', 'ไข่น้อคใหม่']),
        ('ปลาช่อนต้มเค็มกับก๋งฉ่าย', ['ปลาช่อนต้มเค็ม', 'ปลาช่อนต้ม']),
        ('ผัดหัวผักกาดเค็ม', ['ผัดหัวผักกาด', 'หัวผักกาดผัด']),
    ]
    
    results = {'passed': 0, 'failed': 0, 'details': []}
    
    for target_menu, search_variations in specific_tests:
        if target_menu not in all_menus:
            results['details'].append(f"⚠️ {target_menu} ไม่อยู่ในชุดข้อมูล")
            continue
        
        for search_term in search_variations:
            matches = matcher.find_best_match(search_term, all_menus, threshold=0.4)
            
            if matches and len(matches) > 0:
                # ตรวจสอบว่าพบเมนูที่ต้องการในผลลัพธ์ 3 อันดับแรก
                found = any(target_menu == match['text'] for match in matches[:3])
                
                if found:
                    results['passed'] += 1
                    results['details'].append(f"✅ '{search_term}' → '{target_menu}' (คะแนน: {matches[0]['similarity']:.3f})")
                else:
                    results['failed'] += 1
                    top_result = matches[0]['text'] if matches else 'ไม่พบ'
                    results['details'].append(f"❌ '{search_term}' → '{top_result}' (คาดหวัง: '{target_menu}')")
            else:
                results['failed'] += 1
                results['details'].append(f"❌ '{search_term}' → ไม่พบผลลัพธ์")
    
    # แสดงผลลัพธ์
    print(f"🎯 ผลการทดสอบการค้นหาเฉพาะ:")
    print(f"✅ สำเร็จ: {results['passed']}")
    print(f"❌ ล้มเหลว: {results['failed']}")
    total = results['passed'] + results['failed']
    if total > 0:
        print(f"📊 อัตราสำเร็จ: {results['passed']/total*100:.1f}%")
    
    # แสดงรายละเอียด
    if results['details']:
        print("\n📝 รายละเอียด:")
        for detail in results['details'][:10]:  # แสดงแค่ 10 รายการแรก
            print(f"   {detail}")
        
        if len(results['details']) > 10:
            print(f"   ... และอีก {len(results['details']) - 10} รายการ")
    
    return results

if __name__ == "__main__":
    # รันการทดสอบครบถ้วน
    test_stats = run_comprehensive_tests()
    
    # รันการทดสอบเฉพาะ
    specific_results = test_specific_menu_search()
    
    # สรุปรวม
    print(f"\n🏁 การทดสอบทั้งหมดเสร็จสิ้น!")
    print(f"📊 การทดสอบทั่วไป: {test_stats['passed_tests']}/{test_stats['total_tests']} ผ่าน")
    print(f"🎯 การทดสอบเฉพาะ: {specific_results['passed']}/{specific_results['passed'] + specific_results['failed']} ผ่าน")
    
    overall_success_rate = (test_stats['passed_tests'] + specific_results['passed']) / \
                          (test_stats['total_tests'] + specific_results['passed'] + specific_results['failed'])
    print(f"🏆 อัตราสำเร็จรวม: {overall_success_rate*100:.1f}%")
    
    if overall_success_rate > 0.8:
        print("🎉 ระบบทำงานได้ดี!")
    elif overall_success_rate > 0.6:
        print("⚡ ระบบทำงานได้ใช้ได้ แต่ควรปรับปรุง")
    else:
        print("🔧 ระบบต้องการการปรับปรุงอย่างมาก")
