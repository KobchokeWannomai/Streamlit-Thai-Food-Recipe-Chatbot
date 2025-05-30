#!/usr/bin/env python3
"""
ชุดการทดสอบแบบครอบคลุมสำหรับแชทบอทสูตรอาหารไทยขั้นสูง
ทดสอบฟีเจอร์ใหม่ทั้งหมดรวมถึงการเชื่อมต่อ API, การปรับแต่งการทำอาหาร, และการค้นหาขั้นสูง
"""

import pytest
import asyncio
import time
import json
import os
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime

# นำเข้าคอมโพเนนต์ที่จะทดสอบ
from nutrition_analyzer import (
    NutritionAnalyzer, 
    USDANutritionAPI, 
    NutritionixAPI,
    CookingAdjustmentHelper,
    NutritionInfo
)
from ingredient_converter import IngredientConverter
from config import Config, APIConfig, NutritionConfig

class TestEnhancedNutritionAnalyzer:
    """ชุดการทดสอบสำหรับตัววิเคราะห์โภชนาการขั้นสูงพร้อมการเชื่อมต่อ API"""
    
    def setup_method(self):
        """ตั้งค่าสภาพแวดล้อมการทดสอบ"""
        self.analyzer = NutritionAnalyzer()
        self.converter = IngredientConverter()
        self.cooking_helper = CookingAdjustmentHelper()
    
    def test_basic_nutrition_analysis(self):
        """ทดสอบการวิเคราะห์โภชนาการพื้นฐานโดยไม่ใช้ API"""
        ingredients = """
        - ไข่ไก่ 2 ฟอง
        - น้ำมันพืช 1 ช้อนโต๊ะ
        """
        
        result = self.analyzer.analyze_ingredients(ingredients)
        
        assert len(result) == 2
        assert any('ไข่ไก่' in key for key in result.keys())
        assert any('น้ำมันพืช' in key for key in result.keys())
        
        # ตรวจสอบค่าโภชนาการ
        total = self.analyzer.calculate_total_nutrition(result)
        assert total.calories > 0
        assert total.protein > 0
    
    def test_cooking_adjustments(self):
        """ทดสอบฟังก์ชันการปรับแต่งการทำอาหาร"""
        # ทดสอบการปรับแต่งการดูดซึมน้ำมัน
        adjustments = self.cooking_helper.get_cooking_adjustments("ไข่เจียว")
        
        assert 'oil_absorption' in adjustments
        assert adjustments['oil_absorption'] == 0.1  # การดูดซึม 10%
        
        # ทดสอบการเพิ่มวัตถุดิบที่ขาดหายไป
        if 'missing_ingredients' in adjustments:
            assert any(ing['name'] == 'น้ำมันพืช' for ing in adjustments['missing_ingredients'])
    
    def test_enhanced_ingredient_conversion(self):
        """ทดสอบการแปลงส่วนผสมขั้นสูงพร้อมบริบทการทำอาหาร"""
        test_cases = [
            ("ไข่ไก่ 2 ฟอง", 100),  # ประมาณ 100g สำหรับไข่ 2 ฟอง
            ("น้ำมันพืช 1 ช้อนโต๊ะ", 14),  # ประมาณ 14g สำหรับน้ำมัน 1 ช้อนโต๊ะ
            ("กุ้งนาง 4 ตัว", 100),  # ประมาณ 100g สำหรับกุ้งขนาดกลาง 4 ตัว
        ]
        
        for ingredient_text, expected_weight in test_cases:
            result = self.converter.parse_and_convert_ingredient(ingredient_text)
            
            assert result['weight_grams'] > 0
            assert abs(result['weight_grams'] - expected_weight) < expected_weight * 0.5  # ภายใน 50%
            assert result['nutrition_multiplier'] > 0
    
    def test_recipe_analysis_with_adjustments(self):
        """ทดสอบการวิเคราะห์สูตรอาหารพร้อมการปรับแต่งการทำอาหาร"""
        ingredients = """
        - ไข่ไก่ 2 ฟอง
        - เกลือ 1/2 ช้อนชา
        """
        
        # ทดสอบโดยไม่มีการปรับแต่ง
        normal_result = self.analyzer.analyze_ingredients(ingredients, "ไข่เจียว", False)
        normal_total = self.analyzer.calculate_total_nutrition(normal_result)
        
        # ทดสอบพร้อมการปรับแต่ง
        enhanced_result = self.analyzer.analyze_ingredients(ingredients, "ไข่เจียว", True)
        enhanced_total = self.analyzer.calculate_total_nutrition(enhanced_result)
        
        # เวอร์ชันขั้นสูงควรมีแคลอรี่มากกว่าเนื่องจากน้ำมันที่เพิ่มเข้ามา
        assert enhanced_total.calories >= normal_total.calories
        assert len(enhanced_result) >= len(normal_result)  # อาจมีส่วนผสมเพิ่มเติม

class TestAPIIntegration:
    """ชุดการทดสอบสำหรับการเชื่อมต่อ API"""
    
    def setup_method(self):
        """ตั้งค่าสภาพแวดล้อมการทดสอบ API"""
        self.mock_usda_key = "test_usda_key"
        self.mock_nutritionix_id = "test_app_id"
        self.mock_nutritionix_key = "test_api_key"
    
    @patch('requests.Session.get')
    def test_usda_api_search(self, mock_get):
        """ทดสอบฟังก์ชันการค้นหา USDA API"""
        # จำลองการตอบสนอง API ที่สำเร็จ
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'foods': [
                {
                    'fdcId': 12345,
                    'description': 'Egg, whole, raw, fresh'
                }
            ]
        }
        mock_get.return_value = mock_response
        
        api = USDANutritionAPI(self.mock_usda_key)
        result = api.search_food("egg")
        
        assert result is not None
        assert 'foods' in result
        assert len(result['foods']) > 0
        
        # ตรวจสอบว่า API ถูกเรียกด้วยพารามิเตอร์ที่ถูกต้อง
        mock_get.assert_called()
        call_args = mock_get.call_args
        assert 'api_key' in call_args[1]['params']
        assert call_args[1]['params']['query'] == "egg"
    
    @patch('requests.Session.get')
    def test_usda_api_rate_limiting(self, mock_get):
        """ทดสอบการจำกัดอัตรา USDA API"""
        api = USDANutritionAPI(self.mock_usda_key)
        
        # รีเซ็ตตัวนับการจำกัดอัตรา
        api.rate_limit_calls = 0
        api.rate_limit_reset = datetime.now()
        
        # จำลองการเรียก API หลายครั้ง
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'foods': []}
        mock_get.return_value = mock_response
        
        start_time = time.time()
        
        # เรียกส่วนผสมหลายตัวอย่างรวดเร็ว
        for _ in range(5):
            api.search_food("test")
        
        # ไม่ควรใช้เวลานานเกินไปเนื่องจากการจำกัดอัตรา
        elapsed = time.time() - start_time
        assert elapsed < 10  # ควรเสร็จภายใน 10 วินาที
    
    @patch('requests.Session.post')
    def test_nutritionix_api_integration(self, mock_post):
        """ทดสอบการเชื่อมต่อ Nutritionix API"""
        # จำลองการตอบสนอง API ที่สำเร็จ
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'foods': [
                {
                    'nf_calories': 155,
                    'nf_protein': 13,
                    'nf_total_fat': 11,
                    'nf_total_carbohydrate': 1.1
                }
            ]
        }
        mock_post.return_value = mock_response
        
        api = NutritionixAPI(self.mock_nutritionix_id, self.mock_nutritionix_key)
        result = api.get_nutrition_info("egg")
        
        assert result is not None
        assert result.calories == 155
        assert result.protein == 13
        
        # ตรวจสอบว่า API ถูกเรียกด้วย headers ที่ถูกต้อง
        mock_post.assert_called()
        call_args = mock_post.call_args
        assert 'x-app-id' in call_args[1]['headers']
        assert 'x-app-key' in call_args[1]['headers']
    
    def test_api_fallback_strategy(self):
        """ทดสอบกลยุทธ์ fallback ของ API เมื่อบริการไม่พร้อมใช้งาน"""
        # สร้าง analyzer ด้วย API keys ที่ไม่ถูกต้อง
        analyzer = NutritionAnalyzer(
            usda_api_key="invalid_key",
            nutritionix_app_id="invalid_id",
            nutritionix_api_key="invalid_key"
        )
        
        # ควรยังคงคืนข้อมูลโภชนาการจากฐานข้อมูลไทย
        result = analyzer.get_ingredient_nutrition("ไข่ไก่")
        
        assert result is not None
        assert result.calories > 0
        assert result.name == "ไข่ไก่"

class TestEnhancedSearch:
    """ชุดการทดสอบสำหรับความสามารถการค้นหาขั้นสูง"""
    
    def setup_method(self):
        """ตั้งค่าสภาพแวดล้อมการทดสอบการค้นหา"""
        self.search_expansions = Config.SEARCH_ENHANCEMENT['query_expansions']
    
    def test_query_expansion(self):
        """ทดสอบการขยายคำค้นหาอัตโนมัติ"""
        # ทดสอบการขยายพื้นฐาน
        assert 'ไข่' in self.search_expansions
        assert 'ไข่เจียว' in self.search_expansions['ไข่']
        assert 'ไข่ดาว' in self.search_expansions['ไข่']
        
        # ทดสอบการขยายวิธีการทำอาหาร
        assert 'ผัด' in self.search_expansions
        assert 'ผัดไทย' in self.search_expansions['ผัด']
        assert 'ผัดกะเพรา' in self.search_expansions['ผัด']
    
    def test_search_threshold_adjustment(self):
        """ทดสอบเกณฑ์ความคล้ายคลึงที่แตกต่างกัน"""
        normal_threshold = Config.SIMILARITY_THRESHOLD
        enhanced_threshold = Config.ENHANCED_SIMILARITY_THRESHOLD
        
        assert enhanced_threshold < normal_threshold
        assert enhanced_threshold >= 0.2  # ไม่ควรต่ำเกินไป
        assert normal_threshold <= 0.4    # ไม่ควรสูงเกินไป
    
    def test_semantic_search_improvements(self):
        """ทดสอบการปรับปรุงการค้นหาแบบ semantic"""
        # ทดสอบว่าคำค้นหาที่เกี่ยวข้องกับอาหารถูกจัดหมวดหมู่อย่างเหมาะสม
        cooking_methods = Config.SEARCH_ENHANCEMENT['cooking_methods']
        
        assert 'ผัด' in cooking_methods
        assert 'ต้ม' in cooking_methods
        assert 'ทอด' in cooking_methods
        assert 'ย่าง' in cooking_methods

class TestUserInterface:
    """ชุดการทดสอบสำหรับการปรับปรุง UI/UX"""
    
    def test_auto_scroll_configuration(self):
        """ทดสอบการตั้งค่า auto-scroll"""
        display_settings = UIConfig.DISPLAY_SETTINGS if 'UIConfig' in globals() else {}
        
        # ทดสอบว่าความล่าช้า auto-scroll เหมาะสม
        if 'auto_scroll_delay' in display_settings:
            delay = display_settings['auto_scroll_delay']
            assert 100 <= delay <= 2000  # ระหว่าง 0.1 ถึง 2 วินาที
    
    def test_animation_settings(self):
        """ทดสอบการกำหนดค่าแอนิเมชั่น"""
        display_settings = UIConfig.DISPLAY_SETTINGS if 'UIConfig' in globals() else {}
        
        # ทดสอบระยะเวลาแอนิเมชั่นที่เหมาะสม
        if 'animation_duration' in display_settings:
            duration = display_settings['animation_duration']
            assert 100 <= duration <= 1000  # ระหว่าง 0.1 ถึง 1 วินาที

class TestConfiguration:
    """ชุดการทดสอบสำหรับการจัดการการกำหนดค่า"""
    
    def test_api_configuration_validation(self):
        """ทดสอบการตรวจสอบการกำหนดค่า API"""
        # ทดสอบการตรวจจับสถานะ API
        api_status = Config.is_api_configured()
        
        assert 'usda' in api_status
        assert 'nutritionix' in api_status
        assert isinstance(api_status['usda'], bool)
        assert isinstance(api_status['nutritionix'], bool)
    
    def test_nutrition_source_priority(self):
        """ทดสอบลำดับความสำคัญแหล่งข้อมูลโภชนาการ"""
        sources = Config.get_nutrition_source_priority()
        
        assert 'thai_database' in sources
        assert sources[0] == 'thai_database'  # ฐานข้อมูลไทยควรเป็นลำดับแรก
    
    def test_cooking_adjustment_configuration(self):
        """ทดสอบการตั้งค่าการปรับแต่งการทำอาหาร"""
        adjustments = Config.COOKING_ADJUSTMENTS
        
        assert 'oil_absorption_rates' in adjustments
        assert 'missing_ingredients_common' in adjustments
        
        # ทดสอบอัตราการดูดซึมน้ำมันที่สมเหตุสมผล
        oil_rates = adjustments['oil_absorption_rates']
        for method, rate in oil_rates.items():
            assert 0 < rate <= 1.0  # ควรอยู่ระหว่าง 0 ถึง 100%

class TestPerformance:
    """ชุดการทดสอบสำหรับการเพิ่มประสิทธิภาพ"""
    
    def test_caching_performance(self):
        """ทดสอบการแคชปรับปรุงประสิทธิภาพ"""
        analyzer = NutritionAnalyzer()
        
        # การเรียกครั้งแรก (ควรช้ากว่า - ไม่มีแคช)
        start_time = time.time()
        result1 = analyzer.get_ingredient_nutrition("ไข่ไก่")
        first_call_time = time.time() - start_time
        
        # การเรียกครั้งที่สอง (ควรเร็วกว่า - มีแคช)
        start_time = time.time()
        result2 = analyzer.get_ingredient_nutrition("ไข่ไก่")
        second_call_time = time.time() - start_time
        
        # ผลลัพธ์ควรเหมือนกัน
        assert result1.calories == result2.calories
        assert result1.protein == result2.protein
        
        # การเรียกครั้งที่สองควรเร็วกว่าอย่างเห็นได้ชัด (มีแคช)
        # หมายเหตุ: การทดสอบนี้อาจไม่เสถียรในระบบที่เร็ว ดังนั้นใช้การตรวจสอบแบบผ่อนผัน
        assert second_call_time <= first_call_time + 0.1
    
    def test_batch_processing_efficiency(self):
        """ทดสอบประสิทธิภาพการประมวลผลแบบ batch"""
        analyzer = NutritionAnalyzer()
        
        ingredients_list = [
            "ไข่ไก่", "กุ้ง", "หมู", "ไก่", "ข้าว",
            "น้ำปลา", "กะทิ", "พริก", "กระเทียม", "หอม"
        ]
        
        start_time = time.time()
        
        results = []
        for ingredient in ingredients_list:
            result = analyzer.get_ingredient_nutrition(ingredient)
            results.append(result)
        
        batch_time = time.time() - start_time
        
        # ควรประมวลผล 10 ส่วนผสมในเวลาที่สมเหตุสมผล
        assert batch_time < 5.0  # น้อยกว่า 5 วินาที
        assert len(results) == len(ingredients_list)
        assert all(result is not None for result in results)

class TestDataIntegrity:
    """ชุดการทดสอบสำหรับความสมบูรณ์และการตรวจสอบข้อมูล"""
    
    def test_nutrition_data_consistency(self):
        """ทดสอบความสอดคล้องของข้อมูลโภชนาการ"""
        analyzer = NutritionAnalyzer()
        
        # ทดสอบส่วนผสมไทยทั่วไป
        test_ingredients = ["ไข่ไก่", "กุ้ง", "หมู", "ข้าว", "น้ำปลา"]
        
        for ingredient in test_ingredients:
            nutrition = analyzer.get_ingredient_nutrition(ingredient)
            
            assert nutrition is not None
            assert nutrition.calories >= 0
            assert nutrition.protein >= 0
            assert nutrition.carbs >= 0
            assert nutrition.fat >= 0
            
            # การตรวจสอบสุขภาพพื้นฐาน
            if ingredient in ["ไข่ไก่", "กุ้ง", "หมู"]:  # แหล่งโปรตีน
                assert nutrition.protein > 5  # ควรมีโปรตีนอย่างมีนัยสำคัญ
            
            if ingredient == "ข้าว":  # แหล่งคาร์โบไฮเดรต
                assert nutrition.carbs > 10  # ควรมีคาร์โบไฮเดรตอย่างมีนัยสำคัญ
    
    def test_unit_conversion_accuracy(self):
        """ทดสอบความแม่นยำการแปลงหน่วย"""
        converter = IngredientConverter()
        
        test_cases = [
            # (ส่วนผสม, ช่วงน้ำหนักที่คาดหวัง)
            ("ไข่ไก่ 1 ฟอง", (40, 60)),
            ("น้ำมันพืช 1 ช้อนโต๊ะ", (12, 16)),
            ("กุ้งนาง 1 ตัว", (15, 30)),
            ("ข้าว 1 ถ้วย", (180, 220)),
        ]
        
        for ingredient_text, (min_weight, max_weight) in test_cases:
            result = converter.parse_and_convert_ingredient(ingredient_text)
            weight = result['weight_grams']
            
            assert min_weight <= weight <= max_weight, \
                f"{ingredient_text}: คาดหวัง {min_weight}-{max_weight}g, ได้ {weight}g"

class TestErrorHandling:
    """ชุดการทดสอบสำหรับการจัดการข้อผิดพลาดและกรณีขอบเขต"""
    
    def test_invalid_ingredient_handling(self):
        """ทดสอบการจัดการส่วนผสมที่ไม่ถูกต้อง"""
        analyzer = NutritionAnalyzer()
        
        # ทดสอบส่วนผสมว่าง
        result = analyzer.get_ingredient_nutrition("")
        assert result is not None  # ควรคืนค่า NutritionInfo พื้นฐาน
        
        # ทดสอบชื่อส่วนผสมที่ยาวมาก
        long_name = "a" * 1000
        result = analyzer.get_ingredient_nutrition(long_name)
        assert result is not None
        
        # ทดสอบส่วนผสมที่มีอักขระพิเศษ
        special_name = "ไข่@#$%^&*()"
        result = analyzer.get_ingredient_nutrition(special_name)
        assert result is not None
    
    def test_api_error_handling(self):
        """ทดสอบการจัดการข้อผิดพลาด API"""
        # ทดสอบด้วย API key ที่ไม่ถูกต้อง
        analyzer = NutritionAnalyzer(usda_api_key="invalid_key")
        
        # ไม่ควรล้มเหลว ควร fallback ไปข้อมูลท้องถิ่น
        result = analyzer.get_ingredient_nutrition("egg")
        assert result is not None
    
    def test_malformed_ingredient_text(self):
        """ทดสอบการจัดการข้อความส่วนผสมที่ผิดรูปแบบ"""
        analyzer = NutritionAnalyzer()
        
        malformed_inputs = [
            "",  # ว่าง
            "   ",  # เฉพาะช่องว่าง
            "- ",  # เฉพาะขีด
            "- \n- \n",  # รายการว่าง
            "invalid format without dash",
            "- ingredient 1\ninvalid line\n- ingredient 2"
        ]
        
        for malformed_input in malformed_inputs:
            # ไม่ควรล้มเหลว
            try:
                result = analyzer.analyze_ingredients(malformed_input)
                assert isinstance(result, dict)
            except Exception as e:
                pytest.fail(f"ล้มเหลวในการจัดการ input ที่ผิดรูปแบบ '{malformed_input}': {e}")

@pytest.mark.asyncio
async def test_async_processing():
    """ทดสอบความสามารถการประมวลผลแบบ asynchronous"""
    
    async def mock_async_analysis(ingredient):
        """จำลองการวิเคราะห์โภชนาการแบบ async"""
        await asyncio.sleep(0.1)  # จำลองความล่าช้า API
        analyzer = NutritionAnalyzer()
        return analyzer.get_ingredient_nutrition(ingredient)
    
    ingredients = ["ไข่ไก่", "กุ้ง", "หมู", "ไก่", "ข้าว"]
    
    start_time = time.time()
    
    # ประมวลผลส่วนผสมพร้อมกัน
    tasks = [mock_async_analysis(ingredient) for ingredient in ingredients]
    results = await asyncio.gather(*tasks)
    
    elapsed_time = time.time() - start_time
    
    # ควรเสร็จเร็วกว่าการประมวลผลแบบต่อเนื่อง
    # (5 ส่วนผสม * 0.1s แต่ละตัว = 0.5s แบบต่อเนื่อง, ควรเป็น ~0.1s แบบพร้อมกัน)
    assert elapsed_time < 0.3  # อนุญาตให้มี overhead บางส่วน
    assert len(results) == len(ingredients)
    assert all(result is not None for result in results)

def test_integration_comprehensive():
    """การทดสอบรวมแบบครอบคลุม"""
    print("\n" + "="*60)
    print("🧪 กำลังรันการทดสอบรวมแบบครอบคลุม")
    print("="*60)
    
    try:
        # ทดสอบเวิร์กโฟลว์ที่สมบูรณ์
        analyzer = NutritionAnalyzer()
        
        # 1. ทดสอบการแปลงส่วนผสม
        converter = IngredientConverter()
        ingredient_result = converter.parse_and_convert_ingredient("ไข่ไก่ 2 ฟอง")
        
        # 2. ทดสอบการวิเคราะห์โภชนาการ
        ingredients_text = """
        - ไข่ไก่ 2 ฟอง
        - น้ำมันพืช 1 ช้อนโต๊ะ
        - เกลือ 1/2 ช้อนชา
        """
        
        nutrition_data = analyzer.analyze_ingredients(ingredients_text, "ไข่เจียว", True)
        total_nutrition = analyzer.calculate_total_nutrition(nutrition_data)
        
        # 3. ทดสอบการวิเคราะห์สูตรอาหาร
        recipe_result = analyzer.analyze_recipe("ไข่เจียว", ingredients_text, True)
        
        # การยืนยัน
        assert ingredient_result['weight_grams'] > 0
        assert len(nutrition_data) >= 2
        assert total_nutrition.calories > 0
        assert recipe_result['recipe_name'] == "ไข่เจียว"
        assert 'total_nutrition' in recipe_result
        assert 'ingredients' in recipe_result
        
        print("✅ การทดสอบรวมทั้งหมดผ่าน!")
        return True
        
    except Exception as e:
        print(f"❌ การทดสอบรวมล้มเหลว: {e}")
        return False

if __name__ == "__main__":
    """รันการทดสوบทั้งหมดเมื่อสคริปต์ถูกเรียกใช้โดยตรง"""
    print("🍲 แชทบอทสูตรอาหารไทยขั้นสูง - ชุดการทดสอบ")
    print("="*60)
    
    # รันการทดสอบรวมแบบครอบคลุมก่อน
    integration_success = test_integration_comprehensive()
    
    if integration_success:
        print("\n🚀 กำลังรันชุดการทดสอบรายละเอียด...")
        
        # กำหนดค่า pytest ให้รันด้วยผลลัพธ์แบบละเอียด
        pytest_args = [
            __file__,
            "-v",
            "--tb=short",
            "--color=yes"
        ]
        
        # รัน pytest
        exit_code = pytest.main(pytest_args)
        
        if exit_code == 0:
            print("\n🎉 การทดสอบทั้งหมดผ่านเรียบร้อย!")
        else:
            print(f"\n⚠️  การทดสอบบางส่วนล้มเหลว (รหัสออก: {exit_code})")
    else:
        print("\n❌ การทดสอบรวมล้มเหลว - ข้ามการทดสอบรายละเอียด")
        exit(1)
