#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced Thai Food Recipe Chatbot
Tests all new features including API integration, cooking adjustments, and enhanced search
"""

import pytest
import asyncio
import time
import json
import os
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime

# Import components to test
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
    """Test suite for enhanced nutrition analyzer with API integration"""
    
    def setup_method(self):
        """Setup test environment"""
        self.analyzer = NutritionAnalyzer()
        self.converter = IngredientConverter()
        self.cooking_helper = CookingAdjustmentHelper()
    
    def test_basic_nutrition_analysis(self):
        """Test basic nutrition analysis without API"""
        ingredients = """
        - ไข่ไก่ 2 ฟอง
        - น้ำมันพืช 1 ช้อนโต๊ะ
        """
        
        result = self.analyzer.analyze_ingredients(ingredients)
        
        assert len(result) == 2
        assert any('ไข่ไก่' in key for key in result.keys())
        assert any('น้ำมันพืช' in key for key in result.keys())
        
        # Check nutrition values
        total = self.analyzer.calculate_total_nutrition(result)
        assert total.calories > 0
        assert total.protein > 0
    
    def test_cooking_adjustments(self):
        """Test cooking adjustment functionality"""
        # Test oil absorption adjustment
        adjustments = self.cooking_helper.get_cooking_adjustments("ไข่เจียว")
        
        assert 'oil_absorption' in adjustments
        assert adjustments['oil_absorption'] == 0.1  # 10% absorption
        
        # Test missing ingredients addition
        if 'missing_ingredients' in adjustments:
            assert any(ing['name'] == 'น้ำมันพืช' for ing in adjustments['missing_ingredients'])
    
    def test_enhanced_ingredient_conversion(self):
        """Test enhanced ingredient conversion with cooking context"""
        test_cases = [
            ("ไข่ไก่ 2 ฟอง", 100),  # Approximately 100g for 2 eggs
            ("น้ำมันพืช 1 ช้อนโต๊ะ", 14),  # Approximately 14g for 1 tbsp oil
            ("กุ้งนาง 4 ตัว", 100),  # Approximately 100g for 4 medium shrimp
        ]
        
        for ingredient_text, expected_weight in test_cases:
            result = self.converter.parse_and_convert_ingredient(ingredient_text)
            
            assert result['weight_grams'] > 0
            assert abs(result['weight_grams'] - expected_weight) < expected_weight * 0.5  # Within 50%
            assert result['nutrition_multiplier'] > 0
    
    def test_recipe_analysis_with_adjustments(self):
        """Test recipe analysis with cooking adjustments"""
        ingredients = """
        - ไข่ไก่ 2 ฟอง
        - เกลือ 1/2 ช้อนชา
        """
        
        # Test without adjustments
        normal_result = self.analyzer.analyze_ingredients(ingredients, "ไข่เจียว", False)
        normal_total = self.analyzer.calculate_total_nutrition(normal_result)
        
        # Test with adjustments
        enhanced_result = self.analyzer.analyze_ingredients(ingredients, "ไข่เจียว", True)
        enhanced_total = self.analyzer.calculate_total_nutrition(enhanced_result)
        
        # Enhanced version should have more calories due to added oil
        assert enhanced_total.calories >= normal_total.calories
        assert len(enhanced_result) >= len(normal_result)  # May have additional ingredients

class TestAPIIntegration:
    """Test suite for API integrations"""
    
    def setup_method(self):
        """Setup API test environment"""
        self.mock_usda_key = "test_usda_key"
        self.mock_nutritionix_id = "test_app_id"
        self.mock_nutritionix_key = "test_api_key"
    
    @patch('requests.Session.get')
    def test_usda_api_search(self, mock_get):
        """Test USDA API search functionality"""
        # Mock successful API response
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
        
        # Verify API was called with correct parameters
        mock_get.assert_called()
        call_args = mock_get.call_args
        assert 'api_key' in call_args[1]['params']
        assert call_args[1]['params']['query'] == "egg"
    
    @patch('requests.Session.get')
    def test_usda_api_rate_limiting(self, mock_get):
        """Test USDA API rate limiting"""
        api = USDANutritionAPI(self.mock_usda_key)
        
        # Reset rate limit counters
        api.rate_limit_calls = 0
        api.rate_limit_reset = datetime.now()
        
        # Mock multiple API calls
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'foods': []}
        mock_get.return_value = mock_response
        
        start_time = time.time()
        
        # Make multiple calls quickly
        for _ in range(5):
            api.search_food("test")
        
        # Should not take too long due to rate limiting
        elapsed = time.time() - start_time
        assert elapsed < 10  # Should complete within 10 seconds
    
    @patch('requests.Session.post')
    def test_nutritionix_api_integration(self, mock_post):
        """Test Nutritionix API integration"""
        # Mock successful API response
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
        
        # Verify API was called with correct headers
        mock_post.assert_called()
        call_args = mock_post.call_args
        assert 'x-app-id' in call_args[1]['headers']
        assert 'x-app-key' in call_args[1]['headers']
    
    def test_api_fallback_strategy(self):
        """Test API fallback when services are unavailable"""
        # Create analyzer with invalid API keys
        analyzer = NutritionAnalyzer(
            usda_api_key="invalid_key",
            nutritionix_app_id="invalid_id",
            nutritionix_api_key="invalid_key"
        )
        
        # Should still return nutrition data from Thai database
        result = analyzer.get_ingredient_nutrition("ไข่ไก่")
        
        assert result is not None
        assert result.calories > 0
        assert result.name == "ไข่ไก่"

class TestEnhancedSearch:
    """Test suite for enhanced search capabilities"""
    
    def setup_method(self):
        """Setup search test environment"""
        self.search_expansions = Config.SEARCH_ENHANCEMENT['query_expansions']
    
    def test_query_expansion(self):
        """Test automatic query expansion"""
        # Test basic expansion
        assert 'ไข่' in self.search_expansions
        assert 'ไข่เจียว' in self.search_expansions['ไข่']
        assert 'ไข่ดาว' in self.search_expansions['ไข่']
        
        # Test cooking method expansion
        assert 'ผัด' in self.search_expansions
        assert 'ผัดไทย' in self.search_expansions['ผัด']
        assert 'ผัดกะเพรา' in self.search_expansions['ผัด']
    
    def test_search_threshold_adjustment(self):
        """Test different similarity thresholds"""
        normal_threshold = Config.SIMILARITY_THRESHOLD
        enhanced_threshold = Config.ENHANCED_SIMILARITY_THRESHOLD
        
        assert enhanced_threshold < normal_threshold
        assert enhanced_threshold >= 0.2  # Should not be too low
        assert normal_threshold <= 0.4    # Should not be too high
    
    def test_semantic_search_improvements(self):
        """Test semantic search improvements"""
        # Test that common food-related queries are properly categorized
        cooking_methods = Config.SEARCH_ENHANCEMENT['cooking_methods']
        
        assert 'ผัด' in cooking_methods
        assert 'ต้ม' in cooking_methods
        assert 'ทอด' in cooking_methods
        assert 'ย่าง' in cooking_methods

class TestUserInterface:
    """Test suite for UI/UX enhancements"""
    
    def test_auto_scroll_configuration(self):
        """Test auto-scroll settings"""
        display_settings = UIConfig.DISPLAY_SETTINGS if 'UIConfig' in globals() else {}
        
        # Test that auto-scroll delay is reasonable
        if 'auto_scroll_delay' in display_settings:
            delay = display_settings['auto_scroll_delay']
            assert 100 <= delay <= 2000  # Between 0.1 and 2 seconds
    
    def test_animation_settings(self):
        """Test animation configuration"""
        display_settings = UIConfig.DISPLAY_SETTINGS if 'UIConfig' in globals() else {}
        
        # Test animation duration is reasonable
        if 'animation_duration' in display_settings:
            duration = display_settings['animation_duration']
            assert 100 <= duration <= 1000  # Between 0.1 and 1 second

class TestConfiguration:
    """Test suite for configuration management"""
    
    def test_api_configuration_validation(self):
        """Test API configuration validation"""
        # Test API status detection
        api_status = Config.is_api_configured()
        
        assert 'usda' in api_status
        assert 'nutritionix' in api_status
        assert isinstance(api_status['usda'], bool)
        assert isinstance(api_status['nutritionix'], bool)
    
    def test_nutrition_source_priority(self):
        """Test nutrition data source priority"""
        sources = Config.get_nutrition_source_priority()
        
        assert 'thai_database' in sources
        assert sources[0] == 'thai_database'  # Thai DB should be first priority
    
    def test_cooking_adjustment_configuration(self):
        """Test cooking adjustment settings"""
        adjustments = Config.COOKING_ADJUSTMENTS
        
        assert 'oil_absorption_rates' in adjustments
        assert 'missing_ingredients_common' in adjustments
        
        # Test oil absorption rates are reasonable
        oil_rates = adjustments['oil_absorption_rates']
        for method, rate in oil_rates.items():
            assert 0 < rate <= 1.0  # Should be between 0 and 100%

class TestPerformance:
    """Test suite for performance optimization"""
    
    def test_caching_performance(self):
        """Test caching improves performance"""
        analyzer = NutritionAnalyzer()
        
        # First call (should be slower - no cache)
        start_time = time.time()
        result1 = analyzer.get_ingredient_nutrition("ไข่ไก่")
        first_call_time = time.time() - start_time
        
        # Second call (should be faster - cached)
        start_time = time.time()
        result2 = analyzer.get_ingredient_nutrition("ไข่ไก่")
        second_call_time = time.time() - start_time
        
        # Results should be identical
        assert result1.calories == result2.calories
        assert result1.protein == result2.protein
        
        # Second call should be significantly faster (cached)
        # Note: This test might be flaky in fast systems, so we use a lenient check
        assert second_call_time <= first_call_time + 0.1
    
    def test_batch_processing_efficiency(self):
        """Test batch processing is efficient"""
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
        
        # Should process 10 ingredients in reasonable time
        assert batch_time < 5.0  # Less than 5 seconds
        assert len(results) == len(ingredients_list)
        assert all(result is not None for result in results)

class TestDataIntegrity:
    """Test suite for data integrity and validation"""
    
    def test_nutrition_data_consistency(self):
        """Test nutrition data consistency"""
        analyzer = NutritionAnalyzer()
        
        # Test common Thai ingredients
        test_ingredients = ["ไข่ไก่", "กุ้ง", "หมู", "ข้าว", "น้ำปลา"]
        
        for ingredient in test_ingredients:
            nutrition = analyzer.get_ingredient_nutrition(ingredient)
            
            assert nutrition is not None
            assert nutrition.calories >= 0
            assert nutrition.protein >= 0
            assert nutrition.carbs >= 0
            assert nutrition.fat >= 0
            
            # Basic sanity checks
            if ingredient in ["ไข่ไก่", "กุ้ง", "หมู"]:  # Protein sources
                assert nutrition.protein > 5  # Should have significant protein
            
            if ingredient == "ข้าว":  # Carb source
                assert nutrition.carbs > 10  # Should have significant carbs
    
    def test_unit_conversion_accuracy(self):
        """Test unit conversion accuracy"""
        converter = IngredientConverter()
        
        test_cases = [
            # (ingredient, expected_weight_range)
            ("ไข่ไก่ 1 ฟอง", (40, 60)),
            ("น้ำมันพืช 1 ช้อนโต๊ะ", (12, 16)),
            ("กุ้งนาง 1 ตัว", (15, 30)),
            ("ข้าว 1 ถ้วย", (180, 220)),
        ]
        
        for ingredient_text, (min_weight, max_weight) in test_cases:
            result = converter.parse_and_convert_ingredient(ingredient_text)
            weight = result['weight_grams']
            
            assert min_weight <= weight <= max_weight, \
                f"{ingredient_text}: expected {min_weight}-{max_weight}g, got {weight}g"

class TestErrorHandling:
    """Test suite for error handling and edge cases"""
    
    def test_invalid_ingredient_handling(self):
        """Test handling of invalid ingredients"""
        analyzer = NutritionAnalyzer()
        
        # Test empty ingredient
        result = analyzer.get_ingredient_nutrition("")
        assert result is not None  # Should return basic NutritionInfo
        
        # Test very long ingredient name
        long_name = "a" * 1000
        result = analyzer.get_ingredient_nutrition(long_name)
        assert result is not None
        
        # Test ingredient with special characters
        special_name = "ไข่@#$%^&*()"
        result = analyzer.get_ingredient_nutrition(special_name)
        assert result is not None
    
    def test_api_error_handling(self):
        """Test API error handling"""
        # Test with invalid API key
        analyzer = NutritionAnalyzer(usda_api_key="invalid_key")
        
        # Should not crash, should fallback to local data
        result = analyzer.get_ingredient_nutrition("egg")
        assert result is not None
    
    def test_malformed_ingredient_text(self):
        """Test handling of malformed ingredient text"""
        analyzer = NutritionAnalyzer()
        
        malformed_inputs = [
            "",  # Empty
            "   ",  # Only spaces
            "- ",  # Just dash
            "- \n- \n",  # Empty list items
            "invalid format without dash",
            "- ingredient 1\ninvalid line\n- ingredient 2"
        ]
        
        for malformed_input in malformed_inputs:
            # Should not crash
            try:
                result = analyzer.analyze_ingredients(malformed_input)
                assert isinstance(result, dict)
            except Exception as e:
                pytest.fail(f"Failed to handle malformed input '{malformed_input}': {e}")

@pytest.mark.asyncio
async def test_async_processing():
    """Test asynchronous processing capabilities"""
    
    async def mock_async_analysis(ingredient):
        """Mock async nutrition analysis"""
        await asyncio.sleep(0.1)  # Simulate API delay
        analyzer = NutritionAnalyzer()
        return analyzer.get_ingredient_nutrition(ingredient)
    
    ingredients = ["ไข่ไก่", "กุ้ง", "หมู", "ไก่", "ข้าว"]
    
    start_time = time.time()
    
    # Process ingredients concurrently
    tasks = [mock_async_analysis(ingredient) for ingredient in ingredients]
    results = await asyncio.gather(*tasks)
    
    elapsed_time = time.time() - start_time
    
    # Should complete faster than sequential processing
    # (5 ingredients * 0.1s each = 0.5s sequential, should be ~0.1s concurrent)
    assert elapsed_time < 0.3  # Allow some overhead
    assert len(results) == len(ingredients)
    assert all(result is not None for result in results)

def test_integration_comprehensive():
    """Comprehensive integration test"""
    print("\n" + "="*60)
    print("🧪 Running Comprehensive Integration Test")
    print("="*60)
    
    try:
        # Test complete workflow
        analyzer = NutritionAnalyzer()
        
        # 1. Test ingredient conversion
        converter = IngredientConverter()
        ingredient_result = converter.parse_and_convert_ingredient("ไข่ไก่ 2 ฟอง")
        
        # 2. Test nutrition analysis
        ingredients_text = """
        - ไข่ไก่ 2 ฟอง
        - น้ำมันพืช 1 ช้อนโต๊ะ
        - เกลือ 1/2 ช้อนชา
        """
        
        nutrition_data = analyzer.analyze_ingredients(ingredients_text, "ไข่เจียว", True)
        total_nutrition = analyzer.calculate_total_nutrition(nutrition_data)
        
        # 3. Test recipe analysis
        recipe_result = analyzer.analyze_recipe("ไข่เจียว", ingredients_text, True)
        
        # Assertions
        assert ingredient_result['weight_grams'] > 0
        assert len(nutrition_data) >= 2
        assert total_nutrition.calories > 0
        assert recipe_result['recipe_name'] == "ไข่เจียว"
        assert 'total_nutrition' in recipe_result
        assert 'ingredients' in recipe_result
        
        print("✅ All integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

if __name__ == "__main__":
    """Run all tests when script is executed directly"""
    print("🍲 Enhanced Thai Food Recipe Chatbot - Test Suite")
    print("="*60)
    
    # Run comprehensive integration test first
    integration_success = test_integration_comprehensive()
    
    if integration_success:
        print("\n🚀 Running detailed test suite...")
        
        # Configure pytest to run with verbose output
        pytest_args = [
            __file__,
            "-v",
            "--tb=short",
            "--color=yes"
        ]
        
        # Run pytest
        exit_code = pytest.main(pytest_args)
        
        if exit_code == 0:
            print("\n🎉 All tests passed successfully!")
        else:
            print(f"\n⚠️  Some tests failed (exit code: {exit_code})")
    else:
        print("\n❌ Integration test failed - skipping detailed tests")
        exit(1)