# 🔑 Complete API Setup Guide

สำหรับ Enhanced Thai Food Recipe Chatbot with Advanced Nutrition Analysis

## 📖 Table of Contents

1. [Overview](#overview)
2. [USDA FoodData Central API (FREE)](#usda-fooddata-central-api-free)
3. [Nutritionix API (Premium)](#nutritionix-api-premium)
4. [Optional APIs](#optional-apis)
5. [Setting Up in the App](#setting-up-in-the-app)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)

---

## Overview

การเชื่อมต่อ API จะช่วยให้แอปพลิเคชันมีข้อมูลโภชนาการที่แม่นยำและครบถ้วนมากขึ้น โดยสามารถทำงานได้แม้ไม่มี API keys แต่จะมีข้อจำกัดในการวิเคราะห์

### 🎯 Benefits of API Integration

| Feature | Without API | With USDA API | With Multiple APIs |
|---------|-------------|---------------|-------------------|
| Thai ingredients | ✅ Full | ✅ Full | ✅ Full |
| International foods | ⚠️ Limited | ✅ Extensive | ✅ Comprehensive |
| Branded products | ❌ None | ⚠️ Some | ✅ Many |
| Data accuracy | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Cost | FREE | FREE | Mixed |

---

## USDA FoodData Central API (FREE)

### 🌟 Why Choose USDA API?

- ✅ **Completely FREE** with no usage limits
- ✅ **Government-backed** reliable nutrition data
- ✅ **Comprehensive database** with 300,000+ foods
- ✅ **International coverage** including Asian foods
- ✅ **No credit card required**
- ✅ **Instant activation**

### 📝 Step-by-Step Setup

#### 1. Visit the USDA Website
```
🔗 https://fdc.nal.usda.gov/api-guide.html
```

#### 2. Request API Key
1. Click **"Get an API Key"** button
2. Fill out the registration form:

```
Name: [Your Full Name]
Email: [Your Email Address]
Organization: Personal Use (or your company)
Intended Use: Recipe Nutrition Analysis for Thai Food Chatbot
```

#### 3. Confirm Email
- Check your email inbox
- Click the confirmation link
- API key will be sent to your email

#### 4. Example Email Response
```
Subject: Your FoodData Central API Key

Dear [Your Name],

Your API Key: abcd1234-5678-90ef-ghij-klmnopqrstuv

API Documentation: https://fdc.nal.usda.gov/api-guide.html
Rate Limits: No limits for public use
```

#### 5. Test Your API Key
```bash
curl "https://api.nal.usda.gov/fdc/v1/foods/search?api_key=YOUR_API_KEY&query=banana&pageSize=1"
```

Expected response:
```json
{
  "totalHits": 1234,
  "currentPage": 1,
  "totalPages": 1234,
  "foods": [...]
}
```

---

## Nutritionix API (Premium)

### 🥇 Why Consider Nutritionix?

- ✅ **Natural language queries** ("1 cup cooked rice")
- ✅ **Branded food database** (restaurant chains, packaged foods)
- ✅ **Recipe analysis** with automatic ingredient recognition
- ✅ **200 free requests/day** (enough for personal use)
- ⚠️ **Paid plans** for higher usage

### 📝 Step-by-Step Setup

#### 1. Create Account
```
🔗 https://www.nutritionix.com/business/api
```

#### 2. Choose Plan
- **Free Plan**: 200 requests/day
- **Basic Plan**: $49/month for 5,000 requests/day
- **Pro Plan**: $299/month for 50,000 requests/day

#### 3. Get API Credentials
After signup, you'll receive:
```
Application ID: 12345678
Application Key: abcdef1234567890abcdef1234567890
```

#### 4. Test Your Credentials
```bash
curl -X POST \
  https://trackapi.nutritionix.com/v2/natural/nutrients \
  -H 'Content-Type: application/json' \
  -H 'x-app-id: YOUR_APP_ID' \
  -H 'x-app-key: YOUR_APP_KEY' \
  -d '{"query": "1 cup rice"}'
```

---

## Optional APIs

### 🍴 Spoonacular API

**Best for**: Recipe suggestions, meal planning, grocery lists

**Free Tier**: 150 requests/day
**Pricing**: From $7/month

**Setup**:
1. Visit: https://spoonacular.com/food-api
2. Register for free account
3. Get API key from dashboard

### 🥗 Edamam Nutrition Analysis API

**Best for**: Recipe analysis, nutrition facts labels

**Free Tier**: 1,000 requests/month
**Pricing**: From $49/month

**Setup**:
1. Visit: https://developer.edamam.com/
2. Create developer account
3. Get App ID and App Key

---

## Setting Up in the App

### 🖥️ Using the Settings Sidebar

#### 1. Start the App
```bash
streamlit run streamlit_app.py
```

#### 2. Open Settings
- Look for the **⚙️ Settings** button in the left sidebar
- Click to expand the settings panel

#### 3. API Configuration Section

**For USDA API:**
1. ✅ Check "Enable USDA API"
2. 📝 Paste your API key in the text field
3. 🔍 Click "Test Connection"
4. ✅ Look for green status indicator

**For Nutritionix API:**
1. ✅ Check "Enable Nutritionix API"
2. 📝 Enter your App ID
3. 📝 Enter your API Key
4. 🔍 Click "Test Connection"
5. ✅ Look for green status indicator

#### 4. Advanced Features
Enable these for enhanced functionality:
- ✅ **Use external recipe data** - Enhanced cooking calculations
- ✅ **Accurate cooking calculation** - Oil absorption, water loss
- ✅ **Enhanced search scope** - Better search results

### 🔧 Using Environment Variables (.env file)

#### 1. Copy Example File
```bash
cp .env.example .env
```

#### 2. Edit .env File
```bash
# API Keys
USDA_API_KEY=your_actual_usda_api_key_here
NUTRITIONIX_API_KEY=your_actual_nutritionix_api_key_here
NUTRITIONIX_APP_ID=your_actual_nutritionix_app_id_here

# Enable Features
ENABLE_API_INTEGRATION=true
ENABLE_ENHANCED_SEARCH=true
ENABLE_COOKING_ADJUSTMENTS=true
```

#### 3. Restart the App
```bash
# Stop the app (Ctrl+C)
# Start again
streamlit run streamlit_app.py
```

---

## Troubleshooting

### 🔴 Common Issues and Solutions

#### Issue: "API Connection Failed"
**Possible causes:**
- Invalid API key
- Network connection issues
- API service temporarily down
- Rate limit exceeded

**Solutions:**
1. Double-check API key spelling
2. Test with curl command
3. Check internet connection
4. Wait a few minutes and retry

#### Issue: "Rate Limit Exceeded"
**For USDA**: No rate limits, this shouldn't happen
**For Nutritionix**: You've exceeded daily limit

**Solutions:**
1. Wait until next day (for free plans)
2. Upgrade to paid plan
3. Use USDA API as primary source

#### Issue: "No Data Found"
**Possible causes:**
- API key not configured
- Ingredient not in database
- Spelling errors in ingredient names

**Solutions:**
1. Check API configuration in settings
2. Try alternative ingredient names
3. Use built-in Thai database as fallback

### 🔍 Testing API Connection

#### Manual Testing Script
Create a file `test_api.py`:
```python
import requests
import os
from dotenv import load_dotenv

load_dotenv()

def test_usda_api():
    api_key = os.getenv('USDA_API_KEY')
    if not api_key:
        print("❌ USDA API key not found")
        return
    
    url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    params = {
        'api_key': api_key,
        'query': 'banana',
        'pageSize': 1
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            print("✅ USDA API working")
        else:
            print(f"❌ USDA API error: {response.status_code}")
    except Exception as e:
        print(f"❌ USDA API connection error: {e}")

def test_nutritionix_api():
    app_id = os.getenv('NUTRITIONIX_APP_ID')
    api_key = os.getenv('NUTRITIONIX_API_KEY')
    
    if not app_id or not api_key:
        print("❌ Nutritionix credentials not found")
        return
    
    url = "https://trackapi.nutritionix.com/v2/natural/nutrients"
    headers = {
        'x-app-id': app_id,
        'x-app-key': api_key,
        'Content-Type': 'application/json'
    }
    data = {'query': '1 banana'}
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        if response.status_code == 200:
            print("✅ Nutritionix API working")
        else:
            print(f"❌ Nutritionix API error: {response.status_code}")
    except Exception as e:
        print(f"❌ Nutritionix API connection error: {e}")

if __name__ == "__main__":
    test_usda_api()
    test_nutritionix_api()
```

Run the test:
```bash
python test_api.py
```

---

## Best Practices

### 🛡️ Security

#### 1. Protect Your API Keys
```bash
# Never commit API keys to git
echo ".env" >> .gitignore

# Use environment variables in production
export USDA_API_KEY="your_key_here"
```

#### 2. Rotate Keys Regularly
- Change API keys every 3-6 months
- Use different keys for development/production
- Monitor usage for unusual activity

#### 3. Limit Access
- Use least-privilege principle
- Monitor API usage dashboards
- Set up alerts for unusual usage

### ⚡ Performance

#### 1. Caching Strategy
```python
# App automatically caches API responses
# Cache duration: 24 hours (configurable)
# Local cache: nutrition_cache.db
```

#### 2. Rate Limiting
```python
# Built-in rate limiting
# USDA: No limits (use responsibly)
# Nutritionix: Respects API limits
```

#### 3. Fallback Strategy
```
1st: Check local cache
2nd: Thai nutrition database
3rd: USDA API
4th: Nutritionix API
5th: Basic nutrition estimation
```

### 📊 Monitoring

#### 1. Usage Tracking
- Monitor API call counts
- Track response times
- Log error rates

#### 2. Health Checks
- Regular connection tests
- Automated status monitoring
- Alert notifications

#### 3. Performance Metrics
- Cache hit rates
- Average response times
- API success rates

---

## 🆘 Support and Resources

### 📚 Official Documentation
- **USDA FoodData Central**: https://fdc.nal.usda.gov/api-guide.html
- **Nutritionix**: https://developer.nutritionix.com/
- **Spoonacular**: https://spoonacular.com/food-api/docs
- **Edamam**: https://developer.edamam.com/

### 💬 Community Support
- **GitHub Issues**: Report bugs and get help
- **Stack Overflow**: Tag with `thai-food-chatbot`
- **Discord Community**: Real-time chat support

### 📧 Contact
- **Email**: support@thai-food-chatbot.com
- **GitHub**: https://github.com/your-repo/thai-food-chatbot

---

## 🎉 Quick Start Checklist

- [ ] Get USDA API key (5 minutes, FREE)
- [ ] Test API connection
- [ ] Configure in app settings
- [ ] Enable advanced features
- [ ] Try enhanced search
- [ ] Enjoy accurate nutrition data!

**Remember**: The app works great even without API keys, but APIs unlock the full potential of advanced nutrition analysis! 🚀