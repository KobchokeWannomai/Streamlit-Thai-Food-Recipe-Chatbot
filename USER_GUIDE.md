# 📖 Complete User Guide - Enhanced Features

สำหรับ Thai Food Recipe Chatbot with Advanced Nutrition Analysis

## 🎯 Table of Contents

1. [Getting Started](#getting-started)
2. [Settings Sidebar Guide](#settings-sidebar-guide)
3. [Enhanced Search Features](#enhanced-search-features)
4. [Advanced Nutrition Analysis](#advanced-nutrition-analysis)
5. [UI/UX Improvements](#uiux-improvements)
6. [Tips and Best Practices](#tips-and-best-practices)
7. [Troubleshooting](#troubleshooting)

---

## Getting Started

### 🚀 Quick Start

1. **Launch the App**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Open Settings** (New!)
   - Look for **⚙️ การตั้งค่า** in the left sidebar
   - Click to expand the settings panel

3. **Try Enhanced Search**
   - Type "ไข่" and see automatic expansion to ไข่เจียว, ไข่ดาว
   - Try "อาหารง่ายๆ" for semantic search

4. **Enable Advanced Features**
   - Set up API keys for enhanced nutrition data
   - Enable cooking adjustments for accurate calculations

---

## Settings Sidebar Guide

### 🔌 API Configuration Section

#### USDA FoodData Central API
```
✅ เปิดใช้งาน USDA API
📝 USDA API Key: [your_api_key_here]
🔍 [ทดสอบการเชื่อมต่อ USDA]
🟢 เชื่อมต่อแล้ว / 🔴 ไม่ได้เชื่อมต่อ
```

**How to use:**
1. Check "เปิดใช้งาน USDA API"
2. Paste your API key in the text field
3. Click "ทดสอบการเชื่อมต่อ USDA"
4. Look for green status indicator

#### Nutritionix API
```
✅ เปิดใช้งาน Nutritionix API
📝 Nutritionix App ID: [your_app_id]
📝 Nutritionix API Key: [your_api_key]
🔍 [ทดสอบการเชื่อมต่อ Nutritionix]
🟢 เชื่อมต่อแล้ว / 🔴 ไม่ได้เชื่อมต่อ
```

### 🧪 ตัวเลือกขั้นสูง

#### Enhanced Features Toggle
```
✅ ใช้ข้อมูลสูตรอาหารจาก API ภายนอก
   💡 ปรับปรุงการคำนวณโภชนาการด้วยข้อมูลเพิ่มเติม

✅ คำนวณการบริโภควัตถุดิบในการทำอาหารอย่างแม่นยำ
   💡 ปรับปรุงการคำนวณสำหรับวัตถุดิบที่ใช้ในการปรุงอาหาร

✅ เพิ่มขอบเขตการค้นหา
   💡 ค้นหาอาหารในขอบเขตที่กว้างขึ้น รวมถึงวัตถุดิบและคำอธิบาย
```

### 📊 สถานะระบบ

```
📊 สถานะระบบ
┌─────────────────┬──────────────┐
│ สูตรอาหาร      │ 100+         │
│ ฐานข้อมูลโภชนาการ │ 150+ วัตถุดิบ │
│ การค้นหาในเซสชันนี้ │ 25           │
└─────────────────┴──────────────┘
```

---

## Enhanced Search Features

### 🔍 Smart Query Expansion

#### Before (Traditional Search)
```
Query: "ไข่"
Results: Only recipes with exact "ไข่" match
```

#### After (Enhanced Search) ✨
```
Query: "ไข่"
Auto-expanded to: "ไข่ ไข่เจียว ไข่ดาว ไข่ต้ม ไข่ไก่ ไข่เป็ด"
Results: All egg-related recipes
```

### 🧠 Semantic Search Examples

| Your Query | Understanding | Results |
|------------|---------------|---------|
| "อาหารง่ายๆ" | Simple dishes | ไข่เจียว, ไข่ดาว, ข้าวผัด |
| "เมนูเช้า" | Breakfast food | ข้าวต้ม, โจ๊ก, ขนมปัง |
| "อาหารเผ็ด" | Spicy food | ส้มตำ, ลาบ, น้ำพริก |
| "เมนูทอด" | Fried dishes | ไข่เจียว, ปลาทอด, กุ้งทอด |

### 📝 Example Search Buttons (New!)

Click these example searches for instant results:
```
[ไข่เจียว] [ต้มยำกุ้ง] [ผัดไทย] [ส้มตำ] [แกงเขียวหวาน]
[เมนูแคลอรี่ไม่เกิน 300] [อาหารโปรตีนสูง] [เมนูลดน้ำหนัก]
[อาหารทอดง่ายๆ] [แกงเผ็ด]
```

---

## Advanced Nutrition Analysis

### 🧪 Cooking Adjustments Feature

#### Oil Absorption Calculation
```
Traditional: ไข่เจียว (ไข่ 2 ฟอง)
📊 155 kcal, 13g โปรตีน, 11g ไขมัน

Enhanced: ไข่เจียว (ไข่ 2 ฟอง + น้ำมันทอด)
📊 280 kcal, 13g โปรตีน, 25g ไขมัน
💡 +125 kcal จากน้ำมันดูดซึม (10%)
```

#### Missing Ingredients Detection
```
Recipe: ไข่ดาว
Listed: ไข่ไก่ 1 ฟอง

Enhanced Detection:
✅ ไข่ไก่ 1 ฟอง
➕ น้ำมันพืช 2 ช้ช. (สำหรับทอด) [เพิ่มเติม]
💡 บริโภคจริง: 20% = ~3 kcal
```

### 📊 Enhanced Nutrition Display

#### Basic Nutrition Panel
```
🥗 ข้อมูลโภชนาการ (ต่อหนึ่งที่)
┌─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐
│ แคลอรี่     │ โปรตีน      │ คาร์โบไฮเดรต │ ไขมัน       │ ใยอาหาร    │
│ 280 kcal    │ 13.0 g      │ 1.5 g       │ 25.0 g      │ 0.2 g       │
└─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘
```

#### Enhanced Nutrition Panel ✨
```
📈 ข้อมูลที่ปรับปรุงแล้วด้วย API ภายนอก - คำนวณปริมาณการบริโภคจริงแล้ว

🥗 ข้อมูลโภชนาการ (ต่อหนึ่งที่)
┌─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐
│ แคลอรี่     │ โปรตีน      │ คาร์โบไฮเดรต │ ไขมัน       │ ใยอาหาร    │
│ 280 kcal    │ 13.0 g      │ 1.5 g       │ 25.0 g      │ 0.2 g       │
└─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘

วิตามินและแร่ธาตุ:
🔸 วิตามิน A: 140.0 mcg  🔸 วิตามิน D: 2.0 mcg  🔸 เหล็ก: 1.8 mg
🔸 แคลเซียม: 50.0 mg     🔸 โซเดียม: 140 mg
```

### 📋 Detailed Ingredient Breakdown

#### Enhanced Ingredient Details
```
📋 รายละเอียดโภชนาการแต่ละวัตถุดิบ ▼

ไข่ไก่ (2 ฟอง)
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ แคลอรี่: 155 │ โปรตีน: 13g │ คาร์โบ: 1.1g │ ไขมัน: 11g  │
└─────────────┴─────────────┴─────────────┴─────────────┘
สารอาหารอื่นๆ: วิตามิน A: 140 mcg, วิตามิน D: 2.0 mcg, เหล็ก: 1.8 mg

น้ำมันพืช (2 ช้ช.) (ใช้ 20%) [เพิ่มเติม]
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ แคลอรี่: 125 │ โปรตีน: 0g  │ คาร์โบ: 0g   │ ไขมัน: 14g  │
└─────────────┴─────────────┴─────────────┴─────────────┘
🔧 ปรับปรุงการคำนวณปริมาณการบริโภคแล้ว
```

---

## UI/UX Improvements

### 📱 Auto-Scroll Features (New!)

#### Automatic Scrolling
- แอปจะเลื่อนไปข้อความตอบกลับล่าสุดโดยอัตโนมัติ
- ไม่ต้องเลื่อนหาข้อความใหม่เอง

#### Manual Scroll Button
```
                                    [↓]  ← ปุ่มเลื่อนไปข้อความล่าสุด
                                         (มุมขวาล่าง)
```

### 🎯 Smart Expander Behavior (New!)

#### Position Memory
```
📋 รายละเอียดโภชนาการแต่ละวัตถุดิบ ▼
[เนื้อหาโภชนาการ...]

เมื่อปิด expander:
✅ กลับไปตำแหน่งเดิมโดยอัตโนมัติ
❌ ไม่เลื่อนไปด้านบนของหน้า
```

### 🟢 Real-time Status Indicators

#### API Connection Status
```
USDA API: 🟢 เชื่อมต่อแล้ว
Nutritionix API: 🔴 ไม่ได้เชื่อมต่อ
External Data: 🟡 กำลังทดสอบ...
```

#### Processing Status
```
🔄 กำลังค้นหาและวิเคราะห์...
📈 กำลังวิเคราะห์ข้อมูลโภชนาการขั้นสูง...
🔌 กำลังเชื่อมต่อ API...
```

---

## Tips and Best Practices

### 🎯 Getting the Best Results

#### 1. Search Tips
```
✅ Good: "ผัดกะเพรา", "อาหารโปรตีนสูง", "เมนูแคลอรี่ต่ำ"
❌ Avoid: Single letters, very vague terms
```

#### 2. API Setup Priority
```
1st Priority: USDA API (FREE, comprehensive)
2nd Priority: Enhanced search features
3rd Priority: Nutritionix API (if you need branded foods)
```

#### 3. Feature Optimization
```
For Accuracy: Enable all cooking adjustments
For Speed: Use basic mode without API
For Completeness: Enable external recipe data
```

### ⚡ Performance Tips

#### 1. Cache Management
- แอปจะจำข้อมูลที่เคยค้นหาแล้ว
- การค้นหาครั้งที่ 2 จะเร็วกว่า
- Cache จะอัปเดตทุก 24 ชั่วโมง

#### 2. API Usage Optimization
```
USDA API: ไม่จำกัดการใช้งาน ✅
Nutritionix API: 200 requests/day (Free plan) ⚠️
```

#### 3. Search Strategy
```
Specific Terms: "ผัดกะเพรา" → Fast, accurate
General Terms: "อาหารไทย" → Slower, more results  
Nutrition Terms: "แคลอรี่ต่ำ" → API-enhanced results
```

---

## Troubleshooting

### 🔧 Common Issues

#### Issue: Settings Sidebar Not Visible
**Solution:**
1. Check screen width (sidebar hides on narrow screens)
2. Refresh the page (Ctrl+F5)
3. Try different browser

#### Issue: API Connection Failed
**Symptoms:**
- 🔴 Red status indicator
- "API connection failed" message

**Solutions:**
1. **Check API Key:**
   ```
   ❌ your_usda_api_key_here
   ✅ abcd1234-5678-90ef-ghij-klmnopqrstuv
   ```

2. **Test Connection:**
   - Click "ทดสอบการเชื่อมต่อ" button
   - Wait for status update

3. **Check Internet:**
   - Ensure stable internet connection
   - Try accessing https://api.nal.usda.gov directly

#### Issue: Enhanced Features Not Working
**Symptoms:**
- No improvement in search results
- Same nutrition data as before

**Check List:**
```
✅ API keys configured correctly
✅ "เพิ่มขอบเขตการค้นหา" enabled
✅ "คำนวณการบริโภคอย่างแม่นยำ" enabled
✅ "ใช้ข้อมูลจาก API ภายนอก" enabled
```

### 🚨 Error Messages

#### "ไม่พบสูตรอาหารที่ตรงกับคำค้นหา"
**Possible Causes:**
- Very specific search terms
- Typos in ingredient names
- Enhanced search disabled

**Solutions:**
1. Try broader terms: "ไข่" instead of "ไข่เจียวใส่ผักชี"
2. Enable enhanced search in settings
3. Try related terms: "กะเพรา" → "ผัดกะเพรา"

#### "การเชื่อมต่อ API ล้มเหลว"
**Immediate Actions:**
1. Check API key validity
2. Test internet connection
3. Try again in a few minutes
4. Use app without API (still works!)

### 🔄 Reset and Recovery

#### Reset Settings to Default
1. Close the app (Ctrl+C)
2. Delete `.streamlit/secrets.toml` (if exists)
3. Delete `.env` file (if exists)
4. Restart the app
5. Reconfigure settings

#### Clear Cache
1. Go to Settings sidebar
2. Look for "Clear Cache" option (if available)
3. Or restart the app completely

#### Emergency Mode
If app won't start:
```bash
# Use minimal requirements
pip install streamlit pandas numpy sentence-transformers
streamlit run streamlit_app.py
```

---

## 🎓 Advanced Usage Examples

### Example 1: Complete Nutrition Analysis Workflow
```
1. Open app → Settings → Enable USDA API
2. Search: "ผัดกะเพรา"
3. View enhanced nutrition with cooking adjustments
4. Compare with similar recipes
5. Export results (if needed)
```

### Example 2: Meal Planning with Nutrition Goals
```
1. Set nutrition criteria: "เมนูแคลอรี่ไม่เกิน 400"
2. Find suitable recipes
3. Check protein content for balanced meal
4. Use cooking adjustments for accurate planning
```

### Example 3: API Comparison Study
```
1. Enable both USDA and Nutritionix APIs
2. Search same ingredient: "กุ้ง"
3. Compare nutrition data from different sources
4. Choose most appropriate for your needs
```

---

## 📈 What's Next?

### 🔮 Upcoming Features
- Dark Mode Support
- Voice Input Integration  
- PDF Export of Nutrition Reports
- Meal Planning Calendar
- Multi-language Support

### 🚀 Stay Updated
- Check GitHub for latest releases
- Follow update notifications in app
- Join community discussions

---

## 💬 Support

### 📧 Get Help
- **GitHub Issues**: Technical problems
- **Email**: General questions
- **Community**: Tips and tricks sharing

### 🤝 Contribute
- Report bugs and suggestions
- Share your API setup experience
- Help improve documentation

---

**🍲 Enjoy your enhanced Thai food cooking journey! 🇹🇭**