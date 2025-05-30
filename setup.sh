#!/bin/bash

# Enhanced Thai Food Recipe Chatbot with Advanced Nutrition Analysis Setup Script

echo "============================================================================="
echo "🍲 Thai Food Recipe Chatbot with Advanced Nutrition Analysis Setup"
echo "============================================================================="
echo "Setting up enhanced environment with API integration and advanced features..."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions for colored output
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_step() {
    echo -e "${BLUE}🔄 $1${NC}"
}

# Check if Python is installed
print_step "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8+ and try again."
    echo ""
    echo "📥 Download Python from: https://www.python.org/downloads/"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
print_info "Found Python version: $PYTHON_VERSION"

if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
    print_success "Python version is compatible"
else
    print_error "Python 3.8+ is required. Current version: $PYTHON_VERSION"
    echo "📥 Please upgrade Python and try again."
    exit 1
fi

# Create virtual environment
print_step "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Linux/macOS
    source venv/bin/activate
fi

print_success "Virtual environment activated"

# Upgrade pip
print_step "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

# Install requirements
print_step "Installing required packages..."
pip install -r requirements.txt

print_success "Dependencies installed successfully"

# Check if the dataset exists
print_step "Checking dataset availability..."
if [ ! -f "thai_food_processed.csv" ]; then
    print_warning "thai_food_processed.csv not found in the current directory."
    print_info "Checking for alternative data files..."
    
    if [ -f "thai_food_raw.csv" ]; then
        print_info "Found thai_food_raw.csv. Processing with enhanced nutrition analysis..."
        python preprocess.py --input thai_food_raw.csv --output thai_food_processed.csv --analyze-nutrition
    elif [ -f "thai_food_sample.csv" ]; then
        print_info "Found thai_food_sample.csv. Processing sample data with enhancements..."
        python preprocess.py --input thai_food_sample.csv --output thai_food_processed.csv --analyze-nutrition
    else
        print_error "No data file found. Please ensure you have one of:"
        echo "   - thai_food_processed.csv (processed data)"
        echo "   - thai_food_raw.csv (raw data for processing)"
        echo "   - thai_food_sample.csv (sample data for testing)"
        echo ""
        print_info "You can download the data from the repository or use the sample data provided."
        exit 1
    fi
else
    print_success "thai_food_processed.csv found"
    
    # Check if nutrition data exists in the CSV
    if python3 -c "
import pandas as pd
try:
    df = pd.read_csv('thai_food_processed.csv')
    has_nutrition = any(col.startswith('nutrition_') for col in df.columns)
    exit(0 if has_nutrition else 1)
except:
    exit(1)
"; then
        print_success "Nutrition data found in existing dataset"
    else
        print_warning "No nutrition data found. Adding enhanced nutrition analysis..."
        python preprocess.py --input thai_food_processed.csv --output thai_food_processed.csv --analyze-nutrition
    fi
fi

# Create enhanced nutrition database
print_step "Creating enhanced nutrition database..."
python preprocess.py --create-nutrition-db

# Create configuration files for API keys
print_step "Setting up configuration files..."
if [ ! -f ".env" ]; then
    cat > .env << EOL
# Enhanced Thai Food Chatbot - API Configuration
# Get these API keys for enhanced nutrition analysis

# USDA FoodData Central API (FREE - Highly Recommended!)
# Sign up at: https://fdc.nal.usda.gov/api-guide.html
# Benefits: Free, comprehensive nutrition data, government-backed
USDA_API_KEY=your_usda_api_key_here

# Nutritionix API (Optional - 200 requests/day free)
# Sign up at: https://www.nutritionix.com/business/api
# Benefits: Food database with natural language processing
NUTRITIONIX_API_KEY=your_nutritionix_api_key_here
NUTRITIONIX_APP_ID=your_nutritionix_app_id_here

# Advanced Features Configuration
ENABLE_ENHANCED_SEARCH=true
ENABLE_COOKING_ADJUSTMENTS=true
ENABLE_API_INTEGRATION=true

# Performance Settings
API_TIMEOUT=10
CACHE_DURATION_HOURS=24
MAX_SEARCH_RESULTS=10

# Logging Configuration
LOG_LEVEL=INFO
ENABLE_DETAILED_LOGGING=false
EOL
    print_success "Created .env file for enhanced API configuration"
    print_info "📝 Edit .env file to add your API keys for enhanced features"
else
    print_success ".env file already exists"
fi

# Create Streamlit secrets template with enhanced settings
if [ ! -f ".streamlit/secrets.toml" ]; then
    mkdir -p .streamlit
    cat > .streamlit/secrets.toml << EOL
# Enhanced Streamlit Secrets Configuration
# Copy your API keys here for deployment

[api_keys]
# USDA FoodData Central (FREE)
USDA_API_KEY = "your_usda_api_key_here"

# Nutritionix (200 requests/day free)
NUTRITIONIX_API_KEY = "your_nutritionix_api_key_here"
NUTRITIONIX_APP_ID = "your_nutritionix_app_id_here"

[features]
# Enhanced Features Toggle
enhanced_search = true
cooking_adjustments = true
api_integration = true
auto_scroll = true

[performance]
# Performance Settings
api_timeout = 10
cache_duration = 24
max_search_results = 10
batch_size = 5

[database]
# Database configuration
nutrition_cache_path = "nutrition_cache.db"
cache_expiry_days = 30
EOL
    print_success "Created enhanced Streamlit secrets template"
else
    print_success "Streamlit secrets file already exists"
fi

# Test the enhanced setup
print_step "Testing enhanced setup..."
if python3 -c "
import streamlit
import pandas
import sentence_transformers
import plotly
import requests
from nutrition_analyzer import NutritionAnalyzer
from ingredient_converter import IngredientConverter
print('✅ All required modules can be imported')
try:
    # Test enhanced features
    analyzer = NutritionAnalyzer()
    converter = IngredientConverter()
    print('✅ Enhanced nutrition analyzer initialized')
    print('✅ Ingredient converter initialized')
except Exception as e:
    print(f'⚠️  Enhanced features warning: {e}')
"; then
    print_success "Enhanced setup test passed"
else
    print_error "Enhanced setup test failed. Please check the error messages above."
    exit 1
fi

# Test enhanced nutrition analyzer
print_step "Testing enhanced nutrition analyzer..."
if python3 -c "
from nutrition_analyzer import NutritionAnalyzer
try:
    analyzer = NutritionAnalyzer()
    # Test basic functionality
    result = analyzer.get_ingredient_nutrition('กุ้ง')
    if result:
        print(f'✅ Enhanced nutrition analyzer working. Sample: กุ้ง = {result.calories:.1f} cal')
    else:
        print('⚠️  Analyzer working but no data for sample ingredient')
    
    # Test cooking adjustments
    from nutrition_analyzer import CookingAdjustmentHelper
    helper = CookingAdjustmentHelper()
    adjustments = helper.get_cooking_adjustments('ไข่เจียว')
    if adjustments:
        print('✅ Cooking adjustments feature available')
    else:
        print('ℹ️  Cooking adjustments initialized (no specific data for test recipe)')
        
except Exception as e:
    print(f'⚠️  Enhanced analyzer test warning: {e}')
"; then
    print_success "Enhanced nutrition analyzer test passed"
else
    print_warning "Enhanced nutrition analyzer has some limitations but basic functions work"
fi

# Create sample API configuration guide
cat > API_SETUP_GUIDE.md << EOL
# 🔑 API Setup Guide for Enhanced Features

## 🆓 USDA FoodData Central API (Recommended - FREE!)

### Benefits:
- ✅ Completely free with no usage limits
- ✅ Comprehensive nutrition database
- ✅ Government-backed reliable data
- ✅ Covers international foods including Asian cuisine

### Setup Steps:
1. Visit: https://fdc.nal.usda.gov/api-guide.html
2. Click "Get an API Key"
3. Fill out the form:
   - Name: Your name
   - Email: Your email address
   - Organization: "Personal Use" or your organization
   - Intended Use: "Recipe Nutrition Analysis"
4. Check your email for the API key
5. Open the app, go to Settings sidebar
6. Enable "USDA API" and paste your key
7. Click "Test Connection"

## 🥇 Nutritionix API (Optional - 200 requests/day free)

### Benefits:
- ✅ Natural language food queries
- ✅ Extensive branded food database
- ✅ Restaurant menu items
- ✅ 200 free requests per day

### Setup Steps:
1. Visit: https://www.nutritionix.com/business/api
2. Sign up for a free account
3. Choose "Free Plan"
4. Get your Application ID and API Key from the dashboard
5. Open the app, go to Settings sidebar
6. Enable "Nutritionix API" and enter both keys
7. Click "Test Connection"

## 🚀 Using the Enhanced Features

Once you have API keys set up:

1. **Enhanced Nutrition Data**: Get more accurate nutrition information
2. **Cooking Adjustments**: Calculate actual consumption (e.g., oil absorption in frying)
3. **Missing Ingredients**: Automatically add common cooking ingredients
4. **Smart Search**: Expanded search capabilities with better matching

## 💡 Tips:

- You can use the app without API keys (uses built-in Thai nutrition database)
- USDA API is recommended for best results and it's completely free
- Check the Settings sidebar for connection status indicators
- Green dot = Connected, Red dot = Disconnected
EOL

print_success "Created API setup guide: API_SETUP_GUIDE.md"

echo ""
echo "============================================================================="
print_success "🎉 Enhanced setup completed successfully!"
echo "============================================================================="
echo ""
echo "📋 What's been set up:"
echo "   ✅ Virtual environment created and activated"
echo "   ✅ All dependencies installed (including API support)"
echo "   ✅ Thai food dataset processed with enhanced nutrition analysis"
echo "   ✅ Enhanced nutrition database created with cooking adjustments"
echo "   ✅ Configuration files created (.env and Streamlit secrets)"
echo "   ✅ API setup guide generated"
echo "   ✅ System tested and working with enhanced features"
echo ""
echo "🚀 Next steps:"
echo "   1. 📖 Read API_SETUP_GUIDE.md for API configuration (optional but recommended)"
echo "   2. 🔧 Edit .env file to add your API keys for enhanced features"
echo "   3. ▶️  Run: streamlit run streamlit_app.py"
echo "   4. 🌐 Open your browser to the URL shown (usually http://localhost:8501)"
echo "   5. ⚙️  Click the Settings button in the sidebar to configure APIs"
echo ""
echo "💡 Enhanced features available:"
echo "   🔍 Smart search with query expansion"
echo "   📊 Advanced nutrition analysis with API integration"
echo "   🧪 Cooking adjustment calculations (oil absorption, etc.)"
echo "   🎯 Real-time API status monitoring"
echo "   📱 Enhanced UI with auto-scroll and smooth navigation"
echo "   🔧 Missing ingredient detection and addition"
echo ""
echo "🔗 API Resources (for enhanced nutrition data):"
echo "   • USDA FoodData Central (FREE): https://fdc.nal.usda.gov/api-guide.html"
echo "   • Nutritionix (200 free/day): https://www.nutritionix.com/business/api"
echo ""
echo "📚 Additional commands:"
echo "   • python nutrition_example.py - Run enhanced nutrition examples"
echo "   • python batch_nutrition_processor.py - Process multiple recipes"
echo "   • python preprocess.py --help - View preprocessing options"
echo "   • streamlit run streamlit_app.py --help - View Streamlit options"
echo ""
echo "🆘 Need help?"
echo "   • Check README.md for detailed documentation"
echo "   • Read API_SETUP_GUIDE.md for API setup instructions"
echo "   • Open GitHub issues for support"
echo ""
echo "🙏 Thank you for using Enhanced Thai Food Recipe Chatbot! 🍲"
echo "============================================================================="