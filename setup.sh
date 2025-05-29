#!/bin/bash

# Thai Food Recipe Chatbot with Nutrition Analysis Setup Script

echo "===== Thai Food Recipe Chatbot with Nutrition Analysis Setup ====="
echo "Setting up environment with nutrition analysis capabilities..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8+ and try again."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Found Python version: $PYTHON_VERSION"

if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
    echo "✅ Python version is compatible"
else
    echo "❌ Python 3.8+ is required. Current version: $PYTHON_VERSION"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Linux/macOS
    source venv/bin/activate
fi

echo "✅ Virtual environment activated"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing required packages..."
pip install -r requirements.txt

echo "✅ Dependencies installed successfully"

# Check if the dataset exists
if [ ! -f "thai_food_processed.csv" ]; then
    echo "⚠️  thai_food_processed.csv not found in the current directory."
    echo "Checking for alternative data files..."
    
    if [ -f "thai_food_raw.csv" ]; then
        echo "Found thai_food_raw.csv. Processing with nutrition analysis..."
        python preprocess.py --input thai_food_raw.csv --output thai_food_processed.csv --analyze-nutrition
    elif [ -f "thai_food_sample.csv" ]; then
        echo "Found thai_food_sample.csv. Processing sample data..."
        python preprocess.py --input thai_food_sample.csv --output thai_food_processed.csv --analyze-nutrition
    else
        echo "❌ No data file found. Please ensure you have one of:"
        echo "   - thai_food_processed.csv (processed data)"
        echo "   - thai_food_raw.csv (raw data for processing)"
        echo "   - thai_food_sample.csv (sample data for testing)"
        echo ""
        echo "You can download the data from the repository or use the sample data provided."
        exit 1
    fi
else
    echo "✅ thai_food_processed.csv found"
    
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
        echo "✅ Nutrition data found in existing dataset"
    else
        echo "⚠️  No nutrition data found. Adding nutrition analysis..."
        python preprocess.py --input thai_food_processed.csv --output thai_food_processed.csv --analyze-nutrition
    fi
fi

# Create nutrition database with Thai ingredients
echo "Creating nutrition database..."
python preprocess.py --create-nutrition-db

# Create configuration file for API keys
echo "Creating configuration files..."
if [ ! -f ".env" ]; then
    cat > .env << EOL
# API Keys for Nutrition Data (Optional)
# Get USDA API key from: https://fdc.nal.usda.gov/api-guide.html
USDA_API_KEY=your_usda_api_key_here

# Get Nutritionix API from: https://www.nutritionix.com/business/api
NUTRITIONIX_API_KEY=your_nutritionix_api_key_here
NUTRITIONIX_APP_ID=your_nutritionix_app_id_here

# Streamlit secrets (for deployment)
# Copy these to your Streamlit secrets if deploying online
EOL
    echo "✅ Created .env file for API keys configuration"
    echo "📝 Edit .env file to add your API keys (optional but recommended)"
else
    echo "✅ .env file already exists"
fi

# Create Streamlit secrets template
if [ ! -f ".streamlit/secrets.toml" ]; then
    mkdir -p .streamlit
    cat > .streamlit/secrets.toml << EOL
# Streamlit Secrets Configuration
# Copy your API keys here for deployment

[api_keys]
USDA_API_KEY = "your_usda_api_key_here"
NUTRITIONIX_API_KEY = "your_nutritionix_api_key_here"
NUTRITIONIX_APP_ID = "your_nutritionix_app_id_here"

[database]
# Database configuration if needed
EOL
    echo "✅ Created Streamlit secrets template"
else
    echo "✅ Streamlit secrets file already exists"
fi

# Test the setup
echo "Testing the setup..."
if python3 -c "
import streamlit
import pandas
import sentence_transformers
import plotly
from nutrition_analyzer import NutritionAnalyzer
print('✅ All required modules can be imported')
"; then
    echo "✅ Setup test passed"
else
    echo "❌ Setup test failed. Please check the error messages above."
    exit 1
fi

# Test nutrition analyzer
echo "Testing nutrition analyzer..."
if python3 -c "
from nutrition_analyzer import NutritionAnalyzer
analyzer = NutritionAnalyzer()
result = analyzer.get_ingredient_nutrition('กุ้ง')
if result:
    print(f'✅ Nutrition analyzer working. Sample: กุ้ง = {result.calories} cal')
else:
    print('⚠️  Nutrition analyzer working but no data for sample ingredient')
"; then
    echo "✅ Nutrition analyzer test passed"
else
    echo "❌ Nutrition analyzer test failed"
    exit 1
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 What's been set up:"
echo "   ✅ Virtual environment created and activated"
echo "   ✅ All dependencies installed"
echo "   ✅ Thai food dataset processed with nutrition analysis"
echo "   ✅ Nutrition database created"
echo "   ✅ Configuration files created"
echo "   ✅ System tested and working"
echo ""
echo "🚀 Next steps:"
echo "   1. (Optional) Edit .env file to add your API keys for better nutrition data"
echo "   2. Run: streamlit run streamlit_app.py"
echo "   3. Open your browser to the URL shown (usually http://localhost:8501)"
echo ""
echo "💡 Features available:"
echo "   🔍 Search recipes by name or ingredients"
echo "   📊 View detailed nutrition analysis for each recipe"
echo "   🎯 Filter recipes by nutrition criteria (calories, protein, etc.)"
echo "   📈 Visualize nutrition data with interactive charts"
echo "   💾 Export nutrition data"
echo ""
echo "📚 Additional commands:"
echo "   • python nutrition_example.py - Run nutrition analysis examples"
echo "   • python preprocess.py --help - View preprocessing options"
echo "   • streamlit run streamlit_app.py --help - View Streamlit options"
echo ""
echo "🔗 For more information:"
echo "   • Check README.md for detailed documentation"
echo "   • Visit the GitHub repository for updates"
echo ""
echo "Thank you for using Thai Food Recipe Chatbot with Nutrition Analysis! 🍲"
