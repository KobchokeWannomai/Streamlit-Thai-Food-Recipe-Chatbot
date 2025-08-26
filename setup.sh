#!/bin/bash

# Thai Food Nutrition Analyzer - Setup Script for Linux/macOS
# สคริปต์ติดตั้งอัตโนมัติสำหรับ Linux/macOS

set -e  # หยุดเมื่อพบข้อผิดพลาด

# สี ANSI สำหรับการแสดงผล
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ฟังก์ชันแสดงข้อความ
print_header() {
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${CYAN}🍲 Thai Food Nutrition Analyzer - การติดตั้งอัตโนมัติ${NC}"
    echo -e "${BLUE}================================================================${NC}"
    echo
}

print_step() {
    echo -e "${PURPLE}$1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${CYAN}📋 $1${NC}"
}

# ฟังก์ชันตรวจสอบคำสั่ง
check_command() {
    if command -v "$1" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# ฟังก์ชันติดตั้งแพ็คเกจระบบ
install_system_packages() {
    print_step "📦 กำลังตรวจสอบและติดตั้งแพ็คเกจระบบที่จำเป็น..."
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if check_command "brew"; then
            print_success "พบ Homebrew"
            brew update
            brew install python@3.9 git curl wget || true
        else
            print_warning "ไม่พบ Homebrew กรุณาติดตั้งจาก https://brew.sh/"
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if check_command "apt-get"; then
            # Ubuntu/Debian
            print_step "อัปเดตแพ็คเกจ apt..."
            sudo apt-get update
            sudo apt-get install -y python3 python3-venv python3-pip python3-dev \
                                    git curl wget build-essential libssl-dev \
                                    libffi-dev libbz2-dev libreadline-dev \
                                    libsqlite3-dev libncurses5-dev \
                                    libncursesw5-dev xz-utils tk-dev
        elif check_command "yum"; then
            # CentOS/RHEL
            sudo yum update -y
            sudo yum install -y python3 python3-venv python3-pip python3-devel \
                               git curl wget gcc openssl-devel libffi-devel \
                               bzip2-devel readline-devel sqlite-devel \
                               ncurses-devel xz-devel tk-devel
        elif check_command "pacman"; then
            # Arch Linux
            sudo pacman -Syu --noconfirm
            sudo pacman -S --noconfirm python python-pip python-virtualenv \
                                       git curl wget base-devel openssl \
                                       libffi bzip2 readline sqlite ncurses xz tk
        fi
    fi
}

# ฟังก์ชันตรวจสอบความต้องการของระบบ
check_requirements() {
    print_step "🔍 กำลังตรวจสอบความต้องการของระบบ..."
    echo
    
    # ตรวจสอบ Python
    if check_command "python3"; then
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
        
        if [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -ge 8 ]]; then
            print_success "พบ Python $PYTHON_VERSION"
        else
            print_error "ต้องการ Python 3.8+ แต่พบ Python $PYTHON_VERSION"
            echo
            print_info "วิธีติดตั้ง Python 3.8+:"
            echo "  - Ubuntu/Debian: sudo apt install python3.8 python3.8-venv python3.8-pip"
            echo "  - macOS: brew install python@3.9"
            echo "  - หรือดาวน์โหลดจาก https://www.python.org/"
            exit 1
        fi
    elif check_command "python"; then
        # ตรวจสอบ python (อาจเป็น Python 2)
        PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
        if [[ $PYTHON_VERSION == 3.* ]]; then
            print_success "พบ Python $PYTHON_VERSION"
        else
            print_error "พบ Python $PYTHON_VERSION แต่ต้องการ Python 3.8+"
            exit 1
        fi
    else
        print_error "ไม่พบ Python! กรุณาติดตั้ง Python 3.8+"
        install_system_packages
        exit 1
    fi
    
    # เลือกคำสั่ง python ที่ถูกต้อง
    if check_command "python3"; then
        PYTHON_CMD="python3"
        PIP_CMD="pip3"
    else
        PYTHON_CMD="python"
        PIP_CMD="pip"
    fi
    
    # ตรวจสอบ pip
    if check_command "$PIP_CMD"; then
        print_success "พบ $PIP_CMD"
    else
        print_error "ไม่พบ $PIP_CMD! กำลังติดตั้ง..."
        $PYTHON_CMD -m ensurepip --default-pip
    fi
    
    # ตรวจสอบ git
    if check_command "git"; then
        GIT_VERSION=$(git --version 2>&1 | cut -d' ' -f3)
        print_success "พบ Git $GIT_VERSION"
    else
        print_warning "ไม่พบ Git (จะข้าม git operations)"
    fi
    
    # ตรวจสอบการเชื่อมต่ออินเทอร์เน็ต
    if curl -s --connect-timeout 5 https://google.com > /dev/null; then
        print_success "การเชื่อมต่ออินเทอร์เน็ตปกติ"
    else
        print_warning "ไม่สามารถเชื่อมต่ออินเทอร์เน็ตได้"
    fi
}

# ฟังก์ชันสร้างโครงสร้างโฟลเดอร์
create_directory_structure() {
    print_step "📁 กำลังสร้างโครงสร้างโฟลเดอร์..."
    
    mkdir -p data model config .streamlit scripts logs
    
    # สร้างไฟล์ .gitignore ถ้าไม่มี
    if [[ ! -f .gitignore ]]; then
        cat > .gitignore << 'EOF'
# Virtual Environment
venv/
env/
.env

# Python
__pycache__/
*.py[cod]
*$py.class
*.so

# Model files (too large for git)
model/
embeddings.pkl

# Data files
*.csv
*.json
*.xlsx

# Logs
logs/
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Streamlit
.streamlit/secrets.toml
EOF
        print_success "สร้างไฟล์ .gitignore"
    fi
    
    print_success "สร้างโครงสร้างโฟลเดอร์เรียบร้อย"
}

# ฟังก์ชันสร้าง Virtual Environment
setup_virtual_environment() {
    print_step "🐍 กำลังสร้าง Virtual Environment..."
    
    if [[ -d "venv" ]]; then
        print_warning "พบ venv เก่า กำลังลบและสร้างใหม่..."
        rm -rf venv
    fi
    
    $PYTHON_CMD -m venv venv
    
    if [[ $? -eq 0 ]]; then
        print_success "สร้าง Virtual Environment สำเร็จ"
    else
        print_error "ไม่สามารถสร้าง Virtual Environment ได้"
        print_info "ลองติดตั้ง python3-venv:"
        echo "  sudo apt install python3-venv  # Ubuntu/Debian"
        echo "  sudo yum install python3-virtualenv  # CentOS/RHEL"
        exit 1
    fi
    
    # เปิดใช้งาน virtual environment
    source venv/bin/activate
    
    if [[ $? -eq 0 ]]; then
        print_success "เปิดใช้งาน Virtual Environment สำเร็จ"
    else
        print_error "ไม่สามารถเปิดใช้งาน Virtual Environment ได้"
        exit 1
    fi
}

# ฟังก์ชันติดตั้ง Python packages
install_python_packages() {
    print_step "📚 กำลังติดตั้ง Python packages..."
    echo "   (กระบวนการนี้อาจใช้เวลา 5-15 นาที)"
    echo
    
    # อัปเกรด pip
    print_step "📦 กำลังอัปเกรด pip..."
    pip install --upgrade pip
    
    # ติดตั้ง packages แบบแยก เพื่อให้เห็นความคืบหน้า
    print_step "🔧 กำลังติดตั้ง Core packages..."
    pip install streamlit pandas numpy
    
    print_step "🤖 กำลังติดตั้ง AI/ML packages..."
    pip install sentence-transformers scikit-learn transformers
    
    print_step "🔥 กำลังติดตั้ง PyTorch..."
    # ตรวจสอบสถาปัตยกรรม CPU
    ARCH=$(uname -m)
    if [[ "$ARCH" == "arm64" ]] || [[ "$ARCH" == "aarch64" ]]; then
        print_info "ตรวจพบสถาปัตยกรรม ARM64 กำลังติดตั้ง PyTorch สำหรับ ARM..."
        pip install torch torchvision
    else
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
    
    print_step "🌐 กำลังติดตั้ง Web packages..."
    pip install requests urllib3
    
    print_step "📊 กำลังติดตั้ง Data packages..."
    pip install openpyxl plotly matplotlib
    
    print_step "🛠️  กำลังติดตั้ง Utility packages..."
    pip install python-dateutil python-dotenv tqdm fuzzywuzzy python-levenshtein numba
    
    if [[ $? -eq 0 ]]; then
        print_success "ติดตั้ง Python packages สำเร็จ!"
    else
        print_warning "การติดตั้งบางส่วนล้มเหลว กำลังลองติดตั้งจาก requirements.txt..."
        if [[ -f "requirements.txt" ]]; then
            pip install -r requirements.txt
            if [[ $? -eq 0 ]]; then
                print_success "ติดตั้งจาก requirements.txt สำเร็จ!"
            else
                print_error "ไม่สามารถติดตั้ง packages ได้"
                print_info "การแก้ปัญหา:"
                echo "  1. ตรวจสอบการเชื่อมต่ออินเทอร์เน็ต"
                echo "  2. ลองรันคำสั่ง: source venv/bin/activate && pip install -r requirements.txt"
                echo "  3. ตรวจสอบ Python version และสถาปัตยกรรม CPU"
                exit 1
            fi
        else
            print_error "ไม่พบไฟล์ requirements.txt"
            exit 1
        fi
    fi
}

# ฟังก์ชันตรวจสอบและเตรียมข้อมูล
setup_data_files() {
    print_step "🗃️  กำลังตรวจสอบไฟล์ข้อมูล..."
    
    if [[ ! -f "thai_food_processed_cleaned.csv" ]]; then
        if [[ -f "thai_food_processed.csv" ]]; then
            print_warning "พบไฟล์ thai_food_processed.csv กำลังประมวลผลใหม่..."
            python preprocess.py --input thai_food_processed.csv --output thai_food_processed_cleaned.csv --enhanced
            if [[ $? -eq 0 ]]; then
                print_success "ประมวลผลข้อมูลสำเร็จ"
            else
                print_warning "การประมวลผลข้อมูลล้มเหลว"
            fi
        elif [[ -f "data/thai_food_processed.csv" ]]; then
            print_warning "พบไฟล์ใน data/ กำลังประมวลผลใหม่..."
            python preprocess.py --input data/thai_food_processed.csv --output thai_food_processed_cleaned.csv --enhanced
        else
            print_warning "ไม่พบไฟล์ข้อมูลหลัก"
            print_info "กรุณาเตรียมไฟล์อย่างใดอย่างหนึ่ง:"
            echo "  - thai_food_processed.csv"
            echo "  - thai_food_processed_cleaned.csv"
            echo "  - data/thai_food_processed.csv"
        fi
    else
        print_success "พบไฟล์ข้อมูลหลัก"
    fi
    
    # ตรวจสอบไฟล์โภชนาการ
    if [[ ! -f "thai_ingredients_nutrition_data.csv" ]]; then
        if [[ -f "data/thai_ingredients_nutrition_data.csv" ]]; then
            cp data/thai_ingredients_nutrition_data.csv .
            print_success "คัดลอกไฟล์โภชนาการจาก data/"
        else
            print_warning "ไม่พบไฟล์ข้อมูลโภชนาการ"
            print_info "สามารถดึงข้อมูลใหม่ได้ด้วย fetch_usda_data.sh"
        fi
    else
        print_success "พบไฟล์ข้อมูลโภชนาการ"
    fi
}

# ฟังก์ชันสร้างไฟล์การตั้งค่า
create_config_files() {
    print_step "🔧 กำลังสร้างไฟล์การตั้งค่า..."
    
    # สร้างไฟล์ Streamlit config
    if [[ ! -f ".streamlit/config.toml" ]]; then
        mkdir -p .streamlit
        cat > .streamlit/config.toml << 'EOF'
[server]
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#ff6b6b"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f0f5"
textColor = "#262730"
font = "sans serif"
EOF
        print_success "สร้างไฟล์ .streamlit/config.toml"
    fi
    
    # สร้างไฟล์ secrets
    if [[ ! -f ".streamlit/secrets.toml" ]]; then
        cat > .streamlit/secrets.toml << 'EOF'
# USDA API Configuration
USDA_API_KEY = "DEMO_KEY"
# Get your free API key from: https://fdc.nal.usda.gov/api-key-signup.html

# Application Settings
[app_settings]
max_search_results = 10
enable_fuzzy_search = true
enable_cooking_adjustments = false

[nutrition_settings]
show_detailed_breakdown = true
decimal_places = 1
EOF
        print_success "สร้างไฟล์ .streamlit/secrets.toml"
        print_info "กรุณาแก้ไข USDA_API_KEY ในไฟล์ .streamlit/secrets.toml"
    fi
    
    # สร้าง environment file
    if [[ ! -f ".env" ]]; then
        cat > .env << 'EOF'
# Environment Variables for Thai Food Nutrition Analyzer
USDA_API_KEY=DEMO_KEY
STREAMLIT_SERVER_PORT=8501
PYTHON_PATH=./venv/bin/python
EOF
        print_success "สร้างไฟล์ .env"
    fi
}

# ฟังก์ชันสร้างสคริปต์เสริม
create_utility_scripts() {
    print_step "📱 กำลังสร้างสคริปต์เสริม..."
    
    # สคริปต์รันแอปพลิเคชัน
    cat > run_app.sh << 'EOF'
#!/bin/bash
# สคริปต์รันแอปพลิเคชัน Thai Food Nutrition Analyzer

cd "$(dirname "$0")"

echo "🍲 กำลังเริ่ม Thai Food Nutrition Analyzer..."
echo "📍 เปิดเว็บเบราว์เซอร์ไปที่: http://localhost:8501"
echo "🛑 กด Ctrl+C เพื่อหยุดแอปพลิเคชัน"
echo

# เปิดใช้งาน virtual environment
source venv/bin/activate

# รันแอปพลิเคชัน
streamlit run app.py
EOF
    chmod +x run_app.sh
    
    # สคริปต์ดึงข้อมูล USDA
    cat > fetch_usda_data.sh << 'EOF'
#!/bin/bash
# สคริปต์ดึงข้อมูลโภชนาการจาก USDA API

cd "$(dirname "$0")"

echo "🌐 กำลังดึงข้อมูลโภชนาการจาก USDA API..."

# เปิดใช้งาน virtual environment
source venv/bin/activate

# ดึงข้อมูล
python usda_nutrition_fetcher.py \
  --recipes thai_food_processed_cleaned.csv \
  --output thai_ingredients_nutrition_data.csv \
  --update thai_ingredients_nutrition_data.csv

echo "✅ เสร็จสิ้น!"
EOF
    chmod +x fetch_usda_data.sh
    
    # สคริปต์อัปเดต
    cat > update.sh << 'EOF'
#!/bin/bash
# สคริปต์อัปเดตแอปพลิเคชัน

cd "$(dirname "$0")"

echo "🔄 กำลังอัปเดต Thai Food Nutrition Analyzer..."

# เปิดใช้งาน virtual environment
source venv/bin/activate

# อัปเดต Python packages
pip install --upgrade -r requirements.txt

# ประมวลผลข้อมูลใหม่ (ถ้ามี)
if [[ -f "thai_food_raw.csv" ]]; then
    python preprocess.py --input thai_food_raw.csv --output thai_food_processed_cleaned.csv --enhanced
fi

# ลบ embeddings เก่าเพื่อให้สร้างใหม่
rm -f embeddings.pkl

echo "✅ อัปเดตเสร็จสิ้น!"
EOF
    chmod +x update.sh
    
    # สคริปต์ทดสอบ
    cat > test_installation.sh << 'EOF'
#!/bin/bash
# สคริปต์ทดสอบการติดตั้ง

cd "$(dirname "$0")"

echo "🧪 กำลังทดสอบการติดตั้ง..."

# เปิดใช้งาน virtual environment
source venv/bin/activate

echo "1. ทดสอบ Python dependencies..."
python -c "
import streamlit
import pandas
import numpy
import sentence_transformers
import sklearn
print('✅ All Python packages OK')
"

echo "2. ทดสอบการโหลดข้อมูล..."
python -c "
import pandas as pd
import os

files_to_check = [
    'thai_food_processed_cleaned.csv',
    'thai_ingredients_nutrition_data.csv'
]

for file in files_to_check:
    if os.path.exists(file):
        df = pd.read_csv(file)
        print(f'✅ {file}: {len(df)} rows')
    else:
        print(f'⚠️  {file}: Not found')
"

echo "3. ทดสอบ imports สำคัญ..."
python -c "
try:
    from sentence_transformers import SentenceTransformer
    print('✅ SentenceTransformer OK')
    
    import torch
    print('✅ PyTorch OK')
    
    from sklearn.metrics.pairwise import cosine_similarity
    print('✅ Scikit-learn OK')
    
    import streamlit as st
    print('✅ Streamlit OK')
    
    print('🎉 All tests passed!')
except Exception as e:
    print(f'❌ Test failed: {e}')
"

echo "✅ การทดสอบเสร็จสิ้น!"
EOF
    chmod +x test_installation.sh
    
    print_success "สร้างสคริปต์เสริม"
    echo "   - run_app.sh: รันแอปพลิเคชัน"
    echo "   - fetch_usda_data.sh: ดึงข้อมูล USDA"
    echo "   - update.sh: อัปเดตระบบ"
    echo "   - test_installation.sh: ทดสอบการติดตั้ง"
}

# ฟังก์ชันทดสอบการติดตั้ง
test_installation() {
    print_step "🚀 กำลังทดสอบการติดตั้ง..."
    
    # ทดสอบ imports สำคัญ
    python -c "
import sys
try:
    import streamlit
    import pandas
    import numpy
    import sentence_transformers
    print('✅ All dependencies OK')
except ImportError as e:
    print(f'❌ Import failed: {e}')
    sys.exit(1)
" 2>/dev/null
    
    if [[ $? -eq 0 ]]; then
        print_success "การทดสอบสำเร็จ!"
    else
        print_warning "การทดสอบล้มเหลว แต่อาจยังใช้งานได้"
    fi
}

# ฟังก์ชันแสดงสรุป
show_summary() {
    echo
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${GREEN}🎉 การติดตั้งเสร็จสิ้น!${NC}"
    echo -e "${BLUE}================================================================${NC}"
    echo
    print_info "📋 สรุปการติดตั้ง:"
    echo "   ✅ Virtual Environment: venv/"
    echo "   ✅ Python Packages: ติดตั้งครบถ้วน"
    echo "   ✅ การตั้งค่า: .streamlit/secrets.toml"
    echo "   ✅ สคริปต์เสริม: run_app.sh, fetch_usda_data.sh"
    echo
    print_info "🚀 วิธีการรันแอปพลิเคชัน:"
    echo "   ./run_app.sh"
    echo "   หรือ:"
    echo "   source venv/bin/activate"
    echo "   streamlit run app.py"
    echo
    print_info "🌐 หลังจากรันแล้ว เปิดเว็บเบราว์เซอร์ไปที่:"
    echo "   http://localhost:8501"
    echo
    print_info "📚 ขั้นตอนถัดไป:"
    echo "   1. แก้ไข USDA_API_KEY ในไฟล์ .streamlit/secrets.toml"
    echo "   2. ดาวน์โหลดข้อมูลใหม่ด้วย ./fetch_usda_data.sh"
    echo "   3. เปิดใช้งานแอปพลิเคชัน"
    echo
    print_info "🧪 ทดสอบการติดตั้ง:"
    echo "   ./test_installation.sh"
    echo
    print_info "🆘 หากพบปัญหา:"
    echo "   - ตรวจสอบ README.md สำหรับคู่มือแก้ปัญหา"
    echo "   - หรือเปิด Issue ใน GitHub"
    echo
}

# ฟังก์ชันหลัก
main() {
    print_header
    
    # ตรวจสอบระบบปฏิบัติการ
    echo -e "${CYAN}🖥️  ระบบปฏิบัติการ: $OSTYPE${NC}"
    echo -e "${CYAN}🏗️  สถาปัตยกรรม: $(uname -m)${NC}"
    echo
    
    # ขั้นตอนการติดตั้ง
    check_requirements
    create_directory_structure
    setup_virtual_environment
    install_python_packages
    setup_data_files
    create_config_files
    create_utility_scripts
    test_installation
    show_summary
}

# รันฟังก์ชันหลัก
main "$@"
