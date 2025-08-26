#!/bin/bash

# Thai Food Nutrition Analyzer - Run Script
# สคริปต์รันแอปพลิเคชันแบบครบถ้วน

set -e

# สี ANSI
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# ฟังก์ชันแสดงข้อความ
print_header() {
    clear
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${CYAN}🍲 Thai Food Nutrition Analyzer${NC}"
    echo -e "${BLUE}================================================================${NC}"
    echo
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

print_step() {
    echo -e "${PURPLE}$1${NC}"
}

# ฟังก์ชันตรวจสอบไฟล์จำเป็น
check_requirements() {
    print_step "🔍 กำลังตรวจสอบความพร้อม..."
    
    local missing_files=()
    local required_files=(
        "app.py"
        "requirements.txt"
        "venv"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -e "$file" ]]; then
            missing_files+=("$file")
        fi
    done
    
    if [[ ${#missing_files[@]} -gt 0 ]]; then
        print_error "ไม่พบไฟล์จำเป็น: ${missing_files[*]}"
        print_info "กรุณารัน ./setup.sh ก่อนใช้งาน"
        exit 1
    fi
    
    print_success "ไฟล์จำเป็นครบถ้วน"
}

# ฟังก์ชันเปิดใช้งาน virtual environment
activate_venv() {
    print_step "🐍 กำลังเปิดใช้งาน Virtual Environment..."
    
    if [[ ! -d "venv" ]]; then
        print_error "ไม่พบ Virtual Environment"
        print_info "กรุณารัน ./setup.sh เพื่อติดตั้ง"
        exit 1
    fi
    
    source venv/bin/activate
    
    if [[ $? -eq 0 ]]; then
        print_success "Virtual Environment พร้อมใช้งาน"
    else
        print_error "ไม่สามารถเปิดใช้งาน Virtual Environment ได้"
        exit 1
    fi
}

# ฟังก์ชันตรวจสอบข้อมูล
check_data_files() {
    print_step "🗃️  กำลังตรวจสอบไฟล์ข้อมูล..."
    
    local data_files=(
        "thai_food_processed_cleaned.csv"
        "thai_ingredients_nutrition_data.csv"
    )
    
    local missing_data=()
    
    for file in "${data_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            missing_data+=("$file")
        fi
    done
    
    if [[ ${#missing_data[@]} -gt 0 ]]; then
        print_warning "ไม่พบไฟล์ข้อมูล: ${missing_data[*]}"
        print_info "แอปพลิเคชันจะทำงานได้แต่อาจมีฟังก์ชันบางส่วนไม่ครบถ้วน"
        
        echo "ต้องการดำเนินการต่อหรือไม่? (y/n)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            print_info "ยกเลิกการรัน"
            exit 0
        fi
    else
        print_success "ไฟล์ข้อมูลครบถ้วน"
    fi
}

# ฟังก์ชันตรวจสอบ dependencies
check_dependencies() {
    print_step "📚 กำลังตรวจสอบ Python packages..."
    
    # ตรวจสอบ packages สำคัญ
    python -c "
import sys
import importlib

required_packages = [
    'streamlit',
    'pandas', 
    'numpy',
    'sentence_transformers',
    'sklearn',
    'requests'
]

missing_packages = []

for package in required_packages:
    try:
        importlib.import_module(package)
    except ImportError:
        if package == 'sklearn':
            try:
                importlib.import_module('scikit-learn')
            except ImportError:
                missing_packages.append(package)
        else:
            missing_packages.append(package)

if missing_packages:
    print(f'Missing packages: {missing_packages}')
    sys.exit(1)
else:
    print('All packages available')
" 2>/dev/null
    
    if [[ $? -eq 0 ]]; then
        print_success "Python packages ครบถ้วน"
    else
        print_error "ไม่พบ Python packages ที่จำเป็น"
        print_info "กำลังติดตั้ง packages..."
        pip install -r requirements.txt
        
        if [[ $? -eq 0 ]]; then
            print_success "ติดตั้ง packages สำเร็จ"
        else
            print_error "ไม่สามารถติดตั้ง packages ได้"
            exit 1
        fi
    fi
}

# ฟังก์ชันตรวจสอบพอร์ต
check_port() {
    local port=${1:-8501}
    
    if command -v lsof >/dev/null 2>&1; then
        if lsof -i :$port >/dev/null 2>&1; then
            print_warning "พอร์ต $port กำลังถูกใช้งาน"
            print_info "จะใช้พอร์ตอื่นแทน..."
            return 1
        fi
    elif command -v netstat >/dev/null 2>&1; then
        if netstat -ln | grep ":$port " >/dev/null 2>&1; then
            print_warning "พอร์ต $port กำลังถูกใช้งาน"
            print_info "จะใช้พอร์ตอื่นแทน..."
            return 1
        fi
    fi
    
    return 0
}

# ฟังก์ชันเปิดเว็บเบราว์เซอร์
open_browser() {
    local url="http://localhost:$1"
    
    # รอให้แอปพลิเคชันเริ่มทำงาน
    sleep 3
    
    # ตรวจสอบระบบปฏิบัติการและเปิดเบราว์เซอร์
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        open "$url" 2>/dev/null &
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command -v xdg-open >/dev/null 2>&1; then
            xdg-open "$url" 2>/dev/null &
        elif command -v firefox >/dev/null 2>&1; then
            firefox "$url" 2>/dev/null &
        elif command -v google-chrome >/dev/null 2>&1; then
            google-chrome "$url" 2>/dev/null &
        elif command -v chromium-browser >/dev/null 2>&1; then
            chromium-browser "$url" 2>/dev/null &
        fi
    fi
}

# ฟังก์ชันแสดงข้อมูลการใช้งาน
show_usage_info() {
    local port=$1
    
    echo
    echo -e "${GREEN}🚀 แอปพลิเคชันกำลังทำงาน!${NC}"
    echo -e "${BLUE}================================================================${NC}"
    echo
    print_info "📱 การเข้าใช้งาน:"
    echo "   🌐 เว็บเบราว์เซอร์: http://localhost:$port"
    echo "   📱 มือถือ (ในเครือข่ายเดียวกัน): http://$(hostname -I | awk '{print $1}'):$port"
    echo
    print_info "⌨️  คำสั่งที่มีประโยชน์:"
    echo "   Ctrl + C: หยุดแอปพลิเคชัน"
    echo "   Ctrl + R: รีเฟรชหน้าเว็บ"
    echo "   F11: เต็มหน้าจอ"
    echo
    print_info "🔧 การแก้ปัญหาเบื้องต้น:"
    echo "   - ถ้าเว็บไม่โหลด: รอ 30 วินาที แล้วรีเฟรช"
    echo "   - ถ้าพอร์ตถูกใช้: แอปจะหาพอร์ตอื่นให้อัตโนมัติ"
    echo "   - ถ้าข้อผิดพลาด: ตรวจสอบ logs ด้านล่าง"
    echo
    echo -e "${BLUE}================================================================${NC}"
    echo
}

# ฟังก์ชันรันแอปพลิเคชัน
run_app() {
    local port=${1:-8501}
    local host="localhost"
    
    print_step "🚀 กำลังเริ่มแอปพลิเคชัน..."
    
    # ตรวจสอบพอร์ต
    local available_port=$port
    while ! check_port $available_port; do
        available_port=$((available_port + 1))
        if [[ $available_port -gt 8600 ]]; then
            print_error "ไม่สามารถหาพอร์ตที่ว่างได้"
            exit 1
        fi
    done
    
    if [[ $available_port -ne $port ]]; then
        print_info "ใช้พอร์ต $available_port แทน"
    fi
    
    # แสดงข้อมูลการใช้งาน
    show_usage_info $available_port
    
    # เปิดเบราว์เซอร์ (รันใน background)
    open_browser $available_port &
    
    # รันแอปพลิเคชัน Streamlit
    streamlit run app.py \
        --server.port $available_port \
        --server.address $host \
        --server.enableCORS false \
        --server.enableXsrfProtection false \
        --browser.gatherUsageStats false \
        --logger.level warning
}

# ฟังก์ชันจัดการสัญญาณ
cleanup() {
    echo
    print_info "🛑 กำลังหยุดแอปพลิเคชัน..."
    
    # ฆ่าโปรเซส streamlit ที่เหลืออยู่
    pkill -f "streamlit run" 2>/dev/null || true
    
    print_success "แอปพลิเคชันหยุดทำงานแล้ว"
    exit 0
}

# ฟังก์ชันแสดงความช่วยเหลือ
show_help() {
    echo "Thai Food Nutrition Analyzer - Run Script"
    echo
    echo "การใช้งาน:"
    echo "  $0 [OPTIONS]"
    echo
    echo "ตัวเลือก:"
    echo "  -p, --port PORT    กำหนดพอร์ต (default: 8501)"
    echo "  -h, --help         แสดงความช่วยเหลือ"
    echo "  --dev              โหมด development (แสดง debug info)"
    echo "  --no-browser       ไม่เปิดเบราว์เซอร์อัตโนมัติ"
    echo "  --check-only       ตรวจสอบระบบเท่านั้น ไม่รันแอป"
    echo
    echo "ตัวอย่าง:"
    echo "  $0                 # รันด้วยการตั้งค่าพื้นฐาน"
    echo "  $0 -p 8080         # รันที่พอร์ต 8080"
    echo "  $0 --dev           # รันในโหมด development"
    echo
}

# ฟังก์ชันหลัก
main() {
    # ตั้งค่าสัญญาณ
    trap cleanup INT TERM
    
    # เปลี่ยนไปยัง directory ของสคริปต์
    cd "$(dirname "$0")"
    
    # ตัวแปรสำหรับตัวเลือก
    local port=8501
    local dev_mode=false
    local no_browser=false
    local check_only=false
    
    # ประมวลผลพารามิเตอร์
    while [[ $# -gt 0 ]]; do
        case $1 in
            -p|--port)
                port="$2"
                if ! [[ "$port" =~ ^[0-9]+$ ]] || [[ $port -lt 1024 ]] || [[ $port -gt 65535 ]]; then
                    print_error "พอร์ตไม่ถูกต้อง: $port (ต้องเป็น 1024-65535)"
                    exit 1
                fi
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            --dev)
                dev_mode=true
                shift
                ;;
            --no-browser)
                no_browser=true
                shift
                ;;
            --check-only)
                check_only=true
                shift
                ;;
            *)
                print_error "ตัวเลือกไม่รู้จัก: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # แสดงหัวเรื่อง
    print_header
    
    # ขั้นตอนการตรวจสอบ
    check_requirements
    activate_venv
    check_data_files
    check_dependencies
    
    if [[ $check_only == true ]]; then
        print_success "ระบบพร้อมใช้งาน!"
        exit 0
    fi
    
    # ตั้งค่าสำหรับ development mode
    if [[ $dev_mode == true ]]; then
        print_info "🔧 Development Mode เปิดใช้งาน"
        export STREAMLIT_LOGGER_LEVEL="debug"
        export STREAMLIT_CLIENT_TOOLBAR_MODE="viewer"
    fi
    
    # รันแอปพลิเคชัน
    run_app $port
}

# เรียกใช้ฟังก์ชันหลัก
main "$@"
