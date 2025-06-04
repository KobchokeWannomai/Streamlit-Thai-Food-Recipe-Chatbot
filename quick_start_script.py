#!/usr/bin/env python3
"""
🚀 Quick Start Script สำหรับ Thai Food Recipe Chatbot
ช่วยให้ผู้ใช้เริ่มต้นใช้งานฟีเจอร์ใหม่ได้อย่างง่ายดาย
"""

import os
import sys
import subprocess
import platform

def print_header():
    """แสดงหัวข้อโปรแกรม"""
    print("🍲 Thai Food Recipe Chatbot - Quick Start")
    print("=" * 60)
    print("🚀 ติดตั้งและเริ่มใช้งานฟีเจอร์ใหม่อย่างง่ายดาย")
    print("=" * 60)
    print()

def check_python_version():
    """ตรวจสอบเวอร์ชัน Python"""
    print("🐍 ตรวจสอบ Python...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ ต้องการ Python 3.8 หรือใหม่กว่า")
        print(f"   ปัจจุบัน: Python {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - รองรับ")
    return True

def check_files():
    """ตรวจสอบไฟล์ที่จำเป็น"""
    print("\n📁 ตรวจสอบไฟล์...")
    
    required_files = [
        "improved_nutrition_api.py",
        "improved_app.py", 
        "nutrition_testing_demo.py",
        "requirements_improved.txt"
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - ไม่พบไฟล์")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️  ไฟล์ที่ขาดหาย: {', '.join(missing_files)}")
        print("💡 ตรวจสอบว่าได้ดาวน์โหลดไฟล์ครบถ้วนแล้ว")
        return False
    
    return True

def install_packages():
    """ติดตั้งแพ็คเกจที่จำเป็น"""
    print("\n📦 ติดตั้งแพ็คเกจ...")
    
    try:
        # ตรวจสอบว่ามี requirements_improved.txt หรือไม่
        if os.path.exists("requirements_improved.txt"):
            requirements_file = "requirements_improved.txt"
        elif os.path.exists("requirements.txt"):
            requirements_file = "requirements.txt"
        else:
            print("❌ ไม่พบไฟล์ requirements")
            return False
        
        print(f"📋 ใช้ไฟล์: {requirements_file}")
        
        # ติดตั้งแพ็คเกจ
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", requirements_file
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ ติดตั้งแพ็คเกจสำเร็จ")
            return True
        else:
            print("❌ เกิดข้อผิดพลาดในการติดตั้ง:")
            print(result.stderr)
            return False
    
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {str(e)}")
        return False

def test_improved_api():
    """ทดสอบ API ที่ปรับปรุงแล้ว"""
    print("\n🧪 ทดสอบ API ที่ปรับปรุงแล้ว...")
    
    try:
        from improved_nutrition_api import ImprovedNutritionAPI
        
        # ทดสอบสร้าง instance
        nutrition_api = ImprovedNutritionAPI()
        print("✅ สร้าง ImprovedNutritionAPI สำเร็จ")
        
        # ทดสอบคำนวณโภชนาการ
        test_ingredients = "- ไข่ไก่ 2 ฟอง\n- น้ำปลา 1 ช้อนชา"
        result = nutrition_api.calculate_recipe_nutrition(test_ingredients)
        
        if result and 'total_nutrition' in result:
            calories = result['total_nutrition']['calories']
            print(f"✅ ทดสอบการคำนวณสำเร็จ (แคลอรี่: {calories:.1f} kcal)")
            return True
        else:
            print("❌ ผลการคำนวณไม่ถูกต้อง")
            return False
            
    except ImportError as e:
        print(f"❌ ไม่สามารถ import ได้: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {str(e)}")
        return False

def run_demo():
    """เรียกใช้ demo การทดสอบ"""
    print("\n🎮 เรียกใช้ demo การทดสอบ...")
    
    try:
        result = subprocess.run([
            sys.executable, "nutrition_testing_demo.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ demo ทำงานสำเร็จ")
            print("\n📋 ผลลัพธ์ demo:")
            # แสดงเฉพาะบางส่วนของผลลัพธ์
            lines = result.stdout.split('\n')
            for line in lines[:20]:  # แสดง 20 บรรทัดแรก
                if line.strip():
                    print(f"   {line}")
            
            if len(lines) > 20:
                print("   ... (ดูเพิ่มเติมโดยรัน: python nutrition_testing_demo.py)")
            
            return True
        else:
            print("❌ demo ทำงานไม่สำเร็จ:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการรัน demo: {str(e)}")
        return False

def start_app():
    """เริ่มต้นแอปพลิเคชัน"""
    print("\n🚀 เริ่มต้นแอปพลิเคชัน...")
    
    # ตรวจสอบว่ามี streamlit หรือไม่
    try:
        import streamlit
        print("✅ พบ Streamlit")
    except ImportError:
        print("❌ ไม่พบ Streamlit")
        print("💡 ลองติดตั้ง: pip install streamlit")
        return False
    
    # เลือกไฟล์แอปที่จะเรียกใช้
    if os.path.exists("improved_app.py"):
        app_file = "improved_app.py"
        print("🆕 ใช้แอปที่ปรับปรุงแล้ว")
    elif os.path.exists("app.py"):
        app_file = "app.py"
        print("📱 ใช้แอปเดิม")
    else:
        print("❌ ไม่พบไฟล์แอป")
        return False
    
    print(f"\n🌐 เปิดเว็บเบราว์เซอร์ไปที่: http://localhost:8501")
    print("⏹️  กด Ctrl+C เพื่อหยุดเซิร์ฟเวอร์")
    print(f"🚀 เริ่มต้น {app_file}...")
    
    try:
        # เรียกใช้ streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", app_file
        ])
        return True
    except KeyboardInterrupt:
        print("\n👋 หยุดการทำงาน")
        return True
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {str(e)}")
        return False

def show_menu():
    """แสดงเมนูตัวเลือก"""
    print("\n📋 เลือกสิ่งที่ต้องการทำ:")
    print("1. 🧪 ทดสอบ API ที่ปรับปรุงแล้ว")
    print("2. 🎮 รัน demo การทดสอบ")
    print("3. 🚀 เริ่มต้นแอปพลิเคชัน")
    print("4. 📦 ติดตั้งแพ็คเกจใหม่")
    print("5. 🔧 ตรวจสอบระบบทั้งหมด")
    print("0. 👋 ออกจากโปรแกรม")
    print()

def main():
    """ฟังก์ชันหลัก"""
    print_header()
    
    # ตรวจสอบเบื้องต้น
    if not check_python_version():
        return
    
    if not check_files():
        return
    
    while True:
        show_menu()
        
        try:
            choice = input("🔍 เลือกตัวเลือก (0-5): ").strip()
            print()
            
            if choice == "0":
                print("👋 ขอบคุณที่ใช้งาน Thai Food Recipe Chatbot!")
                break
            
            elif choice == "1":
                test_improved_api()
            
            elif choice == "2":
                run_demo()
            
            elif choice == "3":
                start_app()
            
            elif choice == "4":
                install_packages()
            
            elif choice == "5":
                print("🔧 ตรวจสอบระบบทั้งหมด...")
                check_python_version()
                check_files()
                install_packages()
                test_improved_api()
                print("✅ การตรวจสอบเสร็จสิ้น")
            
            else:
                print("❌ ตัวเลือกไม่ถูกต้อง กรุณาเลือก 0-5")
            
            # รอให้ผู้ใช้กดปุ่มก่อนแสดงเมนูใหม่
            if choice != "0":
                input("\n⏎ กด Enter เพื่อกลับไปที่เมนูหลัก...")
                print()
        
        except KeyboardInterrupt:
            print("\n\n👋 ออกจากโปรแกรม")
            break
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาด: {str(e)}")
            input("\n⏎ กด Enter เพื่อกลับไปที่เมนูหลัก...")

if __name__ == "__main__":
    main()
