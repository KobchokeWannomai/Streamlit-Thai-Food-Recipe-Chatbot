import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import pickle
import re
import requests
import time
import json
from difflib import SequenceMatcher
from nutrition_analyzer import NutritionAnalyzer
from config import Config

# การกำหนดค่าหน้าเว็บ
st.set_page_config(
    page_title="แชทบอทสูตรอาหารไทย",
    page_icon="🍲",
    layout="wide"
)

# ตั้งค่าฟอนต์ไทยและ CSS ปรับปรุงแล้ว
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@400;700&display=swap');
    html, body, [class*="st-"] {
        font-family: 'Sarabun', sans-serif !important;
    }
    .nutrition-card {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 12px;
        border-left: 5px solid #4CAF50;
        margin: 15px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        width: 100%;
        box-sizing: border-box;
    }
    .nutrition-item {
        display: inline-block;
        margin: 5px 10px;
        padding: 8px 15px;
        background-color: #e8f5e8;
        border-radius: 20px;
        font-size: 0.9em;
        font-weight: 500;
    }
    .vitamin-mineral {
        display: inline;
        color: #555;
        margin-top: 10px;
        line-height: 1.8;
    }
    .vitamin-mineral-label {
        display: inline;
        font-weight: 600;
        margin-right: 10px;
    }
    .recipe-card {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .recipe-title {
        font-size: 1.8em;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 15px;
    }
    .section-title {
        font-size: 1.2em;
        font-weight: 600;
        color: #34495e;
        margin: 15px 0 10px 0;
        padding-bottom: 5px;
        border-bottom: 2px solid #e0e0e0;
    }
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-connected {
        background-color: #4CAF50;
    }
    .status-disconnected {
        background-color: #f44336;
    }
    .status-testing {
        background-color: #ff9800;
    }
    .scroll-to-bottom-btn {
        position: fixed !important;
        bottom: 120px !important;
        right: 30px !important;
        z-index: 99999 !important;
        background: linear-gradient(135deg, #4CAF50, #45a049) !important;
        color: white !important;
        border: none !important;
        border-radius: 50% !important;
        width: 60px !important;
        height: 60px !important;
        cursor: pointer !important;
        box-shadow: 0 4px 20px rgba(76, 175, 80, 0.4) !important;
        font-size: 24px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        user-select: none !important;
        backdrop-filter: blur(10px) !important;
    }
    .scroll-to-bottom-btn:hover {
        background: linear-gradient(135deg, #45a049, #3d8b40) !important;
        transform: scale(1.1) translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(76, 175, 80, 0.6) !important;
    }
    .scroll-to-bottom-btn:active {
        transform: scale(0.95) !important;
    }
    .similarity-score {
        background-color: #e3f2fd;
        color: #1976d2;
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 0.85em;
        font-weight: 600;
        margin-left: 10px;
    }
    .fuzzy-match-score {
        background-color: #fff3e0;
        color: #ef6c00;
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 0.85em;
        font-weight: 600;
        margin-left: 10px;
    }
    .exact-match-score {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 0.85em;
        font-weight: 600;
        margin-left: 10px;
    }
    /* ปรับปรุงการแสดงผลรายการ */
    ul, ol {
        margin-left: 20px;
        line-height: 1.8;
    }
    /* ปรับปรุงการแสดงผล metric */
    [data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    /* Animation สำหรับข้อความใหม่ */
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    .new-message {
        animation: slideInUp 0.5s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# เส้นทางของไฟล์
DATA_PATH = "thai_food_processed.csv"
EMBEDDINGS_PATH = "embeddings.pkl"
MODEL_PATH = "model"

class EnhancedFuzzyMatcher:
    """คลาสสำหรับจับคู่ข้อความที่คล้ายคลึงกันแบบขั้นสูง พร้อมรองรับการพิมพ์ผิดของเมนูไทย"""
    
    def __init__(self):
        # รายการเมนูอาหารไทยที่ครอบคลุมจากข้อมูลที่ให้มา
        self.thai_menu_variations = {
            'กุ้งทาพริกไทยกระเทียม': ['กุ้งทาพริกไทย', 'กุ้งผัดพริกไทย', 'กุ้งกระเทียม'],
            'ข้าวเม่าทอด': ['ข้าวเหม่าทอด', 'ข้าวเม่า', 'ข้าวหม้อทอด'],
            'เปรี้ยวหวานไข่ม้วน': ['เปรี้ยวหวาน', 'ไข่ม้วนเปรี้ยวหวาน', 'ไข่ม้วน'],
            'ไข่จ่อม': ['ไข่จ๋อม', 'ไข่ซ่อม', 'ไข่ดิบ'],
            'งบปลาทู': ['งบปลา', 'ปลาทูแกง', 'แกงปลาทู'],
            'น้ำพริกจิ้มผักดิบ': ['น้ำพริกผักดิบ', 'น้ำพริกจิ้ม', 'น้ำพริกผัก'],
            'ลอยน้ำดอกไม้สด': ['ลอยน้ำดอกไม้', 'ลอยน้ำ', 'ขนมลอยน้ำ'],
            'ยำไข่ปลาดุก': ['ยำไข่ปลา', 'ไข่ปลาดุกยำ', 'ยำไข่ดุก'],
            'ปลาทูทอดปรุง': ['ปลาทูทอด', 'ปลาทูปรุง', 'ปลาทูผัด'],
            'ต้มยำกะทิ': ['ต้มยำน้ำกะทิ', 'ต้มยำใส่กะทิ', 'ต้มยำขาว'],
            'ไก่ยำ': ['ยำไก่', 'ไก่ลาบ', 'ยำไก่สด'],
            'กล้วยบวชชี': ['กล้วยบุชชี', 'กล้วยชุบแป้ง', 'กล้วยทอด'],
            'แกงคั่วฟักทองกับกุ้งตะเข็บ': ['แกงคั่วฟักทอง', 'แกงคั่วกุ้ง', 'ฟักทองแกงคั่ว'],
            'ไส้กรอกหมู': ['ไส้กรอก', 'ไส้กรอกอีสาน', 'ไส้กรอกหมูสด'],
            'เมี่ยงปลาทู': ['เมี่ยงปลา', 'ปลาทูเมี่ยง', 'น้ำเมี่ยงปลาทู'],
            'นกพิราบตุ๋น': ['นกพิราบ', 'นกตุ๋น', 'พิราบตุ๋น'],
            'ปลากุเลาทอดปรุงหน้า': ['ปลากุเลาทอด', 'ปลากุเลา', 'กุเลาทอด'],
            'ห่อหมกหอยแมลงภู่': ['ห่อหมกหอย', 'หอยแมลงภู่ห่อหมก', 'ห่อหมก'],
            'งบไก่': ['แกงไก่', 'ไก่แกง', 'งบไก่ใส'],
            'ไข่กระจัง': ['ไข่กระจัง', 'ไข่กระจัด', 'ไข่ผัด'],
            'หน้าตั้งแขก': ['หน้าตั้ง', 'แขกหน้าตั้ง', 'หน้าตั้งผัด'],
            'ยำปลาหมึกสด': ['ยำปลาหมึก', 'ปลาหมึกยำ', 'ยำหมึก'],
            'ฟักตุ๋น': ['ฟักทองตุ๋น', 'ตุ๋นฟัก', 'ฟักต้ม'],
            'ยำถั่วพู': ['ยำถั่วพู', 'ถั่วพูยำ', 'ยำถั่ว'],
            'ซ้อสมะเขือเทศซุป': ['ซอสมะเขือเทศ', 'ซุปมะเขือเทศ', 'ซ้อสมะเขือ'],
            'ปลาทูชุบแป้งทอด': ['ปลาทูชุบแป้ง', 'ปลาทูทอดแป้ง', 'ปลาทูทอดกรอบ'],
            'บะหมี่ทรงเครื่อง': ['บะหมี่ใส่ของ', 'บะหมี่พิเศษ', 'บะหมี่ครบรส'],
            'มะตูมเชื่อม': ['มะตูม', 'มะตูมหวาน', 'มะตูมแช่อิ่ม'],
            'สังขยา': ['สังขยาใบเตย', 'สังขยาฟักทอง', 'ขนมสังขยา'],
            'กุ้งแห้งปรุงขิง': ['กุ้งแห้งผัดขิง', 'กุ้งแห้งใส่ขิง', 'กุ้งแห้งขิง'],
            'แกงเผ็ดน้ำมันหมู': ['แกงเผ็ดหมู', 'แกงเผ็ดใส่น้ำมันหมู', 'แกงเผ็ดไขมันหมู'],
            'ผัดต้นผักกาดดอง': ['ผัดผักกาดดอง', 'ต้นผักกาดผัด', 'ผักกาดดองผัด'],
            'ยำขมิ้นขาวกับกุ้งเค็ม': ['ยำขมิ้นขาว', 'ขมิ้นขาวยำ', 'ยำขมิ้น'],
            'เกี๊ยวกุ้ง': ['เกี๊ยวหอม', 'เกี้ยวกุ้ง', 'เกี๋ยวกุ้ง'],
            'ข้าวต้มน้ำวุ้น': ['ข้าวต้มวุ้น', 'ข้าวต้มใส', 'ข้าวต้มน้ำใส'],
            'หมี่หน้าเนื้อ': ['หมี่หน้า', 'หมี่ราดหน้าเนื้อ', 'หมี่ผัดหน้าเนื้อ'],
            'ขนมสาลี่โคโก้': ['ขนมสาลี่', 'สาลี่โคโก้', 'ขนมโคโก้'],
            'สาคูเปียก': ['ขนมสาคู', 'สาคูหวาน', 'สาคูน้ำกะทิ'],
            'ห่อหมกไข่': ['ห่อหมกไข่แดง', 'ไข่ห่อหมก', 'ห่อหมกไข่เจียว'],
            'แกงยา': ['แกงยาใต้', 'แกงยาปลา', 'แกงยาผัก'],
            'เนื้อเครื่องเทศทอด': ['เนื้อทอดเครื่องเทศ', 'เนื้อผัดเครื่องเทศ', 'เนื้อปรุงรส'],
            'ขนมต้มแดง': ['ขนมต้ม', 'ต้มแดง', 'ขนมไทยต้มแดง'],
            'ปลาอบ': ['ปลาย่าง', 'ปลาปิ้ง', 'ปลาเผา'],
            'ตับตุ๋น': ['ตับหมูตุ๋น', 'ตับไก่ตุ๋น', 'ตับตุ๋นกะทิ'],
            'ไก่หยอง': ['ไก่หยองใต้', 'ไก่ผัดพริกแกง', 'ไก่ใส่พริกแกง'],
            'สลัดหมูกรอบ': ['สลัดหมู', 'หมูกรอบสลัด', 'ยำหมูกรอบ'],
            'ไข่ดาว': ['ไข่ดาวกรอบ', 'ไข่ทอด', 'ไข่ฟูดาว'],
            'ยอดแคผัดกรอบ': ['ยอดแค', 'ผักยอดแค', 'แคผัด'],
            'ข้าวชวา': ['ข้าวชาววัง', 'ข้าวชาวบ้าน', 'ข้าวผัดชวา'],
            'มะเขือเทศกุ้งเผา': ['มะเขือเทศใส่กุ้ง', 'กุ้งเผามะเขือเทศ', 'มะเขือเทศผัดกุ้ง'],
            'ไข่ต้มปรุงจับฉ่าย': ['ไข่ต้มปรุงรส', 'ไข่ต้มใส่จับฉ่าย', 'ไข่ต้มผัก'],
            'ยำไข่ดาว': ['ไข่ดาวยำ', 'ยำไข่ทอด', 'ไข่ดาวผัด'],
            'นกปากซ่อมสับ': ['นกปากซ่อม', 'นกสับ', 'เนื้อนกสับ'],
            'แกงจืดชนิดตีน้ำมัน': ['แกงจืดใส', 'แกงจืดไม่ใส่กะทิ', 'แกงจืดเรียบ'],
            'กะหรี่พัฟฟ์': ['กะหรี่ปัฟฟ์', 'กะหรี่พาฟ', 'กะหรี่ย่าม'],
            'ปลาทูแนม': ['ปลาทูเค็ม', 'ปลาทูหมัก', 'ปลาทูดอง'],
            'ขนมกลีบลำดวน': ['ขนมกลีบ', 'กลีบลำดวน', 'ขนมไทยกลีบ'],
            'แกงเผ็ดหมู': ['แกงเผ็ดใส่หมู', 'หมูแกงเผ็ด', 'แกงเผ็ดเนื้อหมู'],
            'หมูแนมสด': ['หมูแนม', 'หมูเค็มสด', 'หมูดองสด'],
            'มะเขือยาวเครื่องเทศ': ['มะเขือยาวผัด', 'มะเขือยาวปรุงรส', 'มะเขือผัดเครื่องเทศ'],
            'ไส้กรอกข้าว': ['ไส้กรอกใส่ข้าว', 'ไส้กรอกอีสานข้าว', 'ไส้กรอกข้าวโพด'],
            'น้ำเมี่ยง': ['น้ำเมี่ยงปลา', 'เมี่ยงน้ำ', 'น้ำจิ้มเมี่ยง'],
            'ปลานึ่งกับมะเขือเทศ': ['ปลานึ่งมะเขือเทศ', 'ปลาใส่มะเขือเทศ', 'ปลานึ่งผัก'],
            'น้ำพริกพะม่า': ['น้ำพริกพะหม่า', 'พะม่า', 'น้ำพริกใต้'],
            'ไก่ต้มขนมจีน': ['ไก่ต้มใส', 'ขนมจีนไก่ต้ม', 'ไก่ต้มสด'],
            'ข้าวเม่าคลุก': ['ข้าวเหม่าคลุก', 'ข้าวเม่าผัด', 'ข้าวคลุกเครื่อง'],
            'กุ้งเผากับมะเขือเปราะ': ['กุ้งเผามะเขือ', 'กุ้งผัดมะเขือเปราะ', 'กุ้งใส่มะเขือ'],
            'มะละกอโถบรรจุใส้': ['มะละกอใส้', 'มะละกอไส้', 'มะละกอยัดไส้'],
            'พุดชาจีนเชื่อมไส้เกาลัด': ['พุดชาจีน', 'ขนมจีนหวาน', 'พุดชาเกาลัด'],
            'ข้าวต้มไข่': ['ข้าวต้มใส่ไข่', 'ข้าวต้มไข่ดาว', 'ข้าวต้มไข่เจียว'],
            'ปลาแนม': ['ปลาเค็ม', 'ปลาดอง', 'ปลาหมัก'],
            'แกงต้มกะทิฟักทอง': ['แกงต้มฟักทอง', 'ฟักทองแกงกะทิ', 'แกงฟักทองกะทิ'],
            'ละมุดมีใส้': ['ละมุดไส้', 'ละมุดยัดไส้', 'ละมุดใส้หวาน'],
            'บะหมี่สำเร็จ': ['บะหมี่กึ่งสำเร็จ', 'บะหมี่แพ็ค', 'บะหมี่ผัด'],
            'ก๋วยเตี๋ยวไส้ไข่': ['ก๋วยเตี๋ยวไข่', 'เส้นใส่ไข่', 'ก๋วยเตี๋ยวไข่ดาว'],
            'ต้มหน่อไม้ไผ่ตงกับหมู': ['ต้มหน่อไม้', 'หน่อไม้ต้มหมู', 'ต้มหน่อไผ่'],
            'ผัดห่วงอาลัย': ['ผัดห่วง', 'ห่วงอาลัยผัด', 'ผักห่วงผัด'],
            'ยำพริก': ['ยำพริกสด', 'พริกยำ', 'ยำพริกแห้ง'],
            'น้ำพริกเผา': ['พริกเผา', 'น้ำพริกเผาแห้ง', 'น้ำพริกเผาสด'],
            'หมูทอดเค็ม': ['หมูทอดกรอบ', 'หมูทอดแห้ง', 'หมูเค็มทอด'],
            'เต้าหู้ยี้ปรุงรส': ['เต้าหู้ยี้', 'เต้าหู้ผัด', 'เต้าหู้ปรุง'],
            'กล้วยทอด': ['กล้วยทอดกรอบ', 'กล้วยทอดแป้ง', 'กล้วยชุบแป้งทอด'],
            'แกงต้มส้ม': ['แกงส้ม', 'ต้มส้ม', 'แกงส้มใส'],
            'ต้มยำปลา': ['ต้มยำใส่ปลา', 'ปลาต้มยำ', 'ต้มยำปลาดุก'],
            'แกงเห็ดฟางกับมะเขือเทศ': ['แกงเห็ดฟาง', 'เห็ดฟางแกง', 'แกงเห็ดมะเขือ'],
            'ต้มโคล้งกุ้ง': ['ต้มโคล้ง', 'กุ้งต้มโคล้ง', 'ต้มโคล้งใส่กุ้ง'],
            'แกงจืดลูกชิ้นกับจีฉ่าย': ['แกงจืดลูกชิ้น', 'ลูกชิ้นแกงจืด', 'แกงจืดใส่จีฉ่าย'],
            'ไข่สามชั้น': ['ไข่สามชั้นผัด', 'หมูสามชั้นไข่', 'ไข่ผัดสามชั้น'],
            'มันผรั่งบดใส่ไส้': ['มันผรั่งไส้', 'มันผรั่งยัดไส้', 'มันผรั่งบด'],
            'ไข่ในรัง': ['ไข่ซ่อนรัง', 'ไข่รังนก', 'ไข่ทำรัง'],
            'ปลาโฉมตรู': ['ปลาโฉม', 'ปลาตรู', 'ปลาโฉมผัด'],
            'ไข่เค็มชั้น': ['ไข่เค็มทอด', 'ไข่เค็มผัด', 'ไข่เค็มปรุง'],
            'แกงเลียงขี้เหล็ก': ['แกงเลียง', 'ขี้เหล็กแกงเลียง', 'แกงเลียงผัก'],
            'ผัดคะน้า': ['คะน้าผัด', 'ผักคะน้าผัด', 'คะน้าใส่หมู'],
            'ปลาช่อนต้มเค็มกับก๋งฉ่าย': ['ปลาช่อนต้มเค็ม', 'ปลาช่อนต้ม', 'ปลาช่อนใส่ก๋งฉ่าย'],
            'ก๋วยเตี๋ยวผัด': ['ผัดเส้น', 'เส้นผัด', 'ก๋วยเตี๋ยวคั่ว'],
            'ไข่เค็มทอดกรอบ': ['ไข่เค็มทอด', 'ไข่เค็มกรอบ', 'ไข่เค็มฟู'],
            'แกงจืดต้นคะน้า': ['แกงจืดคะน้า', 'ต้นคะน้าแกงจืด', 'คะน้าต้มใส'],
            'สาเกเชื่อม': ['สาเกหวาน', 'สาเกแช่อิ่ม', 'สาเกขาว'],
            'ไข่สวรรค์': ['ไข่ฟ้า', 'ไข่สวรรค์ทอง', 'ไข่แสงสวรรค์'],
            'มักกะโรนีรังแตน': ['มักกะโรนี', 'รังแตนมักกะโรนี', 'พาสต้ารังแตน'],
            'น้ำพริกเครื่องสด': ['น้ำพริกสด', 'เครื่องสดน้ำพริก', 'น้ำพริกผักสด'],
            'ปลาทูร่องสวน': ['ปลาทูร่อง', 'ปลาทูสวน', 'ปลาทูใส่ผัก'],
            'ผัดกะเพรา': ['กะเพราผัด', 'ผัดใบกะเพรา', 'กะเพราหมูสับ'],
            'ไข่เจียว': ['ไข่เจียวฟู', 'ไข่เจียวกรอบ', 'ไข่เจียวใส่หอม'],
            'แกงเปลือกแตงโม': ['แกงเปลือกแตง', 'เปลือกแตงโมแกง', 'แกงแตงโม'],
            'ต้มยำหอยแมลงภู่': ['ต้มยำหอย', 'หอยแมลงภู่ต้มยำ', 'ต้มยำหอยใหญ่'],
            'แกงเลียง': ['แกงเลียงผัก', 'แกงเลียงกุ้ง', 'แกงเลียงใต้'],
            'แกงต้มกะทิฟันเขียว': ['แกงต้มฟันเขียว', 'ฟันเขียวแกงกะทิ', 'แกงฟันเขียว'],
            'ถั่วแนม': ['ถั่วเค็ม', 'ถั่วดอง', 'ถั่วหมัก'],
            'ผัดไข่ปลาตะเพียน': ['ไข่ปลาตะเพียนผัด', 'ไข่ปลาผัด', 'ตะเพียนไข่ผัด'],
            'ส้มตำแตงร้าน': ['ส้มตำแตง', 'แตงกวาส้มตำ', 'ส้มตำแตงกวา'],
            'ผัดคะน้ากับซีเซ็กฉ่าย': ['ผัดคะน้าซีเซ็ก', 'คะน้าผัดซีเซ็ก', 'ผัดคะน้าฉ่าย'],
            'ต้มโคล้ง': ['ต้มโคล้งผัก', 'โคล้งต้ม', 'ต้มโคล้งกุ้ง'],
            'ยำทวายสมัยใหม่': ['ยำทวาย', 'ทวายยำ', 'ยำทวายใหม่'],
            'ผัดผักกาดขาว': ['ผักกาดขาวผัด', 'ผัดผักกาด', 'ผักกาดผัด'],
            'ผัดหัวผักกาดเค็ม': ['หัวผักกาดผัด', 'ผักกาดเค็มผัด', 'ผัดหัวไชเท้า'],
            'ไข่ช่อนรูป': ['ไข่ช่อน', 'ไข่รูปช่อน', 'ไข่ทำรูป'],
            'ยำไข่เจียวเครื่องหมี่': ['ยำไข่เจียว', 'ไข่เจียวยำ', 'ยำไข่เจียวผัก'],
            'กุ้งแฝง': ['กุ้งแฝงใต้', 'กุ้งผัดแฝง', 'กุ้งปรุงแฝง'],
            'บี๊ฟที': ['บีฟสเต็ก', 'เนื้อทีโบน', 'เนื้อย่าง'],
            'กงเชียงสด': ['กงเชียง', 'ไส้กรอกจีน', 'กงเชียงทอด'],
            'ไข่ตุ๋น': ['ไข่ตุ๋นกะทิ', 'ไข่ตุ๋นหวาน', 'ไข่ตุ๋นนึ่ง'],
            'แกงต้มเค็ม': ['แกงต้มใส', 'ต้มเค็ม', 'แกงต้มผัก'],
            'กุ้งทอดปรุงรส': ['กุ้งทอด', 'กุ้งทอดกรอบ', 'กุ้งทอดเกลือ'],
            'ยำไข่แมงดา': ['ไข่แมงดายำ', 'ยำไข่มด', 'ไข่แมงดาผัด'],
            'ยำส้มโอ': ['ส้มโอยำ', 'ยำส้มโอกุ้ง', 'ส้มโอผัด'],
            'ไข่หวานฝอย': ['ไข่ฝอย', 'ไข่หวาน', 'ฝอยทอง'],
            'ฟักทองทอด': ['ฟักทองทอดกรอบ', 'ฟักทองชุบแป้ง', 'ฟักทองผัด'],
            'แกงไส้กรอกหมูแห้ง': ['แกงไส้กรอก', 'ไส้กรอกแกง', 'แกงหมูแห้ง'],
            'มะเขือเทศหน้านวล': ['มะเขือเทศหน้า', 'มะเขือเทศนวล', 'มะเขือเทศต้ม'],
            'ปลาแห้งปรุงกระเทียมดอง': ['ปลาแห้งปรุง', 'ปลาแห้งกระเทียม', 'ปลาแห้งผัด'],
            'ฉี่ฉู่เมืองปราณ': ['ฉี่ฉู่', 'เมืองปราณ', 'ขนมฉี่ฉู่'],
            'ทองม้วนเค็ม': ['ทองม้วน', 'ไข่ม้วนเค็ม', 'ทองม้วนคาว'],
            'เปลือกส้มโอแช่อิ่ม': ['เปลือกส้มโอ', 'ส้มโอแช่อิ่ม', 'เปลือกส้มโอหวาน'],
            'ยำทวาย': ['ทวายยำ', 'ยำทวายใต้', 'ยำผลไม้'],
            'ไข่น้อค': ['ไข่น้อคใต้', 'ไข่ย่าง', 'ไข่เผา'],
            'เมี่ยงฝัน': ['เมี่ยงหวาน', 'ฝันเมี่ยง', 'เมี่ยงขนม'],
            'ไข่ม้วน': ['ไข่ม้วนหวาน', 'ไข่ม้วนคาว', 'ไข่ผัดม้วน'],
            'แป้งจี่': ['ขนมแป้งจี่', 'แป้งจี่หวาน', 'แป้งย่าง'],
            'น้ำพริกปลาเค็ม': ['น้ำพริกปลา', 'ปลาเค็มน้ำพริก', 'น้ำพริกใส่ปลา'],
            'กงเชียงไก่นา': ['กงเชียงไก่', 'ไก่นากงเชียง', 'กงเชียงผัดไก่'],
            'ไข่ดาวหน้ากุ้ง': ['ไข่ดาวกุ้ง', 'กุ้งไข่ดาว', 'ไข่ดาวใส่กุ้ง'],
            'น้ำพริกปูเค็ม': ['น้ำพริกปู', 'ปูเค็มน้ำพริก', 'น้ำพริกใส่ปู'],
            'เต้าเจี้ยวปรุงรส': ['เต้าเจี้ยว', 'เต้าเจี้ยวผัด', 'เต้าเจี้ยวหวาน'],
            'ขนมจีบหมูสับ': ['ขนมจีบ', 'หมูสับจีบ', 'ขนมจีบหมู'],
            'แกงส้มถั่วฝักยาว': ['แกงส้มถั่ว', 'ถั่วฝักยาวแกงส้ม', 'แกงส้มใส่ถั่ว'],
            'แกงต้มหมูกับสัปรส': ['แกงต้มหมู', 'หมูแกงต้ม', 'แกงต้มสัปรส'],
            'ผัดเต้าหู้เหลือง': ['เต้าหู้เหลืองผัด', 'ผัดเต้าหู้', 'เต้าหู้ผัด'],
            'ไก่ทันสมัย': ['ไก่สมัยใหม่', 'ไก่ผัดทันสมัย', 'ไก่ปรุงใหม่'],
            'เนื้อผัดเทียมแหนม': ['เนื้อเทียมแหนม', 'เนื้อผัดแหนม', 'เนื้อใส่แหนม'],
            'ไข่น้อคอีกอย่างหนึ่ง': ['ไข่น้อคใหม่', 'ไข่น้อคพิเศษ', 'ไข่น้อคแปลก'],
            'น้ำพริกก้อย': ['พริกก้อย', 'น้ำพริกก้อยใต้', 'น้ำพริกผักก้อย'],
            'ขนมเปียกปูน': ['ขนมเปียก', 'เปียกปูน', 'ขนมไทยเปียก'],
            'น้ำเต้าบรรจุไส้': ['น้ำเต้าไส้', 'น้ำเต้าใส้', 'น้ำเต้ายัดไส้'],
            'น้ำพริกไข่เค็ม': ['น้ำพริกใส่ไข่เค็ม', 'ไข่เค็มน้ำพริก', 'น้ำพริกไข่']
        }
        
        # การแก้ไขการพิมพ์ผิดทั่วไป
        self.common_typos = {
            'กระเพรา': 'กะเพรา',
            'ผัดกระเพรา': 'ผัดกะเพรา',
            'ต้มยำ': 'ต้มยำ',
            'ต้มยํา': 'ต้มยำ',
            'มัสมั่น': 'มัสมั่น',
            'มัสมัน': 'มัสมั่น',
            'เขียวหวาน': 'เขียวหวาน',
            'แกงเผ็ด': 'แกงเผ็ด',
            'แกงเปรียว': 'แกงเผ็ด',
            'ส้มตำ': 'ส้มตำ',
            'ส้มตํา': 'ส้มตำ',
            'ยำวุ้นเส้น': 'ยำวุ้นเส้น',
            'ยำวุนเส้น': 'ยำวุ้นเส้น',
            'ผัดไทย': 'ผัดไทย',
            'ผัดไท': 'ผัดไทย',
            'ไข่เจียว': 'ไข่เจียว',
            'ไข่เยียว': 'ไข่เจียว',
            'ไข่ดาว': 'ไข่ดาว',
            'ลาบหมู': 'ลาบหมู',
            'ลาป': 'ลาบ',
            'กุ้งทาพริก': 'กุ้งทาพริกไทยกระเทียม',
            'ข้าวเหม่า': 'ข้าวเม่าทอด',
            'งบปลา': 'งบปลาทู',
            'ยำไข่ปลา': 'ยำไข่ปลาดุก',
            'ปลาทูทอด': 'ปลาทูทอดปรุง',
            'ต้มยำกะทิ': 'ต้มยำกะทิ',
            'กล้วยบุชชี': 'กล้วยบวชชี',
            'แกงคั่วฟักทอง': 'แกงคั่วฟักทองกับกุ้งตะเข็บ',
            'เมี่ยงปลา': 'เมี่ยงปลาทู',
            'ห่อหมกหอย': 'ห่อหมกหอยแมลงภู่',
            'ยำปลาหมึก': 'ยำปลาหมึกสด',
            'ยำถั่ว': 'ยำถั่วพู',
            'บะหมี่ทรง': 'บะหมี่ทรงเครื่อง',
            'เกี๊ยวกุ้ง': 'เกี๊ยวกุ้ง',
            'หมี่หน้า': 'หมี่หน้าเนื้อ',
            'สาคู': 'สาคูเปียก',
            'แกงยา': 'แกงยา',
            'ขนมต้ม': 'ขนมต้มแดง',
            'สลัดหมู': 'สลัดหมูกรอบ',
            'ยอดแค': 'ยอดแคผัดกรอบ',
            'มะเขือเทศกุ้ง': 'มะเขือเทศกุ้งเผา',
            'ยำไข่ดาว': 'ยำไข่ดาว',
            'กะหรี่': 'กะหรี่พัฟฟ์',
            'ขนมกลีบ': 'ขนมกลีบลำดวน',
            'หมูแนม': 'หมูแนมสด',
            'มะเขือยาว': 'มะเขือยาวเครื่องเทศ',
            'ไส้กรอกข้าว': 'ไส้กรอกข้าว',
            'ปลานึ่ง': 'ปลานึ่งกับมะเขือเทศ',
            'น้ำพริกพะม่า': 'น้ำพริกพะม่า',
            'ไก่ต้ม': 'ไก่ต้มขนมจีน',
            'กุ้งเผา': 'กุ้งเผากับมะเขือเปราะ',
            'พุดชาจีน': 'พุดชาจีนเชื่อมไส้เกาลัด',
            'ข้าวต้มไข่': 'ข้าวต้มไข่',
            'ปลาแนม': 'ปลาแนม',
            'แกงต้มกะทิ': 'แกงต้มกะทิฟักทอง',
            'ละมุด': 'ละมุดมีใส้',
            'บะหมี่สำเร็จ': 'บะหมี่สำเร็จ',
            'ก๋วยเตี๋ยวไส้': 'ก๋วยเตี๋ยวไส้ไข่',
            'ต้มหน่อไม้': 'ต้มหน่อไม้ไผ่ตงกับหมู',
            'ผัดห่วง': 'ผัดห่วงอาลัย',
            'ยำพริก': 'ยำพริก',
            'หมูทอดเค็ม': 'หมูทอดเค็ม',
            'เต้าหู้ยี้': 'เต้าหู้ยี้ปรุงรส',
            'กล้วยทอด': 'กล้วยทอด',
            'แกงต้มส้ม': 'แกงต้มส้ม',
            'ต้มยำปลา': 'ต้มยำปลา',
            'แกงเห็ดฟาง': 'แกงเห็ดฟางกับมะเขือเทศ',
            'ต้มโคล้งกุ้ง': 'ต้มโคล้งกุ้ง',
            'แกงจืดลูกชิ้น': 'แกงจืดลูกชิ้นกับจีฉ่าย',
            'ไข่สามชั้น': 'ไข่สามชั้น',
            'มันผรั่ง': 'มันผรั่งบดใส่ไส้',
            'ไข่ในรัง': 'ไข่ในรัง',
            'ปลาโฉม': 'ปลาโฉมตรู',
            'ไข่เค็มชั้น': 'ไข่เค็มชั้น',
            'แกงเลียงขี้เหล็ก': 'แกงเลียงขี้เหล็ก',
            'ผัดคะน้า': 'ผัดคะน้า',
            'ปลาช่อนต้ม': 'ปลาช่อนต้มเค็มกับก๋งฉ่าย',
            'ก๋วยเตี๋ยวผัด': 'ก๋วยเตี๋ยวผัด',
            'ไข่เค็มทอด': 'ไข่เค็มทอดกรอบ',
            'แกงจืดต้นคะน้า': 'แกงจืดต้นคะน้า',
            'สาเก': 'สาเกเชื่อม',
            'ไข่สวรรค์': 'ไข่สวรรค์',
            'มักกะโรนี': 'มักกะโรนีรังแตน',
            'น้ำพริกเครื่อง': 'น้ำพริกเครื่องสด',
            'ปลาทูร่อง': 'ปลาทูร่องสวน',
            'แกงเปลือกแตง': 'แกงเปลือกแตงโม',
            'ต้มยำหอย': 'ต้มยำหอยแมลงภู่',
            'แกงเลียง': 'แกงเลียง',
            'แกงต้มกะทิฟัน': 'แกงต้มกะทิฟันเขียว',
            'ถั่วแนม': 'ถั่วแนม',
            'ผัดไข่ปลา': 'ผัดไข่ปลาตะเพียน',
            'ส้มตำแตง': 'ส้มตำแตงร้าน',
            'ผัดคะน้าซี': 'ผัดคะน้ากับซีเซ็กฉ่าย',
            'ต้มโคล้ง': 'ต้มโคล้ง',
            'ยำทวายสมัย': 'ยำทวายสมัยใหม่',
            'ผัดผักกาดขาว': 'ผัดผักกาดขาว',
            'ผัดหัวผักกาด': 'ผัดหัวผักกาดเค็ม',
            'ไข่ช่อน': 'ไข่ช่อนรูป',
            'ยำไข่เจียวเครื่อง': 'ยำไข่เจียวเครื่องหมี่',
            'กุ้งแฝง': 'กุ้งแฝง',
            'บี๊ฟที': 'บี๊ฟที',
            'กงเชียงสด': 'กงเชียงสด',
            'ไข่ตุ๋น': 'ไข่ตุ๋น',
            'แกงต้มเค็ม': 'แกงต้มเค็ม',
            'กุ้งทอดปรุง': 'กุ้งทอดปรุงรส',
            'ยำไข่แมงดา': 'ยำไข่แมงดา',
            'ยำส้มโอ': 'ยำส้มโอ',
            'ไข่หวานฝอย': 'ไข่หวานฝอย',
            'ฟักทองทอด': 'ฟักทองทอด',
            'แกงไส้กรอก': 'แกงไส้กรอกหมูแห้ง',
            'มะเขือเทศหน้า': 'มะเขือเทศหน้านวล',
            'ปลาแห้งปรุง': 'ปลาแห้งปรุงกระเทียมดอง',
            'ฉี่ฉู่': 'ฉี่ฉู่เมืองปราณ',
            'ทองม้วนเค็ม': 'ทองม้วนเค็ม',
            'เปลือกส้มโอ': 'เปลือกส้มโอแช่อิ่ม',
            'ยำทวาย': 'ยำทวาย',
            'ไข่น้อค': 'ไข่น้อค',
            'เมี่ยงฝัน': 'เมี่ยงฝัน',
            'แป้งจี่': 'แป้งจี่',
            'น้ำพริกปลาเค็ม': 'น้ำพริกปลาเค็ม',
            'กงเชียงไก่': 'กงเชียงไก่นา',
            'ไข่ดาวหน้า': 'ไข่ดาวหน้ากุ้ง',
            'น้ำพริกปูเค็ม': 'น้ำพริกปูเค็ม',
            'เต้าเจี้ยวปรุง': 'เต้าเจี้ยวปรุงรส',
            'ขนมจีบหมู': 'ขนมจีบหมูสับ',
            'แกงส้มถั่ว': 'แกงส้มถั่วฝักยาว',
            'แกงต้มหมู': 'แกงต้มหมูกับสัปรส',
            'ผัดเต้าหู้เหลือง': 'ผัดเต้าหู้เหลือง',
            'ไก่ทันสมัย': 'ไก่ทันสมัย',
            'เนื้อผัดเทียม': 'เนื้อผัดเทียมแหนม',
            'ไข่น้อคอีก': 'ไข่น้อคอีกอย่างหนึ่ง',
            'น้ำพริกก้อย': 'น้ำพริกก้อย',
            'ขนมเปียกปูน': 'ขนมเปียกปูน',
            'น้ำเต้าบรรจุ': 'น้ำเต้าบรรจุไส้',
            'น้ำพริกไข่เค็ม': 'น้ำพริกไข่เค็ม'
        }
    
    def calculate_similarity(self, s1, s2):
        """คำนวณความคล้ายคลึงระหว่างสองสตริงด้วยวิธีการหลายแบบ"""
        # ความคล้ายคลึงพื้นฐาน
        basic_similarity = SequenceMatcher(None, s1.lower(), s2.lower()).ratio()
        
        # ความคล้ายคลึงแบบละเว้นพื้นที่ว่าง
        s1_no_space = re.sub(r'\s+', '', s1.lower())
        s2_no_space = re.sub(r'\s+', '', s2.lower())
        no_space_similarity = SequenceMatcher(None, s1_no_space, s2_no_space).ratio()
        
        # ความคล้ายคลึงแบบคำ
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())
        if words1 and words2:
            word_similarity = len(words1.intersection(words2)) / len(words1.union(words2))
        else:
            word_similarity = 0
        
        # คำนวณคะแนนรวม
        final_score = max(basic_similarity, no_space_similarity, word_similarity * 0.8)
        
        return final_score
    
    def fix_common_typos(self, text):
        """แก้ไขการพิมพ์ผิดที่พบบ่อย"""
        fixed_text = text
        for typo, correct in self.common_typos.items():
            if typo in fixed_text:
                fixed_text = fixed_text.replace(typo, correct)
        
        return fixed_text
    
    def find_menu_variations(self, query):
        """หาเมนูที่ตรงกับหรือคล้ายกับคำค้นหา"""
        query_lower = query.lower().strip()
        matches = []
        
        # ตรวจสอบการตรงกันแบบตรงตัว
        for main_menu, variations in self.thai_menu_variations.items():
            if query_lower == main_menu.lower():
                matches.append({
                    'menu': main_menu,
                    'similarity': 1.0,
                    'match_type': 'exact'
                })
            elif query_lower in [v.lower() for v in variations]:
                matches.append({
                    'menu': main_menu,
                    'similarity': 0.95,
                    'match_type': 'variation'
                })
            else:
                # ตรวจสอบความคล้ายคลึง
                main_similarity = self.calculate_similarity(query_lower, main_menu.lower())
                if main_similarity >= 0.7:
                    matches.append({
                        'menu': main_menu,
                        'similarity': main_similarity,
                        'match_type': 'fuzzy_main'
                    })
                
                # ตรวจสอบกับรูปแบบต่างๆ
                for variation in variations:
                    var_similarity = self.calculate_similarity(query_lower, variation.lower())
                    if var_similarity >= 0.7:
                        matches.append({
                            'menu': main_menu,
                            'similarity': var_similarity,
                            'match_type': 'fuzzy_variation'
                        })
        
        # เรียงลำดับตามความคล้ายคลึง
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        # ลบรายการซ้ำ
        seen_menus = set()
        unique_matches = []
        for match in matches:
            if match['menu'] not in seen_menus:
                unique_matches.append(match)
                seen_menus.add(match['menu'])
        
        return unique_matches[:5]  # คืนค่าสูงสุด 5 ผลลัพธ์
    
    def find_best_match(self, query, candidates, threshold=0.6):
        """หาผลลัพธ์ที่ตรงกันมากที่สุดจากรายการผู้สมัคร"""
        # แก้ไขการพิมพ์ผิดก่อน
        query = self.fix_common_typos(query.lower())
        
        # หาเมนูที่ตรงกับรูปแบบต่างๆ ก่อน
        menu_matches = self.find_menu_variations(query)
        
        best_matches = []
        
        # เพิ่มผลลัพธ์จากการจับคู่เมนู
        for menu_match in menu_matches:
            # หาเมนูนี้ในรายการผู้สมัคร
            for i, candidate in enumerate(candidates):
                if menu_match['menu'].lower() in candidate.lower() or candidate.lower() in menu_match['menu'].lower():
                    best_matches.append({
                        'index': i,
                        'text': candidate,
                        'similarity': menu_match['similarity'],
                        'match_type': f"menu_{menu_match['match_type']}"
                    })
        
        # เพิ่มการจับคู่แบบทั่วไป
        for i, candidate in enumerate(candidates):
            candidate_clean = candidate.lower()
            similarity = self.calculate_similarity(query, candidate_clean)
            
            if similarity >= threshold:
                # ตรวจสอบว่าไม่ซ้ำกับที่มีอยู่แล้ว
                is_duplicate = any(match['index'] == i for match in best_matches)
                if not is_duplicate:
                    best_matches.append({
                        'index': i,
                        'text': candidate,
                        'similarity': similarity,
                        'match_type': 'fuzzy'
                    })
        
        # เรียงลำดับตามความคล้ายคลึง
        best_matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        return best_matches

class APIManager:
    """จัดการการเชื่อมต่อ API ต่างๆ"""
    
    def __init__(self):
        self.usda_status = False
        self.nutritionix_status = False
        self.external_recipe_status = False
        
    def test_usda_connection(self, api_key):
        """ทดสอบการเชื่อมต่อ USDA API"""
        if not api_key or api_key == "your_usda_api_key_here":
            return False
            
        try:
            url = "https://api.nal.usda.gov/fdc/v1/foods/search"
            params = {
                "api_key": api_key,
                "query": "apple",
                "pageSize": 1
            }
            response = requests.get(url, params=params, timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def test_nutritionix_connection(self, app_id, api_key):
        """ทดสอบการเชื่อมต่อ Nutritionix API"""
        if not app_id or not api_key or app_id == "your_nutritionix_app_id_here":
            return False
            
        try:
            url = "https://trackapi.nutritionix.com/v2/search/instant"
            headers = {
                'x-app-id': app_id,
                'x-app-key': api_key,
                'Content-Type': 'application/json'
            }
            params = {'query': 'apple'}
            response = requests.get(url, headers=headers, params=params, timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def get_external_recipe_data(self, recipe_name):
        """ดึงข้อมูลสูตรอาหารจาก API ภายนอก (จำลอง)"""
        external_recipes = {
            "ไข่เจียว": {
                "ingredients": [
                    {"name": "ไข่ไก่", "amount": 2, "unit": "ฟอง"},
                    {"name": "น้ำมันพืช", "amount": 3, "unit": "ช้อนโต๊ะ", "cooking_only": True, "consumed": 0.3},
                    {"name": "เกลือ", "amount": 0.5, "unit": "ช้อนชา"},
                    {"name": "พริกไทย", "amount": 0.25, "unit": "ช้อนชา"}
                ],
                "nutrition_adjustments": {
                    "oil_absorption": 0.1
                }
            },
            "ไข่ดาว": {
                "ingredients": [
                    {"name": "ไข่ไก่", "amount": 1, "unit": "ฟอง"},
                    {"name": "น้ำมันพืช", "amount": 2, "unit": "ช้อนโต๊ะ", "cooking_only": True, "consumed": 0.2}
                ],
                "nutrition_adjustments": {
                    "oil_absorption": 0.2
                }
            },
            "ผัดกะเพรา": {
                "ingredients": [
                    {"name": "หมูสับ", "amount": 200, "unit": "กรัม"},
                    {"name": "ใบกะเพรา", "amount": 1, "unit": "ถ้วย"},
                    {"name": "พริกขี้หนู", "amount": 5, "unit": "เม็ด"},
                    {"name": "กระเทียม", "amount": 5, "unit": "กลีบ"},
                    {"name": "น้ำปลา", "amount": 2, "unit": "ช้อนโต๊ะ"},
                    {"name": "น้ำตาล", "amount": 1, "unit": "ช้อนชา"},
                    {"name": "น้ำมันพืช", "amount": 2, "unit": "ช้อนโต๊ะ", "cooking_only": True, "consumed": 0.7}
                ],
                "nutrition_adjustments": {
                    "oil_absorption": 0.7
                }
            }
        }
        
        return external_recipes.get(recipe_name)

class EnhancedNutritionAnalyzer(NutritionAnalyzer):
    """ตัววิเคราะห์โภชนาการที่ปรับปรุงแล้ว"""
    
    def __init__(self, usda_api_key=None, use_api_data=False, api_manager=None):
        super().__init__(usda_api_key)
        self.use_api_data = use_api_data
        self.api_manager = api_manager or APIManager()
        
    def analyze_recipe_enhanced(self, recipe_name, ingredients_text, use_external_data=False):
        """วิเคราะห์โภชนาการแบบปรับปรุง"""
        if use_external_data and self.api_manager:
            external_data = self.api_manager.get_external_recipe_data(recipe_name)
            if external_data:
                return self._analyze_with_external_data(recipe_name, external_data)
        
        return self.analyze_recipe(recipe_name, ingredients_text)
    
    def _analyze_with_external_data(self, recipe_name, external_data):
        """วิเคราะห์โภชนาการด้วยข้อมูลจาก API ภายนอก"""
        nutrition_data = {}
        
        for ingredient_info in external_data['ingredients']:
            ingredient_name = ingredient_info['name']
            amount = ingredient_info['amount']
            unit = ingredient_info['unit']
            
            is_cooking_only = ingredient_info.get('cooking_only', False)
            consumed_ratio = ingredient_info.get('consumed', 1.0)
            
            base_nutrition = self.get_ingredient_nutrition(ingredient_name)
            
            if base_nutrition:
                if is_cooking_only:
                    effective_amount = amount * consumed_ratio
                else:
                    effective_amount = amount
                
                weight_grams = self.converter.convert_to_grams(effective_amount, unit, ingredient_name)
                multiplier = weight_grams / 100.0
                
                adjusted_nutrition = self._create_adjusted_nutrition(
                    ingredient_name, base_nutrition, multiplier, amount, unit, is_cooking_only
                )
                
                key = f"{ingredient_name} ({amount} {unit})"
                if is_cooking_only:
                    key += f" (ใช้ {consumed_ratio*100:.0f}%)"
                
                nutrition_data[key] = adjusted_nutrition
        
        total_nutrition = self.calculate_total_nutrition(nutrition_data)
        
        return {
            'recipe_name': recipe_name,
            'total_nutrition': self._nutrition_to_dict(total_nutrition),
            'ingredients': [{'ingredient': k, 'nutrition': v} for k, v in nutrition_data.items()],
            'ingredient_count': len(nutrition_data),
            'enhanced': True
        }
    
    def _create_adjusted_nutrition(self, name, base_nutrition, multiplier, amount, unit, is_cooking_only):
        """สร้างข้อมูลโภชนาการที่ปรับแล้ว"""
        from nutrition_analyzer import NutritionInfo
        
        return NutritionInfo(
            name=name,
            calories=base_nutrition.calories * multiplier,
            protein=base_nutrition.protein * multiplier,
            carbs=base_nutrition.carbs * multiplier,
            fat=base_nutrition.fat * multiplier,
            fiber=base_nutrition.fiber * multiplier,
            sugar=base_nutrition.sugar * multiplier,
            sodium=base_nutrition.sodium * multiplier,
            vitamin_a=base_nutrition.vitamin_a * multiplier,
            vitamin_c=base_nutrition.vitamin_c * multiplier,
            vitamin_d=base_nutrition.vitamin_d * multiplier,
            vitamin_e=base_nutrition.vitamin_e * multiplier,
            vitamin_k=base_nutrition.vitamin_k * multiplier,
            vitamin_b1=base_nutrition.vitamin_b1 * multiplier,
            vitamin_b2=base_nutrition.vitamin_b2 * multiplier,
            vitamin_b6=base_nutrition.vitamin_b6 * multiplier,
            vitamin_b12=base_nutrition.vitamin_b12 * multiplier,
            folate=base_nutrition.folate * multiplier,
            niacin=base_nutrition.niacin * multiplier,
            calcium=base_nutrition.calcium * multiplier,
            iron=base_nutrition.iron * multiplier,
            magnesium=base_nutrition.magnesium * multiplier,
            phosphorus=base_nutrition.phosphorus * multiplier,
            potassium=base_nutrition.potassium * multiplier,
            zinc=base_nutrition.zinc * multiplier,
            serving_size=f"{amount} {unit}" + (" (ใช้ในการทำอาหาร)" if is_cooking_only else "")
        )
    
    def _nutrition_to_dict(self, nutrition):
        """แปลง NutritionInfo เป็น dict"""
        return {
            'calories': nutrition.calories,
            'protein': nutrition.protein,
            'carbs': nutrition.carbs,
            'fat': nutrition.fat,
            'fiber': nutrition.fiber,
            'vitamins': {
                'วิตามิน A': nutrition.vitamin_a,
                'วิตามิน C': nutrition.vitamin_c,
                'วิตามิน D': nutrition.vitamin_d,
                'วิตามิน E': nutrition.vitamin_e,
                'วิตามิน K': nutrition.vitamin_k,
                'วิตามิน B1': nutrition.vitamin_b1,
                'วิตามิน B2': nutrition.vitamin_b2,
                'วิตามิน B6': nutrition.vitamin_b6,
                'วิตามิน B12': nutrition.vitamin_b12,
            },
            'minerals': {
                'แคลเซียม': nutrition.calcium,
                'เหล็ก': nutrition.iron,
                'แมกนีเซียม': nutrition.magnesium,
                'ฟอสฟอรัส': nutrition.phosphorus,
                'โพแทสเซียม': nutrition.potassium,
                'สังกะสี': nutrition.zinc,
                'โซเดียม': nutrition.sodium,
            }
        }

@st.cache_resource
def load_model():
    """โหลดหรือดาวน์โหลดโมเดล sentence transformer"""
    if os.path.exists(MODEL_PATH):
        return SentenceTransformer(MODEL_PATH)
    else:
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        os.makedirs(MODEL_PATH, exist_ok=True)
        model.save(MODEL_PATH)
        return model

@st.cache_resource
def load_api_manager():
    """โหลด API Manager"""
    return APIManager()

@st.cache_resource
def load_enhanced_fuzzy_matcher():
    """โหลด Enhanced Fuzzy Matcher"""
    return EnhancedFuzzyMatcher()

@st.cache_data
def load_data():
    """โหลดชุดข้อมูลอาหารไทย"""
    return pd.read_csv(DATA_PATH)

@st.cache_data
def get_embeddings(_model, data):
    """รับหรือคำนวณ embeddings สำหรับสูตรอาหารทั้งหมด"""
    if os.path.exists(EMBEDDINGS_PATH):
        with open(EMBEDDINGS_PATH, 'rb') as f:
            return pickle.load(f)
    else:
        texts = []
        for _, row in data.iterrows():
            combined_text = f"{row['name']} {row['ingredient']} {row['method']}"
            texts.append(combined_text)
        
        embeddings = _model.encode(texts)
        
        with open(EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump(embeddings, f)
        
        return embeddings

def create_settings_sidebar():
    """สร้างแถบการตั้งค่าปรับปรุงแล้ว"""
    with st.sidebar:
        st.title("⚙️ การตั้งค่า")
        
        # ส่วน API Configuration
        st.subheader("🔌 การเชื่อมต่อ API")
        
        # USDA API
        st.markdown("**USDA FoodData Central API**")
        usda_enabled = st.checkbox("เปิดใช้งาน USDA API", key="usda_enabled")
        
        if usda_enabled:
            usda_api_key = st.text_input(
                "USDA API Key",
                type="password",
                value=st.session_state.get("usda_api_key", ""),
                key="usda_api_key_input"
            )
            
            if st.button("ทดสอบการเชื่อมต่อ USDA", key="test_usda"):
                api_manager = load_api_manager()
                with st.spinner("กำลังทดสอบ..."):
                    status = api_manager.test_usda_connection(usda_api_key)
                    if status:
                        st.success("✅ เชื่อมต่อสำเร็จ")
                        st.session_state.usda_api_key = usda_api_key
                        st.session_state.usda_status = True
                    else:
                        st.error("❌ การเชื่อมต่อล้มเหลว")
                        st.session_state.usda_status = False
            
            status_usda = st.session_state.get("usda_status", False)
            status_class = "status-connected" if status_usda else "status-disconnected"
            status_text = "เชื่อมต่อแล้ว" if status_usda else "ไม่ได้เชื่อมต่อ"
            st.markdown(f'<div><span class="status-indicator {status_class}"></span>{status_text}</div>', unsafe_allow_html=True)
        
        st.divider()
        
        # Nutritionix API
        st.markdown("**Nutritionix API**")
        nutritionix_enabled = st.checkbox("เปิดใช้งาน Nutritionix API", key="nutritionix_enabled")
        
        if nutritionix_enabled:
            nutritionix_app_id = st.text_input(
                "Nutritionix App ID",
                value=st.session_state.get("nutritionix_app_id", ""),
                key="nutritionix_app_id_input"
            )
            nutritionix_api_key = st.text_input(
                "Nutritionix API Key",
                type="password",
                value=st.session_state.get("nutritionix_api_key", ""),
                key="nutritionix_api_key_input"
            )
            
            if st.button("ทดสอบการเชื่อมต่อ Nutritionix", key="test_nutritionix"):
                api_manager = load_api_manager()
                with st.spinner("กำลังทดสอบ..."):
                    status = api_manager.test_nutritionix_connection(nutritionix_app_id, nutritionix_api_key)
                    if status:
                        st.success("✅ เชื่อมต่อสำเร็จ")
                        st.session_state.nutritionix_app_id = nutritionix_app_id
                        st.session_state.nutritionix_api_key = nutritionix_api_key
                        st.session_state.nutritionix_status = True
                    else:
                        st.error("❌ การเชื่อมต่อล้มเหลว")
                        st.session_state.nutritionix_status = False
            
            status_nutritionix = st.session_state.get("nutritionix_status", False)
            status_class = "status-connected" if status_nutritionix else "status-disconnected"
            status_text = "เชื่อมต่อแล้ว" if status_nutritionix else "ไม่ได้เชื่อมต่อ"
            st.markdown(f'<div><span class="status-indicator {status_class}"></span>{status_text}</div>', unsafe_allow_html=True)
        
        st.divider()
        
        # ตัวเลือกที่รวมแล้ว
        st.subheader("⚡ การปรับปรุงคุณภาพ")
        
        enhanced_nutrition_calculation = st.checkbox(
            "ใช้การคำนวณโภชนาการขั้นสูง",
            help="รวมข้อมูลจาก API ภายนอกและคำนวณการบริโภควัตถุดิบอย่างแม่นยำ เช่น ปริมาณน้ำมันที่บริโภคจริงในการทอด",
            value=True,
            key="enhanced_nutrition_calculation"
        )
        
        enhanced_search = st.checkbox(
            "ใช้การค้นหาขั้นสูง",
            help="เปิดใช้งานการค้นหาที่รองรับการพิมพ์ผิดและการจับคู่ที่แม่นยำขึ้น",
            value=True,
            key="enhanced_search"
        )
        
        st.divider()
        
        # สถานะระบบ
        st.subheader("📊 สถานะระบบ")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("สูตรอาหาร", len(load_data()))
        with col2:
            st.metric("ฐานข้อมูลโภชนาการ", "150+ วัตถุดิบ")
        
        if "search_count" not in st.session_state:
            st.session_state.search_count = 0
        
        st.metric("การค้นหาในเซสชันนี้", st.session_state.search_count)
        
        # สถานะการค้นหาขั้นสูง
        if enhanced_search:
            st.info("🔍 การค้นหาขั้นสูงเปิดใช้งาน - รองรับการพิมพ์ผิดและการจับคู่ที่แม่นยำ")
        
        st.divider()
        
        # ข้อมูลเพิ่มเติม
        with st.expander("ℹ️ เกี่ยวกับ API"):
            st.markdown("""
            **USDA FoodData Central API:**
            - ฟรี ไม่มีค่าใช้จ่าย
            - ข้อมูลโภชนาการที่เชื่อถือได้
            - สมัครได้ที่: https://fdc.nal.usda.gov/api-guide.html
            
            **Nutritionix API:**
            - Free Plan: 200 requests/วัน
            - ข้อมูลอาหารที่หลากหลาย
            - สมัครได้ที่: https://www.nutritionix.com/business/api
            """)
        
        with st.expander("🔍 เกี่ยวกับการค้นหาขั้นสูง"):
            st.markdown("""
            **ฟีเจอร์การค้นหาขั้นสูง:**
            - รองรับการพิมพ์ผิดของเมนูอาหารไทย
            - จับคู่ชื่อเมนูที่คล้ายคลึงกัน
            - แก้ไขการพิมพ์ผิดอัตโนมัติ
            - ค้นหาจากชื่อเมนูและรูปแบบต่างๆ
            - รองรับการค้นหาแบบคำย่อ
            """)
        
        return {
            "usda_enabled": usda_enabled and st.session_state.get("usda_status", False),
            "nutritionix_enabled": nutritionix_enabled and st.session_state.get("nutritionix_status", False),
            "usda_api_key": st.session_state.get("usda_api_key", ""),
            "nutritionix_app_id": st.session_state.get("nutritionix_app_id", ""),
            "nutritionix_api_key": st.session_state.get("nutritionix_api_key", ""),
            "enhanced_nutrition_calculation": enhanced_nutrition_calculation,
            "use_external_recipe_data": enhanced_nutrition_calculation,
            "accurate_cooking_calculation": enhanced_nutrition_calculation,
            "enhanced_search": enhanced_search
        }

def format_ingredients(ingredients_text):
    """จัดรูปแบบรายการวัตถุดิบเพื่อการแสดงผลที่ดีขึ้น"""
    ingredients = ingredients_text.split('\n')
    formatted = "<ul style='line-height: 1.8;'>"
    for item in ingredients:
        if item.strip():
            cleaned_item = item.strip()
            if cleaned_item.startswith('- '):
                cleaned_item = cleaned_item[2:]
            formatted += f"<li>{cleaned_item}</li>"
    formatted += "</ul>"
    return formatted

def format_cooking_method(method_text):
    """จัดรูปแบบวิธีการทำอาหารเพื่อการแสดงผลที่ดีขึ้น"""
    has_numbered_steps = bool(re.search(r'^\s*\d+\.', method_text, re.MULTILINE))
    
    lines = method_text.split('\n')
    formatted = ""
    current_section = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('##'):
            if current_section:
                formatted += format_section(current_section, has_numbered_steps)
                current_section = []
            formatted += f"<div class='section-title' style='font-size: 1.0em; margin-top: 15px;'>{line.replace('##', '').strip()}</div>"
        elif line.startswith('#'):
            if current_section:
                formatted += format_section(current_section, has_numbered_steps)
                current_section = []
            formatted += f"<div class='section-title' style='font-size: 1.0em; margin-top: 15px;'>{line.replace('#', '').strip()}</div>"
        elif line.startswith('**หมายเหตุ'):
            if current_section:
                formatted += format_section(current_section, has_numbered_steps)
                current_section = []
            formatted += f"<div style='margin-top: 15px; padding: 10px; background-color: #f9f9f9; border-left: 3px solid #ffa500;'><strong>หมายเหตุ</strong> {line.replace('**หมายเหตุ**', '').replace('**หมายเหตุ', '').strip()}</div>"
        else:
            current_section.append(line)
    
    if current_section:
        formatted += format_section(current_section, has_numbered_steps)
    
    return formatted

def format_section(lines, has_numbered_steps):
    """จัดรูปแบบส่วนของวิธีทำ"""
    if not lines:
        return ""
    
    if has_numbered_steps:
        formatted = "<ol style='line-height: 1.8;'>"
        for line in lines:
            if re.match(r'^\d+\.', line):
                clean_line = re.sub(r'^\d+\.\s*', '', line)
                formatted += f"<li>{clean_line}</li>"
            else:
                if formatted.endswith("</li>"):
                    formatted = formatted[:-5] + f" {line}</li>"
                else:
                    formatted += f"<li>{line}</li>"
        formatted += "</ol>"
    else:
        text = ' '.join(lines)
        sentences = re.split(r'(?<=[ๆ.])\s+', text)
        
        formatted = "<div style='line-height: 1.8; text-align: justify;'>"
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                formatted += sentence.strip()
                if i < len(sentences) - 1:
                    formatted += " "
        formatted += "</div>"
    
    return formatted

def display_nutrition_info(nutrition_data):
    """แสดงข้อมูลโภชนาการในรูปแบบที่สวยงาม"""
    if not nutrition_data:
        return
    
    total_nutrition = nutrition_data.get('total_nutrition', {})
    is_enhanced = nutrition_data.get('enhanced', False)
    
    st.markdown("### 🥗 ข้อมูลโภชนาการ (ต่อหนึ่งที่)")
    
    if is_enhanced:
        st.info("📈 ข้อมูลที่ปรับปรุงแล้วด้วยการคำนวณขั้นสูง - คำนวณปริมาณการบริโภคจริงแล้ว")
    
    # สารอาหารหลัก
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("แคลอรี่", f"{total_nutrition.get('calories', 0):.0f} kcal")
    with col2:
        st.metric("โปรตีน", f"{total_nutrition.get('protein', 0):.1f} g")
    with col3:
        st.metric("คาร์โบไฮเดรต", f"{total_nutrition.get('carbs', 0):.1f} g")
    with col4:
        st.metric("ไขมัน", f"{total_nutrition.get('fat', 0):.1f} g")
    with col5:
        st.metric("ใยอาหาร", f"{total_nutrition.get('fiber', 0):.1f} g")
    
    # วิตามินและแร่ธาตุ
    vitamins = total_nutrition.get('vitamins', {})
    minerals = total_nutrition.get('minerals', {})
    
    if vitamins or minerals:
        nutrition_items = []
        
        vitamin_units = {
            'วิตามิน A': 'mcg', 'วิตามิน C': 'mg', 'วิตามิน D': 'mcg',
            'วิตามิน E': 'mg', 'วิตามิน K': 'mcg', 'วิตามิน B1': 'mg',
            'วิตามิน B2': 'mg', 'วิตามิน B6': 'mg', 'วิตามิน B12': 'mcg'
        }
        
        for vitamin, amount in vitamins.items():
            if amount > 0.1:
                unit = vitamin_units.get(vitamin, 'mg')
                nutrition_items.append(f"<span class='nutrition-item'>{vitamin}: {amount:.1f} {unit}</span>")
        
        mineral_units = {
            'แคลเซียม': 'mg', 'เหล็ก': 'mg', 'แมกนีเซียม': 'mg',
            'ฟอสฟอรัส': 'mg', 'โพแทสเซียม': 'mg', 'สังกะสี': 'mg', 'โซเดียม': 'mg'
        }
        
        for mineral, amount in minerals.items():
            if amount > 0.1:
                unit = mineral_units.get(mineral, 'mg')
                if mineral == 'โซเดียม':
                    nutrition_items.append(f"<span class='nutrition-item'>{mineral}: {amount:.0f} {unit}</span>")
                else:
                    nutrition_items.append(f"<span class='nutrition-item'>{mineral}: {amount:.1f} {unit}</span>")
        
        if nutrition_items:
            st.markdown(f"<div><span class='vitamin-mineral-label'>วิตามินและแร่ธาตุ:</span><span class='vitamin-mineral'>{''.join(nutrition_items)}</span></div>", unsafe_allow_html=True)
    
    # รายละเอียดวัตถุดิบ
    with st.expander("📋 รายละเอียดโภชนาการแต่ละวัตถุดิบ"):
        for ingredient_info in nutrition_data.get('ingredients', []):
            ingredient_full_name = ingredient_info['ingredient']
            nutrition = ingredient_info['nutrition']
            
            st.write(f"**{ingredient_full_name}**")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.write(f"แคลอรี่: {nutrition.calories:.0f} kcal")
            with col2:
                st.write(f"โปรตีน: {nutrition.protein:.1f} g")
            with col3:
                st.write(f"คาร์โบไฮเดรต: {nutrition.carbs:.1f} g")
            with col4:
                st.write(f"ไขมัน: {nutrition.fat:.1f} g")
            
            # แสดงวิตามินและแร่ธาตุที่มีค่ามากกว่า 0.1
            vitamin_mineral_text = []
            
            # เช็ควิตามิน
            if hasattr(nutrition, 'vitamin_a') and nutrition.vitamin_a > 0.1:
                vitamin_mineral_text.append(f"วิตามิน A: {nutrition.vitamin_a:.1f} mcg")
            if hasattr(nutrition, 'vitamin_c') and nutrition.vitamin_c > 0.1:
                vitamin_mineral_text.append(f"วิตามิน C: {nutrition.vitamin_c:.1f} mg")
            if hasattr(nutrition, 'vitamin_d') and nutrition.vitamin_d > 0.1:
                vitamin_mineral_text.append(f"วิตามิน D: {nutrition.vitamin_d:.1f} mcg")
            if hasattr(nutrition, 'vitamin_e') and nutrition.vitamin_e > 0.1:
                vitamin_mineral_text.append(f"วิตามิน E: {nutrition.vitamin_e:.1f} mg")
            if hasattr(nutrition, 'vitamin_k') and nutrition.vitamin_k > 0.1:
                vitamin_mineral_text.append(f"วิตามิน K: {nutrition.vitamin_k:.1f} mcg")
            if hasattr(nutrition, 'vitamin_b1') and nutrition.vitamin_b1 > 0.1:
                vitamin_mineral_text.append(f"วิตามิน B1: {nutrition.vitamin_b1:.1f} mg")
            if hasattr(nutrition, 'vitamin_b6') and nutrition.vitamin_b6 > 0.1:
                vitamin_mineral_text.append(f"วิตามิน B6: {nutrition.vitamin_b6:.1f} mg")
            if hasattr(nutrition, 'vitamin_b12') and nutrition.vitamin_b12 > 0.1:
                vitamin_mineral_text.append(f"วิตามิน B12: {nutrition.vitamin_b12:.1f} mcg")
            
            # เช็คแร่ธาตุ
            if hasattr(nutrition, 'calcium') and nutrition.calcium > 0.1:
                vitamin_mineral_text.append(f"แคลเซียม: {nutrition.calcium:.1f} mg")
            if hasattr(nutrition, 'iron') and nutrition.iron > 0.1:
                vitamin_mineral_text.append(f"เหล็ก: {nutrition.iron:.1f} mg")
            if hasattr(nutrition, 'sodium') and nutrition.sodium > 0.1:
                vitamin_mineral_text.append(f"โซเดียม: {nutrition.sodium:.0f} mg")
            if hasattr(nutrition, 'potassium') and nutrition.potassium > 0.1:
                vitamin_mineral_text.append(f"โพแทสเซียม: {nutrition.potassium:.0f} mg")
            if hasattr(nutrition, 'zinc') and nutrition.zinc > 0.1:
                vitamin_mineral_text.append(f"สังกะสี: {nutrition.zinc:.1f} mg")
            if hasattr(nutrition, 'phosphorus') and nutrition.phosphorus > 0.1:
                vitamin_mineral_text.append(f"ฟอสฟอรัส: {nutrition.phosphorus:.0f} mg")
            if hasattr(nutrition, 'magnesium') and nutrition.magnesium > 0.1:
                vitamin_mineral_text.append(f"แมกนีเซียม: {nutrition.magnesium:.0f} mg")
            
            if hasattr(nutrition, 'fiber') and nutrition.fiber > 0.1:
                vitamin_mineral_text.append(f"ใยอาหาร: {nutrition.fiber:.1f} g")
            
            if vitamin_mineral_text:
                st.write("สารอาหารอื่นๆ: " + ", ".join(vitamin_mineral_text))
            
            if hasattr(nutrition, 'sodium') and nutrition.sodium > 500:
                st.caption("⚠️ มีโซเดียมสูง")
                
            if is_enhanced and "(ใช้" in ingredient_full_name:
                st.caption("🔧 ปรับปรุงการคำนวณปริมาณการบริโภคแล้ว")
                
            st.divider()

def search_recipes_enhanced(query, model, data, embeddings, nutrition_analyzer, settings, top_k=5):
    """ฟังก์ชันค้นหาสูตรอาหารที่ปรับปรุงแล้วด้วย enhanced fuzzy matching"""
    query_lower = query.lower().strip()
    
    # โหลด Enhanced Fuzzy Matcher
    enhanced_matcher = load_enhanced_fuzzy_matcher()
    
    # ขยายคำค้นหาด้วยคำที่เกี่ยวข้อง
    query_expansions = {
        'ไข่': ['ไข่ไก่', 'ไข่เป็ด', 'ไข่ดาว', 'ไข่เจียว', 'ไข่ต้ม'],
        'หมู': ['เนื้อหมู', 'หมูสับ', 'หมูย่าง', 'หมูทอด'],
        'ไก่': ['เนื้อไก่', 'ไก่ย่าง', 'ไก่ทอด', 'ไก่ต้ม'],
        'กุ้ง': ['กุ้งนาง', 'กุ้งฝอย', 'กุ้งแห้ง'],
        'ผัด': ['ผัดไทย', 'ผัดกะเพรา', 'ผัดซีอิ๊ว'],
        'แกง': ['แกงเขียวหวาน', 'แกงเผ็ด', 'แกงส้ม', 'แกงมัสมั่น'],
        'ต้ม': ['ต้มยำ', 'ต้มข่า', 'ต้มจืด'],
        'ยำ': ['ยำวุ้นเส้น', 'ยำถั่วพู', 'ยำมะม่วง'],
        'ส้ม': ['ส้มตำ', 'ส้มตำไทย', 'ส้มตำปู'],
        'ลาบ': ['ลาบหมู', 'ลาบไก่', 'ลาบเนื้อ']
    }
    
    # เพิ่มการขยายคำค้นหา
    expanded_terms = [query_lower]
    for key, expansions in query_expansions.items():
        if key in query_lower:
            expanded_terms.extend(expansions)
    
    # สร้างรายการชื่อเมนูสำหรับ enhanced fuzzy matching
    recipe_names = data['name'].tolist()
    
    # ใช้ Enhanced Fuzzy Matcher
    if settings.get('enhanced_search', True):
        fuzzy_matches = enhanced_matcher.find_best_match(query, recipe_names, threshold=0.6)
    else:
        # ใช้ fuzzy matcher เดิม
        basic_matcher = FuzzyMatcher()
        fuzzy_matches = basic_matcher.find_best_match(query, recipe_names, threshold=0.6)
    
    # ค้นหาแบบตรงตัว (exact match)
    exact_matches = []
    
    for idx, row in data.iterrows():
        recipe_name = row['name'].lower()
        ingredients = row['ingredient'].lower()
        method = row['method'].lower()
        
        # ตรวจสอบการตรงกันแบบต่างๆ
        if query_lower == recipe_name:
            exact_matches.append({'index': idx, 'score': 1.0, 'match_type': 'exact_name'})
        elif query_lower in recipe_name or any(term in recipe_name for term in query_lower.split()):
            query_words = query_lower.split()
            matching_words = sum(1 for word in query_words if word in recipe_name)
            score = 0.9 * (matching_words / len(query_words))
            exact_matches.append({'index': idx, 'score': score, 'match_type': 'partial_name'})
        elif any(term in ingredients for term in expanded_terms):
            ingredient_words = ingredients.split()
            matching_ingredients = sum(1 for term in expanded_terms if term in ingredients)
            score = 0.7 * (matching_ingredients / len(expanded_terms))
            exact_matches.append({'index': idx, 'score': score, 'match_type': 'ingredient'})
    
    # ค้นหาแบบ semantic search
    search_text = ' '.join(expanded_terms)
    query_embedding = model.encode([search_text])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # รวมผลลัพธ์
    results = []
    used_indices = set()
    
    # เพิ่ม enhanced fuzzy matches ก่อน (มีคะแนนความคล้ายคลึงสูง)
    for match in fuzzy_matches[:3]:  # เอาแค่ 3 ผลลัพธ์ที่ดีที่สุด
        if len(results) >= top_k:
            break
            
        idx = match['index']
        if idx not in used_indices:
            recipe_name = data.iloc[idx]['name']
            ingredients = data.iloc[idx]['ingredient']
            
            nutrition_data = nutrition_analyzer.analyze_recipe_enhanced(
                recipe_name, ingredients,
                use_external_data=settings.get('use_external_recipe_data', False)
            ) if hasattr(nutrition_analyzer, 'analyze_recipe_enhanced') else nutrition_analyzer.analyze_recipe(recipe_name, ingredients)
            
            # กำหนดประเภทการจับคู่ที่ชัดเจนขึ้น
            if match.get('match_type', '').startswith('menu_'):
                match_type = 'menu_match'
            else:
                match_type = 'enhanced_fuzzy'
            
            results.append({
                'name': recipe_name,
                'similarity': match['similarity'],
                'match_type': match_type,
                'ingredients': ingredients,
                'method': data.iloc[idx]['method'],
                'nutrition': nutrition_data
            })
            used_indices.add(idx)
    
    # เพิ่ม exact matches
    for match in sorted(exact_matches, key=lambda x: x['score'], reverse=True):
        if len(results) >= top_k:
            break
            
        idx = match['index']
        if idx not in used_indices:
            recipe_name = data.iloc[idx]['name']
            ingredients = data.iloc[idx]['ingredient']
            
            nutrition_data = nutrition_analyzer.analyze_recipe_enhanced(
                recipe_name, ingredients,
                use_external_data=settings.get('use_external_recipe_data', False)
            ) if hasattr(nutrition_analyzer, 'analyze_recipe_enhanced') else nutrition_analyzer.analyze_recipe(recipe_name, ingredients)
            
            results.append({
                'name': recipe_name,
                'similarity': match['score'],
                'match_type': match['match_type'],
                'ingredients': ingredients,
                'method': data.iloc[idx]['method'],
                'nutrition': nutrition_data
            })
            used_indices.add(idx)
    
    # เพิ่มผลลัพธ์จาก semantic search ถ้าต้องการเพิ่มเติม
    if len(results) < top_k:
        threshold = 0.4
        top_indices = np.argsort(-similarities)
        
        for idx in top_indices:
            if len(results) >= top_k:
                break
                
            if idx not in used_indices and similarities[idx] > threshold:
                recipe_name = data.iloc[idx]['name']
                ingredients = data.iloc[idx]['ingredient']
                
                nutrition_data = nutrition_analyzer.analyze_recipe_enhanced(
                    recipe_name, ingredients,
                    use_external_data=settings.get('use_external_recipe_data', False)
                ) if hasattr(nutrition_analyzer, 'analyze_recipe_enhanced') else nutrition_analyzer.analyze_recipe(recipe_name, ingredients)
                
                results.append({
                    'name': recipe_name,
                    'similarity': similarities[idx],
                    'match_type': 'semantic',
                    'ingredients': ingredients,
                    'method': data.iloc[idx]['method'],
                    'nutrition': nutrition_data
                })
                used_indices.add(idx)
    
    return results

def search_by_nutrition_criteria(data, nutrition_analyzer, criteria, settings):
    """ค้นหาสูตรอาหารตามเกณฑ์โภชนาการ"""
    results = []
    
    for _, row in data.iterrows():
        recipe_name = row['name']
        ingredients = row['ingredient']
        
        nutrition_data = nutrition_analyzer.analyze_recipe_enhanced(
            recipe_name, ingredients,
            use_external_data=settings.get('use_external_recipe_data', False)
        ) if hasattr(nutrition_analyzer, 'analyze_recipe_enhanced') else nutrition_analyzer.analyze_recipe(recipe_name, ingredients)
        
        total_nutrition = nutrition_data.get('total_nutrition', {})
        
        # ตรวจสอบเกณฑ์
        match = True
        if 'max_calories' in criteria and total_nutrition.get('calories', 0) > criteria['max_calories']:
            match = False
        if 'min_calories' in criteria and total_nutrition.get('calories', 0) < criteria['min_calories']:
            match = False
        if 'min_protein' in criteria and total_nutrition.get('protein', 0) < criteria['min_protein']:
            match = False
        
        if match:
            results.append({
                'recipe_name': recipe_name,
                'calories': total_nutrition.get('calories', 0),
                'protein': total_nutrition.get('protein', 0),
                'carbs': total_nutrition.get('carbs', 0),
                'fat': total_nutrition.get('fat', 0),
                'fiber': total_nutrition.get('fiber', 0),
                'ingredients': ingredients,
                'method': row['method'],
                'nutrition': nutrition_data
            })
    
    results.sort(key=lambda x: x['calories'])
    return results

def detect_nutrition_search(query):
    """ตรวจจับว่าการค้นหาเป็นการค้นหาตามโภชนาการหรือไม่"""
    nutrition_keywords = [
        'แคลอรี่', 'แคลอรี', 'calorie', 'cal', 'kcal',
        'โปรตีน', 'protein', 'คาร์โบ', 'คาร์โบไฮเดรต', 'carb', 'carbohydrate',
        'ไขมัน', 'fat', 'ใยอาหาร', 'fiber', 'ลดน้ำหนัก', 'diet', 'healthy', 'เฮลธ์ตี้',
        'โภชนาการ', 'nutrition', 'วิตามิน', 'vitamin', 'แร่ธาตุ', 'mineral',
        'ต่ำ', 'สูง', 'น้อย', 'เยอะ', 'มาก', 'ไม่เกิน', 'มากกว่า', 'น้อยกว่า'
    ]
    
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in nutrition_keywords)

def extract_nutrition_criteria_from_text(query):
    """แยกเกณฑ์โภชนาการจากข้อความค้นหา"""
    criteria = {}
    query_lower = query.lower()
    
    # ค้นหาแคลอรี่
    calorie_patterns = [
        r'แคลอรี่.*?ไม่เกิน.*?(\d+)', r'ไม่เกิน.*?(\d+).*?แคลอรี่',
        r'แคลอรี่.*?น้อยกว่า.*?(\d+)', r'น้อยกว่า.*?(\d+).*?แคลอรี่',
        r'แคลอรี.*?ไม่เกิน.*?(\d+)', r'ไม่เกิน.*?(\d+).*?แคลอรี'
    ]
    
    for pattern in calorie_patterns:
        match = re.search(pattern, query_lower)
        if match:
            criteria['max_calories'] = int(match.group(1))
            break
    
    # ค้นหาโปรตีน
    protein_patterns = [
        r'โปรตีน.*?มากกว่า.*?(\d+)', r'มากกว่า.*?(\d+).*?โปรตีน',
        r'โปรตีน.*?สูง.*?(\d+)', r'โปรตีน.*?เยอะ.*?(\d+)', r'โปรตีน.*?อย่างน้อย.*?(\d+)'
    ]
    
    for pattern in protein_patterns:
        match = re.search(pattern, query_lower)
        if match:
            criteria['min_protein'] = int(match.group(1))
            break
    
    # เกณฑ์พื้นฐาน
    if 'ลดน้ำหนัก' in query_lower or 'diet' in query_lower:
        criteria.update({'max_calories': 400, 'min_protein': 15})
    elif 'แคลอรี่ต่ำ' in query_lower or 'แคลอรีต่ำ' in query_lower:
        criteria['max_calories'] = 300
    elif 'โปรตีนสูง' in query_lower:
        criteria['min_protein'] = 20
    elif 'เฮลธ์ตี้' in query_lower or 'healthy' in query_lower:
        criteria['max_calories'] = 350
    
    return criteria

def create_enhanced_scroll_button():
    """สร้างปุ่มเลื่อนที่ปรับปรุงแล้วพร้อมระบบ auto-scroll ที่เสถียร"""
    scroll_button_html = """
    <div id="scroll-to-bottom-container"></div>
    <script>
    // ตัวแปรสำหรับควบคุมการทำงาน
    let scrollBtnCreated = false;
    let autoScrollActive = false;
    let lastMessageCount = 0;
    let scrollCheckInterval = null;
    
    // สร้างปุ่มเลื่อน
    function createEnhancedScrollButton() {
        if (scrollBtnCreated) return;
        
        try {
            // ลบปุ่มเก่าถ้ามี
            const existingBtn = document.getElementById('enhanced-scroll-btn');
            if (existingBtn) {
                existingBtn.remove();
            }
            
            // สร้างปุ่มใหม่
            const scrollBtn = document.createElement('button');
            scrollBtn.id = 'enhanced-scroll-btn';
            scrollBtn.className = 'scroll-to-bottom-btn';
            scrollBtn.innerHTML = '↓';
            scrollBtn.title = 'เลื่อนไปข้อความล่าสุด';
            
            // เพิ่ม event listener
            scrollBtn.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                scrollToLatestMessage();
            });
            
            // เพิ่มปุ่มเข้าไปใน DOM
            document.body.appendChild(scrollBtn);
            scrollBtnCreated = true;
            
            console.log('✅ สร้างปุ่มเลื่อนขั้นสูงสำเร็จ');
        } catch (error) {
            console.error('❌ เกิดข้อผิดพลาดในการสร้างปุ่ม:', error);
        }
    }
    
    // ฟังก์ชันเลื่อนไปข้อความล่าสุด
    function scrollToLatestMessage() {
        try {
            // รอให้เนื้อหาโหลดเสร็จก่อน
            setTimeout(() => {
                // หาข้อความล่าสุด
                const chatMessages = document.querySelectorAll('[data-testid="stChatMessage"]');
                const expanders = document.querySelectorAll('[data-testid="stExpander"]');
                const allMessages = [...chatMessages, ...expanders];
                
                if (allMessages.length > 0) {
                    // เลื่อนไปข้อความล่าสุด
                    const lastMessage = allMessages[allMessages.length - 1];
                    lastMessage.scrollIntoView({ 
                        behavior: 'smooth', 
                        block: 'end',
                        inline: 'nearest'
                    });
                    console.log('📜 เลื่อนไปข้อความล่าสุดแล้ว');
                } else {
                    // ไม่มีข้อความ ให้เลื่อนไปด้านล่าง
                    window.scrollTo({
                        top: document.body.scrollHeight,
                        behavior: 'smooth'
                    });
                    console.log('📜 เลื่อนไปด้านล่างแล้ว');
                }
            }, 200);
        } catch (error) {
            console.error('❌ เกิดข้อผิดพลาดในการเลื่อน:', error);
            // Fallback
            window.scrollTo({
                top: document.body.scrollHeight,
                behavior: 'smooth'
            });
        }
    }
    
    // ตรวจสอบข้อความใหม่และ auto-scroll
    function checkForNewMessages() {
        try {
            const currentMessages = document.querySelectorAll('[data-testid="stChatMessage"]');
            const currentCount = currentMessages.length;
            
            if (currentCount > lastMessageCount && currentCount > 0) {
                console.log(`🔔 พบข้อความใหม่: ${currentCount} ข้อความ`);
                lastMessageCount = currentCount;
                
                // Auto-scroll หลังจากมีข้อความใหม่
                if (!autoScrollActive) {
                    autoScrollActive = true;
                    setTimeout(() => {
                        scrollToLatestMessage();
                        setTimeout(() => {
                            autoScrollActive = false;
                        }, 1000);
                    }, 500);
                }
            } else if (currentCount > 0) {
                lastMessageCount = currentCount;
            }
        } catch (error) {
            console.error('❌ เกิดข้อผิดพลาดในการตรวจสอบข้อความ:', error);
        }
    }
    
    // ตั้งค่า MutationObserver สำหรับตรวจจับการเปลี่ยนแปลง
    function setupAdvancedAutoScroll() {
        try {
            const observer = new MutationObserver(function(mutations) {
                let shouldScroll = false;
                
                mutations.forEach(function(mutation) {
                    if (mutation.type === 'childList') {
                        mutation.addedNodes.forEach(function(node) {
                            if (node.nodeType === 1) {
                                // ตรวจสอบว่ามีการเพิ่มข้อความใหม่หรือไม่
                                if (node.querySelector && 
                                    (node.querySelector('[data-testid="stChatMessage"]') ||
                                     node.getAttribute('data-testid') === 'stChatMessage' ||
                                     node.querySelector('[data-testid="stExpander"]') ||
                                     node.getAttribute('data-testid') === 'stExpander')) {
                                    shouldScroll = true;
                                }
                            }
                        });
                    }
                });
                
                if (shouldScroll && !autoScrollActive) {
                    autoScrollActive = true;
                    setTimeout(() => {
                        scrollToLatestMessage();
                        setTimeout(() => {
                            autoScrollActive = false;
                        }, 1500);
                    }, 300);
                }
            });
            
            // เริ่มการสังเกตการณ์
            observer.observe(document.body, {
                childList: true,
                subtree: true
            });
            
            console.log('🎯 ตั้งค่า auto-scroll ขั้นสูงแล้ว');
        } catch (error) {
            console.error('❌ เกิดข้อผิดพลาดในการตั้งค่า auto-scroll:', error);
        }
    }
    
    // เริ่มระบบตรวจสอบข้อความใหม่
    function startMessageMonitoring() {
        // ตรวจสอบทุก 1 วินาที
        scrollCheckInterval = setInterval(checkForNewMessages, 1000);
        console.log('⏰ เริ่มระบบตรวจสอบข้อความใหม่');
    }
    
    // ฟังก์ชันเริ่มต้น
    function initializeEnhancedScrollSystem() {
        createEnhancedScrollButton();
        setupAdvancedAutoScroll();
        startMessageMonitoring();
        
        // ตรวจสอบและสร้างปุ่มซ้ำทุก 5 วินาที
        setInterval(() => {
            if (!document.getElementById('enhanced-scroll-btn')) {
                scrollBtnCreated = false;
                createEnhancedScrollButton();
            }
        }, 5000);
    }
    
    // เรียกใช้งานเมื่อ DOM พร้อม
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            setTimeout(initializeEnhancedScrollSystem, 1000);
        });
    } else {
        // DOM โหลดเสร็จแล้ว
        setTimeout(initializeEnhancedScrollSystem, 1000);
    }
    
    // เมื่อออกจากหน้า ให้ทำความสะอาด
    window.addEventListener('beforeunload', function() {
        if (scrollCheckInterval) {
            clearInterval(scrollCheckInterval);
        }
    });
    </script>
    """
    
    st.components.v1.html(scroll_button_html, height=0)

def main():
    # สร้างแถบการตั้งค่า
    settings = create_settings_sidebar()
    
    # โหลดโมเดลและข้อมูล
    model = load_model()
    data = load_data()
    embeddings = get_embeddings(model, data)
    api_manager = load_api_manager()
    
    # สร้าง nutrition analyzer
    usda_api_key = settings.get('usda_api_key') if settings.get('usda_enabled') else None
    nutrition_analyzer = EnhancedNutritionAnalyzer(
        usda_api_key=usda_api_key,
        use_api_data=settings.get('usda_enabled') or settings.get('nutritionix_enabled'),
        api_manager=api_manager
    )
    
    # แอปหลัก
    st.title("🍲 แชทบอทสูตรอาหารไทย")
    st.markdown("**ค้นหาสูตรอาหารไทยพร้อมข้อมูลโภชนาการขั้นสูง** - ถามเกี่ยวกับวิธีทำอาหารไทยหรือค้นหาตามโภชนาการได้เลย!")
    
    # แสดงสถานะการค้นหาขั้นสูง
    if settings.get('enhanced_search', True):
        st.info("🔍 การค้นหาขั้นสูงเปิดใช้งาน - รองรับการพิมพ์ผิดและการจับคู่ที่แม่นยำของเมนูอาหารไทย")
    
    # สร้างปุ่มเลื่อนขั้นสูง
    create_enhanced_scroll_button()
    
    # ตัวอย่างการค้นหา - เพิ่มเมนูจากชุดข้อมูลที่กำหนด
    example_queries = [
        "ไข่เจียว", "ต้มยำกุ้ง", "ผัดไทย", "ส้มตำ", "แกงเขียวหวาน",
        "กุ้งทาพริกไทยกระเทียม", "ข้าวเม่าทอด", "เปรี้ยวหวานไข่ม้วน", 
        "ไข่จ่อม", "งบปลาทู", "ยำไข่ปลาดุก", "กล้วยบวชชี",
        "เมนูแคลอรี่ไม่เกิน 300", "อาหารโปรตีนสูง", "เมนูลดน้ำหนัก"
    ]
    
    with st.expander("💡 ตัวอย่างการค้นหา", expanded=False):
        cols = st.columns(5)
        for i, query in enumerate(example_queries):
            with cols[i % 5]:
                if st.button(query, key=f"example_{i}"):
                    st.session_state.example_query = query
                    st.rerun()
    
    # เริ่มต้นประวัติการสนทนา
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # แสดงประวัติการสนทนา
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "recipe" in message:
                recipe = message["recipe"]
                
                # แสดงชื่อเมนูพร้อมค่าความเกี่ยวข้อง
                similarity_score = recipe.get('similarity', 0)
                match_type = recipe.get('match_type', 'semantic')
                
                title_html = f'<div class="recipe-title">{recipe["name"]}'
                if similarity_score > 0:
                    if match_type == 'enhanced_fuzzy' or match_type == 'menu_match':
                        title_html += f'<span class="exact-match-score">การจับคู่: {similarity_score:.2f}</span>'
                    elif match_type == 'fuzzy':
                        title_html += f'<span class="fuzzy-match-score">ความคล้ายคลึง: {similarity_score:.2f}</span>'
                    else:
                        title_html += f'<span class="similarity-score">ความเกี่ยวข้อง: {similarity_score:.2f}</span>'
                title_html += '</div>'
                
                st.markdown(title_html, unsafe_allow_html=True)
                
                display_nutrition_info(recipe['nutrition'])
                
                st.markdown('<div class="section-title">📝 วัตถุดิบ</div>', unsafe_allow_html=True)
                st.markdown(format_ingredients(recipe["ingredients"]), unsafe_allow_html=True)
                
                st.markdown('<div class="section-title">👩‍🍳 วิธีทำ</div>', unsafe_allow_html=True)
                st.markdown(format_cooking_method(recipe["method"]), unsafe_allow_html=True)
                
            elif message["role"] == "assistant" and "nutrition_results" in message:
                results = message["nutrition_results"]
                st.markdown(f"พบ **{len(results)}** สูตรอาหารที่ตรงเกณฑ์:")
                
                for i, result in enumerate(results[:5], 1):
                    with st.expander(f"{i}. {result['recipe_name']} ({result['calories']:.0f} kcal)"):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.write(f"🔥 แคลอรี่: {result['calories']:.0f} kcal")
                        with col2:
                            st.write(f"🥩 โปรตีน: {result['protein']:.1f} g")
                        with col3:
                            st.write(f"🍞 คาร์โบไฮเดรต: {result['carbs']:.1f} g")
                        with col4:
                            st.write(f"🧈 ไขมัน: {result['fat']:.1f} g")
                        
                        if st.button(f"ดูสูตรอาหาร", key=f"recipe_{i}_{len(st.session_state.messages)}"):
                            st.write("### วัตถุดิบ")
                            st.write(result['ingredients'])
                            st.write("### วิธีทำ")
                            st.write(result['method'])
                
                if len(results) > 5:
                    st.markdown(f"*และอีก {len(results) - 5} สูตร...*")
            else:
                st.markdown(message["content"])
    
    # ตรวจสอบตัวอย่างการค้นหาที่ถูกคลิก
    if "example_query" in st.session_state and st.session_state.example_query:
        prompt = st.session_state.example_query
        st.session_state.pop("example_query", None)
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
    else:
        prompt = None
    
    # ช่องใส่ข้อความ
    if user_input := st.chat_input("ค้นหาสูตรอาหารไทย..."):
        prompt = user_input
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
    
    # ประมวลผลคำค้นหา
    if prompt:
        st.session_state.search_count += 1
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 กำลังค้นหาและวิเคราะห์..."):
                if detect_nutrition_search(prompt):
                    # การค้นหาตามโภชนาการ
                    criteria = extract_nutrition_criteria_from_text(prompt)
                    
                    if criteria:
                        nutrition_results = search_by_nutrition_criteria(data, nutrition_analyzer, criteria, settings)
                        
                        if nutrition_results:
                            response = f"พบ {len(nutrition_results)} สูตรอาหารที่ตรงกับเกณฑ์โภชนาการที่ต้องการ"
                            st.markdown(response)
                            
                            st.markdown("### 🔍 ผลการค้นหา")
                            for i, result in enumerate(nutrition_results[:5], 1):
                                with st.expander(f"{i}. {result['recipe_name']} ({result['calories']:.0f} kcal)"):
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.write(f"🔥 แคลอรี่: {result['calories']:.0f} kcal")
                                    with col2:
                                        st.write(f"🥩 โปรตีน: {result['protein']:.1f} g")
                                    with col3:
                                        st.write(f"🍞 คาร์โบไฮเดรต: {result['carbs']:.1f} g")
                                    with col4:
                                        st.write(f"🧈 ไขมัน: {result['fat']:.1f} g")
                            
                            if len(nutrition_results) > 5:
                                st.markdown(f"*และอีก {len(nutrition_results) - 5} สูตร...*")
                            
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response,
                                "nutrition_results": nutrition_results
                            })
                        else:
                            response = "ขออภัย ไม่พบสูตรอาหารที่ตรงกับเกณฑ์โภชนาการที่ต้องการ ลองปรับเกณฑ์ใหม่"
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        response = "กรุณาระบุเกณฑ์โภชนาการให้ชัดเจนขึ้น เช่น 'เมนูแคลอรี่ไม่เกิน 300' หรือ 'อาหารโปรตีนสูงมากกว่า 20 กรัม'"
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    # การค้นหาทั่วไป
                    results = search_recipes_enhanced(prompt, model, data, embeddings, nutrition_analyzer, settings)
                    
                    if results:
                        best_match = results[0]
                        
                        # ตรวจสอบคุณภาพของผลลัพธ์
                        if best_match.get('match_type') in ['menu_match', 'enhanced_fuzzy']:
                            threshold = 0.3  # ลดเกณฑ์สำหรับการจับคู่ที่แม่นยำ
                        else:
                            threshold = 0.4
                            
                        if best_match["similarity"] > threshold:
                            similarity_score = best_match["similarity"]
                            match_type = best_match.get("match_type", "semantic")
                            
                            if match_type in ['menu_match', 'enhanced_fuzzy']:
                                response = f"พบสูตรอาหารที่ตรงกับที่คุณค้นหา: **{best_match['name']}**"
                            elif match_type == 'fuzzy':
                                response = f"พบสูตรอาหารที่คล้ายกับที่คุณค้นหา: **{best_match['name']}**"
                            else:
                                response = f"พบสูตรอาหารที่เกี่ยวข้อง: **{best_match['name']}**"
                            
                            st.markdown(response)
                            
                            # แสดงชื่อเมนูพร้อมค่าความเกี่ยวข้อง
                            title_html = f'<div class="recipe-title">{best_match["name"]}'
                            if match_type in ['menu_match', 'enhanced_fuzzy']:
                                title_html += f'<span class="exact-match-score">การจับคู่: {similarity_score:.2f}</span>'
                            elif match_type == 'fuzzy':
                                title_html += f'<span class="fuzzy-match-score">ความคล้ายคลึง: {similarity_score:.2f}</span>'
                            else:
                                title_html += f'<span class="similarity-score">ความเกี่ยวข้อง: {similarity_score:.2f}</span>'
                            title_html += '</div>'
                            
                            st.markdown(title_html, unsafe_allow_html=True)
                            
                            display_nutrition_info(best_match['nutrition'])
                            
                            st.markdown('<div class="section-title">📝 วัตถุดิบ</div>', unsafe_allow_html=True)
                            st.markdown(format_ingredients(best_match["ingredients"]), unsafe_allow_html=True)
                            
                            st.markdown('<div class="section-title">👩‍🍳 วิธีทำ</div>', unsafe_allow_html=True)
                            st.markdown(format_cooking_method(best_match["method"]), unsafe_allow_html=True)
                            
                            # แสดงเมนูที่เกี่ยวข้อง
                            if len(results) > 1:
                                st.markdown("### 🍽️ เมนูที่เกี่ยวข้อง")
                                for i, related in enumerate(results[1:4], 1):
                                    similarity = related['similarity']
                                    match_type_related = related.get('match_type', 'semantic')
                                    if match_type_related in ['menu_match', 'enhanced_fuzzy']:
                                        st.markdown(f"{i}. **{related['name']}** (การจับคู่: {similarity:.2f})")
                                    elif match_type_related == 'fuzzy':
                                        st.markdown(f"{i}. **{related['name']}** (ความคล้ายคลึง: {similarity:.2f})")
                                    else:
                                        st.markdown(f"{i}. **{related['name']}** (ความเกี่ยวข้อง: {similarity:.2f})")
                            
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response, 
                                "recipe": best_match
                            })
                        else:
                            response = "ขออภัย ฉันไม่พบสูตรอาหารที่ตรงกับคำค้นหาของคุณ กรุณาลองคำค้นหาอื่น"
                            st.markdown(response)
                            
                            # แสดงเมนูแนะนำ
                            st.markdown("### 🍽️ เมนูแนะนำ")
                            random_recipes = data.sample(5)
                            for _, recipe in random_recipes.iterrows():
                                st.markdown(f"- {recipe['name']}")
                            
                            st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        response = "ขออภัย ฉันไม่สามารถค้นหาสูตรอาหารได้ในขณะนี้ กรุณาลองใหม่อีกครั้ง"
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
