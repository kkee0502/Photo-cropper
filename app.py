import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io

# 1. 모델 불러오기 (파일 이름이 best.pt여야 함)
try:
    model = YOLO('best.pt')
except Exception as e:
    st.error(f"모델 로드 실패: {e}")

st.title("🦷 구내 사진 자동 편집기")
st.write("Before 사진을 올리면 학습된 스타일로 크롭합니다.")

# 여백 조절 슬라이더
padding = st.sidebar.slider("여백 조절 (Pixel)", 0, 100, 30)

uploaded_files = st.file_uploader("사진을 올려주세요", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # 이미지 읽기
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        # AI 탐지
        results = model.predict(img, conf=0.4)
        
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            if len(boxes) > 0:
                x1, y1, x2, y2 = map(int, boxes[0])
                
                # 여백 추가 및 크롭
                h, w = img.shape[:2]
                x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
                x2, y2 = min(w, x2 + padding), min(h, y2 + padding)
                
                cropped = img[y1:y2, x1:x2]
                cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                
                # 결과 표시
                st.image(cropped_rgb, caption=f"편집 완료: {uploaded_file.name}")
                
                # 다운로드 버튼
                res_img = Image.fromarray(cropped_rgb)
                buf = io.BytesIO()
                res_img.save(buf, format="JPEG")
                st.download_button(label="📥 다운로드", data=buf.getvalue(), file_name=f"crop_{uploaded_file.name}")
            else:
                st.warning(f"{uploaded_file.name}: 치아 영역을 찾지 못했습니다.")
