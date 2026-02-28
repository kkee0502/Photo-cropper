import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import os

# 1. 모델 로드 (파일명 주의: 반드시 best.pt여야 함)
model_path = 'best.pt'
if not os.path.exists(model_path):
    st.error("⚠️ 'best.pt' 파일이 없습니다. GitHub에 파일을 올려주세요.")
    st.stop()

# 모델을 메모리에 한 번만 올리도록 설정
@st.cache_resource
def get_model():
    return YOLO(model_path)

model = get_model()

st.title("🦷 가로형 3:2 비율 자동 크롭기")
st.write("모든 사진을 가로가 긴 3:2 비율로 정밀하게 크롭합니다.")

# 영역 확장 슬라이더 (사용자 취향대로 조절)
padding = st.sidebar.slider("영역 확장 정도", 1.0, 3.0, 1.8, step=0.1)

uploaded_files = st.file_uploader("사진을 선택하세요", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # 이미지 읽기
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        h_orig, w_orig = img.shape[:2]
        
        # [고정] 가로가 긴 3:2 비율 (3 / 2 = 1.5)
        target_ratio = 1.5 

        results = model.predict(img, conf=0.4)
        
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            if len(boxes) > 0:
                # AI가 찾은 박스 정보
                x1, y1, x2, y2 = boxes[0]
                box_w, box_h = x2 - x1, y2 - y1
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                
                # 3:2 비율에 맞춘 새로운 크기 계산
                if box_w / box_h > target_ratio:
                    # 박스가 가로로 더 넓은 경우 -> 가로 기준 확장
                    new_w = box_w * padding
                    new_h = new_w / target_ratio
                else:
                    # 박스가 세로로 더 긴 경우 -> 세로 기준 확장
                    new_h = box_h * padding
                    new_w = new_h * target_ratio

                # 좌표 확정 (이미지 밖으로 나가지 않게 조절)
                nx1 = int(max(0, cx - new_w / 2))
                ny1 = int(max(0, cy - new_h / 2))
                nx2 = int(min(w_orig, nx1 + new_w))
                # 삐져나온 만큼 다시 nx1 조정
                if nx2 == w_orig: nx1 = int(max(0, nx2 - new_w))
                
                ny2 = int(min(h_orig, ny1 + (nx2 - nx1) / target_ratio))
                # 삐져나온 만큼 다시 ny1 조정
                if ny2 == h_orig: ny1 = int(max(0, ny2 - (nx2 - nx1) / target_ratio))

                # 최종 크롭 및 출력
                cropped = img[ny1:ny2, nx1:nx2]
                cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                
                st.image(cropped_rgb, caption=f"3:2 가로형 크롭: {uploaded_file.name}")
                
                # 다운로드 버튼
                res_img = Image.fromarray(cropped_rgb)
                buf = io.BytesIO()
                res_img.save(buf, format="JPEG", quality=95)
                st.download_button(label=f"📥 {uploaded_file.name} 다운로드", 
                                   data=buf.getvalue(), 
                                   file_name=f"3x2_{uploaded_file.name}")
            else:
                st.warning(f"{uploaded_file.name}: 영역 인식 실패")
