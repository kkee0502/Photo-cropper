import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io

# 1. 모델 불러오기
try:
    model = YOLO('best.pt')
except Exception as e:
    st.error(f"모델 로드 실패: {e}")

st.title("🦷 비율 유지 자동 크롭기")
st.write("원본 사진의 가로세로 비율을 유지하면서 타겟 영역을 크롭합니다.")

padding_percent = st.sidebar.slider("추가 여백 (%)", 0, 50, 10)

uploaded_files = st.file_uploader("사진을 올려주세요", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        h_orig, w_orig = img.shape[:2]
        orig_ratio = w_orig / h_orig  # 원본 비율 (가로/세로)

        results = model.predict(img, conf=0.4)
        
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            if len(boxes) > 0:
                # AI가 찾은 박스 좌표
                x1, y1, x2, y2 = boxes[0]
                box_w = x2 - x1
                box_h = y2 - y1
                
                # 1. 박스의 중심점 계산
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                
                # 2. 원본 비율에 맞게 크롭 영역 결정
                # 박스 비율이 원본보다 가로로 길면 가로 기준, 세로로 길면 세로 기준 확장
                if box_w / box_h > orig_ratio:
                    crop_w = box_w * (1 + padding_percent/100)
                    crop_h = crop_w / orig_ratio
                else:
                    crop_h = box_h * (1 + padding_percent/100)
                    crop_w = crop_h * orig_ratio

                # 3. 최종 좌표 계산 (이미지 경계 벗어나지 않게 조정)
                nx1 = max(0, int(cx - crop_w / 2))
                ny1 = max(0, int(cy - crop_h / 2))
                nx2 = min(w_orig, int(cx + crop_w / 2))
                ny2 = min(h_orig, int(cy + crop_h / 2))
                
                # 경계에 걸려 비율이 깨지는 경우 대비 다시 정밀 조정
                final_crop = img[ny1:ny2, nx1:nx2]
                final_rgb = cv2.cvtColor(final_crop, cv2.COLOR_BGR2RGB)
                
                st.image(final_rgb, caption=f"비율 유지 크롭: {uploaded_file.name}")
                
                # 다운로드 설정
                res_img = Image.fromarray(final_rgb)
                buf = io.BytesIO()
                res_img.save(buf, format="JPEG", quality=95)
                st.download_button(label="📥 다운로드", data=buf.getvalue(), file_name=f"fixed_{uploaded_file.name}")
            else:
                st.warning(f"{uploaded_file.name}: 영역을 찾지 못했습니다.")
