import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import os

# 1. 모델 로드
model_path = 'best.pt'
if not os.path.exists(model_path):
    st.error("⚠️ 'best.pt' 파일이 없습니다.")
    st.stop()

@st.cache_resource
def get_model():
    return YOLO(model_path)

model = get_model()

st.title("🦷 좌우 균등 자동 크롭기 (3:2)")
st.write("치아를 중앙에 배치하며, 오류 없이 안정적으로 작동합니다.")

# 여백 조절 슬라이더
margin = st.sidebar.slider("여백 크기", 1.2, 3.5, 2.0, step=0.1)

uploaded_files = st.file_uploader("사진을 선택하세요", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            if img is None: continue
            
            h_orig, w_orig = img.shape[:2]
            target_ratio = 1.5 # 3:2

            results = model.predict(img, conf=0.4, verbose=False)
            
            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy()
                if len(boxes) > 0:
                    x1, y1, x2, y2 = boxes[0]
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    bw, bh = x2 - x1, y2 - y1
                    
                    # [대칭 계산] 중심에서 박스 끝까지의 거리
                    dx = max(cx - x1, x2 - cx)
                    dy = max(cy - y1, y2 - cy)
                    
                    # 3:2 비율 적용
                    if (dx * 2) / (dy * 2) > target_ratio:
                        cw, ch = (dx * 2) * margin, ((dx * 2) * margin) / target_ratio
                    else:
                        ch, cw = (dy * 2) * margin, ((dy * 2) * margin) * target_ratio

                    # 초기 좌표
                    nx1, nx2 = int(cx - cw / 2), int(cx + cw / 2)
                    ny1, ny2 = int(cy - ch / 2), int(cy + ch / 2)

                    # [안전장치] 사진 범위를 벗어나면 '대칭'보다 '표시'를 우선함
                    if nx1 < 0 or nx2 > w_orig or ny1 < 0 or ny2 > h_orig:
                        nx1, ny1 = max(0, nx1), max(0, ny1)
                        nx2, ny2 = min(w_orig, nx2), min(h_orig, ny2)
                        # 잘린 후 비율이 깨졌을 수 있으므로 다시 한번 3:2 강제 조정
                        new_w = nx2 - nx1
                        new_h = int(new_w / target_ratio)
                        ny2 = min(h_orig, ny1 + new_h)

                    # 실제 자르기
                    cropped = img[ny1:ny2, nx1:nx2]
                    if cropped.size == 0: 
                        st.warning(f"{uploaded_file.name}: 크롭 영역 계산 오류")
                        continue
                    
                    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                    st.image(cropped_rgb, caption=f"완료: {uploaded_file.name}")
                    
                    # 다운로드 버튼
                    res_img = Image.fromarray(cropped_rgb)
                    buf = io.BytesIO()
                    res_img.save(buf, format="JPEG", quality=95)
                    st.download_button(label=f"📥 {uploaded_file.name} 받기", 
                                       data=buf.getvalue(), 
                                       file_name=f"fixed_{uploaded_file.name}")
                else:
                    st.warning(f"{uploaded_file.name}: 치아를 찾지 못했습니다.")
        except Exception as e:
            st.error(f"에러 발생 ({uploaded_file.name}): {e}")
