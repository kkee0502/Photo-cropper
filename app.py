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
st.write("치아를 정중앙에 배치하며, 좌우 여백을 완벽하게 대칭으로 맞춥니다.")

# 여백 조절 슬라이더
margin_factor = st.sidebar.slider("여백 크기", 1.0, 3.0, 1.5, step=0.1)

uploaded_files = st.file_uploader("사진을 선택하세요", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            if img is None: continue
            
            h_orig, w_orig = img.shape[:2]
            target_ratio = 1.5  # 3:2

            results = model.predict(img, conf=0.4, verbose=False)
            
            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy()
                if len(boxes) > 0:
                    # 가장 큰 박스(보통 전체 치아) 선택
                    x1, y1, x2, y2 = boxes[0]
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    
                    # 1. 객체를 감싸는 최소 반폭(Half-width/height) 계산
                    half_bw = (x2 - x1) / 2 * margin_factor
                    half_bh = (y2 - y1) / 2 * margin_factor
                    
                    # 2. 3:2 비율에 따른 반폭 조정
                    if half_bw / half_bh > target_ratio:
                        half_bh = half_bw / target_ratio
                    else:
                        half_bw = half_bh * target_ratio
                    
                    # 3. [핵심] 사진 경계를 넘지 않는 최대 대칭 반폭 결정
                    # 중심에서 각 변까지의 거리 중 가장 짧은 곳을 기준으로 삼음
                    limit_w = min(cx, w_orig - cx)
                    limit_h = min(cy, h_orig - cy)
                    
                    final_half_w = min(half_bw, limit_w)
                    final_half_h = final_half_w / target_ratio
                    
                    # 높이가 사진을 벗어나면 다시 조정
                    if final_half_h > limit_h:
                        final_half_h = limit_h
                        final_half_w = final_half_h * target_ratio

                    # 4. 최종 좌표 확정 (정수 변환)
                    nx1, nx2 = int(cx - final_half_w), int(cx + final_half_w)
                    ny1, ny2 = int(cy - final_half_h), int(cy + final_half_h)

                    # 실제 자르기
                    cropped = img[ny1:ny2, nx1:nx2]
                    
                    if cropped.size == 0:
                        st.warning(f"{uploaded_file.name}: 영역 계산 오류")
                        continue
                    
                    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                    st.image(cropped_rgb, caption=f"대칭 정렬 완료: {uploaded_file.name}")
                    
                    # 다운로드 버튼
                    res_img = Image.fromarray(cropped_rgb)
                    buf = io.BytesIO()
                    res_img.save(buf, format="JPEG", quality=95)
                    st.download_button(label=f"📥 {uploaded_file.name} 받기", 
                                       data=buf.getvalue(), 
                                       file_name=f"balanced_{uploaded_file.name}")
                else:
                    st.warning(f"{uploaded_file.name}: 치아를 찾지 못했습니다.")
        except Exception as e:
            st.error(f"에러 발생 ({uploaded_file.name}): {e}")

# ---------------------------------------------------------
# 공간이 남아서 채우는 기호: ----------------------------------
# ---------------------------------------------------------
