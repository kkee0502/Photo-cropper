import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import os
import zipfile

# 1. 모델 로드
model_path = 'best.pt'
if not os.path.exists(model_path):
    st.error("⚠️ 'best.pt' 파일이 없습니다.")
    st.stop()

@st.cache_resource
def get_model():
    return YOLO(model_path)

model = get_model()

st.title("🦷 좌우 균등 자동 크롭기 (일괄 저장)")
st.write("치아 정중선을 기준으로 대칭 크롭 후, 원본 이름 그대로 한꺼번에 저장합니다.")

# 여백 조절 슬라이더
margin_factor = st.sidebar.slider("여백 크기", 1.0, 3.5, 2.0, step=0.1)

uploaded_files = st.file_uploader("사진을 선택하세요", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if uploaded_files:
    # 모든 결과물을 담을 리스트 (파일명, 바이너리 데이터)
    processed_results = []
    
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
                    
                    desired_w = (x2 - x1) * margin_factor
                    desired_h = (y2 - y1) * margin_factor
                    
                    if desired_w / desired_h > target_ratio:
                        final_w = desired_w
                        final_h = final_w / target_ratio
                    else:
                        final_h = desired_h
                        final_w = final_h * target_ratio
                    
                    # 대칭 한계값 계산
                    max_half_w = min(cx, w_orig - cx)
                    max_half_h = min(cy, h_orig - cy)
                    
                    half_w = min(final_w / 2, max_half_w)
                    half_h = half_w / target_ratio
                    
                    if half_h > max_half_h:
                        half_h = max_half_h
                        half_w = half_h * target_ratio

                    nx1, nx2 = int(cx - half_w), int(cx + half_w)
                    ny1, ny2 = int(cy - half_h), int(cy + half_h)

                    cropped = img[ny1:ny2, nx1:nx2]
                    if cropped.size == 0: continue
                    
                    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                    
                    # 결과물 이미지화 및 버퍼 저장
                    res_img = Image.fromarray(cropped_rgb)
                    buf = io.BytesIO()
                    res_img.save(buf, format="JPEG", quality=95)
                    
                    # 리스트에 추가 (압축용)
                    processed_results.append((uploaded_file.name, buf.getvalue()))
                    
                    # 화면에 미리보기 출력
                    st.image(cropped_rgb, caption=f"처리됨: {uploaded_file.name}")
                else:
                    st.warning(f"{uploaded_file.name}: 치아를 찾지 못했습니다.")
        except Exception as e:
            st.error(f"에러 ({uploaded_file.name}): {e}")

    # --- 일괄 다운로드 버튼 ---
    if processed_results:
        st.divider()
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            for filename, data in processed_results:
                zip_file.writestr(filename, data)
        
        st.download_button(
            label="📂 모든 사진 원본 이름으로 다운로드 (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="processed_dental_images.zip",
            mime="application/zip",
            use_container_width=True
        )

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
