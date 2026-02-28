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

st.title("🦷 치아 보호 좌우 대칭 크롭기")
st.write("치아가 잘리지 않는 선에서 최대한의 좌우 대칭을 맞춥니다.")

# 여백 조절 슬라이더
margin_factor = st.sidebar.slider("여백 크기 (치아 대비)", 1.0, 3.5, 1.8, step=0.1)

uploaded_files = st.file_uploader("사진을 선택하세요", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if uploaded_files:
    processed_results = []
    first_file_base_name = os.path.splitext(uploaded_files[0].name)[0]
    
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

                    # [해결책 1] 치아 본체는 무조건 포함하는 최소 반폭 설정
                    min_half_w = bw / 2
                    min_half_h = bh / 2

                    # [해결책 2] 사용자가 원하는 여백 적용
                    desired_half_w = (bw * margin_factor) / 2
                    desired_half_h = desired_half_w / target_ratio

                    # [해결책 3] 사진 경계를 넘지 않는 최대 허용 대칭폭 계산
                    # 중심에서 좌우 끝까지의 거리 중 짧은 쪽을 기준으로 함
                    limit_half_w = min(cx, w_orig - cx)
                    limit_half_h = min(cy, h_orig - cy)

                    # 최종 반폭 결정: (원하는 폭 vs 한계 폭) 중 작은 값 선택 
                    # 단, 치아(min_half_w)보다는 커야 함
                    final_half_w = max(min_half_w, min(desired_half_w, limit_half_w))
                    final_half_h = final_half_w / target_ratio

                    # 높이 제약 조건 확인
                    if final_half_h > limit_half_h:
                        final_half_h = limit_half_h
                        final_half_w = final_half_h * target_ratio
                    
                    # 다시 한번 치아 폭 보호 (최종 확인)
                    if final_half_w < min_half_w:
                        final_half_w = min_half_w
                        final_half_h = final_half_w / target_ratio

                    # 좌표 확정 (이미지 범위를 절대 벗어나지 않도록 clip)
                    nx1 = int(np.clip(cx - final_half_w, 0, w_orig))
                    nx2 = int(np.clip(cx + final_half_w, 0, w_orig))
                    ny1 = int(np.clip(cy - final_half_h, 0, h_orig))
                    ny2 = int(np.clip(cy + final_half_h, 0, h_orig))

                    cropped = img[ny1:ny2, nx1:nx2]
                    if cropped.size == 0: continue
                    
                    # 3:2 비율 강제 리사이즈 (자르기 후 미세 오차 조정)
                    cropped_resized = cv2.resize(cropped, (int((ny2-ny1)*target_ratio), ny2-ny1))
                    
                    cropped_rgb = cv2.cvtColor(cropped_resized, cv2.COLOR_BGR2RGB)
                    res_img = Image.fromarray(cropped_rgb)
                    buf = io.BytesIO()
                    res_img.save(buf, format="JPEG", quality=95)
                    
                    processed_results.append((uploaded_file.name, buf.getvalue()))
                    st.image(cropped_rgb, caption=f"치아 보호 대칭 완료: {uploaded_file.name}")
                else:
                    st.warning(f"{uploaded_file.name}: 치아를 찾지 못했습니다.")
        except Exception as e:
            st.error(f"에러 ({uploaded_file.name}): {e}")

    if processed_results:
        st.divider()
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            for filename, data in processed_results:
                zip_file.writestr(filename, data)
        
        st.download_button(
            label=f"📂 '{first_file_base_name}.zip' 일괄 다운로드",
            data=zip_buffer.getvalue(),
            file_name=f"{first_file_base_name}.zip",
            mime="application/zip",
            use_container_width=True
        )

# --------------------------------------------------------------------------------------------------
# 공간이 남아서 채우는 기호: --------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
