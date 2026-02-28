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

st.title("🦷 양 끝 치아 기준 균등 여백 크롭기")
st.write("가장 바깥쪽 치아에서 사진 끝까지의 거리를 좌우 똑같이 맞춥니다.")

# 여백 조절 슬라이더 (치아 바깥쪽 추가 공간)
extra_margin = st.sidebar.slider("추가 여백 (픽셀)", 10, 500, 100, step=10)

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
                    # 모든 탐지된 치아를 포함하는 전체 바운딩 박스 계산
                    # (여러 치아가 각각 탐지될 경우를 대비해 전체 영역을 잡음)
                    all_x1 = np.min(boxes[:, 0])
                    all_y1 = np.min(boxes[:, 1])
                    all_x2 = np.max(boxes[:, 2])
                    all_y2 = np.max(boxes[:, 3])
                    
                    tooth_width = all_x2 - all_x1
                    tooth_height = all_y2 - all_y1
                    
                    # [핵심 로직] 양 끝 치아 기준 여백 설정
                    # 1. 치아를 포함하고 좌우에 extra_margin만큼의 공간을 확보한 폭
                    desired_w = tooth_width + (extra_margin * 2)
                    # 2. 3:2 비율을 위한 높이 계산
                    desired_h = desired_w / target_ratio
                    
                    # 만약 계산된 높이가 실제 치아 높이보다 작으면 높이 기준으로 재계산
                    if desired_h < tooth_height + (extra_margin * 2 / target_ratio):
                        desired_h = tooth_height + (extra_margin * 2 / target_ratio)
                        desired_w = desired_h * target_ratio

                    # 3. 중심점 설정 (치아 뭉치의 정중앙)
                    cx, cy = (all_x1 + all_x2) / 2, (all_y1 + all_y2) / 2
                    
                    # 4. 좌표 계산 및 패딩 처리
                    nx1, nx2 = int(cx - desired_w / 2), int(cx + desired_w / 2)
                    ny1, ny2 = int(cy - desired_h / 2), int(cy + desired_h / 2)

                    # 사진 범위를 벗어나면 패딩 생성 (검은색)
                    pad_l = max(0, -nx1)
                    pad_r = max(0, nx2 - w_orig)
                    pad_t = max(0, -ny1)
                    pad_b = max(0, ny2 - h_orig)

                    padded_img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, 
                                                   cv2.BORDER_CONSTANT, value=[0, 0, 0])

                    # 패딩된 이미지에서 크롭
                    final_cropped = padded_img[ny1+pad_t : ny2+pad_t, nx1+pad_l : nx2+pad_l]
                    
                    if final_cropped.size == 0: continue
                    
                    cropped_rgb = cv2.cvtColor(final_cropped, cv2.COLOR_BGR2RGB)
                    res_img = Image.fromarray(cropped_rgb)
                    buf = io.BytesIO()
                    res_img.save(buf, format="JPEG", quality=95)
                    
                    processed_results.append((uploaded_file.name, buf.getvalue()))
                    st.image(cropped_rgb, caption=f"양 끝 여백 정렬 완료: {uploaded_file.name}")
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

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
