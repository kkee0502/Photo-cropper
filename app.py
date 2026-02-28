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

st.title("🦷 픽셀 정밀 대칭 크롭기")
st.write("AI 감지 후 픽셀 분석을 통해 좌우 여백을 1px 단위로 맞춥니다.")

# 여백 조절 슬라이더
margin_px = st.sidebar.slider("치아 끝단 기준 추가 여백 (px)", 20, 600, 150, step=10)

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

            # YOLO 예측
            results = model.predict(img, conf=0.35, verbose=False)
            
            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy()
                if len(boxes) > 0:
                    # 1. YOLO 박스 영역 추출
                    yolo_x1, yolo_y1, yolo_x2, yolo_y2 = np.min(boxes[:, 0]), np.min(boxes[:, 1]), np.max(boxes[:, 2]), np.max(boxes[:, 3])
                    
                    # 2. [정밀 분석] 박스 내부에서 실제 '밝은 치아' 영역 재탐색
                    roi = img[int(yolo_y1):int(yolo_y2), int(yolo_x1):int(yolo_x2)]
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    _, binary_roi = cv2.threshold(gray_roi, 120, 255, cv2.THRESH_BINARY) # 밝은 부분만 남김
                    
                    coords = cv2.findNonZero(binary_roi)
                    if coords is not None:
                        rx, ry, rw, rh = cv2.boundingRect(coords)
                        # 원본 이미지 기준 실제 치아 끝단 좌표
                        real_x1 = yolo_x1 + rx
                        real_x2 = yolo_x1 + rx + rw
                        real_y1 = yolo_y1 + ry
                        real_y2 = yolo_y1 + ry + rh
                    else:
                        real_x1, real_x2, real_y1, real_y2 = yolo_x1, yolo_x2, yolo_y1, yolo_y2

                    # 3. 실제 치아 끝단을 기준으로 한 중심축(Midline)
                    midline_x = (real_x1 + real_x2) / 2
                    midline_y = (real_y1 + real_y2) / 2
                    
                    # 4. 좌우 여백을 똑같이 맞춘 최종 폭 계산
                    # (치아 실제 폭 + 양쪽 동일 여백)
                    final_w = (real_x2 - real_x1) + (margin_px * 2)
                    final_h = final_w / target_ratio
                    
                    # 5. 좌표 설정 및 패딩(이미지 부족 시 보완)
                    nx1, nx2 = int(midline_x - final_w/2), int(midline_x + final_w/2)
                    ny1, ny2 = int(midline_y - final_h/2), int(midline_y + final_h/2)
                    
                    pad_l, pad_r = max(0, -nx1), max(0, nx2 - w_orig)
                    pad_t, pad_b = max(0, -ny1), max(0, ny2 - h_orig)
                    
                    padded_img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                    
                    # 6. 최종 크롭
                    final_cropped = padded_img[ny1+pad_t : ny2+pad_t, nx1+pad_l : nx2+pad_l]
                    
                    if final_cropped.size == 0: continue
                    
                    cropped_rgb = cv2.cvtColor(final_cropped, cv2.COLOR_BGR2RGB)
                    res_img = Image.fromarray(cropped_rgb)
                    buf = io.BytesIO()
                    res_img.save(buf, format="JPEG", quality=95)
                    
                    processed_results.append((uploaded_file.name, buf.getvalue()))
                    st.image(cropped_rgb, caption=f"정밀 정렬 완료: {uploaded_file.name}")
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
