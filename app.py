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

st.title("🦷 치아 외곽 픽셀 기준 완벽 대칭기")
st.write("치아의 가장 바깥쪽 픽셀을 찾아 사진 경계까지의 거리를 1:1로 맞춥니다.")

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

            results = model.predict(img, conf=0.3, verbose=False)
            
            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy()
                if len(boxes) > 0:
                    # [단계 1] YOLO가 찾은 영역을 조금 더 넓게 잡아 픽셀 분석 준비
                    y1, y2 = int(np.min(boxes[:, 1])), int(np.max(boxes[:, 3]))
                    x1, x2 = int(np.min(boxes[:, 0])), int(np.max(boxes[:, 2]))
                    
                    # [단계 2] 픽셀 분석으로 '진짜' 치아 끝점 찾기
                    # 치아는 밝기 때문에 그레이스케일에서 특정 임계값 이상의 범위를 찾음
                    roi = img[y1:y2, :] # 가로는 전체를 보고 정확한 끝단 탐색
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(gray_roi, 130, 255, cv2.THRESH_BINARY)
                    
                    # 픽셀이 존재하는 모든 x좌표 추출
                    coords = cv2.findNonZero(thresh)
                    if coords is not None:
                        # 실제 치아가 존재하는 가장 왼쪽(min_x)과 오른쪽(max_x) 픽셀 위치
                        pixel_x1 = np.min(coords[:, :, 0])
                        pixel_x2 = np.max(coords[:, :, 0])
                    else:
                        pixel_x1, pixel_x2 = x1, x2

                    # [단계 3] 현재 이미지 기준 실제 여백 측정
                    current_L = pixel_x1
                    current_R = w_orig - pixel_x2
                    
                    # [단계 4] 양쪽 여백을 동일하게 맞춤 (좁은 쪽 기준)
                    target_margin = min(current_L, current_R)
                    
                    # 새로운 크롭 경계 (치아 끝점에서 동일한 여백만큼 확장)
                    final_x1 = int(pixel_x1 - target_margin)
                    final_x2 = int(pixel_x2 + target_margin)
                    
                    # [단계 5] 3:2 비율 유지하며 세로 범위 계산
                    new_w = final_x2 - final_x1
                    new_h = new_w / target_ratio
                    
                    mid_y = (y1 + y2) / 2
                    final_y1 = int(max(0, mid_y - new_h / 2))
                    final_y2 = int(min(h_orig, mid_y + new_h / 2))

                    # [단계 6] 크롭 및 결과 생성
                    final_cropped = img[final_y1:final_y2, final_x1:final_x2]
                    
                    if final_cropped.size == 0: continue
                    
                    cropped_rgb = cv2.cvtColor(final_cropped, cv2.COLOR_BGR2RGB)
                    res_img = Image.fromarray(cropped_rgb)
                    buf = io.BytesIO()
                    res_img.save(buf, format="JPEG", quality=95)
                    
                    processed_results.append((uploaded_file.name, buf.getvalue()))
                    st.image(cropped_rgb, caption=f"완벽 대칭(L=R): {uploaded_file.name}")
                    st.write(f"📏 적용된 여백: {target_margin}px (좌우 동일)")
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
# --------------------------------------------------------------------------------------------------
