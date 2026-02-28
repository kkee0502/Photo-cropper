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

st.title("🦷 좌우 여백 실측 동일화기")
st.write("치아 끝단에서 사진 경계까지의 거리를 측정하여 양쪽을 똑같이 맞춥니다.")

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

            results = model.predict(img, conf=0.35, verbose=False)
            
            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy()
                if len(boxes) > 0:
                    # 1. 모든 치아를 포함하는 영역 탐지
                    tx1, ty1, tx2, ty2 = np.min(boxes[:, 0]), np.min(boxes[:, 1]), np.max(boxes[:, 2]), np.max(boxes[:, 3])
                    
                    # 2. [실측] 왼쪽 여백(L)과 오른쪽 여백(R) 측정
                    left_margin = tx1
                    right_margin = w_orig - tx2
                    
                    # 3. [동일화] 더 좁은 쪽의 여백을 기준으로 설정
                    min_margin = min(left_margin, right_margin)
                    
                    # 4. 새로운 크롭 범위 설정 (여백을 동일하게 적용)
                    nx1 = tx1 - min_margin
                    nx2 = tx2 + min_margin
                    
                    # 5. 3:2 비율을 맞추기 위한 높이 계산 (중심축 유지)
                    new_width = nx2 - nx1
                    new_height = new_width / target_ratio
                    
                    cy = (ty1 + ty2) / 2
                    ny1 = int(max(0, cy - new_height / 2))
                    ny2 = int(min(h_orig, cy + new_height / 2))
                    
                    # 가로 좌표 정수화
                    nx1, nx2 = int(nx1), int(nx2)

                    # 6. 최종 자르기
                    final_cropped = img[ny1:ny2, nx1:nx2]
                    
                    if final_cropped.size == 0: continue
                    
                    # 결과물 변환 및 저장
                    cropped_rgb = cv2.cvtColor(final_cropped, cv2.COLOR_BGR2RGB)
                    res_img = Image.fromarray(cropped_rgb)
                    buf = io.BytesIO()
                    res_img.save(buf, format="JPEG", quality=95)
                    
                    processed_results.append((uploaded_file.name, buf.getvalue()))
                    st.image(cropped_rgb, caption=f"여백 실측 동기화 완료: {uploaded_file.name} (여백: {int(min_margin)}px)")
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
