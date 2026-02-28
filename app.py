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

st.title("🦷 정중앙 대칭 크롭기 (3:2)")
st.write("치아를 정중앙에 배치하고 상하좌우 여백을 균등하게 조절합니다.")

# 여백 조절 (값이 커질수록 치아가 작아지고 배경이 넓어짐)
margin_scale = st.sidebar.slider("전체 여백 크기", 1.2, 4.0, 2.0, step=0.1)

uploaded_files = st.file_uploader("사진을 선택하세요", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        h_orig, w_orig = img.shape[:2]
        target_ratio = 1.5 # 가로 3 : 세로 2

        results = model.predict(img, conf=0.4)
        
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            if len(boxes) > 0:
                x1, y1, x2, y2 = boxes[0]
                # 1. AI가 찾은 영역의 중심점 계산
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                
                # 2. 중심으로부터 가장 먼 경계까지의 거리 계산 (대칭을 위해)
                dist_x = max(cx - x1, x2 - cx)
                dist_y = max(cy - y1, y2 - cy)
                
                # 3. 3:2 비율을 유지하면서 박스를 모두 포함하는 최소 반경 계산
                if dist_x / dist_y > target_ratio:
                    # 가로가 더 지배적인 경우
                    half_w = dist_x * margin_scale
                    half_h = half_w / target_ratio
                else:
                    # 세로가 더 지배적인 경우
                    half_h = dist_y * margin_scale
                    half_w = half_h * target_ratio

                # 4. 최종 좌표 (중심에서 양쪽으로 동일하게 확장)
                nx1 = int(cx - half_w)
                nx2 = int(cx + half_w)
                ny1 = int(cy - half_h)
                ny2 = int(cy + half_h)

                # 5. 이미지 경계를 벗어나는 경우, 대칭을 유지하며 최대 크기로 축소
                if nx1 < 0 or nx2 > w_orig or ny1 < 0 or ny2 > h_orig:
                    # 넘치는 비율 중 가장 큰 값을 찾아 전체를 축소 (대칭 유지용)
                    shrink_factor = max(
                        (-nx1 if nx1 < 0 else 0) / half_w,
                        (nx2 - w_orig if nx2 > w_orig else 0) / half_w,
                        (-ny1 if ny1 < 0 else 0) / half_h,
                        (ny2 - h_orig if ny2 > h_orig else 0) / half_h
                    )
                    scale = 1 - shrink_factor
                    half_w *= scale
                    half_h *= scale
                    nx1, nx2 = int(cx - half_w), int(cx + half_w)
                    ny1, ny2 = int(cy - half_h), int(cy + half_h)

                # 실제 자르기
                cropped = img[ny1:ny2, nx1:nx2]
                if cropped.size == 0: continue
                
                cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                st.image(cropped_rgb, caption=f"대칭 크롭 완료: {uploaded_file.name}")
                
                # 다운로드
                res_img = Image.fromarray(cropped_rgb)
                buf = io.BytesIO()
                res_img.save(buf, format="JPEG", quality=95)
                st.download_button(label=f"📥 {uploaded_file.name} 다운로드", 
                                   data=buf.getvalue(), 
                                   file_name=f"centered_{uploaded_file.name}")
            else:
                st.warning(f"{uploaded_file.name}: 영역 인식 실패")
