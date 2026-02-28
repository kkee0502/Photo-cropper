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
    st.error("⚠️ 'best.pt' 파일을 찾을 수 없습니다.")
    st.stop()

@st.cache_resource
def get_model():
    return YOLO(model_path)

model = get_model()

st.title("🦷 좌우 완벽 대칭 크롭기 (3:2)")
st.write("치아를 정중앙에 두고 좌우 여백을 동일하게 맞춥니다.")

# 여백 조절 (기본값 1.8 정도로 추천)
margin = st.sidebar.slider("전체 여백 크기", 1.2, 3.5, 1.8, step=0.1)

uploaded_files = st.file_uploader("사진을 선택하세요", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        h_orig, w_orig = img.shape[:2]
        target_ratio = 1.5 # 3:2 비율

        results = model.predict(img, conf=0.4)
        
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            if len(boxes) > 0:
                x1, y1, x2, y2 = boxes[0]
                
                # 1. 치아 박스의 중심점 계산
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                
                # 2. 박스 자체의 크기
                bw, bh = x2 - x1, y2 - y1
                
                # 3. [핵심] 좌우 대칭을 위한 반경 계산
                # 중심에서 박스 끝까지의 거리 중 큰 값을 선택하여 2배를 해줌
                side_dist = max(cx - x1, x2 - cx)
                top_dist = max(cy - y1, y2 - cy)
                
                # 4. 3:2 비율에 맞춰 자를 영역의 크기(Width, Height) 결정
                if (side_dist * 2) / (top_dist * 2) > target_ratio:
                    cw = (side_dist * 2) * margin
                    ch = cw / target_ratio
                else:
                    ch = (top_dist * 2) * margin
                    cw = ch * target_ratio

                # 5. 좌표 계산 (중심점에서 정확히 반반씩 확장)
                nx1 = int(cx - cw / 2)
                nx2 = int(cx + cw / 2)
                ny1 = int(cy - ch / 2)
                ny2 = int(cy + ch / 2)

                # 6. 사진 경계를 벗어날 경우, 대칭을 유지하며 전체 크기를 축소
                # (한쪽이 닿으면 반대쪽도 그만큼만 남게 함)
                offset_x = max(0, -nx1, nx2 - w_orig)
                offset_y = max(0, -ny1, ny2 - h_orig)
                
                if offset_x > 0 or offset_y > 0:
                    # 경계에 부딪힌 비율만큼 전체 폭/높이를 줄임
                    reduction_w = offset_x * 2
                    reduction_h = reduction_w / target_ratio
                    cw -= reduction_w
                    ch -= reduction_h
                    
                    # 다시 좌표 계산
                    nx1 = int(cx - cw / 2)
                    nx2 = int(cx + cw / 2)
                    ny1 = int(cy - ch / 2)
                    ny2 = int(cy + ch / 2)

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
