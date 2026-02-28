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

st.title("🦷 정밀 자동 크롭기 (3:2)")
st.write("치아를 중심으로 여백을 넉넉하고 균등하게 확보합니다.")

# 여백 조절 슬라이더 (기본값을 1.8 정도로 넉넉히 잡았습니다)
margin = st.sidebar.slider("여백 크기 (높을수록 치아가 작아짐)", 1.2, 3.5, 2.2, step=0.1)

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
                
                # 1. AI가 찾은 영역의 크기와 중심점
                bw, bh = x2 - x1, y2 - y1
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                
                # 2. 3:2 비율에 맞춰서 자를 영역의 크기 결정
                # 치아 박스의 가로/세로 중 더 긴 쪽을 기준으로 여백(margin)을 곱함
                if bw / bh > target_ratio:
                    cw = bw * margin
                    ch = cw / target_ratio
                else:
                    ch = bh * margin
                    cw = ch * target_ratio
                
                # 3. 좌표 계산 (중심점 기준)
                nx1 = int(cx - cw / 2)
                ny1 = int(cy - ch / 2)
                nx2 = int(nx1 + cw)
                ny2 = int(ny1 + ch)
                
                # 4. [매우 중요] 사진 경계를 벗어나는 경우 '밀어내기' (잘림 방지)
                if nx1 < 0: nx2 -= nx1; nx1 = 0
                if ny1 < 0: ny2 -= ny1; ny1 = 0
                if nx2 > w_orig: nx1 -= (nx2 - w_orig); nx2 = w_orig
                if ny2 > h_orig: ny1 -= (ny2 - h_orig); ny2 = h_orig
                
                # 5. 밀어내기 후에도 혹시나 범위를 벗어나면 강제 조정 (최종 방어선)
                nx1, ny1 = max(0, nx1), max(0, ny1)
                nx2, ny2 = min(w_orig, nx2), min(h_orig, ny2)

                # 실제 자르기
                cropped = img[ny1:ny2, nx1:nx2]
                if cropped.size == 0: continue
                
                cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                st.image(cropped_rgb, caption=f"크롭 완료: {uploaded_file.name}")
                
                # 다운로드
                res_img = Image.fromarray(cropped_rgb)
                buf = io.BytesIO()
                res_img.save(buf, format="JPEG", quality=95)
                st.download_button(label=f"📥 {uploaded_file.name} 다운로드", 
                                   data=buf.getvalue(), 
                                   file_name=f"crop_{uploaded_file.name}")
            else:
                st.warning(f"{uploaded_file.name}: 영역 인식 실패")
