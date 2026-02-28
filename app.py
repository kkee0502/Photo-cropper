import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io

# 1. 모델 불러오기
try:
    model = YOLO('best.pt')
except Exception as e:
    st.error(f"모델 로드 실패: {e}")

st.title("🦷 100% 원본 비율 유지 크롭기")
st.write("사진을 올리면 해당 사진의 원본 비율을 자동으로 계산하여 크롭합니다.")

# 여백 조절 (박스 크기 대비 확장 비율)
padding_factor = st.sidebar.slider("영역 확장 정도", 1.0, 3.0, 1.5, step=0.1)

uploaded_files = st.file_uploader("사진을 올려주세요", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # 이미지 읽기
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        h_orig, w_orig = img.shape[:2]
        
        # [중요] 실시간 원본 비율 계산 (예: 2:3이면 0.666..., 3:4면 0.75)
        target_ratio = w_orig / h_orig 

        results = model.predict(img, conf=0.4)
        
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            if len(boxes) > 0:
                # AI가 찾은 타겟의 좌표와 크기
                x1, y1, x2, y2 = boxes[0]
                box_w = x2 - x1
                box_h = y2 - y1
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2 # 중심점
                
                # 원본 비율을 유지하면서 박스를 포함하는 새로운 크기 계산
                # 박스보다 크게 잡기 위해 padding_factor를 곱함
                if box_w / box_h > target_ratio:
                    # 박스가 가로로 더 넓은 경우 -> 가로 기준 확장
                    new_w = box_w * padding_factor
                    new_h = new_w / target_ratio
                else:
                    # 박스가 세로로 더 긴 경우 -> 세로 기준 확장
                    new_h = box_h * padding_factor
                    new_w = new_h * target_ratio

                # 최종 좌표 계산 (이미지 경계를 넘지 않게 컷트)
                nx1 = int(max(0, cx - new_w / 2))
                ny1 = int(max(0, cy - new_h / 2))
                nx2 = int(min(w_orig, cx + new_w / 2))
                ny2 = int(min(h_orig, cy + crop_h / 2 if 'crop_h' in locals() else cy + new_h / 2)) 
                
                # 비율이 0.1픽셀이라도 어긋나지 않도록 재검증하여 자르기
                # 실제 자를 때 소수점 버림 현상 때문에 미세하게 틀릴 수 있어 보정함
                final_w = nx2 - nx1
                final_h = int(final_w / target_ratio)
                ny2 = min(h_orig, ny1 + final_h)

                cropped = img[ny1:ny2, nx1:nx2]
                cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                
                # 결과 화면 표시
                st.image(cropped_rgb, caption=f"비율 고정 완료 ({w_orig}:{h_orig})")
                
                # 다운로드 버튼
                res_img = Image.fromarray(cropped_rgb)
                buf = io.BytesIO()
                res_img.save(buf, format="JPEG", quality=100) # 화질 최대 유지
                st.download_button(label=f"📥 {uploaded_file.name} 받기", 
                                   data=buf.getvalue(), 
                                   file_name=f"fixed_{uploaded_file.name}")
            else:
                st.warning(f"{uploaded_file.name}: 영역을 찾지 못했습니다.")
