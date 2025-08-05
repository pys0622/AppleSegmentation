import os
import cv2
import numpy as np
from glob import glob

# 경로 설정
image_dir = "dataset/images/test"
label_dir = "dataset/labels/test"
output_dir = "visualization/test"
os.makedirs(output_dir, exist_ok=True)

# 모든 이미지에 대해 반복
image_paths = glob(os.path.join(image_dir, "*.jpg"))  # 또는 .png
for img_path in image_paths:
    fname = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(label_dir, f"{fname}.txt")
    if not os.path.exists(label_path):
        continue

    # 이미지 불러오기
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # 마스크 라벨 읽기
    with open(label_path, "r") as f:
        lines = f.read().strip().splitlines()

    for line in lines:
        tokens = line.strip().split()
        cls_id = int(tokens[0])
        coords = list(map(float, tokens[1:]))

        # 폴리곤 좌표 복원
        points = np.array([(int(x * w), int(y * h)) for x, y in zip(coords[::2], coords[1::2])], dtype=np.int32)
        cv2.polylines(img, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.fillPoly(img, [points], color=(0, 255, 0, 50))  # 반투명 마스크

    # 저장
    out_path = os.path.join(output_dir, f"{fname}_gt.png")
    cv2.imwrite(out_path, img)

print(f"✅ GT 마스크 시각화 완료: {output_dir}/")