import os
import cv2
import numpy as np
from glob import glob

# 경로 설정
img_dir = "dataset/images/test"
gt_label_dir = "dataset/labels/test"
pred_label_dir = "inference/results/pred_masks/labels"
output_dir = "inference/visualization"
os.makedirs(output_dir, exist_ok=True)

# 공통 파일 리스트
img_paths = glob(os.path.join(img_dir, "*.jpg"))
for img_path in img_paths:
    fname = os.path.splitext(os.path.basename(img_path))[0]
    gt_label_path = os.path.join(gt_label_dir, f"{fname}.txt")
    pred_label_path = os.path.join(pred_label_dir, f"{fname}.txt")

    if not os.path.exists(gt_label_path) or not os.path.exists(pred_label_path):
        continue

    # 이미지 불러오기
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    gt_img = img.copy()
    pred_img = img.copy()

    def draw_mask(label_path, image, color):
        with open(label_path, "r") as f:
            lines = f.read().strip().splitlines()

        for line in lines:
            tokens = line.strip().split()
            if len(tokens) < 7:
                continue
            coords = list(map(float, tokens[1:]))
            points = np.array([(int(x * w), int(y * h)) for x, y in zip(coords[::2], coords[1::2])], dtype=np.int32)
            cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
            cv2.fillPoly(image, [points], color=color)

    # GT: 초록색, Pred: 파란색
    draw_mask(gt_label_path, gt_img, color=(0, 255, 0))
    draw_mask(pred_label_path, pred_img, color=(255, 0, 0))

    # 이미지 병합
    combined = cv2.hconcat([gt_img, pred_img])

    # 저장
    out_path = os.path.join(output_dir, f"{fname}_compare.png")
    cv2.imwrite(out_path, combined)

print(f"✅ 비교 이미지 저장 완료! 위치: {output_dir}/")