import os
import json
import shutil
import random
from sklearn.model_selection import train_test_split

# 1. 경로 설정
coco_json_path = '../downloaded/train/_annotations.coco.json'  # COCO json 경로
image_dir = '../downloaded/train'  # 이미지 경로
output_dir = 'dataset'   # YOLO 형식으로 변환될 경로
os.makedirs(output_dir, exist_ok=True)

# 2. COCO JSON 불러오기
with open(coco_json_path, 'r') as f:
    coco = json.load(f)

images = coco['images']
annotations = coco['annotations']
categories = {cat['id']: cat['name'] for cat in coco['categories']}

# 3. 이미지 ID → 파일명 매핑
image_id_to_info = {img['id']: img for img in images}
image_id_to_annots = {}
for ann in annotations:
    image_id_to_annots.setdefault(ann['image_id'], []).append(ann)

# 4. 이미지 리스트 분할 (train/val/test)
image_ids = list(image_id_to_info.keys())
train_ids, test_ids = train_test_split(image_ids, test_size=0.2, random_state=42)
train_ids, val_ids = train_test_split(train_ids, test_size=0.2, random_state=42)

splits = {'train': train_ids, 'val': val_ids, 'test': test_ids}
for split in splits:
    os.makedirs(f'{output_dir}/images/{split}', exist_ok=True)
    os.makedirs(f'{output_dir}/labels/{split}', exist_ok=True)

# 5. COCO → YOLO Segmentation 변환
def convert_segmentation_to_yolo(segmentation, width, height):
    yolo_seg = []
    for i in range(0, len(segmentation), 2):
        x = segmentation[i] / width
        y = segmentation[i + 1] / height
        yolo_seg.extend([x, y])
    return yolo_seg

for split, ids in splits.items():
    for img_id in ids:
        img_info = image_id_to_info[img_id]
        file_name = img_info['file_name']
        width = img_info['width']
        height = img_info['height']

        # 이미지 복사
        src_img = os.path.join(image_dir, file_name)
        dst_img = os.path.join(output_dir, f'images/{split}/{file_name}')
        shutil.copy2(src_img, dst_img)

        # 라벨 생성
        label_lines = []
        for ann in image_id_to_annots.get(img_id, []):
            cls_id = ann['category_id']
            segs = ann['segmentation']
            if not segs or type(segs[0]) != list:
                continue  # 예외처리: RLE 형식은 무시
            for seg in segs:
                yolo_seg = convert_segmentation_to_yolo(seg, width, height)
                label_line = f"{cls_id} " + " ".join(map(str, yolo_seg))
                label_lines.append(label_line)

        # 라벨 파일 저장
        label_name = os.path.splitext(file_name)[0] + ".txt"
        label_path = os.path.join(output_dir, f'labels/{split}/{label_name}')
        with open(label_path, 'w') as f:
            f.write("\n".join(label_lines))