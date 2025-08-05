from ultralytics import YOLO

# 1. 모델 로드
model = YOLO("runs/segment/yolov8n-seg-apple/weights/best.pt")

# 2. 추론 실행
results = model.predict(
    source="dataset/images/test",  # 추론 대상 이미지 경로
    conf=0.25,
    save=True,                     # 추론 결과 이미지 저장
    project="inference/results",  # 결과 저장 폴더
    name="pred_masks",            # 하위 폴더 이름
    save_txt=True,                # 예측 마스크 txt 저장
    save_conf=True                # confidence도 함께 저장
)

print("✅ 추론 이미지 저장 위치:", results[0].save_dir)