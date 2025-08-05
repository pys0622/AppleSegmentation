from ultralytics import YOLO

# 1. 사용할 모델 불러오기 (yolov8n-seg, yolov8s-seg 등 선택 가능)
model = YOLO("yolov8n-seg.pt")  # or yolov8s-seg.pt, yolov8m-seg.pt 등

# 2. 학습 수행
results = model.train(
    data="dataset/data.yaml",  # ← 변환된 data.yaml 경로
    epochs=50,
    imgsz=640,
    batch=16,
    name="yolov8n-seg-apple",       # 실험 이름 (결과 디렉토리에 사용됨)
    project="runs/segment",         # 결과 저장 위치 (기본: runs/segment)
    workers=4                       # 병렬 처리용 워커 수
)

# 3. 학습 결과 요약 출력
print("✅ 학습 완료!")
print(f"📁 결과 저장 위치: {results.save_dir}")