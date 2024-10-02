from ultralytics import YOLO

model = YOLO('/path/to/best.pt')

results = model('0', show=True, save=True)  # Use video path or '0' for webcam

