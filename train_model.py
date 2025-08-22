from ultralytics import YOLO

def run_training():
    model = YOLO("yolov8n.pt")  # или твоя модель
    res = model.train(data='data.yaml', epochs=100, imgsz=640, workers=0)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # особенно важно при сборке .exe, но можно оставить
    run_training()