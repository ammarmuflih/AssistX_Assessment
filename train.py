from ultralytics import YOLO
from multiprocessing import freeze_support

data = r"C:\Users\Muhammad Ammar M\Documents\Daftar Kerja\AssitsX\Dataset\data.yaml"

def train():
    model = YOLO('yolov8s')
    model.train(
        data=data,
        imgsz=640,
        batch=16,
        epochs=1,
        name='test' 
    )

if __name__ == '__main__':
    freeze_support()
    train()