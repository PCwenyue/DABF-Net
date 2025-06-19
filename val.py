import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('runs/train/DABF-Net/weights/best.pt')
    model.val(data='dataset/NCHU-A2G-SIRST/NCHU-A2G-SIRST.yaml',
              split='test',
              imgsz=640,
              batch=4,
              project='runs/val',
              name='exp',
              )