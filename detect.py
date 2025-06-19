import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('runs/train/DABF-Net/weights/best.pt') # select your model.pt path
    model.predict(source='dataset/NCHU-A2G-SIRST/images/test',
                  project='runs/detect',
                  name='exp',
                  save=True,
                #   visualize=True # visualize model features maps
                  )