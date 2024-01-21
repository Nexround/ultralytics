from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model.train(cfg="test.yaml",data="mydataset.yaml")
results = model('bus.jpg')
