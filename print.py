from ultralytics import YOLO

model = YOLO("/Users/sarawit/Documents/Year2/sem2/Robotics_Lab/weapon_datection/best.pt")

print(model.names)
