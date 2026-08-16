import torch
torch.hub.set_dir("D:/torch_cache")
from ultralytics import YOLO

def main():
    model = YOLO('yolov5s.pt')

    results = model.train(
        data='data.yaml',           
        epochs=50,                   
        imgsz=640,                  
        batch=16,                    
        device=0,                   
        workers=8,                
        patience=50,            
        save=True,               
        project='runs/train',        
        name='vehicle',        
        exist_ok=True,       
        pretrained=True,          
        optimizer='auto',      
        verbose=True,         
        seed=42,                  
        deterministic=True,     
        amp=True,             
    )

if __name__ == "__main__":
    main()
