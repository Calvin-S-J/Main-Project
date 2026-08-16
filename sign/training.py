from ultralytics import YOLO
import torch

if __name__ == '__main__':
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    
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
        name='traffic_signs',        
        exist_ok=True,       
        pretrained=True,          
        optimizer='auto',      
        verbose=True,         
        seed=42,                  
        deterministic=True,     
        amp=True,             
    )

  
    print("Training completed!")
    print(f"Results saved to: {results.save_dir}")