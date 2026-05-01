from pathlib import Path
from loguru import logger
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.config import settings


def download_yolov8_model():
    try:
        from ultralytics import YOLO
        
        model_dir = settings.MODEL_DIR
        model_dir.mkdir(parents=True, exist_ok=True)
        
        yolo_model_path = model_dir / "yolov8n.pt"
        
        if yolo_model_path.exists():
            logger.info(f"YOLOv8n already exists at {yolo_model_path}")
            return
        
        logger.info("Downloading YOLOv8n model...")
        model = YOLO("yolov8n.pt")
        
        downloaded_path = Path("yolov8n.pt")
        if downloaded_path.exists():
            import shutil
            shutil.move(str(downloaded_path), str(yolo_model_path))
            logger.info(f"Saved YOLOv8n to {yolo_model_path}")
        else:
            logger.warning("YOLOv8n download may have failed")
            
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
    except Exception as e:
        logger.error(f"Failed to download YOLOv8: {e}")


def download_hcit_model():
    model_dir = settings.MODEL_DIR
    hcit_path = model_dir / "hcit_model.pth"
    
    if hcit_path.exists():
        logger.info(f"HCiT model already exists at {hcit_path}")
        return
    
    logger.warning(f"HCiT model not found at {hcit_path}")
    logger.info("You need to train the HCiT model first: python train_hcit.py")


if __name__ == "__main__":
    logger.info("Downloading required models...")
    download_yolov8_model()
    download_hcit_model()
    logger.info("Done")
