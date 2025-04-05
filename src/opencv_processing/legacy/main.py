import cv2
from brightness_analyzer import ThreadBrightnessAnalyzer
from camera_processor import CameraProcessor
from PyQt6.QtWidgets import QApplication
import sys    

def main():
    app = QApplication(sys.argv)

    analyzer = ThreadBrightnessAnalyzer()
    processor = CameraProcessor(0)
    
    while True:
        frame, mask = processor.frame()
        if frame is None:
            continue
        
        analyzer.analyze(frame, mask, thickness=2)

        # Проверка нажатия клавиш
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC - выход
            break
    
    # Освобождаем ресурсы
    analyzer.close()
    processor.close()
    # app.exec() -- нужно запускать одновременно этот и основной поток

if __name__ == "__main__":
    main()
