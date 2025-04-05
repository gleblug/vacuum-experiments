import cv2
import numpy as np
from collections import deque

class CameraProcessor:
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)

        self.drawing = False  # Индикатор того, что мы сейчас рисуем
        self.start_x, self.start_y = -1, -1  # Начальная точка
        self.end_x, self.end_y = -1, -1  # Конечная точка
        self.roi_selected = False  # Флаг выбранной области
    
    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        # Создаем копию кадра для отображения
        display_frame = frame.copy()
        cv2.imshow('Video', frame)
        
        # Если в процессе выделения, рисуем текущий прямоугольник
        if self.drawing:
            cv2.rectangle(display_frame, (self.start_x, self.start_y), (self.end_x, self.end_y), (0, 255, 0), 2)
        # Если область выделена, обрабатываем её
        if self.roi_selected:
            # Определяем координаты прямоугольника в правильном порядке
            x1, y1 = min(self.start_x, self.end_x), min(self.start_y, self.end_y)
            x2, y2 = max(self.start_x, self.end_x), max(self.start_y, self.end_y)
            
            # Проверяем, что область имеет размер
            if x1 != x2 and y1 != y2:
                # Выделяем выбранную область
                roi = frame[y1:y2, x1:x2]
                
                # Здесь ваша логика обработки ROI
                mask = self.wire_brightness(roi)
                cv2.imshow('Mask', cv2.bitwise_and(roi, roi, mask=mask))
                
                # Рисуем рамку выделенной области
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                key = cv2.waitKey(1) & 0xFF
                return roi, mask
        return None, None

    def zoom_at(img, zoom, coord=None):
        h, w, _ = [ zoom * i for i in img.shape ]
        
        if coord is None: cx, cy = w/2, h/2
        else: cx, cy = [ zoom*c for c in coord ]
        
        img = cv2.resize( img, (0, 0), fx=zoom, fy=zoom)
        img = img[ int(round(cy - h/zoom * .5)) : int(round(cy + h/zoom * .5)),
                int(round(cx - w/zoom * .5)) : int(round(cx + w/zoom * .5)),
                : ]
        
        return img

    def wire_brightness(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Бинаризация изображения для выделения тонкой нити
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # В случае, если нить белая на темном фоне, инвертируем бинарное изображение
        white_pixels = cv2.countNonZero(binary)
        black_pixels = binary.size - white_pixels
        if white_pixels > black_pixels:
            binary = cv2.bitwise_not(binary)

        # Удаление шумов с помощью морфологических операций (опционально)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Поиск контуров нити
        contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Если нет контуров, выходим из программы
        if len(contours) == 0:
            return 0, None

        # Находим наибольший контур по площади (предполагаем, что это нить)
        largest_contour = max(contours, key=cv2.contourArea)

        # Создание маски для нити
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        return mask

    def mouse_callback(self, event, x, y, flags, param):        
        # Начало выделения области - нажатие левой кнопки мыши
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_x, self.start_y = x, y
            self.end_x, self.end_y = x, y
            self.roi_selected = False
        
        # Отслеживание перемещения мыши во время выделения
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_x, self.end_y = x, y
        
        # Завершение выделения - отпускание кнопки мыши
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_x, self.end_y = x, y
            self.roi_selected = True
