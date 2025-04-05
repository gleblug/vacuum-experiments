import cv2
import numpy as np
import sys
import os
from datetime import datetime
import csv
from typing import Tuple, List, Optional, Union

# PyQt импорты
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QFileDialog, QTabWidget, QGraphicsView, QGraphicsScene, 
    QSizePolicy
)
from PyQt6.QtCore import (
    QThread, pyqtSignal, pyqtSlot, Qt, QRectF, QPointF, QPoint
)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor

# Научные библиотеки
from skimage.morphology import skeletonize
from scipy.interpolate import splprep, splev
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector


class ZoomableImageView(QGraphicsView):
    """
    Виджет для отображения изображения с возможностью масштабирования и выделения ROI.
    
    Сигналы:
        mouse_pressed: Сигнализирует о нажатии кнопки мыши (x, y)
        mouse_moved: Сигнализирует о перемещении мыши (x, y)
        mouse_released: Сигнализирует об отпускании кнопки мыши (x, y)
    """
    mouse_pressed = pyqtSignal(int, int)
    mouse_moved = pyqtSignal(int, int)
    mouse_released = pyqtSignal(int, int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._init_variables()
    
    def _setup_ui(self):
        """Настройка пользовательского интерфейса"""
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        # Настройки масштабирования
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        
        # Настройки для производительности
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setOptimizationFlags(QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing)
        
        # Отключаем полосы прокрутки
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Отслеживание движений мыши
        self.setMouseTracking(True)
    
    def _init_variables(self):
        """Инициализация переменных"""
        # Масштаб
        self.zoom_factor = 1.15
        self.current_zoom = 1.0
        self.first_image_loaded = False
        
        # Переменные для рисования прямоугольника
        self.drawing = False
        self.start_point = QPointF(0, 0)
        self.current_point = QPointF(0, 0)
        
        # Переменные для перемещения
        self.panning = False
        self.last_pan_point = QPoint()
        
        # Изображение
        self.pixmap_item = None
        
        # Запоминаем текущую трансформацию
        self.current_transform = self.transform()
    
    def setImage(self, image: np.ndarray) -> None:
        """
        Устанавливает новое изображение для отображения
        
        Args:
            image: Изображение в формате NumPy array
        """
        if image is None:
            return
        
        # Сохраняем текущую трансформацию
        old_transform = self.transform()
        
        # Преобразуем изображение OpenCV в QPixmap
        if len(image.shape) == 3:
            # Цветное изображение
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        else:
            # Черно-белое изображение
            h, w = image.shape
            q_image = QImage(image.data, w, h, w, QImage.Format.Format_Grayscale8)
        
        pixmap = QPixmap.fromImage(q_image)
        
        # Очищаем сцену и добавляем новое изображение
        self.scene.clear()
        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.scene.setSceneRect(QRectF(0, 0, w, h))
        
        # Только при первой загрузке подгоняем под размер и сбрасываем трансформацию
        if not self.first_image_loaded:
            self.resetTransform()
            self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.first_image_loaded = True
        else:
            # Восстанавливаем предыдущую трансформацию
            self.setTransform(old_transform)
    
    def wheelEvent(self, event):
        """Обработка колесика мыши для зумирования"""
        if self.pixmap_item is None:
            return
        
        # Определяем направление и коэффициент масштабирования
        factor = self.zoom_factor if event.angleDelta().y() > 0 else 1 / self.zoom_factor
        self.current_zoom *= factor
        
        # Ограничение масштаба
        if self.current_zoom > 10.0:
            factor = 10.0 / (self.current_zoom / factor)
            self.current_zoom = 10.0
        elif self.current_zoom < 0.1:
            factor = 0.1 / (self.current_zoom / factor)
            self.current_zoom = 0.1
        
        # Применяем масштабирование
        self.scale(factor, factor)
        
        # Сохраняем текущую трансформацию
        self.current_transform = self.transform()
    
    def mousePressEvent(self, event):
        """Обработка нажатия кнопки мыши"""
        if not self.pixmap_item:
            return super().mousePressEvent(event)
        
        if event.button() == Qt.MouseButton.LeftButton:
            # Начинаем рисование прямоугольника
            self.drawing = True
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.start_point = self.mapToScene(event.position().toPoint())
            self.current_point = self.start_point
            
            # Сигнализируем о начале выделения
            self.mouse_pressed.emit(int(self.start_point.x()), int(self.start_point.y()))
            
            # Обновляем виджет для рисования прямоугольника
            self.viewport().update()
        
        elif event.button() == Qt.MouseButton.RightButton:
            # Инициализируем перемещение (панорамирование)
            self.panning = True
            self.last_pan_point = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        else:
            super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Обработка движения мыши"""
        if self.drawing:
            # Обновляем текущую точку при рисовании прямоугольника
            self.current_point = self.mapToScene(event.position().toPoint())
            
            # Сигнализируем о движении
            self.mouse_moved.emit(int(self.current_point.x()), int(self.current_point.y()))
            
            # Обновляем виджет для перерисовки прямоугольника
            self.viewport().update()
        elif self.panning:
            # Перемещение (панорамирование)
            new_pos = event.position().toPoint()
            delta = new_pos - self.last_pan_point
            self.last_pan_point = new_pos
            
            # Применяем смещение
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
        else:
            super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Обработка отпускания кнопки мыши"""
        if event.button() == Qt.MouseButton.LeftButton and self.drawing:
            # Завершаем рисование прямоугольника
            self.current_point = self.mapToScene(event.position().toPoint())
            self.drawing = False
            
            # Сигнализируем о завершении выделения
            self.mouse_released.emit(int(self.current_point.x()), int(self.current_point.y()))
            
            # Обновляем виджет, чтобы удалить временный прямоугольник
            self.viewport().update()
        
        elif event.button() == Qt.MouseButton.RightButton and self.panning:
            # Завершаем перемещение
            self.panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            super().mouseReleaseEvent(event)
    
    def paintEvent(self, event):
        """Переопределяем метод отрисовки для добавления прямоугольника выделения"""
        # Сначала рисуем все остальное (изображение и т.д.)
        super().paintEvent(event)
        
        # Теперь рисуем прямоугольник выделения, если в процессе рисования
        if self.drawing and self.start_point != self.current_point:
            painter = QPainter(self.viewport())
            painter.setPen(QPen(QColor(0, 255, 0), 2, Qt.PenStyle.SolidLine))
            
            # Преобразуем координаты сцены в координаты виджета для рисования
            start_pos = self.mapFromScene(self.start_point)
            current_pos = self.mapFromScene(self.current_point)
            
            # Рисуем прямоугольник (используя целочисленные координаты)
            x = min(start_pos.x(), current_pos.x())
            y = min(start_pos.y(), current_pos.y())
            width = abs(start_pos.x() - current_pos.x())
            height = abs(start_pos.y() - current_pos.y())
            
            painter.drawRect(x, y, width, height)
            painter.end()


class VideoThread(QThread):
    """
    Поток для захвата и обработки видео с камеры.
    
    Сигналы:
        change_pixmap_signal: Передает кадр видео
        roi_signal: Передает ROI и маску
        mask_signal: Передает обработанную маску
    """
    change_pixmap_signal = pyqtSignal(np.ndarray)
    roi_signal = pyqtSignal(np.ndarray, np.ndarray)
    mask_signal = pyqtSignal(np.ndarray)
    
    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = True
        self.drawing = False
        self.start_x, self.start_y = -1, -1
        self.end_x, self.end_y = -1, -1
        self.roi_selected = False
        self.original_frame = None
        self.current_mask = None
        self.current_roi = None
    
    def run(self):
        """Основной цикл захвата и обработки видео"""
        cap = cv2.VideoCapture(self.camera_index)
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Сохраняем оригинальный кадр для вырезания ROI
            self.original_frame = frame.copy()
            
            # Отправляем кадр для отображения
            self.change_pixmap_signal.emit(frame)
            
            # Если область выделена, обрабатываем её
            if self.roi_selected and self.original_frame is not None:
                self._process_roi()
            
            # Небольшая задержка, чтобы не перегружать CPU
            self.msleep(30)
        
        cap.release()

    def _process_roi(self):
        """Обработка выделенной области интереса (ROI)"""
        # Определяем координаты прямоугольника в правильном порядке
        x1, y1 = min(self.start_x, self.end_x), min(self.start_y, self.end_y)
        x2, y2 = max(self.start_x, self.end_x), max(self.start_y, self.end_y)
        
        # Проверяем границы изображения
        h, w = self.original_frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Проверяем, что область имеет размер
        if x1 < x2 and y1 < y2:
            # Выделяем выбранную область
            roi = self.original_frame[y1:y2, x1:x2]
            self.current_roi = roi.copy()
            
            # Проверяем, что ROI не пустой
            if roi.size > 0:
                # Обработка ROI
                mask = self.wire_brightness(roi)
                self.current_mask = mask  # Сохраняем маску
                
                # Отправляем сигнал с ROI и маской
                if mask is not None:
                    self.roi_signal.emit(roi, mask)
                    
                    # Отправляем маску для отображения
                    # Преобразуем маску в цветное изображение для лучшей визуализации
                    colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    # Подсвечиваем маску зеленым цветом
                    colored_mask[:,:,0] = 0
                    colored_mask[:,:,2] = 0
                    self.mask_signal.emit(colored_mask)
    
    def wire_brightness(self, image: np.ndarray) -> np.ndarray:
        """
        Выделяет нить на изображении и создает бинарную маску
        
        Args:
            image: Входное изображение
            
        Returns:
            Бинарная маска выделенной нити
        """
        if image is None or image.size == 0:
            return None
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Бинаризация изображения для выделения тонкой нити
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # В случае, если нить белая на темном фоне, инвертируем бинарное изображение
        white_pixels = cv2.countNonZero(binary)
        black_pixels = binary.size - white_pixels
        if white_pixels > black_pixels:
            binary = cv2.bitwise_not(binary)

        # Удаление шумов с помощью морфологических операций
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Поиск контуров нити
        contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Если нет контуров, возвращаем пустую маску
        if not contours:
            return np.zeros_like(gray)

        # Находим наибольший контур по площади
        largest_contour = max(contours, key=cv2.contourArea)

        # Создание маски для нити
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        return mask
    
    def handle_mouse_event(self, event_type: str, x: int, y: int) -> None:
        """
        Обработка событий мыши
        
        Args:
            event_type: Тип события ("press", "move", "release")
            x: Координата X
            y: Координата Y
        """
        if event_type == "press":
            self.drawing = True
            self.start_x, self.start_y = x, y
            self.end_x, self.end_y = x, y
            self.roi_selected = False
        
        elif event_type == "move" and self.drawing:
            self.end_x, self.end_y = x, y
        
        elif event_type == "release":
            self.end_x, self.end_y = x, y
            # Проверяем, что выделена реальная область, а не точка
            if abs(self.end_x - self.start_x) > 5 and abs(self.end_y - self.start_y) > 5:
                self.drawing = False
                self.roi_selected = True
            else:
                # Если выделение слишком маленькое, сбрасываем
                self.drawing = False
                self.roi_selected = False

    def stop(self):
        """Останавливает поток обработки видео"""
        self.running = False
        self.wait()


class ThreadBrightnessAnalyzer(QThread):
    """
    Поток для анализа яркости нити
    
    Сигналы:
        update_plot_signal: Передает данные для обновления графика
        update_image_signal: Передает визуализацию результата анализа
    """
    update_plot_signal = pyqtSignal(np.ndarray, np.ndarray)
    update_image_signal = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self.roi = None
        self.mask = None
        self.running = True
        self.process_now = False
        self.result_image = None
    
    def set_data(self, roi: np.ndarray, mask: np.ndarray) -> None:
        """
        Устанавливает данные для анализа
        
        Args:
            roi: Область интереса (изображение)
            mask: Бинарная маска нити
        """
        self.roi = roi
        self.mask = mask
    
    def trigger_analysis(self) -> None:
        """Запускает процесс анализа"""
        self.process_now = True
    
    def run(self):
        """Основной цикл анализа"""
        while self.running:
            if self.process_now and self.roi is not None and self.mask is not None:
                print("Запуск анализа нити...")
                smoothed_points, brightness_values = self.analyze(self.roi, self.mask)
                if smoothed_points is not None and brightness_values is not None:
                    # Вычисляем расстояние вдоль нити
                    x_points, y_points = smoothed_points
                    dx = np.diff(x_points)
                    dy = np.diff(y_points)
                    segment_lengths = np.sqrt(dx**2 + dy**2)
                    distances = np.zeros_like(x_points)
                    distances[1:] = np.cumsum(segment_lengths)
                    
                    self.update_plot_signal.emit(distances, brightness_values)
                self.process_now = False
            
            self.msleep(100)
    
    def analyze(self, image: np.ndarray, thread_mask: np.ndarray, thickness: int = 5) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray]], Optional[np.ndarray]]:
        """
        Анализирует изображение и строит график яркости вдоль нити
        
        Args:
            image: Входное изображение
            thread_mask: Бинарная маска нити
            thickness: Толщина линии для измерения яркости
            
        Returns:
            Кортеж из (smoothed_points, brightness_values)
        """        
        # Получаем скелет маски нити
        skeleton = self._get_skeleton(thread_mask)
        
        # Получаем упорядоченные точки центральной линии нити
        centerline_points = self._get_centerline_points(skeleton)
        
        # Аппроксимируем центральную линию сплайном
        smoothed_points = self._smooth_centerline(centerline_points)
        
        # Если недостаточно точек, выходим
        if smoothed_points is None or len(smoothed_points[0]) < 2:
            return None, None
        
        # Измеряем яркость вдоль аппроксимированной центральной линии
        distances, brightness_values = self._measure_brightness(image, smoothed_points, thickness)
        
        # Визуализируем результаты
        vis_image = self._visualize_results(image, smoothed_points)
        self.result_image = vis_image.copy()  # Сохраняем для будущего использования
        self.update_image_signal.emit(vis_image)
        
        return smoothed_points, brightness_values
    
    def _get_skeleton(self, mask: np.ndarray) -> np.ndarray:
        """
        Получает скелет маски нити
        
        Args:
            mask: Бинарная маска нити
            
        Returns:
            Скелетизированная маска
        """
        binary_mask = mask.astype(bool)
        skeleton = skeletonize(binary_mask)
        return skeleton.astype(np.uint8) * 255
    
    def _get_centerline_points(self, skeleton: np.ndarray) -> np.ndarray:
        """
        Извлекает и упорядочивает точки центральной линии
        
        Args:
            skeleton: Скелетизированная маска
            
        Returns:
            Массив координат точек центральной линии
        """
        # Находим точки скелета
        y_coords, x_coords = np.where(skeleton > 0)
        
        if len(x_coords) == 0:
            return np.array([])
            
        # Объединяем координаты
        points = np.column_stack((x_coords, y_coords))
        
        # Находим две наиболее удаленные точки (концы нити)
        max_dist = 0
        endpoints = None
        
        # Если много точек, выбираем подмножество для ускорения
        if len(points) > 100:
            subset = points[np.random.choice(len(points), 100, replace=False)]
        else:
            subset = points
            
        for i in range(len(subset)):
            for j in range(i+1, len(subset)):
                dist = np.sum((subset[i] - subset[j])**2)
                if dist > max_dist:
                    max_dist = dist
                    endpoints = (subset[i], subset[j])
        
        if endpoints is None:
            return points
            
        # Ищем путь между концами нити
        ordered_points = self._find_path_between_endpoints(skeleton, endpoints)
        
        return np.array(ordered_points)
    
    def _find_path_between_endpoints(self, skeleton: np.ndarray, endpoints: Tuple[np.ndarray, np.ndarray]) -> List[Tuple[int, int]]:
        """
        Находит путь между двумя концами нити на скелете
        
        Args:
            skeleton: Скелетизированная маска
            endpoints: Кортеж из двух точек - начало и конец нити
            
        Returns:
            Список точек пути
        """
        start, end = endpoints
        start = tuple(start)
        end = tuple(end)
        
        # Создаем копию скелета для поиска пути
        visited = np.zeros_like(skeleton, dtype=bool)
        queue = [(start, [start])]
        visited[start[1], start[0]] = True
        
        # Смещения для 8-связных соседей
        neighbors = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        while queue:
            (x, y), path = queue.pop(0)
            
            # Если дошли до конечной точки
            if (x, y) == end:
                return path
                
            # Проверяем соседей
            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy
                
                # Проверяем границы изображения
                if nx < 0 or ny < 0 or nx >= skeleton.shape[1] or ny >= skeleton.shape[0]:
                    continue
                    
                # Если точка на скелете и еще не посещена
                if skeleton[ny, nx] > 0 and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append(((nx, ny), path + [(nx, ny)]))
        
        # Если путь не найден, возвращаем все точки скелета
        y_coords, x_coords = np.where(skeleton > 0)
        return list(zip(x_coords, y_coords))
    
    def _smooth_centerline(self, points: np.ndarray, smoothing: int = 50) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Аппроксимирует центральную линию нити сплайном
        
        Args:
            points: Массив точек центральной линии
            smoothing: Параметр сглаживания сплайна
            
        Returns:
            Кортеж из массивов x и y координат сглаженной линии
        """
        if len(points) < 4:
            return None
            
        # Разделяем x и y координаты
        x = points[:, 0]
        y = points[:, 1]
        
        # Используем сплайн для сглаживания
        try:
            tck, u = splprep([x, y], s=smoothing)
            # Генерируем равномерно распределенные точки вдоль сплайна
            u_new = np.linspace(0, 1, 200)
            x_new, y_new = splev(u_new, tck)
            return np.array([x_new, y_new])
        except Exception as e:
            print(f"Ошибка при сглаживании: {e}")
            # Если не удалось аппроксимировать сплайном, используем полином
            try:
                # Сортируем точки по x-координате
                sorted_idx = np.argsort(x)
                x_sorted = x[sorted_idx]
                y_sorted = y[sorted_idx]
                
                # Аппроксимируем полиномом
                degree = min(5, len(x) - 1)
                coeffs = np.polyfit(x_sorted, y_sorted, degree)
                poly = np.poly1d(coeffs)
                
                # Генерируем равномерно распределенные точки
                x_new = np.linspace(min(x), max(x), 200)
                y_new = poly(x_new)
                return np.array([x_new, y_new])
            except Exception as e:
                print(f"Ошибка при аппроксимации полиномом: {e}")
                return None
    
    def _measure_brightness(self, image: np.ndarray, smoothed_points: Tuple[np.ndarray, np.ndarray], thickness: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Измеряет яркость изображения вдоль нити
        
        Args:
            image: Изображение
            smoothed_points: Кортеж из массивов x и y координат
            thickness: Толщина линии для измерения
            
        Returns:
            Кортеж из массивов расстояний и значений яркости
        """
        # Конвертируем в grayscale, если изображение цветное
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        x_points, y_points = smoothed_points
        brightness = np.zeros(len(x_points))
        
        # Вычисляем расстояние вдоль нити
        dx = np.diff(x_points)
        dy = np.diff(y_points)
        segment_lengths = np.sqrt(dx**2 + dy**2)
        distances = np.zeros_like(x_points)
        distances[1:] = np.cumsum(segment_lengths)
        
        # Для каждой точки центральной линии
        for i in range(len(x_points)):
            x, y = int(round(x_points[i])), int(round(y_points[i]))
            
            # Проверяем границы изображения
            if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
                # Определяем область вокруг точки
                x_min = max(0, x - thickness // 2)
                x_max = min(gray.shape[1], x + thickness // 2 + 1)
                y_min = max(0, y - thickness // 2)
                y_max = min(gray.shape[0], y + thickness // 2 + 1)
                
                # Извлекаем область и измеряем среднюю яркость
                region = gray[y_min:y_max, x_min:x_max]
                if region.size > 0:
                    brightness[i] = np.mean(region)
        
        return distances, brightness
    
    def _visualize_results(self, image: np.ndarray, smoothed_points: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Визуализирует результаты анализа
        
        Args:
            image: Исходное изображение
            smoothed_points: Кортеж из массивов x и y координат
            
        Returns:
            Изображение с визуализацией
        """
        # Визуализируем центральную линию на изображении
        vis_image = image.copy()
        x_points, y_points = smoothed_points
        
        # Рисуем центральную линию
        points = np.column_stack((x_points.astype(np.int32), y_points.astype(np.int32)))
        for i in range(1, len(points)):
            cv2.line(vis_image, tuple(points[i-1]), tuple(points[i]), (0, 255, 0), 1)
        
        # Добавляем точки для наглядности
        for i in range(0, len(points), 10):
            cv2.circle(vis_image, tuple(points[i]), 3, (0, 0, 255), -1)
        
        return vis_image
    
    def stop(self):
        """Останавливает поток анализа"""
        self.running = False
        self.wait()


class MatplotlibCanvas(FigureCanvasQTAgg):
    """Canvas для отображения графиков matplotlib"""
    def __init__(self, parent=None, width=5, height=4, dpi=100, xlabel="X", ylabel="Y"):
        # Создаем фигуру matplotlib
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super(MatplotlibCanvas, self).__init__(self.fig)
        
        # Инициализация графика
        self.line, = self.ax.plot([], [], 'r-', lw=2)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.grid(True)


class GraphApp(QWidget):
    """Виджет для отображения и анализа графиков"""
    def __init__(self, xlabel="X", ylabel="Y"):
        super().__init__()
        
        # Данные графика
        self.x_data = np.array([])
        self.y_data = np.array([])
        
        # Сохраняем метки осей
        self.xlabel = xlabel
        self.ylabel = ylabel
        
        # Инициализация UI
        self.init_ui()
        
    def init_ui(self):
        """Инициализация пользовательского интерфейса"""
        # Создание и добавление компоновки
        main_layout = QVBoxLayout(self)
        
        # Создание и добавление холста matplotlib
        self.canvas = MatplotlibCanvas(self, width=8, height=6, dpi=100, 
                                      xlabel=self.xlabel, ylabel=self.ylabel)
        main_layout.addWidget(self.canvas)
        
        # Создание компоновки для кнопок
        button_layout = QHBoxLayout()
        
        # Кнопки сохранения
        self.save_image_btn = QPushButton("Сохранить как изображение")
        self.save_image_btn.clicked.connect(self.save_as_image)
        button_layout.addWidget(self.save_image_btn)
        
        self.save_data_btn = QPushButton("Сохранить точки данных")
        self.save_data_btn.clicked.connect(self.save_data_points)
        button_layout.addWidget(self.save_data_btn)
        
        # Добавление компоновки кнопок в главную компоновку
        main_layout.addLayout(button_layout)
        
        # Добавление строки состояния
        self.status_label = QLabel("Готов")
        main_layout.addWidget(self.status_label)
        
        # Добавление SpanSelector для выбора диапазона
        self.span_selector = SpanSelector(
            self.canvas.ax, 
            self.on_range_selected, 
            'horizontal', 
            useblit=True,
            props=dict(alpha=0.5, facecolor='lightblue'),
            interactive=True,
            drag_from_anywhere=True
        )
    
    def update_plot(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        """
        Обновление графика с переданными данными
        
        Args:
            x_data: Данные по оси X
            y_data: Данные по оси Y
        """
        self.x_data = x_data
        self.y_data = y_data
        
        # Обновление графика
        self.canvas.line.set_data(self.x_data, self.y_data)
        self.canvas.ax.relim()
        self.canvas.ax.autoscale_view(True, True, True)
        self.canvas.draw()
    
    def save_as_image(self):
        """Сохранение графика как изображения"""
        if len(self.x_data) == 0:
            self.status_label.setText("Нет данных для сохранения")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"график_{timestamp}.png"
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Сохранить график как изображение", default_name, 
            "Изображения (*.png *.jpg *.jpeg *.pdf *.svg);;Все файлы (*)"
        )
        
        if filename:
            self.canvas.fig.savefig(filename, dpi=300, bbox_inches='tight')
            self.status_label.setText(f"График сохранен как {os.path.basename(filename)}")
    
    def save_data_points(self):
        """Сохранение точек данных в CSV файл"""
        if len(self.x_data) == 0:
            self.status_label.setText("Нет данных для сохранения")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"данные_{timestamp}.csv"
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Сохранить данные как CSV", default_name, 
            "CSV файлы (*.csv);;Все файлы (*)"
        )
        
        if filename:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([self.xlabel, self.ylabel])
                for x, y in zip(self.x_data, self.y_data):
                    writer.writerow([x, y])
            
            self.status_label.setText(f"Данные сохранены как {os.path.basename(filename)}")
    
    def on_range_selected(self, min_val: float, max_val: float) -> None:
        """
        Обработка выбора диапазона и расчет среднего значения
        
        Args:
            min_val: Минимальное значение выбранного диапазона
            max_val: Максимальное значение выбранного диапазона
        """
        if len(self.x_data) == 0 or len(self.y_data) == 0:
            return
            
        # Находим индексы данных в выбранном диапазоне
        indices = [i for i, x in enumerate(self.x_data) if min_val <= x <= max_val]
        
        if not indices:
            self.status_label.setText("В выбранном диапазоне нет данных")
            return
            
        # Рассчитываем среднее значение y в выбранном диапазоне
        selected_y_values = [self.y_data[i] for i in indices]
        mean_value = sum(selected_y_values) / len(selected_y_values)
        
        # Показываем информацию в статусной строке
        self.status_label.setText(
            f"Диапазон X: [{min_val:.4f}, {max_val:.4f}], "
            f"Среднее значение {self.ylabel}: {mean_value:.4f}, "
            f"Количество точек: {len(indices)}"
        )


class MainWindow(QMainWindow):
    """Основное окно приложения"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Анализ яркости нити")
        self.setGeometry(100, 100, 1600, 900)
        
        # Инициализация интерфейса
        self._init_ui()
        
        # Инициализация рабочих потоков
        self._init_threads()
        
        # Инициализация обработчиков событий
        self._init_event_handlers()
    
    def _init_ui(self):
        """Инициализация пользовательского интерфейса"""
        # Создаем вкладки
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # Создаем вкладку для анализа
        self.analysis_tab = QWidget()
        self.tabs.addTab(self.analysis_tab, "Анализ изображения")
        
        # Создаем вкладку для графика
        self.graph_app = GraphApp('Расстояние вдоль нити (пиксели)', 'Яркость')
        self.tabs.addTab(self.graph_app, "График")
        
        # Компоновка для вкладки анализа
        analysis_layout = QVBoxLayout(self.analysis_tab)
        
        # Верхняя панель с видео
        top_panel = QWidget()
        top_layout = QVBoxLayout(top_panel)
        
        # Виджет для отображения видео с возможностью зума
        self.video_view = ZoomableImageView()
        self.video_view.setMinimumSize(800, 600)
        top_layout.addWidget(self.video_view)
        
        # Добавляем подсказку по использованию
        help_label = QLabel("Колесико мыши для масштабирования, ЛКМ для выделения области, ПКМ для перемещения")
        help_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top_layout.addWidget(help_label)
        
        # Панель с кнопками
        button_panel = QWidget()
        button_layout = QHBoxLayout(button_panel)
        
        # Кнопка для запуска анализа
        self.analyze_button = QPushButton("Анализировать нить")
        self.analyze_button.clicked.connect(self.start_analysis)
        button_layout.addWidget(self.analyze_button)
        
        # Кнопка для сохранения всех изображений
        self.save_all_button = QPushButton("Сохранить все изображения")
        self.save_all_button.clicked.connect(self.save_all_images)
        button_layout.addWidget(self.save_all_button)
        
        top_layout.addWidget(button_panel)
        analysis_layout.addWidget(top_panel)
        
        # Нижняя панель для ROI, маски и анализа
        bottom_panel = QWidget()
        bottom_layout = QHBoxLayout(bottom_panel)
        
        # ROI виджет с возможностью зума
        self.roi_view = ZoomableImageView()
        self.roi_view.setMinimumSize(320, 240)
        bottom_layout.addWidget(self.roi_view)
        
        # Виджет для маски с возможностью зума
        self.mask_view = ZoomableImageView()
        self.mask_view.setMinimumSize(320, 240)
        bottom_layout.addWidget(self.mask_view)
        
        # Виджет анализа с возможностью зума
        self.analysis_view = ZoomableImageView()
        self.analysis_view.setMinimumSize(320, 240)
        bottom_layout.addWidget(self.analysis_view)
        
        analysis_layout.addWidget(bottom_panel)
        
        # Статус бар
        self.statusBar().showMessage("Готов")
    
    def _init_threads(self):
        """Инициализация рабочих потоков"""
        # Создаем поток для обработки видео
        self.video_thread = VideoThread(0)
        
        # Создаем обработчик для анализа яркости
        self.analyzer = ThreadBrightnessAnalyzer()
        
        # Запускаем потоки
        self.video_thread.start()
        self.analyzer.start()
    
    def _init_event_handlers(self):
        """Инициализация обработчиков событий"""
        # Соединяем сигналы и слоты
        self.video_thread.change_pixmap_signal.connect(self.update_video)
        self.video_thread.roi_signal.connect(self.update_roi)
        self.video_thread.mask_signal.connect(self.update_mask)
        self.analyzer.update_plot_signal.connect(self.update_plot)
        self.analyzer.update_image_signal.connect(self.update_analysis)
        
        # Соединяем сигналы мыши
        self.video_view.mouse_pressed.connect(self.on_mouse_pressed)
        self.video_view.mouse_moved.connect(self.on_mouse_moved)
        self.video_view.mouse_released.connect(self.on_mouse_released)
    
    @pyqtSlot(np.ndarray)
    def update_video(self, cv_img: np.ndarray) -> None:
        """
        Обновляет изображение с камеры
        
        Args:
            cv_img: Изображение с камеры
        """
        self.video_view.setImage(cv_img)
    
    @pyqtSlot(np.ndarray, np.ndarray)
    def update_roi(self, roi: np.ndarray, mask: np.ndarray) -> None:
        """
        Обновляет ROI и отправляет данные для анализа
        
        Args:
            roi: Выделенная область интереса
            mask: Маска нити
        """
        self.roi_view.setImage(roi)
        self.analyzer.set_data(roi, mask)
        if roi is not None:
            h, w = roi.shape[:2]
            self.statusBar().showMessage(f"Выделена область: {w}x{h} пикселей")
    
    @pyqtSlot(np.ndarray)
    def update_mask(self, mask: np.ndarray) -> None:
        """
        Обновляет отображение маски
        
        Args:
            mask: Обработанная маска
        """
        self.mask_view.setImage(mask)
    
    def start_analysis(self) -> None:
        """Запуск анализа нити по нажатию кнопки"""
        if self.video_thread.current_mask is not None:
            self.statusBar().showMessage("Запуск анализа нити...")
            self.analyzer.trigger_analysis()
        else:
            self.statusBar().showMessage("Нет выделенной области для анализа", 5000)
    
    def save_all_images(self) -> None:
        """Сохранить все изображения с одним базовым именем"""
        if (self.video_thread.original_frame is None or 
            self.video_thread.current_roi is None or 
            self.video_thread.current_mask is None or 
            self.analyzer.result_image is None):
            self.statusBar().showMessage("Нет изображений для сохранения", 5000)
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"анализ_нити_{timestamp}"
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Задайте базовое имя файла", default_name, 
            "Все файлы (*)"
        )
        
        if filename:
            base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
            
            cv2.imwrite(f"{base_name}_original.png", self.video_thread.original_frame)
            cv2.imwrite(f"{base_name}_roi.png", self.video_thread.current_roi)
            cv2.imwrite(f"{base_name}_mask.png", self.video_thread.current_mask)
            cv2.imwrite(f"{base_name}_analysis.png", self.analyzer.result_image)
            
            self.graph_app.canvas.fig.savefig(f"{base_name}_graph.png", dpi=300, bbox_inches='tight')
            
            with open(f"{base_name}_data.csv", 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([self.graph_app.xlabel, self.graph_app.ylabel])
                for x, y in zip(self.graph_app.x_data, self.graph_app.y_data):
                    writer.writerow([x, y])
            
            self.statusBar().showMessage(f"Все изображения сохранены с базовым именем {os.path.basename(base_name)}", 5000)
    
    @pyqtSlot(np.ndarray, np.ndarray)
    def update_plot(self, distances: np.ndarray, brightness: np.ndarray) -> None:
        """
        Обновляет график с новыми данными
        
        Args:
            distances: Расстояния вдоль нити
            brightness: Значения яркости
        """
        self.graph_app.update_plot(distances, brightness)
        self.statusBar().showMessage("Анализ завершен")
        self.tabs.setCurrentIndex(1)
    
    @pyqtSlot(np.ndarray)
    def update_analysis(self, image: np.ndarray) -> None:
        """
        Обновляет изображение анализа
        
        Args:
            image: Изображение с визуализацией анализа
        """
        self.analysis_view.setImage(image)
    
    def on_mouse_pressed(self, x: int, y: int) -> None:
        """
        Обработчик нажатия кнопки мыши на видео
        
        Args:
            x: Координата X
            y: Координата Y
        """
        self.video_thread.handle_mouse_event("press", x, y)
    
    def on_mouse_moved(self, x: int, y: int) -> None:
        """
        Обработчик движения мыши на видео
        
        Args:
            x: Координата X
            y: Координата Y
        """
        self.video_thread.handle_mouse_event("move", x, y)
    
    def on_mouse_released(self, x: int, y: int) -> None:
        """
        Обработчик отпускания кнопки мыши на видео
        
        Args:
            x: Координата X
            y: Координата Y
        """
        self.video_thread.handle_mouse_event("release", x, y)
    
    def keyPressEvent(self, event):
        """Обработчик нажатия клавиш"""
        if event.key() == Qt.Key.Key_Escape:
            self.close()
    
    def closeEvent(self, event):
        """Обработка закрытия окна"""
        self.video_thread.stop()
        self.analyzer.stop()
        event.accept()


def main():
    """Точка входа в приложение"""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
