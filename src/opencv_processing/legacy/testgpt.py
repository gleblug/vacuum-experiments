import cv2
import numpy as np
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QStatusBar, QFileDialog,
                            QTabWidget, QScrollArea)
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot, Qt, QRect, QPoint, QSize
from PyQt6.QtGui import QImage, QPixmap, QMouseEvent, QKeyEvent, QPainter, QPen, QColor
from skimage.morphology import skeletonize
from scipy.interpolate import splprep, splev
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from datetime import datetime
import csv
import os
from matplotlib.widgets import SpanSelector

class VideoThread(QThread):
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
        cap = cv2.VideoCapture(self.camera_index)
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Сохраняем оригинальный кадр для вырезания ROI
            self.original_frame = frame.copy()
            
            # Рисуем прямоугольник выделения на кадре, если выделение активно
            display_frame = frame.copy()
            if self.drawing or self.roi_selected:
                x1, y1 = min(self.start_x, self.end_x), min(self.start_y, self.end_y)
                x2, y2 = max(self.start_x, self.end_x), max(self.start_y, self.end_y)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Отправляем кадр для отображения
            self.change_pixmap_signal.emit(display_frame)
            
            # Если область выделена, обрабатываем её
            if self.roi_selected and self.original_frame is not None:
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
            
            # Немного задержка, чтобы не перегружать CPU
            self.msleep(30)
        
        cap.release()
    
    def wire_brightness(self, image):
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
        if len(contours) == 0:
            return np.zeros_like(gray)

        # Находим наибольший контур по площади
        largest_contour = max(contours, key=cv2.contourArea)

        # Создание маски для нити
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        return mask
    
    def handle_mouse_event(self, event_type, x, y):
        # Обработка событий мыши
        if event_type == "press":
            self.drawing = True
            self.start_x, self.start_y = x, y
            self.end_x, self.end_y = x, y
            self.roi_selected = False
        
        elif event_type == "move":
            if self.drawing:
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
        self.running = False
        self.wait()

# Класс для QLabel с поддержкой событий мыши для отрисовки прямоугольника выделения
class VideoWidget(QLabel):
    mouse_pressed = pyqtSignal(int, int)
    mouse_moved = pyqtSignal(int, int)
    mouse_released = pyqtSignal(int, int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.start_pos = None
        self.current_pos = None
        self.drawing = False
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.start_pos = event.position()
            self.current_pos = event.position()
            self.mouse_pressed.emit(int(event.position().x()), int(event.position().y()))
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        if self.drawing:
            self.current_pos = event.position()
            self.update()  # Запрашиваем перерисовку для отображения прямоугольника
        self.mouse_moved.emit(int(event.position().x()), int(event.position().y()))
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.drawing:
            self.drawing = False
            self.current_pos = event.position()
            self.mouse_released.emit(int(event.position().x()), int(event.position().y()))
            self.update()  # Перерисовка для обновления или удаления временного прямоугольника
        super().mouseReleaseEvent(event)
    
    def paintEvent(self, event):
        super().paintEvent(event)
        
        # Рисуем прямоугольник непосредственно на виджете
        if self.drawing and self.start_pos and self.current_pos:
            painter = QPainter(self)
            painter.setPen(QPen(QColor(0, 255, 0), 2, Qt.PenStyle.SolidLine))
            
            # Создаем прямоугольник между начальной и текущей точками
            rect = QRect(
                QPoint(int(self.start_pos.x()), int(self.start_pos.y())), 
                QPoint(int(self.current_pos.x()), int(self.current_pos.y()))
            )
            painter.drawRect(rect.normalized())  # normalized() обеспечивает правильные координаты даже при рисовании в обратном направлении

class ThreadBrightnessAnalyzer(QThread):
    update_plot_signal = pyqtSignal(np.ndarray, np.ndarray)
    update_image_signal = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self.roi = None
        self.mask = None
        self.running = True
        self.process_now = False
        self.result_image = None
        
    def set_data(self, roi, mask):
        self.roi = roi
        self.mask = mask
    
    def trigger_analysis(self):
        self.process_now = True
    
    def run(self):
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
    
    def analyze(self, image, thread_mask, thickness=5):
        """
        Анализирует изображение и строит график яркости вдоль нити
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
    
    def _get_skeleton(self, mask):
        """Получает скелет маски нити"""
        binary_mask = mask.astype(bool)
        skeleton = skeletonize(binary_mask)
        return skeleton.astype(np.uint8) * 255
    
    def _get_centerline_points(self, skeleton):
        """Извлекает и упорядочивает точки центральной линии"""
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
    
    def _find_path_between_endpoints(self, skeleton, endpoints):
        """Находит путь между двумя концами нити на скелете"""
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
        return np.column_stack((x_coords, y_coords))
    
    def _smooth_centerline(self, points, smoothing=50):
        """Аппроксимирует центральную линию нити сплайном"""
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
    
    def _measure_brightness(self, image, smoothed_points, thickness):
        """Измеряет яркость изображения вдоль нити"""
        # Конвертируем в grayscale, если изображение цветное
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
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
    
    def _visualize_results(self, image, smoothed_points):
        """Визуализирует результаты анализа и возвращает визуализацию"""
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
        self.running = False
        self.wait()

# Canvas для matplotlib
class MatplotlibCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100, xlabel="X", ylabel="Y"):
        # Создаем фигуру matplotlib
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super(MatplotlibCanvas, self).__init__(self.fig)
        
        # Инициализация графика с настраиваемыми метками осей
        self.line, = self.ax.plot([], [], 'r-', lw=2)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.grid(True)

class GraphApp(QWidget):
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
    
    def update_plot(self, x_data, y_data):
        """Обновление графика с переданными данными"""
        self.x_data = x_data
        self.y_data = y_data
        
        # Обновление графика
        self.canvas.line.set_data(self.x_data, self.y_data)
        self.canvas.ax.relim()      # Пересчет границ
        self.canvas.ax.autoscale_view(True, True, True)  # Автомасштабирование
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
                writer.writerow([self.xlabel, self.ylabel])  # Заголовок с настроенными названиями осей
                for x, y in zip(self.x_data, self.y_data):
                    writer.writerow([x, y])
            
            self.status_label.setText(f"Данные сохранены как {os.path.basename(filename)}")
    
    def on_range_selected(self, min_val, max_val):
        """Обработка выбора диапазона и расчет среднего значения"""
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

# Класс для отображения изображения с сохранением исходного размера и добавлением прокрутки
class ScrollableImageLabel(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        
        # Создаем QLabel для отображения изображения
        self.imageLabel = QLabel()
        self.imageLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Устанавливаем политику размера
        from PyQt6.QtWidgets import QSizePolicy
        self.imageLabel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Устанавливаем QLabel как виджет для QScrollArea
        self.setWidget(self.imageLabel)
    
    def setPixmap(self, pixmap):
        # Установка изображения без масштабирования
        self.imageLabel.setPixmap(pixmap)
        self.imageLabel.adjustSize()  # Установка размера метки под размер изображения


# Основной класс приложения
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Анализ яркости нити")
        self.setGeometry(100, 100, 1600, 900)
        
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
        
        # Виджет для отображения видео с поддержкой прокрутки
        self.scroll_video = ScrollableImageLabel()
        self.video_widget = VideoWidget()
        self.scroll_video.setWidget(self.video_widget)
        top_layout.addWidget(self.scroll_video)
        
        # Панель с кнопками
        button_panel = QWidget()
        button_layout = QHBoxLayout(button_panel)
        
        # Кнопка для запуска анализа
        self.analyze_button = QPushButton("Анализировать нить")
        self.analyze_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        self.analyze_button.clicked.connect(self.start_analysis)
        button_layout.addWidget(self.analyze_button)
        
        # Кнопка для сохранения всех изображений
        self.save_all_button = QPushButton("Сохранить все изображения")
        self.save_all_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
            QPushButton:pressed {
                background-color: #0a69b7;
            }
        """)
        self.save_all_button.clicked.connect(self.save_all_images)
        button_layout.addWidget(self.save_all_button)
        
        top_layout.addWidget(button_panel)
        analysis_layout.addWidget(top_panel)
        
        # Нижняя панель для ROI, маски и анализа
        bottom_panel = QWidget()
        bottom_layout = QHBoxLayout(bottom_panel)
        
        # ROI виджет
        self.roi_widget = QLabel()
        self.roi_widget.setMinimumSize(320, 240)
        self.roi_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.roi_widget.setStyleSheet("border: 1px solid #cccccc;")
        self.roi_widget.setText("ROI будет отображен здесь")
        bottom_layout.addWidget(self.roi_widget)
        
        # Виджет для маски
        self.mask_widget = QLabel()
        self.mask_widget.setMinimumSize(320, 240)
        self.mask_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mask_widget.setStyleSheet("border: 1px solid #cccccc;")
        self.mask_widget.setText("Маска будет отображена здесь")
        bottom_layout.addWidget(self.mask_widget)
        
        # Виджет анализа
        self.analysis_widget = QLabel()
        self.analysis_widget.setMinimumSize(320, 240)
        self.analysis_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.analysis_widget.setStyleSheet("border: 1px solid #cccccc;")
        self.analysis_widget.setText("Анализ будет отображен здесь")
        bottom_layout.addWidget(self.analysis_widget)
        
        analysis_layout.addWidget(bottom_panel)
        
        # Статус бар
        self.statusBar().showMessage("Готов")
        
        # Создаем поток для обработки видео
        self.video_thread = VideoThread(0)
        
        # Создаем обработчик для анализа яркости
        self.analyzer = ThreadBrightnessAnalyzer()
        
        # Соединяем сигналы и слоты
        self.video_thread.change_pixmap_signal.connect(self.update_video)
        self.video_thread.roi_signal.connect(self.update_roi)
        self.video_thread.mask_signal.connect(self.update_mask)
        self.analyzer.update_plot_signal.connect(self.update_plot)
        self.analyzer.update_image_signal.connect(self.update_analysis)
        
        # Соединяем сигналы мыши
        self.video_widget.mouse_pressed.connect(self.on_mouse_pressed)
        self.video_widget.mouse_moved.connect(self.on_mouse_moved)
        self.video_widget.mouse_released.connect(self.on_mouse_released)
        
        # Запускаем потоки
        self.video_thread.start()
        self.analyzer.start()
    
    def convert_cv_qt(self, cv_img):
        """Конвертирует изображение OpenCV в QImage"""
        if cv_img is None:
            return QPixmap()
            
        if len(cv_img.shape) == 2:  # Если изображение в градациях серого
            h, w = cv_img.shape
            # Преобразуем grayscale в формат, который может использовать QImage
            qimage = QImage(cv_img.data.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
            return QPixmap.fromImage(qimage)
        else:  # Цветное изображение
            # Преобразуем BGR (OpenCV) в RGB (Qt)
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qimage = QImage(rgb_image.data.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888)
            return QPixmap.fromImage(qimage)
    
    @pyqtSlot(np.ndarray)
    def update_video(self, cv_img):
        """Обновляем изображение с камеры"""
        qt_img = self.convert_cv_qt(cv_img)
        self.video_widget.setPixmap(qt_img)
        # Устанавливаем размер видеовиджета по размеру изображения для отображения без масштабирования
        self.video_widget.setFixedSize(qt_img.size())
    
    @pyqtSlot(np.ndarray, np.ndarray)
    def update_roi(self, roi, mask):
        """Обновляем ROI и отправляем данные для анализа"""
        # Показываем ROI
        roi_pixmap = self.convert_cv_qt(roi)
        self.roi_widget.setPixmap(roi_pixmap)
        self.roi_widget.setFixedSize(roi_pixmap.size())
        
        # Отправляем данные в анализатор
        self.analyzer.set_data(roi, mask)
        
        # Показываем в статусной строке информацию о размере ROI
        if roi is not None:
            h, w = roi.shape[:2]
            self.statusBar().showMessage(f"Выделена область: {w}x{h} пикселей")
    
    @pyqtSlot(np.ndarray)
    def update_mask(self, mask):
        """Обновляем отображение маски"""
        mask_pixmap = self.convert_cv_qt(mask)
        self.mask_widget.setPixmap(mask_pixmap)
        self.mask_widget.setFixedSize(mask_pixmap.size())
    
    def start_analysis(self):
        """Запуск анализа нити по нажатию кнопки"""
        if self.video_thread.current_mask is not None:
            self.statusBar().showMessage("Запуск анализа нити...")
            self.analyzer.trigger_analysis()
        else:
            self.statusBar().showMessage("Нет выделенной области для анализа", 5000)
    
    def save_all_images(self):
        """Сохранить все изображения с одним базовым именем"""
        if (self.video_thread.original_frame is None or 
            self.video_thread.current_roi is None or 
            self.video_thread.current_mask is None or 
            self.analyzer.result_image is None):
            self.statusBar().showMessage("Нет изображений для сохранения", 5000)
            return
            
        # Получаем базовое имя файла от пользователя
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"анализ_нити_{timestamp}"
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Задайте базовое имя файла", default_name, 
            "Все файлы (*)"
        )
        
        if filename:
            # Получаем базовое имя без расширения
            base_name = filename
            if '.' in filename:
                base_name = filename.rsplit('.', 1)[0]
            
            # Сохраняем исходное изображение
            original_name = f"{base_name}_original.png"
            cv2.imwrite(original_name, self.video_thread.original_frame)
            
            # Сохраняем ROI
            roi_name = f"{base_name}_roi.png"
            cv2.imwrite(roi_name, self.video_thread.current_roi)
            
            # Сохраняем маску
            mask_name = f"{base_name}_mask.png"
            cv2.imwrite(mask_name, self.video_thread.current_mask)
            
            # Сохраняем результат анализа
            analysis_name = f"{base_name}_analysis.png"
            cv2.imwrite(analysis_name, self.analyzer.result_image)
            
            # Сохраняем график
            graph_name = f"{base_name}_graph.png"
            self.graph_app.canvas.fig.savefig(graph_name, dpi=300, bbox_inches='tight')
            
            # Сохраняем данные графика
            data_name = f"{base_name}_data.csv"
            with open(data_name, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([self.graph_app.xlabel, self.graph_app.ylabel])
                for x, y in zip(self.graph_app.x_data, self.graph_app.y_data):
                    writer.writerow([x, y])
            
            self.statusBar().showMessage(f"Все изображения сохранены с базовым именем {os.path.basename(base_name)}", 5000)
    
    @pyqtSlot(np.ndarray, np.ndarray)
    def update_plot(self, distances, brightness):
        """Обновляем график с новыми данными"""
        self.graph_app.update_plot(distances, brightness)
        self.statusBar().showMessage("Анализ завершен")
        # Переключаемся на вкладку с графиком
        self.tabs.setCurrentIndex(1)
    
    @pyqtSlot(np.ndarray)
    def update_analysis(self, image):
        """Обновляем изображение анализа"""
        pixmap = self.convert_cv_qt(image)
        self.analysis_widget.setPixmap(pixmap)
        self.analysis_widget.setFixedSize(pixmap.size())
    
    def on_mouse_pressed(self, x, y):
        """Обработчик нажатия кнопки мыши на видео"""
        self.video_thread.handle_mouse_event("press", x, y)
    
    def on_mouse_moved(self, x, y):
        """Обработчик движения мыши на видео"""
        self.video_thread.handle_mouse_event("move", x, y)
    
    def on_mouse_released(self, x, y):
        """Обработчик отпускания кнопки мыши на видео"""
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
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
