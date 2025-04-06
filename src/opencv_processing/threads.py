import cv2
import numpy as np
from typing import Tuple, List, Optional, Union

from PyQt6.QtCore import QThread, pyqtSignal

from skimage.morphology import skeletonize
from scipy.interpolate import splprep, splev


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
        
        # Улучшение контраста для лучшего выделения тусклых участков
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Бинаризация с пониженным порогом для захвата тусклых участков
        otsu_threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        lower_threshold = max(0, otsu_threshold - 30)  # Снижаем порог для тусклых участков
        _, binary = cv2.threshold(gray, lower_threshold, 255, cv2.THRESH_BINARY)

        # В случае, если нить белая на темном фоне, инвертируем бинарное изображение
        white_pixels = cv2.countNonZero(binary)
        black_pixels = binary.size - white_pixels
        if white_pixels > black_pixels:
            binary = cv2.bitwise_not(binary)

        # Удаление шумов с помощью морфологических операций
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Соединение разрывов в нити
        binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, kernel)

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
                distances, brightness_values = self.analyze(self.roi, self.mask)
                if distances is not None and brightness_values is not None:
                    self.update_plot_signal.emit(distances, brightness_values)
                self.process_now = False
            
            self.msleep(100)

    def analyze(self, image: np.ndarray, wire_mask: np.ndarray, thickness: int = 3) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray]], Optional[np.ndarray]]:
        """
        Анализирует изображение и строит график яркости вдоль нити
        
        Args:
            image: Входное изображение
            thread_mask: Бинарная маска нити
            thickness: Толщина линии для измерения яркости
            
        Returns:
            Кортеж из (distances, brightness_values)
        """        
        # Получаем скелет маски нити
        skeleton = self._get_skeleton(wire_mask)
        
        # Получаем упорядоченные точки центральной линии нити
        centerline_points = self._get_centerline_points(skeleton)
        
        # Аппроксимируем центральную линию сплайном
        smoothed_points = self._smooth_centerline(centerline_points)
        
        # Если недостаточно точек, выходим
        if smoothed_points is None or len(smoothed_points[0]) < 2:
            return None, None
        
        # Измеряем яркость вдоль аппроксимированной центральной линии
        distances, brightness_values, bright_points = self._measure_brightness(image, smoothed_points, thickness)
        
        # Визуализируем результаты
        vis_image = self._visualize_results(image, bright_points)
        self.result_image = vis_image.copy()  # Сохраняем для будущего использования
        self.update_image_signal.emit(vis_image)
        
        return distances, brightness_values

    def _measure_brightness(self, image: np.ndarray, smoothed_points: Tuple[np.ndarray, np.ndarray], thickness: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Измеряет яркость изображения вдоль самых ярких точек нити
        
        Args:
            image: Изображение
            smoothed_points: Кортеж из массивов x и y координат
            thickness: Толщина области для поиска самого яркого пикселя
            
        Returns:
            Кортеж из массивов расстояний, значений яркости и координат ярких точек
        """
        # Конвертируем в grayscale, если изображение цветное
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        x_points, y_points = smoothed_points
        num_points = len(x_points)
        brightness = np.zeros(num_points)
        
        # Создаем массив для хранения координат ярких точек
        bright_points = np.zeros((num_points, 2), dtype=np.float32)
        
        # Вычисляем расстояние вдоль нити
        dx = np.diff(x_points)
        dy = np.diff(y_points)
        segment_lengths = np.sqrt(dx**2 + dy**2)
        distances = np.zeros(num_points)
        distances[1:] = np.cumsum(segment_lengths)
        
        # Вычисляем направление нити для определения перпендикуляра
        directions = np.zeros((num_points, 2))
        directions[:-1, 0] = dx / (segment_lengths + 1e-10)
        directions[:-1, 1] = dy / (segment_lengths + 1e-10)
        directions[-1] = directions[-2]  # Для последней точки используем предыдущее направление
        
        # Вычисляем перпендикулярные векторы
        perpendicular = np.zeros_like(directions)
        perpendicular[:, 0] = -directions[:, 1]  # перпендикулярный x = -y
        perpendicular[:, 1] = directions[:, 0]   # перпендикулярный y = x
        
        # Для каждой точки центральной линии ищем самую яркую точку поперек нити
        search_radius = max(thickness * 2, 5)  # Область поиска
        
        for i in range(num_points):
            x_center, y_center = int(round(x_points[i])), int(round(y_points[i]))
            
            # Проверяем границы изображения
            if not (0 <= x_center < gray.shape[1] and 0 <= y_center < gray.shape[0]):
                bright_points[i] = [x_center, y_center]  # Используем исходную точку
                continue
            
            # Ищем самую яркую точку в перпендикулярном направлении
            perp_x, perp_y = perpendicular[i]
            max_brightness = 0
            max_x, max_y = x_center, y_center
            
            for offset in range(-search_radius, search_radius + 1):
                x = int(round(x_center + offset * perp_x))
                y = int(round(y_center + offset * perp_y))
                
                if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
                    pixel_value = gray[y, x]
                    if pixel_value > max_brightness:
                        max_brightness = pixel_value
                        max_x, max_y = x, y
            
            bright_points[i] = [max_x, max_y]  # Сохраняем координаты яркой точки
            
            # Измеряем окончательную яркость в небольшой области вокруг найденной яркой точки
            window_size = min(3, thickness)
            x_min = max(0, max_x - window_size // 2)
            x_max = min(gray.shape[1], max_x + window_size // 2 + 1)
            y_min = max(0, max_y - window_size // 2)
            y_max = min(gray.shape[0], max_y + window_size // 2 + 1)
            
            region = gray[y_min:y_max, x_min:x_max]
            if region.size > 0:
                brightness[i] = np.mean(region)
        
        return distances, brightness, bright_points

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
        tck, u = splprep([x, y], s=smoothing)
        
        # Генерируем равномерно распределенные точки вдоль сплайна
        u_new = np.linspace(0, 1, 200)
        x_new, y_new = splev(u_new, tck)
        return np.array([x_new, y_new])
    
    def _visualize_results(self, image: np.ndarray, bright_points: np.ndarray) -> np.ndarray:
        """
        Визуализирует результаты анализа
        
        Args:
            image: Исходное изображение
            bright_points: Массив координат точек максимальной яркости
            
        Returns:
            Изображение с визуализацией
        """
        # Визуализируем линию измерения на изображении
        vis_image = image.copy()
        
        # Рисуем линию через яркие точки (синим)
        bright_points_int = bright_points.astype(np.int32)
        for i in range(10, len(bright_points_int), 10):
            cv2.arrowedLine(vis_image, tuple(bright_points_int[i-10]), tuple(bright_points_int[i]), (255, 0, 255), 1, 
                       tipLength=0.5, line_type=cv2.LINE_AA)

        # Добавляем контрольные точки для наглядности
        for i in range(0, len(bright_points_int), 5):
            # Контрольная точка на линии ярких точек (синяя точка)
            cv2.circle(vis_image, tuple(bright_points_int[i]), 2, (255, 0, 0))
        
        return vis_image
    
    def stop(self):
        """Останавливает поток анализа"""
        self.running = False
        self.wait()

