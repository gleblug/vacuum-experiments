from skimage.morphology import skeletonize
from scipy.interpolate import splprep, splev
from scipy.signal import savgol_filter
import cv2
import numpy as np

from interactive_plot import GraphApp

class ThreadBrightnessAnalyzer:
    """Класс для анализа яркости нити произвольной формы"""
    
    def __init__(self):
        # Настройка интерактивного отображения
        # self.plot = GraphApp('Расстояние вдоль нити (пиксели)', 'Яркость')
        
        # Создание окон визуализации
        cv2.namedWindow('wire analysis', cv2.WINDOW_NORMAL)
    
    def analyze(self, image, thread_mask, thickness=5):
        """
        Анализирует изображение и строит график яркости вдоль нити
        
        Args:
            image: Исходное изображение (BGR или grayscale)
            thread_mask: Бинарная маска нити
            thickness: Толщина для измерения яркости вдоль нити
            
        Returns:
            Координаты центральной линии нити и значения яркости
        """        
        # Получаем скелет маски нити
        skeleton = self._get_skeleton(thread_mask)
        
        # Получаем упорядоченные точки центральной линии нити
        centerline_points = self._get_centerline_points(skeleton)
        
        # Аппроксимируем центральную линию сплайном
        smoothed_points = self._smooth_centerline(centerline_points)
        
        # Если недостаточно точек, выходим
        if smoothed_points is None or len(smoothed_points[0]) < 2:
            print("Не удалось определить центральную линию нити")
            return None, None
        
        # Измеряем яркость вдоль аппроксимированной центральной линии
        distances, brightness_values = self._measure_brightness(
            image, smoothed_points, thickness)
        
        # Визуализируем результаты
        self._visualize_results(
            image, smoothed_points, distances, brightness_values)
        
        return smoothed_points, brightness_values
    def _get_skeleton(self, mask):
        """Получает скелет маски нити"""
        binary_mask = mask.astype(bool)
        skeleton = skeletonize(binary_mask)
        return skeleton.astype(np.uint8)
    
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
    
    def _visualize_results(self, image, smoothed_points, distances, brightness):
        """Визуализирует результаты анализа"""
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
        
        # Отображаем визуализацию
        cv2.imshow('wire analysis', vis_image)
        
        # Обновляем график
        if cv2.waitKey(1) & 0xFF == ord('p'):
            self.plot.update_plot(distances, brightness)
    
    def close(self):
        """Закрывает все окна и освобождает ресурсы"""
        cv2.destroyAllWindows()
