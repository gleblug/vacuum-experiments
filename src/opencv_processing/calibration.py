import numpy as np
import cv2

def second_order_calibration(image):
    """
    Применяет калибровку второго порядка к изображению методом наименьших квадратов.
    Приводит ненулевые пиксели к среднему значению.
    
    Параметры:
        image (numpy.ndarray): Входное изображение в градациях серого
        
    Возвращает:
        numpy.ndarray: Калиброванное изображение
    """
    # Получаем координаты ненулевых пикселей
    y, x = np.where(image > 0)
    values = image[y, x]
    
    # Создаем матрицу для полинома второго порядка (ax² + bx + c)
    A = np.column_stack((x**2, x, np.ones_like(x)))
    
    # Решаем методом наименьших квадратов
    coeffs, _, _, _ = np.linalg.lstsq(A, values, rcond=None)
    
    # Вычисляем предсказанные значения
    predicted = np.dot(A, coeffs)
    
    # Вычисляем среднее значение
    mean_val = np.mean(values)
    
    # Вычисляем калибровочные коэффициенты
    calibration_coeffs = mean_val / predicted
    
    # Применяем калибровку
    calibrated_image = np.zeros_like(image, dtype=np.float32)
    calibrated_image[y, x] = image[y, x] * calibration_coeffs
    
    # Нормализуем и конвертируем обратно в исходный тип
    calibrated_image = cv2.normalize(calibrated_image, None, 0, 255, cv2.NORM_MINMAX)
    return calibrated_image.astype(image.dtype)

# Пример использования:
# image = cv2.imread('input.png', cv2.IMREAD_GRAYSCALE)
# calibrated = second_order_calibration(image)
# cv2.imwrite('calibrated.png', calibrated)
