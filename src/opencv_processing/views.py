import cv2
import numpy as np
import os
from datetime import datetime
import csv

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QFileDialog, QGraphicsView, QGraphicsScene
)
from PyQt6.QtCore import (
    pyqtSignal, Qt, QRectF, QPointF, QPoint
)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor

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
        self.canvas.ax.set_ylim(0, 255)
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
