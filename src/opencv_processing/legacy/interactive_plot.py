import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                            QHBoxLayout, QWidget, QFileDialog, QStatusBar)
from datetime import datetime
import csv
from matplotlib.widgets import SpanSelector

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

class GraphApp(QMainWindow):
    def __init__(self, xlabel="X", ylabel="Y"):
        super().__init__()
        
        # Настройка главного окна
        self.setWindowTitle("Интерактивный график (PyQt6)")
        self.setGeometry(100, 100, 900, 600)
        
        # Данные графика
        self.x_data = np.array([])
        self.y_data = np.array([])
        
        # Сохраняем метки осей
        self.xlabel = xlabel
        self.ylabel = ylabel
        
        # Инициализация UI
        self.init_ui()
        
    def init_ui(self):
        # Создание главного виджета и компоновки
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Создание и добавление холста matplotlib
        self.canvas = MatplotlibCanvas(self, width=8, height=5, dpi=100, 
                                      xlabel=self.xlabel, ylabel=self.ylabel)
        main_layout.addWidget(self.canvas)
        
        # Добавление панели инструментов matplotlib
        toolbar = NavigationToolbar2QT(self.canvas, self)
        main_layout.addWidget(toolbar)
        
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
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Готов")
        
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
            self.status_bar.showMessage("Нет данных для сохранения", 5000)
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"график_{timestamp}.png"
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Сохранить график как изображение", default_name, 
            "Изображения (*.png *.jpg *.jpeg *.pdf *.svg);;Все файлы (*)"
        )
        
        if filename:
            self.canvas.fig.savefig(filename, dpi=300, bbox_inches='tight')
            self.status_bar.showMessage(f"График сохранен как {os.path.basename(filename)}", 5000)
    
    def save_data_points(self):
        """Сохранение точек данных в CSV файл"""
        if len(self.x_data) == 0:
            self.status_bar.showMessage("Нет данных для сохранения", 5000)
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
            
            self.status_bar.showMessage(f"Данные сохранены как {os.path.basename(filename)}", 5000)
    
    def on_range_selected(self, min_val, max_val):
        """Обработка выбора диапазона и расчет среднего значения"""
        if len(self.x_data) == 0 or len(self.y_data) == 0:
            return
            
        # Находим индексы данных в выбранном диапазоне
        indices = [i for i, x in enumerate(self.x_data) if min_val <= x <= max_val]
        
        if not indices:
            self.status_bar.showMessage("В выбранном диапазоне нет данных", 5000)
            return
            
        # Рассчитываем среднее значение y в выбранном диапазоне
        selected_y_values = [self.y_data[i] for i in indices]
        mean_value = sum(selected_y_values) / len(selected_y_values)
        
        # Показываем информацию в статусной строке
        self.status_bar.showMessage(
            f"Диапазон X: [{min_val:.4f}, {max_val:.4f}], "
            f"Среднее значение {self.ylabel}: {mean_value:.4f}, "
            f"Количество точек: {len(indices)}"
        )
