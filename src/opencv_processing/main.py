import cv2
import numpy as np
import sys
import os
from datetime import datetime
import csv

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QFileDialog, QTabWidget, QComboBox
)
from PyQt6.QtCore import pyqtSlot, Qt

from threads import VideoThread, ThreadBrightnessAnalyzer
from views import ZoomableImageView, GraphApp

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
        
        # Верхняя панель с видео и выбором камеры
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

        # Добавляем выпадающий список камер
        self.camera_combo = QComboBox()
        
        self.available_cameras = self.get_available_cameras()
        for i, camera_name in enumerate(self.available_cameras):
            self.camera_combo.addItem(f"{i}: {camera_name}")
        
        self.camera_combo.currentIndexChanged.connect(self.camera_changed)
        button_layout.addWidget(self.camera_combo)
        
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
        
        # Создаем поток для обработки видео с выбранной камерой
        self.current_camera_index = 0  # По умолчанию используем первую камеру
        self.video_thread = VideoThread(self.current_camera_index)
        
        # Создаем обработчик для анализа яркости
        self.analyzer = ThreadBrightnessAnalyzer()
        
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
        
        # Запускаем потоки
        self.video_thread.start()
        self.analyzer.start()
    
    def get_available_cameras(self):
        """Получает список доступных камер"""
        camera_names = []
        
        # Проверяем до 5 возможных камер
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Пытаемся получить имя устройства (не всегда доступно)
                # В некоторых системах можно использовать cap.get(cv2.CAP_PROP_DEVICE_NAME)
                camera_names.append(f"Камера {i}")
                cap.release()
            else:
                break
        
        # Если не найдено ни одной камеры, добавляем заглушку
        if not camera_names:
            camera_names.append("Камера не найдена")
        
        return camera_names
    
    def camera_changed(self, index):
        """Обработчик изменения выбранной камеры"""
        if index < 0 or index >= len(self.available_cameras):
            return
        
        # Если выбрана новая камера, перезапускаем поток
        if self.current_camera_index != index:
            self.current_camera_index = index
            
            # Останавливаем текущий поток
            self.video_thread.stop()
            
            # Создаем и запускаем новый поток с выбранной камерой
            self.video_thread = VideoThread(self.current_camera_index)
            self.video_thread.change_pixmap_signal.connect(self.update_video)
            self.video_thread.roi_signal.connect(self.update_roi)
            self.video_thread.mask_signal.connect(self.update_mask)
            self.video_thread.start()
            
            self.statusBar().showMessage(f"Переключение на камеру {index}: {self.available_cameras[index]}")
    
    @pyqtSlot(np.ndarray)
    def update_video(self, cv_img):
        """Обновляем изображение с камеры"""
        self.video_view.setImage(cv_img)
    
    @pyqtSlot(np.ndarray, np.ndarray)
    def update_roi(self, roi, mask):
        """Обновляем ROI и отправляем данные для анализа"""
        self.roi_view.setImage(roi)
        self.analyzer.set_data(roi, mask)
        if roi is not None:
            h, w = roi.shape[:2]
            self.statusBar().showMessage(f"Выделена область: {w}x{h} пикселей")
    
    @pyqtSlot(np.ndarray)
    def update_mask(self, mask):
        """Обновляем отображение маски"""
        self.mask_view.setImage(mask)
    
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
            
        timestamp = datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")
        directory = "out"
        if not os.path.exists(directory):
            os.makedirs(directory)
        default_name = f"{directory}/data_{timestamp}"
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Задайте базовое имя файла", default_name, 
            "Все файлы (*)"
        )
        
        if filename:
            base_name = filename
            if '.' in filename:
                base_name = filename.rsplit('.', 1)[0]
            
            cv2.imwrite(f"{base_name}_original.png", self.video_thread.original_frame)
            cv2.imwrite(f"{base_name}_roi.png", self.video_thread.current_roi)
            # cv2.imwrite(f"{base_name}_mask.png", self.video_thread.current_mask)
            cv2.imwrite(f"{base_name}_analysis.png", self.analyzer.result_image)
            
            self.graph_app.canvas.fig.savefig(f"{base_name}_graph.png", dpi=300, bbox_inches='tight')
            
            with open(f"{base_name}_data.csv", 'w', newline='') as csvfile:
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
        self.tabs.setCurrentIndex(1)
    
    @pyqtSlot(np.ndarray)
    def update_analysis(self, image):
        """Обновляем изображение анализа"""
        self.analysis_view.setImage(image)
    
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
    """Точка входа в приложение"""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
