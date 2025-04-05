from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTabWidget, QFileDialog
)
from PyQt6.QtCore import Qt
import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Анализ соответствия компетенций')
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet('''
            QMainWindow {
                background-color: #f5f7fa;
            }
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #1a3e72;
            }
            QPushButton {
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
        ''')

        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.create_analysis_section(layout)
        self.create_result_tabs(layout)

    def create_analysis_section(self, layout):
        analysis = QWidget()
        analysis.setStyleSheet('''
            background-color: white;
            padding: 30px;
        ''')
        analysis_layout = QHBoxLayout(analysis)

        upload_box = QWidget()
        upload_box.setStyleSheet('''
            border: 2px dashed #ccc;
            border-radius: 8px;
            padding: 10px;
        ''')
        upload_layout = QVBoxLayout(upload_box)
        upload_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        upload_icon = QLabel('📁')
        upload_icon.setStyleSheet('font-size: 40px; border: none')
        upload_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        upload_layout.addWidget(upload_icon)

        upload_text = QLabel('Загрузите до трех резюме')
        upload_text.setStyleSheet('font-weight: bold; font-size: 16px; border: none')
        upload_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        upload_layout.addWidget(upload_text)

        upload_text2 = QLabel("Перетащите файл сюда или нажмите для выбора")
        upload_text2.setStyleSheet('font-size: 16px; border: none')
        upload_text2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        upload_layout.addWidget(upload_text2)

        upload_formats = QLabel("Поддерживаемые форматы: PDF, DOCX, TXT")
        upload_formats.setStyleSheet('font-size: 16px; border: none')
        upload_formats.setAlignment(Qt.AlignmentFlag.AlignCenter)
        upload_layout.addWidget(upload_formats)

        self.analyze_btn = QPushButton("Загрузить файл")
        self.analyze_btn.clicked.connect(self.open_file_dialog)
        upload_layout.addWidget(self.analyze_btn)

        self.file_label = QLabel("")
        self.file_label.setStyleSheet("font-size: 14px; background-color: #ff6a1f;")
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        upload_layout.addWidget(self.file_label)

        analysis_layout.addWidget(upload_box)

        result_box = QWidget()
        result_box.setStyleSheet("""
            background-color: #f9f9f9;
            border-radius: 8px;
            padding: 10px;
        """)
        result_layout = QVBoxLayout(result_box)
        result_title = QLabel("Результаты анализа")
        result_title.setStyleSheet("font-weight: bold; font-size: 18px;")
        result_layout.addWidget(result_title)

        result_text = QLabel("Навыки найдены:")
        result_layout.addWidget(result_text)

        skills = ["Python", "DevOps", "Big Data", "Docker", "SQL"]
        skills_layout = QHBoxLayout()
        for skill in skills:
            skill_label = QLabel(skill)
            skill_label.setStyleSheet("""
                background-color: #e1f5fe;
                color: #1a3e72;
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 14px;
            """)
            skills_layout.addWidget(skill_label)
        result_layout.addLayout(skills_layout)

        vacancies_text = QLabel("Рекомендуемые вакансии:")
        result_layout.addWidget(vacancies_text)

        vacancy = QLabel("Middle Java Developer (85% совпадение)")
        result_layout.addWidget(vacancy)

        analysis_layout.addWidget(result_box)
        layout.addWidget(analysis)

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл резюме", "", "Документы (*.pdf *.docx *.txt)"
        )
        if file_path:
            self.file_label.setText(f"Загружено: {file_path.split('/')[-1]}")

    def create_result_tabs(self, layout):
        tabs = QTabWidget()
        tabs.setStyleSheet('''
            QTabWidget::pane {
                border: none;
                padding: 0px;
            }
            QTabBar::tab {
                padding: 8px 16px;
                background: transparent;
                color: #1a3e72;
            }
            QTabBar::tab:selected {
                border-bottom: 3px solid #1a3e72;
                font-weight: bold;
                color: #1a3e72;
            }'''
        )

        matches_tab = QWidget()
        matches_layout = QVBoxLayout(matches_tab)
        matches_title = QLabel('Совпадения с матрицей компетенций')
        matches_title.setAlignment(Qt.AlignmentFlag.AlignTop)
        matches_title.setStyleSheet('font-size: 18px; font-weight: bold;')
        matches_layout.addWidget(matches_title)
        tabs.addTab(matches_tab, 'Совпадение с вакансиями')

        skills_tab = QWidget()
        skills_layout = QVBoxLayout(skills_tab)
        skills_title = QLabel('Рекомендуемые темы для изучения')
        skills_title.setAlignment(Qt.AlignmentFlag.AlignTop)
        skills_title.setStyleSheet('font-size: 18px; font-weight: bold;')
        skills_layout.addWidget(skills_title)
        tabs.addTab(skills_tab, 'Недостающие навыки')

        charts_tab = QWidget()
        charts_layout = QVBoxLayout(charts_tab)
        charts_title = QLabel('Графики соответствия компетенций')
        charts_title.setAlignment(Qt.AlignmentFlag.AlignTop)
        charts_title.setStyleSheet('font-size: 18px; font-weight: bold;')
        charts_layout.addWidget(charts_title)
        tabs.addTab(charts_tab, 'Графики')

        layout.addWidget(tabs)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
