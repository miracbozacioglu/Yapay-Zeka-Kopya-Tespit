import sys
import cv2
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QHBoxLayout
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
from tracker import EyeTracker
from utils import AlertSystem

class ExamGuard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Kopya Tespit Sistemi")
        
        # 1. Pencere Boyutu Sabit
        self.setFixedSize(1000, 800) 

        # Ana Layout
        main_layout = QVBoxLayout()
        
        # --- ÜST BAR ---
        header_layout = QHBoxLayout()
        self.logo_label = QLabel()
        try:
            pixmap = QPixmap("assets/logo.jpg").scaled(50, 50, Qt.AspectRatioMode.KeepAspectRatio)
            self.logo_label.setPixmap(pixmap)
        except:
            pass
            
        header_layout.addWidget(self.logo_label)
        
        title = QLabel("AKILLI SINAV GÜVENLİK PANELİ")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #2c3e50;")
        header_layout.addWidget(title)
        header_layout.addStretch()
        
        main_layout.addLayout(header_layout)

        # --- VİDEO EKRANI (Sorunun Çözüldüğü Yer) ---
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("border: 3px solid #34495e; background-color: #000000; border-radius: 10px;")
        
        # DEĞİŞİKLİK BURADA:
        # Video alanına SABİT bir boyut veriyoruz (640x480 veya 800x600).
        # Bu sayede pencere açılırken "boyut hesaplama animasyonu" oluşmaz.
        self.FIXED_VIDEO_WIDTH = 800
        self.FIXED_VIDEO_HEIGHT = 600
        self.video_label.setFixedSize(self.FIXED_VIDEO_WIDTH, self.FIXED_VIDEO_HEIGHT)
        
        # Video etiketini layout içinde ortala
        main_layout.addWidget(self.video_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # --- DURUM MESAJI ---
        self.status_label = QLabel("SİSTEM BAŞLATILIYOR...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("font-size: 20px; padding: 15px; background-color: #ecf0f1; border-radius: 5px;")
        self.status_label.setFixedHeight(60)
        main_layout.addWidget(self.status_label)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # --- KAMERA AYARLARI ---
        self.cap = cv2.VideoCapture(0)
        
        # Kameraya da standart bir çözünürlük verelim ki açılışta kafası karışmasın.
        # Bu işlem performansı da artırır.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.tracker = EyeTracker()
        self.alert = AlertSystem()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30) 

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Tracker işlemleri
            focused, processed_frame = self.tracker.check_focus(frame)

            if not focused:
                self.status_label.setText("⚠️ LÜTFEN EKRANA ODAKLANIN!")
                self.status_label.setStyleSheet("color: white; background-color: #c0392b; font-weight: bold; font-size: 22px; border-radius: 5px;")
                self.alert.play_alert()
            else:
                self.status_label.setText("✅ SINAV GÜVENLİ - ODAK NORMAL")
                self.status_label.setStyleSheet("color: white; background-color: #27ae60; font-weight: bold; font-size: 18px; border-radius: 5px;")
                self.alert.stop_alert()

            # Renk Dönüşümü
            rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
            # DEĞİŞİKLİK BURADA:
            # Artık dinamik olarak 'self.video_label.width()' demiyoruz.
            # En başta belirlediğimiz sabit değerleri kullanıyoruz.
            # Bu, görüntünün titremesini ve boyut değiştirmesini engeller.
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.FIXED_VIDEO_WIDTH - 6, # Kenarlık payı (border: 3px demiştik, toplam 6px)
                self.FIXED_VIDEO_HEIGHT - 6,
                Qt.AspectRatioMode.KeepAspectRatio
            )
            
            self.video_label.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        self.cap.release()
        self.alert.stop_alert()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ExamGuard()
    window.show()
    sys.exit(app.exec())