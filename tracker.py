import cv2
import mediapipe as mp
import numpy as np
from collections import deque

class EyeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # --- Önemli Landmark Noktaları ---
        self.NOSE_TIP = 1       # Burun ucu
        self.CHIN = 152         # Çene ucu
        self.FOREHEAD = 10      # Alın (Saç çizgisi ortası)
        self.LEFT_CHEEK = 234   # Sol Yanak
        self.RIGHT_CHEEK = 454  # Sağ Yanak

        # --- Hareketli Ortalama Buffer (Gecikmeyi azaltır) ---
        self.pitch_buffer = deque(maxlen=4)  # Son 4 frame'in pitch değerini tut
        self.yaw_buffer = deque(maxlen=4)    # Son 4 frame'in yaw değerini tut
        self.position_buffer = deque(maxlen=3)  # Konum için buffer

        # --- Geliştirilmiş Eşik Değerleri ---
        self.pitch_up_threshold = 0.37      # Yukarı bakış eşiği (daha hassas)
        self.pitch_down_threshold = 0.63    # Aşağı bakış eşiği (daha hassas)
        self.yaw_left_threshold = 0.27      # Sol dönüş eşiği
        self.yaw_right_threshold = 0.73     # Sağ dönüş eşiği

    def check_focus(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        img_h, img_w, _ = frame.shape
        focused = True
        warning_msg = ""

        # --- 1. GÜVENLİ ALAN (Kişinin çerçeve içinde kalması için) ---
        safe_x_min, safe_x_max = 0.20, 0.80 
        safe_y_min, safe_y_max = 0.25, 0.85

        # Çerçeveyi Çiz
        cv2.rectangle(frame, 
                      (int(safe_x_min * img_w), int(safe_y_min * img_h)), 
                      (int(safe_x_max * img_w), int(safe_y_max * img_h)), 
                      (255, 255, 0), 2) 

        if results.multi_face_landmarks:
            mesh = results.multi_face_landmarks[0].landmark

            def get_coords(idx):
                return int(mesh[idx].x * img_w), int(mesh[idx].y * img_h)

            # --- KOORDİNATLAR ---
            nose = mesh[self.NOSE_TIP]
            nose_coords = get_coords(self.NOSE_TIP)
            
            # Yükseklik hesapları için Y değerlerini alıyoruz (0-1 arası)
            y_forehead = mesh[self.FOREHEAD].y
            y_nose = mesh[self.NOSE_TIP].y
            y_chin = mesh[self.CHIN].y

            # --- KONTROL 1: KONUM (Sandalye pozisyonu) ---
            position_data = (nose.x, nose.y)
            self.position_buffer.append(position_data)
            avg_x = np.mean([p[0] for p in self.position_buffer])
            avg_y = np.mean([p[1] for p in self.position_buffer])

            if avg_x < safe_x_min:
                focused = False
                warning_msg = "SAG TARAFA KAYDINIZ"
            elif avg_x > safe_x_max:
                focused = False
                warning_msg = "SOL TARAFA KAYDINIZ"
            elif avg_y < safe_y_min:
                focused = False
                warning_msg = "CERCEVEDEN CIKTINIZ (UST)"
            elif avg_y > safe_y_max:
                focused = False
                warning_msg = "CERCEVEDEN CIKTINIZ (ALT)"

            # --- KONTROL 2: KAFA DÖNÜŞÜ (SAĞ / SOL - YAW) ---
            p_left = np.array(get_coords(self.LEFT_CHEEK))
            p_right = np.array(get_coords(self.RIGHT_CHEEK))
            p_nose = np.array(nose_coords)

            dist_to_left = np.linalg.norm(p_nose - p_left)
            dist_to_right = np.linalg.norm(p_nose - p_right)
            face_width = dist_to_left + dist_to_right
            
            if face_width > 0:
                yaw_ratio = dist_to_left / face_width
                self.yaw_buffer.append(yaw_ratio)
                
                # Hareketli ortalama al
                avg_yaw = np.mean(list(self.yaw_buffer)) if len(self.yaw_buffer) > 0 else yaw_ratio
                
                if avg_yaw < self.yaw_left_threshold:
                    focused = False
                    warning_msg = "BASINIZI CEVIRMEYIN (SOL)"
                elif avg_yaw > self.yaw_right_threshold:
                    focused = False
                    warning_msg = "BASINIZI CEVIRMEYIN (SAG)"

            # --- KONTROL 3: KAFA EĞİMİ (YUKARI / AŞAĞI - PITCH) ---
            face_vertical_height = y_chin - y_forehead
            
            if face_vertical_height > 0:
                nose_position_ratio = (y_nose - y_forehead) / face_vertical_height
                self.pitch_buffer.append(nose_position_ratio)
                
                # Hareketli ortalama al - DAHA HIZLI ALGILAMA İÇİN
                avg_pitch = np.mean(list(self.pitch_buffer)) if len(self.pitch_buffer) > 0 else nose_position_ratio

                if avg_pitch < self.pitch_up_threshold: 
                    focused = False
                    warning_msg = "YUKARI BAKTINIZ"
                elif avg_pitch > self.pitch_down_threshold:
                    focused = False
                    warning_msg = "ASAGI BAKTINIZ"

            # Görselleştirme
            color = (0, 255, 0) if focused else (0, 0, 255)
            cv2.circle(frame, nose_coords, 5, color, -1)

            # Debug için alın ve çeneyi de çizelim
            cv2.circle(frame, get_coords(self.FOREHEAD), 3, (0, 255, 255), -1)
            cv2.circle(frame, get_coords(self.CHIN), 3, (0, 255, 255), -1)

            if warning_msg:
                cv2.putText(frame, warning_msg, (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        else:
            focused = False
            cv2.putText(frame, "YUZ BULUNAMADI!", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        return focused, frame