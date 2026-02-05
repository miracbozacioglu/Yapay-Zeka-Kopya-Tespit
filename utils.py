import pygame

class AlertSystem:
    def __init__(self):
        pygame.mixer.init()
        try:
            # Görseldeki dosya adına göre güncellendi
            self.sound_path = "assets/alarm.mp3"
            pygame.mixer.music.load(self.sound_path)
        except Exception as e:
            print(f"Hata: {self.sound_path} yüklenemedi! {e}")

    def play_alert(self):
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play(-1) # -1 sürekli çalmasını sağlar

    def stop_alert(self):
        pygame.mixer.music.stop()