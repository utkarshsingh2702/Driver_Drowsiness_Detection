# src/alerts.py
import time
import platform
import os

# Try playsound if available; fallback to winsound on Windows or terminal beep
try:
    from playsound import playsound
    PLAYSOUND_OK = True
except Exception:
    PLAYSOUND_OK = False

def play_alert_sound(sound_path=None):
    """
    Play a short alert. If sound_path provided and playsound exists, play it.
    Otherwise use platform beep.
    """
    if sound_path and PLAYSOUND_OK:
        try:
            playsound(sound_path)
            return
        except Exception:
            pass

    # fallback
    if platform.system() == "Windows":
        try:
            import winsound
            winsound.Beep(1650, 1000)
            winsound.Beep(1650, 1000)
            return
        except Exception:
            pass

    # generic fallback: print BEL
    print("\a")  # may produce a beep in terminal
