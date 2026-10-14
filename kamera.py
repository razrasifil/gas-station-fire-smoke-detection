import cv2
import numpy as np
import joblib

# ==============================
# AYARLAR
# ==============================
RESIM_BOYUTU = (64, 64)
MODEL_DOSYA = "tehlike_modeli.pkl"

SVM_ESIK = 0.65
SIGARA_FRAME_ESIK = 6
BOS_SAHNE_STD = 12

# ==============================
# MODEL YÜKLE
# ==============================
model = joblib.load(MODEL_DOSYA)
print("🧠 Model yüklendi")

# ==============================
# KAMERA
# ==============================
cap = cv2.VideoCapture(0)
sigara_sayac = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sigara_tehlike = False
    ates_tehlike = False

    # ==========================
    # 🧱 BOŞ SAHNE KONTROLÜ
    # ==========================
    if np.std(gray) < BOS_SAHNE_STD:
        sigara_sayac = 0
        cv2.putText(frame, "GUVENLI", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                    (0, 255, 0), 3)

        cv2.imshow("Sigara + Ates Algilama", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # ==========================
    # 🔥 ATEŞ ALGILAMA
    # ==========================
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_fire = np.array([0, 150, 180])
    upper_fire = np.array([30, 255, 255])
    fire_mask = cv2.inRange(hsv, lower_fire, upper_fire)

    fire_pixel = cv2.countNonZero(fire_mask)
    fire_ratio = fire_pixel / (frame.shape[0] * frame.shape[1])

    if fire_pixel > 1200 and fire_ratio > 0.01:
        ates_tehlike = True
        sigara_sayac = 0  # 🔴 ateş varken sigara iptal

    # ==========================
    # 🚬 SİGARA ALGILAMA
    # ==========================
    if not ates_tehlike:
        sigara_frame = cv2.resize(gray, RESIM_BOYUTU)
        features = sigara_frame.flatten().astype("float32") / 255.0
        features = features.reshape(1, -1)

        tahmin = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]

        if prob < 0.55:
            sigara_sayac = max(0, sigara_sayac - 1)
        elif tahmin == 1 and prob > SVM_ESIK:
            sigara_sayac += 1
        else:
            sigara_sayac = max(0, sigara_sayac - 1)

        if sigara_sayac >= SIGARA_FRAME_ESIK:
            sigara_tehlike = True

    # ==========================
    # 🖥️ EKRAN YAZILARI
    # ==========================
    if ates_tehlike:
        cv2.putText(frame, "ATES TEHLIKE", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 3)

    if sigara_tehlike:
        cv2.putText(frame, "SIGARA TEHLIKESI", (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 165, 255), 3)

    if ates_tehlike or sigara_tehlike:
        cv2.putText(frame, "GENEL TEHLIKE", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                    (0, 0, 255), 3)
    else:
        cv2.putText(frame, "GUVENLI", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                    (0, 255, 0), 3)

    cv2.imshow("Sigara + Ates Algilama", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
