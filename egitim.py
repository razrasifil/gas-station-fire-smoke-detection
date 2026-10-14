import os
import cv2
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# ==============================
# AYARLAR
# ==============================
RESIM_BOYUTU = (64, 64)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

KLASORLER = {
    "normal": 0,
    "sigara": 1,
    "ates": 2
}

X, y = [], []

# ==============================
# KLASÖR OKUMA (AKILLI)
# ==============================
def klasor_oku(klasor_yolu, label):
    veri, etiket = [], []

    if not os.path.exists(klasor_yolu):
        print(f"❌ Klasör yok: {klasor_yolu}")
        return veri, etiket

    for item in os.listdir(klasor_yolu):
        yol = os.path.join(klasor_yolu, item)

        # 🔹 ALT KLASÖR VARSA
        if os.path.isdir(yol):
            for dosya in os.listdir(yol):
                img_path = os.path.join(yol, dosya)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                img = cv2.resize(img, RESIM_BOYUTU)
                img = img.flatten() / 255.0
                veri.append(img)
                etiket.append(label)

        # 🔹 DİREKT RESİM VARSA
        elif item.lower().endswith((".jpg", ".jpeg", ".png")):
            img = cv2.imread(yol, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, RESIM_BOYUTU)
            img = img.flatten() / 255.0
            veri.append(img)
            etiket.append(label)

    return veri, etiket

# ==============================
# VERİLERİ YÜKLE
# ==============================
print("📂 Veri setleri yükleniyor...\n")

for klasor, label in KLASORLER.items():
    yol = os.path.join(BASE_DIR, klasor)
    v, e = klasor_oku(yol, label)

    print(f"{klasor.capitalize():7}: {len(v)}")

    X.extend(v)
    y.extend(e)

X = np.array(X)
y = np.array(y)

print(f"\nToplam veri: {len(X)}")

# ==============================
# GÜVENLİK KONTROLÜ
# ==============================
sinif_sayisi = len(np.unique(y))
if sinif_sayisi < 2:
    raise ValueError("❌ En az 2 sınıf olmadan model eğitilemez!")

# ==============================
# EĞİTİM / TEST
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ==============================
# MODEL
# ==============================
model = SVC(
    kernel="rbf",
    probability=True,
    class_weight="balanced"
)

print("\n🧠 Model eğitiliyor...")
model.fit(X_train, y_train)

# ==============================
# SONUÇLAR
# ==============================
print(f"\n📈 Eğitim doğruluk: %{model.score(X_train, y_train)*100:.2f}")
print(f"🎯 Test doğruluk  : %{model.score(X_test, y_test)*100:.2f}")

print("\n📊 SINIFLANDIRMA RAPORU")
print("0 → Normal | 1 → Sigara | 2 → Ateş\n")
print(classification_report(y_test, model.predict(X_test)))

# ==============================
# KAYDET
# ==============================
joblib.dump(model, "tehlike_modeli.pkl")
print("\n💾 Model kaydedildi: tehlike_modeli.pkl")
