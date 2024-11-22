import cv2
import numpy as np

# kamerayı aç
cap = cv2.VideoCapture(0)

ret, frame = cap.read()
if not ret:
    print("Kamera açılmadı")
    exit()

# yüz algılama için Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(frame)

# listenin boş olup olmadığını kontrol ediniz
if len(faces) == 0: # length -> boyut
    print("yüz bulunamadı")
    exit()

# ilk yüzü hedef seç
(x, y, w, h) = faces[0]
track_window = (x, y, w, h) # takip penceresi

# region of interest -> roi -> ilgili bölgesi
roi = frame[y:y+h, x:x+w]

hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0.,60.,32.,)), np.array((180., 255. , 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0,180])

cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# meanshift kriterlerini ayarla
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü hsv renk uzayına çevir
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # geri projeksiyon hesapla
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # meanshift uygula ve yeni hedefi bul
    ret, track_window = cv2.meanShift(dst, track_window, term_crit)

    # yeni hedefi çiz
    x, y, w, h = track_window
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

   
    # çıktıyı göster
    cv2.imshow("Yüz Takibi", frame)

    # q ile çıkış yap
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
