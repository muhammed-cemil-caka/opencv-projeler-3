import cv2
import numpy as np
import mediapipe as mp

mp_el = mp.solutions.hands
eller = mp_el.Hands(max_num_hands=1)
mp_ciz = mp.solutions.drawing_utils

kamera = cv2.VideoCapture(0)
tuval = None

renk = (0, 255, 0)
fircaboyutu = 10
silgi_boyutu = 100

onceki_x, onceki_y = None, None

renk_kutulari = {
    'kirmizi': (10, 450, 60, 500),
    'yesil': (70, 450, 120, 500),
    'mavi': (130, 450, 180, 500),
    'silgi': (190, 450, 240, 500)
}

renkler = {
    'kirmizi': (0, 0, 255),
    'yesil': (0, 255, 0),
    'mavi': (255, 0, 0),
    'silgi': (0, 0, 0)
}

mevcut_arac = 'firca'

while True:
    ret, kare = kamera.read()
    if not ret:
        print("Görüntü alınamadı")
        break

    kare = cv2.flip(kare, 1)
    yukseklik, genislik, c = kare.shape
    if tuval is None:
        tuval = np.zeros_like(kare)

    cv2.rectangle(kare, renk_kutulari['kirmizi'][0:2], renk_kutulari['kirmizi'][2:4], (0, 0, 255), -1)
    cv2.rectangle(kare, renk_kutulari['yesil'][0:2], renk_kutulari['yesil'][2:4], (0, 255, 0), -1)
    cv2.rectangle(kare, renk_kutulari['mavi'][0:2], renk_kutulari['mavi'][2:4], (255, 0, 0), -1)
    cv2.rectangle(kare, renk_kutulari['silgi'][0:2], renk_kutulari['silgi'][2:4], (255, 255, 255), -1)
    cv2.putText(kare, 'Silgi', (renk_kutulari['silgi'][0] + 5, renk_kutulari['silgi'][1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    rgb_kare = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)

    sonuc = eller.process(rgb_kare)
    if sonuc.multi_hand_landmarks:
        for el_isaretleri in sonuc.multi_hand_landmarks:
            mp_ciz.draw_landmarks(kare, el_isaretleri, mp_el.HAND_CONNECTIONS)

            isaretler = el_isaretleri.landmark
            isaret_parmagi_uc = isaretler[8]

            isaret_x, isaret_y = int(isaret_parmagi_uc.x * genislik), int(isaret_parmagi_uc.y * yukseklik)

            for anahtar, (x1, y1, x2, y2) in renk_kutulari.items():
                if x1 < isaret_x < x2 and y1 < isaret_y < y2:
                    mevcut_arac = anahtar
                    renk = renkler[anahtar]

            if onceki_x is not None and onceki_y is not None:
                if mevcut_arac == 'silgi':
                    cv2.line(tuval, (onceki_x, onceki_y), (isaret_x, isaret_y), (0, 0, 0), silgi_boyutu)
                else:
                    cv2.line(tuval, (onceki_x, onceki_y), (isaret_x, isaret_y), renk, fircaboyutu)

            onceki_x, onceki_y = isaret_x, isaret_y

    kare = cv2.addWeighted(kare, 0.5, tuval, 0.5, 0)

    cv2.imshow("Kamera Görüntüsü", kare)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()
