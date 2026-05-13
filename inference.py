import cv2
import numpy as np
import tensorflow as tf
from keras.applications.efficientnet import preprocess_input

# 1. Modeli Yükle
model_path = 'final_effnetB3_rafdb_pro.keras'
model = tf.keras.models.load_model(model_path)

# 2. Yüz Tespiti
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 3. Duygu Etiketleri 
emotion_labels = ['Kizgin', 'Tiksinme', 'Korku', 'Mutlu', 'Notr', 'Uzgun', 'Saskin']

# 4. Kamerayı Başlat
cap = cv2.VideoCapture(0)

#  Yumuşatma için liste (buffer) 
pred_buffer = []
buffer_size = 5  # Bu sayıyı artırırsan daha yavaş ve stabil, azaltırsan daha hızlı tepki verir.

print("🚀 Duygu Analizi Başlatıldı... Çıkmak için 'q' tuşuna basın.")

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_color = frame[y:y+h, x:x+w]
        roi_color = cv2.resize(roi_color, (300, 300))
        
        # Eğitimdeki gibi RGB'ye çevirip preprocess yapalım
        roi_rgb = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
        img_array = tf.keras.preprocessing.image.img_to_array(roi_rgb)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Tahmin Al
        prediction = model.predict(img_array, verbose=0)

        # YUMUŞATMA İŞLEMİ
        pred_buffer.append(prediction[0])
        if len(pred_buffer) > buffer_size:
            pred_buffer.pop(0) # En eski tahmini sil

        # Son 10 tahminin ortalamasını al
        smoothed_prediction = np.mean(pred_buffer, axis=0)
        max_index = np.argmax(smoothed_prediction)
        
        emotion = emotion_labels[max_index]
        confidence = smoothed_prediction[max_index]

        # Çizim
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        label = f"{emotion} ({confidence*100:.1f}%)"
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow('Stabil Duygu Analizi', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
