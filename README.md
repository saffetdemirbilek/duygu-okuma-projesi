# EfficientNet-B3 Duygu Tanıma Sistemi

Bu proje, RAF-DB veri seti kullanılarak eğitilmiş, gerçek zamanlı bir duygu tanıma uygulamasıdır.

Kurulum
1. Bu depoyu klonlayın.
2. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```
3. Model dosyasını (`final_effnetB3_rafdb_pro.keras`) ana dizine yerleştirin.

Kullanım
Uygulamayı başlatmak için şu komutu çalıştırın:
```bash
python inference.py
```
- Kamera açıldığında yüzünüzü odaklayın.
- Çıkış yapmak için klavyeden 'q' tuşuna basın.

 Model Performansı
- **Doğrulama Başarısı:** %77.23
- **Mimari:** EfficientNet-B3
