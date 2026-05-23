# Güneş Enerjisi Üretimi Tahmini - Derin Öğrenme Arayüzü

Derin öğrenme ve sinyal ayrıştırma teknikleri kullanarak güneş enerjisi üretimini tahmin etmek için geliştirilmiş kapsamlı bir masaüstü uygulamasıdır. Python, CustomTkinter ve TensorFlow/Keras kullanılarak inşa edilmiştir.

## Özellikler

Uygulama temel olarak dört aşamadan oluşmaktadır:

*   **1. Aşama: Veri Yönetimi ve Ayrıştırma**
    *   Zaman serisi CSV veri setlerini yükleme ve temizleme.
    *   Özelliklerin hedef (Aktif Güç) ile olan korelasyonlarını hesaplama.
    *   Gelişmiş sinyal ayrıştırma (EMD, EEMD, CEEMDAN, VMD) algoritmaları.
    *   Gecikmeli (lag) özellikler oluşturma ve veriyi bölme.

*   **2. Aşama: Ön-Eğitimli Modeller**
    *   1D Evrişimli Sinir Ağları (CNN) kullanarak tablo verisi formatında eğitim.
    *   Desteklenen mimariler: AlexNet-1D, GoogLeNet-1D, ResNet-1D, VGG16-1D, SqueezeNet-1D, ShuffleNet-1D.
    *   Gerçek zamanlı eğitim ilerlemesi ve kayıp (loss) grafiği çizimi.

*   **3. Aşama: Özel Model Tasarımcısı**
    *   Özel yapay sinir ağı mimarilerini katman katman etkileşimli olarak inşa etme.
    *   Desteklenen katmanlar: Conv1D, LSTM, GRU, Dense, Dropout, MaxPooling1D, vb.
    *   Kendi modellerinizi derleme, eğitme ve sonuçlarını canlı takip etme.

*   **4. Aşama: Değerlendirme ve Raporlama**
    *   Eğitilen modellerin test veri seti üzerinde değerlendirilmesi.
    *   Temel performans metriklerinin (R, RMSE, MAE, MAPE) hesaplanması.
    *   Gerçekleşen ve tahmin edilen güç üretiminin grafiksel karşılaştırması.
    *   Sonuçların CSV veya PDF raporu olarak dışa aktarılması.

## Kurulum ve Kullanım

1. Sanal bir ortam (virtual environment) oluşturun ve etkinleştirin.
2. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```
3. Uygulamayı başlatın:
   ```bash
   python main.py
   ```

## Gereksinimler
* Python 3.8+
* TensorFlow / Keras
* CustomTkinter
* Pandas, NumPy, Scikit-learn
* Matplotlib
* PyEMD, vmdpy (sinyal ayrıştırma için)
