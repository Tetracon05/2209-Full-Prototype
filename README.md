# Güneş Enerjisi Üretimi Tahmini - Derin Öğrenme Arayüzü

Derin öğrenme ve sinyal ayrıştırma teknikleri kullanarak güneş enerjisi üretimini tahmin etmek için geliştirilmiş kapsamlı bir masaüstü uygulamasıdır. Python, CustomTkinter ve TensorFlow/Keras kullanılarak inşa edilmiştir.

## Özellikler

Uygulama temel olarak dört aşamadan oluşmaktadır:

*   **1. Aşama: Veri Yönetimi ve Ayrıştırma**
    *   Zaman serisi CSV veri setlerini yükleme ve temizleme.
    *   Özelliklerin hedef (Aktif Güç) ile olan korelasyonlarını hesaplama.
    *   Gelişmiş sinyal ayrıştırma (CEEMDAN, VMD) algoritmaları.
    *   Gecikmeli (lag) özellikler oluşturma ve veriyi bölme.

*   **2. Aşama: Ön-Eğitimli Modeller**
    *   1D Evrişimli Sinir Ağları (CNN) kullanarak tablo verisi formatında eğitim.
    *   Desteklenen mimariler: AlexNet-1D, GoogLeNet-1D, ResNet-1D.
    *   Gerçek zamanlı eğitim ilerlemesi ve kayıp (loss) grafiği çizimi.

*   **3. Aşama: Özel Model Tasarımcısı**
    *   Özel yapay sinir ağı mimarilerini katman katman etkileşimli olarak inşa etme.
    *   Desteklenen katmanlar: Conv1D, LSTM, GRU, Dense, Dropout, MaxPooling1D, vb.
    *   Kendi modellerinizi derleme, eğitme ve sonuçlarını canlı takip etme.

*   **4. Aşama: Değerlendirme ve Raporlama**
    *   Eğitilen modellerin test veri seti üzerinde değerlendirilmesi.
    *   Temel performans metriklerinin (R, RMSE, MAE, sMAPE) hesaplanması.
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

## Örnek Çıktılar
* [Örnek Çıktılar (Google Drive)](https://drive.google.com/drive/folders/1K3n-pZ5do3dP1j4_niZLGN262e-Swh7R?usp=sharing)

## Son Güncellemeler ve Performans Analizi
* **Hata Metriği İyileştirmesi:** Güneş enerjisi üretimindeki "gece/alacakaranlık" saatlerinde oluşan (üretimin sıfıra yakın olduğu anlarda) aşırı yüksek yüzdelik sapmaları engellemek için değerlendirme modülünde MAPE yerine **sMAPE** (Symmetric Mean Absolute Percentage Error) kullanıldı. Anlamlı hesaplama için sadece `> 5.0 W` üretim anları değerlendirmeye alındı.
* **Model Performansları (VMD ile Ayrıştırılmış Veri):**
  * **GoogLeNet-1D (Pre-Trained):** En iyi genel performans (R: `0.9780`, MAE: `15.05`, sMAPE: `%16.40`).
  * **Kombinasyon 2 (Custom Arch):** En iyi Mutlak Ortalama Hata performansı (R: `0.9756`, MAE: `14.82`, sMAPE: `%19.85`).
