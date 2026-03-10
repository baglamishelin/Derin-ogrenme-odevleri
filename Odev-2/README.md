
## 🧠 XOR Problemi ve Çok Katmanlı Sinir Ağları

Bu çalışma, yapay sinir ağlarının tarihsel gelişimindeki en kritik dönüm noktalarından biri olan **XOR (Özel VEYA)** probleminin çözümünü içermektedir. 

### 🚀 Projenin Amacı
Tek katmanlı perceptron modellerinin çözemediği "doğrusal ayrıştırılamaz" (linearly non-separable) problemleri, **gizli katman (hidden layer)** ve **sigmoid aktivasyon fonksiyonu** kullanarak çözebilen bir yapay sinir ağı mimarisi oluşturmaktır.

### 🛠 Teknik Detaylar
* **Mimari:** 2 Giriş, 2 Gizli Nöron ve 1 Çıkış Nöronu.
* **Aktivasyon:** İleri besleme ve geri yayılım (backpropagation) süreçlerinde `Sigmoid` fonksiyonu kullanılmıştır.
* **Optimizasyon:** Gradyan İnişi (Gradient Descent) yöntemiyle ağırlık ve bias değerleri güncellenmektedir.
* **Kütüphane:** Saf Python (math ve random kütüphaneleri dışında bağımlılık içermez).

### 📈 Sonuç
Model, yaklaşık 10,000 epoch sonunda XOR mantıksal kapısının tüm girdilerini %100 doğrulukla öğrenmekte ve karmaşık veri yapılarını öğrenme kapasitesini kanıtlamaktadır.
