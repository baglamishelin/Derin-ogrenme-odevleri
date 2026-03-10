import math
import random

random.seed(42)

# ─────────────────────────────────────────
#  AKTİVASYON FONKSİYONU: SİGMOİD
# ─────────────────────────────────────────
def sigmoid(x):
    # Çok büyük/küçük değerlerde taşmayı önlemek için sınırlandırıyoruz
    x = max(-500, min(500, x))
    return 1 / (1 + math.exp(-x))

def sigmoid_turev(x):
    # Sigmoid'in türevi: s(x) * (1 - s(x))
    s = sigmoid(x)
    return s * (1 - s)

# ─────────────────────────────────────────
#  AĞ AĞIRLIKLARI (rastgele başlatma)
# ─────────────────────────────────────────
# W1[i][j]: j. girişten i. gizli nörona ağırlık
W1 = [[random.uniform(-1, 1) for _ in range(2)] for _ in range(2)]
b1 = [random.uniform(-1, 1) for _ in range(2)]   # gizli katman bias

# W2[i]: i. gizli nörondan çıkış nöronuna ağırlık
W2 = [random.uniform(-1, 1) for _ in range(2)]
b2 = random.uniform(-1, 1)                        # çıkış bias

# ─────────────────────────────────────────
#  EĞİTİM VERİSİ (XOR tablosu)
# ─────────────────────────────────────────
# (giriş_A, giriş_B, beklenen_çıkış)
veri = [
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0),
]

ogrenme_hizi = 0.1
epoch_sayisi = 10000

# ─────────────────────────────────────────
#  FORWARD PASS (ileri besleme)
# ─────────────────────────────────────────
def ileri_besleme(a, b):
    # Gizli katman: her nöron için ağırlıklı toplam + aktivasyon
    z1_0 = a * W1[0][0] + b * W1[0][1] + b1[0]
    z1_1 = a * W1[1][0] + b * W1[1][1] + b1[1]
    h0 = sigmoid(z1_0)   # gizli nöron 1 çıkışı
    h1 = sigmoid(z1_1)   # gizli nöron 2 çıkışı

    # Çıkış katmanı
    z2 = h0 * W2[0] + h1 * W2[1] + b2
    cikis = sigmoid(z2)

    return z1_0, z1_1, h0, h1, z2, cikis

# ─────────────────────────────────────────
#  EĞİTİM DÖNGÜSÜ (backpropagation)
# ─────────────────────────────────────────
print("=" * 50)
print("  XOR Sinir Ağı — Eğitim Başlıyor")
print("=" * 50)

for epoch in range(epoch_sayisi + 1):
    toplam_kayip = 0

    for a, b, y in veri:
        # 1. İleri besleme
        z1_0, z1_1, h0, h1, z2, cikis = ileri_besleme(a, b)

        # 2. Kayıp hesapla (MSE)
        kayip = 0.5 * (cikis - y) ** 2
        toplam_kayip += kayip

        # 3. Geri yayılım — çıkış katmanı
        # dL/d(cikis) * d(cikis)/d(z2)
        delta2 = (cikis - y) * sigmoid_turev(z2)

        # 4. Geri yayılım — gizli katman
        delta1_0 = W2[0] * delta2 * sigmoid_turev(z1_0)
        delta1_1 = W2[1] * delta2 * sigmoid_turev(z1_1)

        # 5. Ağırlıkları güncelle (gradient descent)
        W2[0] -= ogrenme_hizi * delta2 * h0
        W2[1] -= ogrenme_hizi * delta2 * h1
        b2    -= ogrenme_hizi * delta2

        W1[0][0] -= ogrenme_hizi * delta1_0 * a
        W1[0][1] -= ogrenme_hizi * delta1_0 * b
        b1[0]    -= ogrenme_hizi * delta1_0

        W1[1][0] -= ogrenme_hizi * delta1_1 * a
        W1[1][1] -= ogrenme_hizi * delta1_1 * b
        b1[1]    -= ogrenme_hizi * delta1_1

    ortalama_kayip = toplam_kayip / 4

    # Her 1000 epoch'ta bir ilerlemeyi yazdır
    if epoch % 1000 == 0:
        print(f"  Epoch {epoch:5d}  →  Kayıp: {ortalama_kayip:.6f}")

    # Yeterince iyi öğrendiyse dur
    if ortalama_kayip < 0.001:
        print(f"  Epoch {epoch:5d}  →  Kayıp: {ortalama_kayip:.6f}  ✓ Hedef kayba ulaşıldı!")
        break

# ─────────────────────────────────────────
#  SONUÇLARI TEST ET
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("  Eğitim Sonrası Test Sonuçları")
print("=" * 50)
print(f"  {'A':>3}  {'B':>3}  {'Beklenen':>10}  {'Tahmin':>10}  {'Durum':>6}")
print("  " + "-" * 40)

dogru = 0
for a, b, y in veri:
    _, _, _, _, _, cikis = ileri_besleme(a, b)
    tahmin = 1 if cikis > 0.5 else 0
    durum = "✓" if tahmin == y else "✗"
    if tahmin == y:
        dogru += 1
    print(f"  {a:>3}  {b:>3}  {y:>10}  {cikis:>10.4f}  {durum:>6}")

print("  " + "-" * 40)
print(f"\n  Doğruluk: {dogru}/4 = %{dogru * 25}")
print("=" * 50)