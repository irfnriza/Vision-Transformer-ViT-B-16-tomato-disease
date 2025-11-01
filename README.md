# Vision Transformer (ViT) untuk Klasifikasi Penyakit Tomat

Proyek ini menggunakan Vision Transformer (ViT-B/16) untuk mengklasifikasikan penyakit pada tanaman tomat menggunakan deep learning dan transfer learning dari model pretrained ImageNet-1K.

## 📋 Deskripsi

Model ini dapat mengklasifikasikan gambar tanaman tomat ke dalam 3 kategori:
- **Tomato_healthy** - Tomat Sehat
- **Tomato_Late_blight** - Penyakit Late Blight
- **Tomato_YellowLeaf_Curl_Virus** - Virus Curl Daun Kuning

## 🎯 Fitur Utama

- ✅ Transfer learning menggunakan **Vision Transformer (ViT-B/16)** pretrained pada ImageNet-1K
- ✅ **Stratified split** untuk memastikan distribusi kelas yang seimbang (70% train, 15% val, 15% test)
- ✅ **Class weighting** untuk menangani class imbalance
- ✅ **Early stopping** dan **CosineAnnealingLR** scheduler untuk optimasi training
- ✅ **Attention map visualization** untuk interpretabilitas model
- ✅ Comprehensive evaluation dengan confusion matrix dan classification report

## 📊 Dataset

Dataset berisi **8,857 gambar** tomat dengan distribusi:
- Tomato_healthy: 1,591 gambar
- Tomato_Late_blight: 1,909 gambar
- Tomato_YellowLeaf_Curl_Virus: 5,357 gambar

### Struktur Dataset
```
dataset/
├── Tomato_healthy/
│   └── *.JPG
├── Tomato_Late_blight/
│   └── *.JPG
└── Tomato__Tomato_YellowLeaf__Curl_Virus/
    └── *.JPG
```

## 🛠️ Instalasi

### Requirements
- Python 3.8+
- CUDA-capable GPU (opsional, untuk training lebih cepat)

### Setup Environment

1. Clone repository ini:
```bash
git clone <repository-url>
cd tranformers
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Penggunaan

### Training Model

Jalankan notebook Jupyter untuk training model:

```bash
jupyter notebook notebook/notebook.ipynb
```

Atau jalankan semua cell secara otomatis menggunakan Python:
```bash
python -m jupyter nbconvert --to notebook --execute notebook/notebook.ipynb
```

### Konfigurasi Training

Parameter training utama (dapat dimodifikasi di notebook):
- **Batch size**: 32
- **Learning rate**: 1e-4
- **Weight decay**: 1e-4
- **Max epochs**: 100
- **Early stopping patience**: 10
- **Optimizer**: AdamW
- **Scheduler**: CosineAnnealingLR
- **Input size**: 224x224

### Struktur Proyek

```
tranformers/
├── dataset/                              # Dataset gambar tomat
│   ├── Tomato_healthy/
│   ├── Tomato_Late_blight/
│   └── Tomato__Tomato_YellowLeaf__Curl_Virus/
├── model/                                # Output model dan evaluasi
│   ├── best_vit_tomato.pth              # Model checkpoint terbaik
│   ├── vit_tomato_evaluation.txt        # Laporan evaluasi
│   ├── training_curves.png              # Kurva training
│   ├── confusion_matrix.png             # Confusion matrix
│   └── attention_maps/                  # Visualisasi attention maps
├── notebook/
│   └── notebook.ipynb                   # Jupyter notebook utama
├── README.md                            # Dokumentasi ini
└── requirements.txt                     # Python dependencies
```

## 📈 Hasil Training

Model akan menghasilkan:
1. **best_vit_tomato.pth** - Model checkpoint dengan akurasi validasi terbaik
2. **vit_tomato_evaluation.txt** - Laporan evaluasi lengkap
3. **training_curves.png** - Grafik loss dan accuracy selama training
4. **confusion_matrix.png** - Confusion matrix dari test set
5. **attention_maps/** - Visualisasi attention maps untuk interpretabilitas

## 🔍 Evaluasi Model

Model dievaluasi menggunakan metrik:
- Accuracy
- Precision (per-class dan macro average)
- Recall (per-class dan macro average)
- F1-Score (per-class dan macro average)
- Confusion Matrix
- Classification Report

## 💡 Interpretabilitas

Proyek ini menyediakan **attention map visualization** yang menunjukkan bagian mana dari gambar yang diperhatikan oleh model saat membuat prediksi, membantu memahami decision-making process model.

## ⚙️ Arsitektur Model

- **Base Model**: Vision Transformer (ViT-B/16)
- **Pretrained**: ImageNet-1K weights
- **Input Size**: 224x224 pixels
- **Output Classes**: 3 classes
- **Total Parameters**: ~86M parameters
- **Classifier Head**: Modified linear layer untuk 3 classes

## 🔧 Technical Details

### Data Preprocessing
- Resize ke 224x224
- ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Augmentation
- Training: Resize dan normalization
- Validation/Test: Resize dan normalization

### Class Weighting
Model menggunakan class weights untuk menangani imbalance dataset:
```python
class_weights = 1.0 / train_class_counts
class_weights = class_weights / class_weights.sum() * num_classes
```

## 📝 Citation

Jika Anda menggunakan proyek ini, mohon reference:

```
Vision Transformer for Tomato Disease Classification
Using ViT-B/16 pretrained on ImageNet-1K
```


---

**Catatan**: Model ini memerlukan GPU dengan memory yang cukup untuk training. Untuk inferensi, CPU juga dapat digunakan tetapi akan lebih lambat.
