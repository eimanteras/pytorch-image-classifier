# 🖼️ Image Classifier – Cat | Dog | Car

> **Vilniaus universitetas** · Mašininis mokymasis · 2 užduotis  
> Autorius: **Eimantas Šostakas** · LSP: `2016600`

Vaizdų klasifikavimo modelis su **PyTorch ResNet-18** ir transfer learning.  
Duomenys: **OpenImages V6** (Cat, Dog, Car klasės).

---

## Architektūra

```
Input (224×224 RGB)
      ↓
ResNet-18 backbone  ← Frozen (ImageNet weights)
      ↓
Dropout(0.3)
      ↓
Linear(512 → 3)     ← Trained
      ↓
Cat | Dog | Car
```

**Transfer learning** – iš anksto apmokyti ResNet-18 svoriai (ImageNet).  
Mokomi tik paskutinio sluoksnio parametrai (~1 539 iš 11 mln.).

---

## Duomenų augmentacija

| Transformacija | Paskirtis |
|---|---|
| `RandomResizedCrop` | Skirtingi atstumai iki objekto |
| `RandomHorizontalFlip` | Veidrodinis apvertimas |
| `RandomRotation ±15°` | Skirtingi fotografavimo kampai |
| `ColorJitter` | Apšvietimo variacija |
| `RandomGrayscale 10%` | Spalvos nepriklausomybė |
| `RandomErasing` | Dalinės informacijos tolerancija |

---

## Metrikos (testavimo aibė)

| Metrika | Reikšmė |
|---|---|
| Accuracy | – |
| Precision (macro) | – |
| Recall (macro) | – |
| F1-score (macro) | – |

*Užpildyk po apmokymo.*

---

## Paleidimas

```bash
# 1. Sukurti ir aktyvuoti aplinką (Windows)
py -3.12 -m venv .venv
.venv\Scripts\activate

# 2A. CPU variantas
python -m pip install -r requirements.txt

# 2B. GPU variantas (NVIDIA)
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt

# 3. Paleisti
python image_classifier.py
```

Duomenys atsisiunčiami automatiškai iš OpenImages V6.

---

## Failo struktūra

```
├── image_classifier.py       # Pagrindinis kodas
├── best_model.pth            # Išsaugotas modelis (po apmokymo)
├── training_history.png      # Mokymo grafikai
├── confusion_matrix.png      # Klasifikavimo matrica
├── augmentation_preview.png  # Augmentacijos vizualizacija
├── .github/
│   └── agents/
│       └── image-classifier-assistant.agent.md  # Copilot agentas
└── README.md
```

---

## Technologijos

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![OpenImages](https://img.shields.io/badge/Dataset-OpenImages_V6-green)
