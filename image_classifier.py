"""
=============================================================
Vaizdų klasifikavimo modelis – 2 užduotis
=============================================================
Autorius  : <Eimantas Šostakas>
LSP nr.   : <2016600>
Variantas : Cat | Dog | Car  (OpenImages V6)
Versija   : 1.0  (2026-03-20)
=============================================================

Architektūra:
  - Transfer learning su ResNet-18 (ImageNet svoriai)
  - Paskutinis sluoksnis pakeistas į 3 klases
  - Duomenys atsisiųsti per fiftyone biblioteką
  - Metrikų skaičiavimas: confusion matrix, accuracy,
    precision, recall, F1

Paleidimas:
    python -m pip install torch torchvision fiftyone scikit-learn
                                                matplotlib seaborn tqdm numpy pillow
  python image_classifier.py
=============================================================
"""

# ─── Standartinės bibliotekos ────────────────────────────────────────────────
import os
import random
import shutil

# ─── Trečiųjų šalių bibliotekos ──────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights

import fiftyone as fo
import fiftyone.zoo as foz

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from tqdm import tqdm

# ─── Konfigūracija ───────────────────────────────────────────────────────────

# Klasės (mažiausiai 3, reikalavimas įvykdytas)
CLASSES = ["Cat", "Dog", "Car"]

# Katalogų struktūra duomenims
DATA_DIR   = "data"
TRAIN_DIR  = os.path.join(DATA_DIR, "train")
TEST_DIR   = os.path.join(DATA_DIR, "test")

# Mokymo parametrai
NUM_EPOCHS       = 10      # epochų skaičius
BATCH_SIZE       = 32      # batch dydis
LEARNING_RATE    = 1e-3    # mokymosi greitis
TRAIN_SAMPLES    = 300     # pavyzdžių sk. mokymo aibėje (vienai klasei)
TEST_SAMPLES     = 75      # pavyzdžių sk. testavimo aibėje (vienai klasei)
VAL_SPLIT        = 0.15    # validacijos dalis iš treniravimo aibės

# Atsitiktinumo sėkla atkuriamumui
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Įrenginys: GPU jei prieinama, kitaip CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Naudojamas įrenginys: {DEVICE}")


# ─── 1. DUOMENŲ ATSISIUNTIMAS ─────────────────────────────────────────────────

def download_openimages(split: str, samples_per_class: int, out_dir: str) -> None:
    """
    Atsisiunčia OpenImages V6 vaizdus per fiftyone ir
    išsaugo juos į katalogų struktūrą, tinkamą ImageFolder.

    Parametrai
    ----------
    split            : 'train' arba 'test'
    samples_per_class: kiek pavyzdžių atsisiųsti kiekvienai klasei
    out_dir          : kelias, kur išsaugoti vaizdus
    """
    if os.path.exists(out_dir) and len(os.listdir(out_dir)) == len(CLASSES):
        print(f"[INFO] {split} duomenys jau egzistuoja – praleidžiama.")
        return

    print(f"\n[INFO] Atsisiunčiami {split} duomenys ({samples_per_class} vnt./klasei)...")

    dataset = foz.load_zoo_dataset(
        "open-images-v6",
        split=split,
        label_types=["classifications"],
        classes=CLASSES,
        max_samples=samples_per_class * len(CLASSES),
    )

    for cls in CLASSES:
        cls_dir = os.path.join(out_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)

        # Filtruojami tik einamos klasės pavyzdžiai
        view = dataset.filter_labels(
            "positive_labels",
            fo.ViewField("label") == cls,
        ).limit(samples_per_class)

        for sample in view:
            src = sample.filepath
            dst = os.path.join(cls_dir, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

        print(f"  {cls}: {len(os.listdir(cls_dir))} vaizdai → {cls_dir}")


# ─── 2. DUOMENŲ PARUOŠIMAS ────────────────────────────────────────────────────

def get_transforms() -> dict:
    """
    Grąžina transformacijų žodyną mokymo ir vertinimo duomenims.

    Mokymo transformacijos apima duomenų papildymą (data augmentation) –
    dirbtinį duomenų rinkinio didinimą transformuojant esamus vaizdus.
    Tai padeda:
      - išvengti persimokymо (overfitting),
      - padidinti modelio atsparumą įvairioms sąlygoms,
      - kompensuoti ribotą duomenų kiekį.

    Naudojamos augmentacijos:
      RandomResizedCrop   – atsitiktinė apkarpymas ir dydžio keitimas
                            imituoja skirtingus atstumus iki objekto
      RandomHorizontalFlip – horizontalus apvertimas (50% tikimybė)
                            veidrodinis objekto vaizdas
      RandomRotation      – pasukimas iki ±15°
                            objektai gali būti fotografuojami įvairiais kampais
      ColorJitter         – atsitiktinis ryškumo, kontrasto, soties keitimas
                            imituoja skirtingas apšvietimo sąlygas
      RandomGrayscale     – 10% tikimybė konvertuoti į pilkus atspalvius
                            moko modelį nesikliauti vien spalva
      RandomErasing       – atsitiktinis stačiakampio ištrynimas iš vaizdo
                            moko modelį klasifikuoti iš dalinės informacijos

    Testavimo transformacijos NENAUDOJA augmentacijos –
    rezultatai turi būti atkuriami ir objektyvūs.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    return {
        # ── Mokymo aibė: augmentacija + normalizacija ──────────────────────
        "train": transforms.Compose([
            # 1. Atsitiktinis apkarpymas ir dydžio keitimas
            #    scale=(0.7, 1.0) – išsaugo 70–100% pradinio ploto
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),

            # 2. Horizontalus apvertimas (50% tikimybė)
            transforms.RandomHorizontalFlip(p=0.5),

            # 3. Pasukimas iki ±15 laipsnių
            transforms.RandomRotation(degrees=15),

            # 4. Spalvų jitter'is: ryškumas ±30%, kontrastas ±30%,
            #    sočiai ±20%, atspalvis ±10%
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.1,
            ),

            # 5. 10% tikimybė konvertuoti į pilkus atspalvius
            transforms.RandomGrayscale(p=0.1),

            # 6. Konvertavimas į tensorių (privalo būti prieš RandomErasing)
            transforms.ToTensor(),

            # 7. Normalizacija pagal ImageNet statistiką
            transforms.Normalize(imagenet_mean, imagenet_std),

            # 8. Atsitiktinis stačiakampio ištrynimas
            #    p=0.2 – 20% tikimybė, scale=(0.02, 0.15) – ištrinamo ploto dalis
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        ]),

        # ── Testavimo aibė: JOKIOS augmentacijos ───────────────────────────
        "test": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]),
    }


def build_dataloaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Sukuria DataLoader objektus mokymui, validacijai ir testavimui.

    Grąžina
    -------
    train_loader, val_loader, test_loader
    """
    tf = get_transforms()

    # Pilna mokymo aibė
    full_train = datasets.ImageFolder(TRAIN_DIR, transform=tf["train"])

    # Padalinimas į mokymo ir validacijos aibes
    n_total = len(full_train)
    n_val   = int(n_total * VAL_SPLIT)
    n_train = n_total - n_val
    train_set, val_set = torch.utils.data.random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )

    # Testavimo aibė (atskiras katalogas)
    test_set = datasets.ImageFolder(TEST_DIR, transform=tf["test"])

    print(f"\n[INFO] Duomenų aibių dydžiai:")
    print(f"  Mokymo   : {n_train} vaizdai")
    print(f"  Validacijos: {n_val} vaizdai")
    print(f"  Testavimo: {len(test_set)} vaizdai")
    print(f"  Klasės   : {full_train.classes}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


# ─── 3. MODELIO KŪRIMAS ───────────────────────────────────────────────────────

def build_model(num_classes: int = 3) -> nn.Module:
    """
    Sukuria ResNet-18 modelį su transfer learning.

    Strategija:
      1. Užkraunami iš anksto apmokyti svoriai (ImageNet).
      2. Užšaldomi visi sluoksniai (feature extraction).
      3. Paskutinis visiškai sujungtas sluoksnis pakeičiamas
         į naują, su mūsų klasių skaičiumi.

    Parametrai
    ----------
    num_classes : klasifikuojamų klasių skaičius

    Grąžina
    -------
    Sukonfigūruotas PyTorch modelis
    """
    # Užkraunamas iš anksto apmokytas modelis
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # Užšaldomos konvoliucinio karkaso svoriai
    for param in model.parameters():
        param.requires_grad = False

    # Pakeičiamas paskutinis sluoksnis
    in_features = model.fc.in_features          # 512 ResNet-18 atveju
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),                      # Regularizacija overfitting'ui mažinti
        nn.Linear(in_features, num_classes),
    )

    return model.to(DEVICE)


# ─── 4. MOKYMAS ──────────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> tuple[nn.Module, dict]:
    """
    Apmokyti modelį naudojant kryžminę entropiją ir Adam optimizatorių.

    Parametrai
    ----------
    model        : PyTorch modelis
    train_loader : mokymo duomenų DataLoader
    val_loader   : validacijos duomenų DataLoader

    Grąžina
    -------
    Apmokytas modelis ir istorija (nuostoliai + tikslumas)
    """
    # Tik naujo sluoksnio parametrai bus atnaujinami
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Mokymosi greičio sumažinimas, jei validacijos nuostoliai nesimažina
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        # ── Mokymo fazė ──
        model.train()
        train_loss, train_correct = 0.0, 0

        for images, labels in tqdm(train_loader, desc=f"Epocha {epoch}/{NUM_EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        # ── Validacijos fazė ──
        model.eval()
        val_loss, val_correct = 0.0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs   = model(images)
                loss      = criterion(outputs, labels)
                val_loss    += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        # Epochos metrikos
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_val_loss   = val_loss   / len(val_loader.dataset)
        train_acc      = train_correct / len(train_loader.dataset)
        val_acc        = val_correct   / len(val_loader.dataset)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        scheduler.step(avg_val_loss)

        print(
            f"  Nuostoliai – mokymo: {avg_train_loss:.4f} | validacijos: {avg_val_loss:.4f} | "
            f"Tikslumas – mokymo: {train_acc:.4f} | validacijos: {val_acc:.4f}"
        )

        # Geriausio modelio išsaugojimas
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("  [✓] Modelis išsaugotas (best_model.pth)")

    return model, history


# ─── 5. VERTINIMAS ────────────────────────────────────────────────────────────

def evaluate_model(model: nn.Module, test_loader: DataLoader) -> dict:
    """
    Apskaičiuoja visas reikalaujamas metrikas testavimo aibėje.

    Metrikos:
      - Klasifikavimo matrica (confusion matrix)
      - Tikslumas (accuracy)
      - Precizija (precision) – makro vidurkis
      - Atkūrimas (recall)   – makro vidurkis
      - F1 balas             – makro vidurkis

    Parametrai
    ----------
    model       : apmokytas modelis
    test_loader : testavimo DataLoader

    Grąžina
    -------
    Žodynas su visomis metrikomis
    """
    # Užkraunamas geriausias modelis
    model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds   = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Metrikos
    metrics = {
        "accuracy"  : accuracy_score(all_labels, all_preds),
        "precision" : precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall"    : recall_score(all_labels, all_preds,    average="macro", zero_division=0),
        "f1"        : f1_score(all_labels, all_preds,        average="macro", zero_division=0),
        "conf_matrix": confusion_matrix(all_labels, all_preds),
        "preds"     : all_preds,
        "labels"    : all_labels,
    }

    # Išvedimas į konsolę
    print("\n" + "=" * 50)
    print("TESTAVIMO AIBĖS REZULTATAI")
    print("=" * 50)
    print(f"  Tikslumas  (Accuracy) : {metrics['accuracy']:.4f}")
    print(f"  Precizija  (Precision): {metrics['precision']:.4f}")
    print(f"  Atkūrimas  (Recall)   : {metrics['recall']:.4f}")
    print(f"  F1 balas   (F1-Score) : {metrics['f1']:.4f}")
    print("\nDetali ataskaita:")
    print(classification_report(all_labels, all_preds, target_names=CLASSES))

    return metrics


# ─── 6. VIZUALIZACIJA ─────────────────────────────────────────────────────────

def plot_results(history: dict, metrics: dict) -> None:
    """
    Braižo mokymo istoriją ir klasifikavimo matricą.

    Sukuria du paveikslus:
      1. training_history.png – nuostoliai ir tikslumas per epochas
      2. confusion_matrix.png – klasifikavimo matrica
    """
    # ── Mokymo istorija ──
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Mokymas")
    axes[0].plot(epochs, history["val_loss"],   label="Validacija")
    axes[0].set_title("Nuostoliai (Loss)")
    axes[0].set_xlabel("Epocha")
    axes[0].set_ylabel("Nuostoliai")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Mokymas")
    axes[1].plot(epochs, history["val_acc"],   label="Validacija")
    axes[1].set_title("Tikslumas (Accuracy)")
    axes[1].set_xlabel("Epocha")
    axes[1].set_ylabel("Tikslumas")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    print("[INFO] Išsaugota: training_history.png")
    plt.show()

    # ── Klasifikavimo matrica ──
    cm = metrics["conf_matrix"]
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
    )
    plt.title("Klasifikavimo matrica (Confusion Matrix)")
    plt.xlabel("Prognozuota klasė")
    plt.ylabel("Tikroji klasė")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("[INFO] Išsaugota: confusion_matrix.png")
    plt.show()


# ─── 7. ATSKIRO VAIZDO KLASIFIKAVIMAS ────────────────────────────────────────

def visualize_augmentations(image_dir: str, num_images: int = 4) -> None:
    """
    Parodo, kaip augmentacija transformuoja tuos pačius vaizdus.

    Kiekvienam pasirinktam vaizdui rodomas originalas ir
    kelios augmentuotos versijos – naudinga demonstracijai.

    Parametrai
    ----------
    image_dir  : katalogas su klasių pakatalogiais (TRAIN_DIR)
    num_images : kiek vaizdų rodyti
    """
    from PIL import Image

    # Augmentacijos BEZ normalizacijos (vizualizacijai)
    aug_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
    ])

    first_class_dir = os.path.join(image_dir, CLASSES[0])
    image_files = [
        os.path.join(first_class_dir, f)
        for f in os.listdir(first_class_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ][:num_images]

    fig, axes = plt.subplots(num_images, 5, figsize=(14, num_images * 3))
    fig.suptitle(
        f"Duomenų augmentacija – klasė: {CLASSES[0]}\n"
        "Stulpelis 1: originalas | 2–5: augmentuotos versijos",
        fontsize=13,
    )

    for row, img_path in enumerate(image_files):
        original = Image.open(img_path).convert("RGB")
        original_resized = original.resize((224, 224))

        axes[row, 0].imshow(original_resized)
        axes[row, 0].set_title("Originalas", fontsize=9)
        axes[row, 0].axis("off")

        for col in range(1, 5):
            augmented = aug_transform(original)
            axes[row, col].imshow(augmented)
            axes[row, col].set_title(f"Augm. #{col}", fontsize=9)
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig("augmentation_preview.png", dpi=150)
    print("[INFO] Išsaugota: augmentation_preview.png")
    plt.show()


def predict_single_image(model: nn.Module, image_path: str) -> str:
    """
    Klasifikuoja vieną vaizdą.

    Naudojimas:
        label = predict_single_image(model, "cat_photo.jpg")

    Parametrai
    ----------
    model      : apmokytas modelis
    image_path : kelias iki vaizdo failo

    Grąžina
    -------
    Prognozuota klasė (eilutė)
    """
    from PIL import Image

    tf = get_transforms()["test"]
    image = Image.open(image_path).convert("RGB")
    tensor = tf(image).unsqueeze(0).to(DEVICE)   # pridedame batch dimensiją

    model.eval()
    with torch.no_grad():
        output = model(tensor)
        pred_idx = output.argmax(1).item()

    return CLASSES[pred_idx]


# ─── PAGRINDINIS SRAUTAS ──────────────────────────────────────────────────────

def main() -> None:
    """
    Pagrindinė funkcija, kuri:
      1. Atsisiunčia duomenis iš OpenImages V6
      2. Paruošia DataLoader objektus
      3. Sukuria ir apmokyti modelį
      4. Apskaičiuoja metrikas
      5. Braižo grafikus
    """
    print("\n" + "=" * 55)
    print("  Vaizdų klasifikavimas – Cat | Dog | Car")
    print("  Modelis: ResNet-18 (transfer learning)")
    print("=" * 55)

    # 1. Duomenų atsisiuntimas
    download_openimages("train", TRAIN_SAMPLES, TRAIN_DIR)
    download_openimages("test",  TEST_SAMPLES,  TEST_DIR)

    # 2. Augmentacijos vizualizacija (parodoma prieš mokymą)
    print("\n[INFO] Braižoma augmentacijos vizualizacija...")
    visualize_augmentations(TRAIN_DIR)

    # 3. DataLoader'iai
    train_loader, val_loader, test_loader = build_dataloaders()

    # 3. Modelis
    model = build_model(num_classes=len(CLASSES))
    print(f"\n[INFO] Modelio parametrų skaičius: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 4. Mokymas
    print(f"\n[INFO] Pradedamas mokymas ({NUM_EPOCHS} epochų)...\n")
    model, history = train_model(model, train_loader, val_loader)

    # 5. Vertinimas
    metrics = evaluate_model(model, test_loader)

    # 6. Grafikai
    plot_results(history, metrics)

    # 7. Testavimo pavyzdys su atskiru vaizdu
    # (panaudokite, kai dėstytojas atsiųs testinį vaizdą)
    # predicted_class = predict_single_image(model, "test_image.jpg")
    # print(f"Prognozuota klasė: {predicted_class}")


if __name__ == "__main__":
    main()
