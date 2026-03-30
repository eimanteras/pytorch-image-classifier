# LAB2 - Detali kodo analize (image_classifier.py)

## Tikslas
Sis failas skirtas atsiskaitymui: paaiskina ka daro kiekviena svarbi `image_classifier.py` dalis, kodel ji reikalinga ir kaip apie tai aiskiai atsakyti destytojui.

---

## 1. Projekto ideja

Projektas realizuoja 3 klasiu vaizdu klasifikatoriu:
- Cat
- Dog
- Car

Naudojama transfer learning schema su ResNet-18:
1. Uzkraunami ImageNet svoriai
2. Uzsaldomas backbone
3. Permokomas tik paskutinis sluoksnis i 3 klases

Tai leidzia:
- mokyti greiciau
- tureti stabilesni rezultata su mazu duomenu kiekiu
- sumazinti overfitting rizika

---

## 2. Failo struktura ir vykdymo kelias

`image_classifier.py` vyksta tokia tvarka:
1. Konfiguruojami importai ir hiperparametrai
2. Parenkamas irenginys (`cuda` arba `cpu`)
3. Atsisiunciami duomenys is OpenImages
4. Pritaikomos transformacijos
5. Sukuriami DataLoader
6. Sukuriamas modelis
7. Vykdomas mokymas
8. Vykdomas testavimas su metrikomis
9. Sugeneruojami grafikai

Entry point:
- `if __name__ == "__main__":`
- kviecia `main()`

---

## 3. Eiluciu paaiskinimas pagal blokus

## 3.1. Modulio antraste ir importai

### Eilutes 1-22
Komentaru bloke aprasyta:
- darbo variantas
- modelio architektura
- metrikos
- paleidimo komandos

### Eilutes 24-52
Importai skirstomi i:
- standartinius (`os`, `random`, `shutil`)
- ML bibliotekas (`torch`, `torchvision`, `sklearn`)
- duomenu gavima (`fiftyone`)
- vizualizacija (`matplotlib`, `seaborn`)

Kodel svarbu gynime:
- gali paaiskinti, kad kiekviena biblioteka turi konkretu vaidmeni pipeline'e.

---

## 3.2. Konfiguracija

### Eilute 57
`CLASSES = ["Cat", "Dog", "Car"]`

Tai tavo problemos etiketes. Visos funkcijos veliau remiasi sia seka (ypac confusion matrix ir ataskaitos).

### Eilutes 60-62
`DATA_DIR`, `TRAIN_DIR`, `TEST_DIR` nusako kur lokaliai laikomi duomenys.

### Eilutes 65-70
Hiperparametrai:
- `NUM_EPOCHS`
- `BATCH_SIZE`
- `LEARNING_RATE`
- `TRAIN_SAMPLES`
- `TEST_SAMPLES`
- `VAL_SPLIT`

### Eilutes 73-76
Fiksuojamos sekos (`SEED`), kad rezultatai butu kiek galima atkuriami.

### Eilutes 79-80
`DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")`

Automatinis parinkimas:
- jei yra CUDA -> treniruoja ant GPU
- jei nera -> ant CPU

Gynimo sakinys:
"Kodas yra portable: nereikia rankiniu if'u tarp skirtingu kompiuteriu."

---

## 3.3. Duomenu atsisiuntimas

## Funkcija: `download_openimages(...)` (nuo 85 eil.)

Ka daro:
1. Patikrina ar katalogas jau yra su klasiu pakatalogiais
2. Jei yra - praleidzia atsisiuntima
3. Jei nera - kviecia `foz.load_zoo_dataset(...)`
4. Filtruoja kiekviena klase atskirai
5. Kopijuoja failus i `ImageFolder` stiliaus struktura

Kodel toks sprendimas geras:
- `ImageFolder` suderinamas su `torchvision.datasets.ImageFolder`
- nereikia rasyti custom dataset klases

Svarbi pastaba gynimui:
OpenImages ne visada grazina idealiai vienoda klasiu kieki net su `max_samples`, todel realus skaiciai gali skirtis tarp klasiu.

---

## 3.4. Transformacijos (augmentacija)

## Funkcija: `get_transforms()` (nuo 131 eil.)

Grazina zodyna su 2 rinkiniais:
- `train`
- `test`

### Train transformacijos
- `RandomResizedCrop`
- `RandomHorizontalFlip`
- `RandomRotation`
- `ColorJitter`
- `RandomGrayscale`
- `ToTensor`
- `Normalize`
- `RandomErasing`

Kodel tai reikalinga:
- padidina duomenu ivairove
- padeda mazinti overfitting
- modelis tampa atsparesnis realioms salygoms

### Test transformacijos
Naudojama tik:
- `Resize`
- `ToTensor`
- `Normalize`

Kodel be augmentacijos testui:
- testavimas turi buti stabilus ir pakartojamas
- metrikos turi atspindeti tikra modelio kokybe, o ne atsitiktine augmentacija

---

## 3.5. DataLoader sukurimas

## Funkcija: `build_dataloaders()` (nuo 207 eil.)

Ka daro:
1. Sukuria `full_train = ImageFolder(TRAIN_DIR, transform=tf["train"])`
2. Padalina i train/val pagal `VAL_SPLIT`
3. Testui naudoja atskira `TEST_DIR`
4. Sukuria 3 DataLoader:
   - `train_loader` su `shuffle=True`
   - `val_loader` su `shuffle=False`
   - `test_loader` su `shuffle=False`

Kodel `shuffle=True` tik train:
- train metu reikia randomizacijos
- val/test turi buti deterministiniai

---

## 3.6. Modelio konstravimas

## Funkcija: `build_model(...)` (nuo 247 eil.)

Logika:
1. `models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`
2. `for param in model.parameters(): param.requires_grad = False`
3. Pakeiciama `model.fc` i:
   - `Dropout(0.3)`
   - `Linear(in_features, num_classes)`

Kodel taip:
- backbone jau moka universalius vizualinius bruozus
- mokyti tik galva pigiau ir stabiliau
- su mazu dataset tai daznai geresne praktika nei full fine-tuning

Gynimo skaicius:
"Mokomu parametru lieka apie 1539, vietoje desimciu milijonu."

---

## 3.7. Mokymas

## Funkcija: `train_model(...)` (nuo 284 eil.)

Naudojama:
- `Adam` optimizeris tik `model.fc.parameters()`
- `CrossEntropyLoss`
- `ReduceLROnPlateau`

Per kiekviena epocha:
1. `model.train()`
2. ciklas per train batch'us:
   - `zero_grad`
   - forward
   - loss
   - backward
   - step
3. `model.eval()` validacijai su `torch.no_grad()`
4. skaiciuojami:
   - `avg_train_loss`
   - `avg_val_loss`
   - `train_acc`
   - `val_acc`
5. scheduler gauna `avg_val_loss`
6. jei pagerina val loss -> saugo `best_model.pth`

Kodel saugoti pagal `val_loss`, o ne train loss:
- train loss gali gereti net kai modelis persimoko
- validation geriau atspindi generalizacija

---

## 3.8. Vertinimas ir metrikos

## Funkcija: `evaluate_model(...)` (nuo 372 eil.)

Ka daro:
1. Uzkrauna `best_model.pth`
2. Paleidzia inferencija ant test aibes
3. Sukaupia `all_preds` ir `all_labels`
4. Skaiciuoja metrikas:
   - `accuracy`
   - `precision` (macro)
   - `recall` (macro)
   - `f1` (macro)
   - `confusion_matrix`
5. Isveda `classification_report`

Kodel `average="macro"`:
- kiekviena klase turi vienoda svori
- ypac svarbu kai klases gali buti disbalansuotos

---

## 3.9. Vizualizacija

## Funkcija: `plot_results(...)` (nuo 436 eil.)

Sukuriami 2 paveikslai:
1. `training_history.png`
   - train/val loss
   - train/val accuracy
2. `confusion_matrix.png`
   - klasifikavimo matrica

Kodel tai svarbu gynime:
- grafikai leidzia greitai parodyti mokymo dinamika
- confusion matrix parodo kurias klases modelis painioja dazniausiai

---

## 3.10. Augmentaciju demonstracija

## Funkcija: `visualize_augmentations(...)` (nuo 489 eil.)

Ka daro:
1. paima kelis paveikslus is pirmos klases
2. rodo originala ir 4 augmentuotas versijas
3. issaugo `augmentation_preview.png`

Tai labai geras demonstracinis blokas atsiskaitymui, nes vizualiai parodo augmentacijos prasme.

---

## 3.11. Vieno vaizdo klasifikacija

## Funkcija: `predict_single_image(...)` (nuo 546 eil.)

Ka daro:
1. atidaro paveiksla per PIL
2. pritaiko test transform
3. prideda batch dimensija (`unsqueeze(0)`)
4. paleidzia modeli
5. grazina klases pavadinima

Naudojimas patogus, kai destytojas duoda atskira paveiksla gyvai.

---

## 3.12. Pagrindinis srautas

## Funkcija: `main()` (nuo 578 eil.)

Vykdymo seka:
1. `download_openimages("train", ...)`
2. `download_openimages("test", ...)`
3. `visualize_augmentations(...)`
4. `build_dataloaders()`
5. `build_model(...)`
6. `train_model(...)`
7. `evaluate_model(...)`
8. `plot_results(...)`

Paskutines eilutes:
- `if __name__ == "__main__":`
- `main()`

---

## 4. Dazni gynimo klausimai ir trumpi atsakymai

### Q1. Kodel ResNet-18?
A: Geras tikslumo/greicio kompromisas, ypac kai duomenu nera labai daug.

### Q2. Kodel uzsaldytas backbone?
A: Kad naudotume jau ismoktus ImageNet bruozus ir mokytume tik paskutini sluoksni.

### Q3. Kodel augmentacija tik train?
A: Test turi buti deterministinis ir atkartojamas.

### Q4. Kodel macro precision/recall/F1?
A: Nes macro vienodai vertina kiekviena klase.

### Q5. Ka duoda `ReduceLROnPlateau`?
A: Automatiskai mazina mokymosi zingsni, kai validacijos nuostoliai nebegereja.

### Q6. Ka reiskia `best_model.pth`?
A: Isaugota modelio busena su maziausiu validacijos loss.

### Q7. Kodel `torch.no_grad()` validacijoje ir teste?
A: Taupo atminti ir greitina inferencija, nes gradientu nera poreikio skaiciuoti.

---

## 5. Ka butina pamineti per atsiskaityma

1. Turi 3 klases ir transfer learning architektura.
2. Naudoji duomenu augmentacija tik train fazei.
3. Atskirai valdai train/val/test.
4. Saugai geriausia modeli pagal validacijos nuostolius.
5. Skaiciuoji visas svarbias metrikas ir turi confusion matrix.
6. Kodas automatinai pasirenka GPU/CPU pagal aplinka.

---

## 6. Greitas pristatymo scenarijus (1-2 min)

"Sis projektas sprendzia 3 klasiu vaizdu klasifikavimo uzdavini Cat/Dog/Car. Naudoju ResNet-18 su transfer learning: backbone uzsaldytas, mokoma tik klasifikavimo galva su dropout. Duomenis imu is OpenImages per fiftyone, juos suformatuoju i ImageFolder ir train aibei taikau augmentacija. Mokymo metu stebiu train ir validation metrikas, o geriausia modeli saugau pagal maziausia validation loss. Po mokymo skaiciuoju accuracy, macro precision, macro recall, macro F1 ir braizau confusion matrix bei mokymo istorijos grafikus. Kodas veikia tiek ant CPU, tiek ant GPU." 

---

## 7. Super detali line-by-line atmintine (gynimui)

Zemiau pateikta glausta, bet labai tiksli atmintine pagal kodo eigas.

### 7.1. Pradzia ir konfig

- 24-26: importuojami `os`, `random`, `shutil` failu operacijoms ir atkuriamumui.
- 29-36: importuojamas PyTorch ir torchvision modelio mokymui bei transformacijoms.
- 38-39: fiftyone importai duomenu parsisiuntimui is OpenImages.
- 41-52: metriku, grafiku ir progreso juostos importai.
- 57: klasiu sarasas, kurio tvarka veliau naudojama metrikose ir confusion matrix.
- 60-62: katalogu keliai train/test rinkiniams.
- 65-70: hiperparametrai, valdantys mokymo trukme ir sample kieki.
- 73-76: random seklu nustatymas atkuriamumui.
- 79: automatinis irenginio parinkimas cuda/cpu.
- 80: konsoles loginimas, kad iskart matytum ant ko mokoma.

### 7.2. Duomenu gavimas

- 97-100: jei katalogas jau pilnas klasiu pakatalogiu, atsisiuntimas praleidziamas.
- 104-110: kvieciamas OpenImages zoo loader su klasemis ir split.
- 112-114: sukuriami kiekvienos klases katalogai.
- 117-120: filtruojami tik einamos klases pavyzdziai.
- 122-126: kopijuojami failai i vietine train/test struktura.

Trumpas gynimo akcentas:
- sita funkcija padaro OpenImages suderinama su `ImageFolder`, todel nereikia custom dataset klases.

### 7.3. Transformacijos

- 161: `RandomResizedCrop` imituoja skirtinga atstuma iki objekto.
- 164: `RandomHorizontalFlip` didina orientacijos ivairove.
- 167: `RandomRotation` daro modelio orientacijos robustiskuma.
- 171-176: `ColorJitter` imituoja apsvietimo skirtumus.
- 179: `RandomGrayscale` mazina priklausomybe nuo spalvos.
- 182: `ToTensor` konvertuoja i tensor formata.
- 185: `Normalize` su ImageNet statistika.
- 189: `RandomErasing` padeda kai objekto dalis uzdengta.
- 194-198: test pipeline be augmentacijos, tik stabilus preprocessing.

### 7.4. DataLoader

- 219: sukuriamas `ImageFolder` train rinkinys su train transform.
- 222-229: train rinkinys padalinamas i train/val.
- 232: test rinkinys kraunamas is atskiro katalogo.
- 239-241: sukuriami trys DataLoader su tinkamu shuffle.

### 7.5. Modelis

- 267: uzkraunamas `resnet18` su ImageNet svoriais.
- 270-271: visi parametrai uzsaldomi (`requires_grad=False`).
- 274-278: pakeiciamas galutinis `fc` i dropout + linear 3 klasems.
- 280: modelis perkeliamas i pasirinkta irengini.

### 7.6. Mokymo ciklas

- 304: optimizeris gauna tik `model.fc.parameters()`.
- 305: `CrossEntropyLoss` multi-class uzdaviniui.
- 308-310: scheduleris mazina LR jei val loss sustoja.
- 316: epochu ciklas.
- 318: train rezimas.
- 321-331: forward, loss, backward, step ant kiekvieno batch.
- 334: pereinama i eval rezima validacijai.
- 337-344: validacija be gradientu.
- 347-350: epochos vidurkiu skaiciavimas.
- 357: scheduleris atnaujinamas pagal val loss.
- 365-368: geriausio modelio issaugojimas i `best_model.pth`.

### 7.7. Vertinimas

- 391: uzkraunamas geriausias modelis is failo.
- 392: eval rezimas.
- 397-403: inferencija test rinkinyje.
- 410-418: accuracy, precision, recall, f1 ir confusion matrix skaiciavimas.
- 427: `classification_report` su klase pavadinimais.

### 7.8. Grafikai

- 447-461: train/val loss ir accuracy braizymas.
- 464: issaugojamas `training_history.png`.
- 469-481: confusion matrix heatmap.
- 486: issaugojamas `confusion_matrix.png`.

### 7.9. Papildomos funkcijos

- 503-509: augmentaciju demonstracijos transformacijos.
- 516-519: paimami keli paveikslai vizualizacijai.
- 528-533: kiekvienam vaizdui sugeneruojamos 4 augmentuotos versijos.
- 537: issaugomas `augmentation_preview.png`.
- 563-571: vieno paveikslo inferencija ir klases grazinimas.

### 7.10. Main seka

- 590-591: train ir test duomenu atsisiuntimas.
- 595: augmentacijos perziura.
- 598: DataLoader sukurimas.
- 601-603: modelio sukurimas ir mokomu parametru skaicius.
- 607: mokymo startas.
- 610: test metrikos.
- 613: grafiku generavimas.
- 624-625: entry point paleidzia `main()`.

---

## 8. Greiti atsakymai i labai konkretu klausima

### Kodel reikia `model.train()` ir `model.eval()`?
- Nes dropout ir batchnorm elgiasi skirtingai train ir eval rezimuose.

### Kodel `optimizer.zero_grad()` kiekvienam batch?
- Kad gradientai nesikauptu tarp skirtingu batch ir neiskraipytu update.

### Kodel `outputs.argmax(1)`?
- Nes modelis grazina logits per klase, o argmax paima didziausio logit indeksa.

### Kodel `zero_division=0` metrikose?
- Kad precision/recall skaiciavimas nesuluztu jei kuri klase neturi teigiamu prognoziu.

### Kodel validacijos dalis paimama is train?
- Kad turetum nepriklausoma rinkini hiperparametru kontrolei, nelieciant test rinkinio.

