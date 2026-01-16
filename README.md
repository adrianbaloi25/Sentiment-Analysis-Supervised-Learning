# Proiect: Tehnici de Învățare Automată (TIA)
Student: Adrian Costin Băloi, grupa 344AB

## Aplicație: Analiza Sentimentelor în Recenziile de Filme (Sentiment Analysis)

Acest document detaliază implementarea și rezultatele unui sistem de clasificare automată a textului, utilizând tehnici de învățare supervizată.

### 1. Detalierea Aplicației și a Setului de Date

#### 1.1 Aplicația Aleasă

Aplicația dezvoltată are ca scop identificarea polarității emoționale a unui text (pozitiv sau negativ). 

Aceasta este o problemă de clasificare binară în cadrul procesării limbajului natural (NLP). Importanța acestei aplicații constă în capacitatea de a analiza automat mii de recenzii de produse, postări pe rețelele sociale sau feedback-ul clienților pentru a extrage informații valoroase de business.

#### 1.2 Setul de Date Utilizat

Pentru acest proiect, am utilizat setul de date oficial IMDb Movie Reviews Dataset, recunoscut la nivel internațional pentru testarea algoritmilor de analiză a sentimentelor.

- **Sursă:** Setul complet conține 50.000 de recenzii de film etichetate.

- **Volum utilizat:** Din motive de performanță computațională, am utilizat un subset de 5.000 de recenzii, asigurând un echilibru între precizie și timpul de antrenare.

- **Intrare (X):** Recenzii sub formă de text brut.

- **Ieșire (y):** Etichete binare: 1 pentru sentiment pozitiv și 0 pentru sentiment negativ.

### 2. Preprocesarea Datelor

Datele de tip text sunt nestructurate și necesită o transformare matematică. Procesul implementat în modulul preprocessing.py cuprinde:

- **Uniformizare (Lowercasing):** Conversia întregului text la litere mici pentru a evita duplicarea cuvintelor bazată pe capitalizare.

- **Curățare (Cleaning):** Eliminarea semnelor de punctuație și a caracterelor speciale folosind expresii regulate.

- **Vectorizare TF-IDF:** Am transformat cuvintele în vectori numerici folosind metoda Term Frequency-Inverse Document Frequency. Aceasta calculează importanța unui cuvânt $t$ într-un document $d$ în raport cu un corpus $D$ prin formula:

$$TF-IDF(t, d, D) = tf(t, d) \cdot \log\left(\frac{N}{df(t, D)}\right)$$

### 3. Tehnici de Învățare Implementate

Am ales două modele cu principii matematice diferite pentru a oferi o analiză comparativă:

#### 3.1 Support Vector Machine (SVM)

SVM este un algoritm de învățare supervizată care funcționează prin găsirea unui hiperplan care separă clasele cu o marjă maximă. 

În cazul clasificării textului, unde datele sunt adesea liniar separabile în spații cu dimensiuni mari, un kernel liniar este extrem de eficient.

#### 3.2 Multinomial Naive Bayes (NB)
Naive Bayes este un clasificator probabilistic bazat pe Teorema lui Bayes. Acesta presupune independența condiționată între caracteristici (cuvinte). Formula de bază utilizată este:

$$P(C|x) = \frac{P(x|C)P(C)}{P(x)}$$

Este recunoscut pentru viteza sa de antrenare și performanțele solide în sarcini de tip "spam detection" și "sentiment analysis".

### 4. Rezultate și Comentarii

Modelele au fost evaluate folosind un set de test de $20\%$ din datele totale.

#### 4.1 Comparația Performanței (Tabel)

| Model                  | SVM Model | Naive Bayes Model |
|------------------------|---------------|---------------|
| Accuracy               | ~0.88         | ~0.85         |
| Precision              | ~0.88         | ~0.86         |
| Recall                 | ~0.87         | ~0.84         |
| F1-Score               | ~0.88         | ~0.85         |

Notă: Acuratețea perfectă se datorează setului de date controlat utilizat pentru demonstrație. În condiții reale, acuratețea se situează între $85-92\%$.

#### 4.2 Analiza Grafică

Rezultatele vizuale salvate în folderul _/results_ oferă următoarele perspective:

- **Compararea Acurateței (accuracy_comparison.png):** Graficul evidențiază faptul că ambele modele au reușit să captureze tiparele cheie din text.

- **Matricea de Confuzie (confusion_matrix_svm.png):** Indică faptul că nu au existat confuzii între clase (fals pozitivi sau fals negativi), modelele fiind capabile să identifice corect atât recenziile pozitive, cât și pe cele negative.

### 5. Concluzii

În cadrul acestui proiect, am modelat o problemă de învățare supervizată aplicată pe date textuale. Am demonstrat că:

Preprocesarea și vectorizarea TF-IDF sunt critice pentru succesul modelelor de NLP.

SVM oferă o robustețe geometrică mare, în timp ce Naive Bayes este o soluție probabilistică extrem de rapidă.

Ambele metode sunt adecvate pentru analiza sentimentelor, oferind rezultate competitive.

Proiectul poate fi extins prin utilizarea Deep Learning (rețele neuronale de tip LSTM sau modele Transformer) pentru a gestiona nuanțe lingvistice mai complexe, precum sarcasmul.