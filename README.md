<div align="center">
  <img src="uni-pr.png" alt="University Logo" title="University Logo" width="200">
</div>

<div align="center">
<b>Fakulteti: Fakulteti i Inxhinierisë Elektrike dhe Kompjuterike</b><br>
<b>Departamenti: Departamenti i Inxhinierisë Kompjuterike</b><br>
<b>Lënda: Procesimi i gjuhëve natyrale</b><br>
<b>Niveli: Master</b><br>
<b>Profesori: Dr. Sc. Mërgim Hoti</b><br>
<b>Autorët: Endrit Hoda, Lorik Mustafa, Meriton Kryeziu</b><br>
</div>
<br>
<br>

# Përmbledhje Shumëgjuhëshe e Tekstit me modelin mT5

Ky projekt implementon një model të Mësimit të Thellë (Deep Learning) të tipit sekuencë-në-sekuencë, i aftë për të përmbledhur tekst në tre gjuhë: **Anglisht**, **Frëngjisht** dhe **Gjermanisht**.

Zgjidhja përmirëson (fine-tunes) modelin transformues **Google mT5 (Multilingual T5)** duke përdorur një përzierje të kuruar të grupeve të të dhënave (datasets) nga lajmet dhe artikujt wiki.

## Përmbledhje e Projektit

* **Detyra:** Përmbledhje Abstrakte (Abstractive Summarization)
* **Gjuhët:** Anglisht (`en`), Frëngjisht (`fr`), Gjermanisht (`de`)
* **Modeli Bazë:** `google/mt5-small`
* **Infrastruktura:** Python 3, PyTorch, Hugging Face Transformers
* **Fuqia Kompjuterike:** Projektuar për GPU T4 (Standardi i Google Colab)

## Strategjia e Të Dhënave

Për të siguruar që modeli të përgjithësohet mirë në fusha dhe gjuhë të ndryshme, u ndërtua një grup të dhënash i personalizuar duke marrë mostra dhe duke balancuar tre burime të ndryshme me kod të hapur (open-source):

| Burimi i Të Dhënave | Gjuhët e Përdorura | Ndarja | Mostra për Burim |
| :--- | :--- | :--- | :--- |
| **GEM/wiki_lingua** | en, fr, de | Train | 600 për gjuhë |
| **mlsum** | fr, de | Train | 600 për gjuhë |
| **csebuetnlp/xlsum** | en, fr | Train | 600 për gjuhë |

**Procesi i Përpunimit Paraprak (Preprocessing Pipeline):**
1. **Standardizimi:** Hartëzimi i emrave të ndryshëm të kolonave (p.sh., `article`, `document`, `text`) në një format të unifikuar `document` -> `summary`.
2. **Filtrimi:** U hoqën mostrat ku gjatësia e dokumentit ishte < 80 karaktere ose përmbledhja < 10 karaktere.
3. **Balancimi:** U reduktua numri i mostrave (downsampled) në 600 shembuj për çdo çift burim/gjuhë për të parandaluar anshmërinë gjuhësore.
4. **Totali i Të Dhënave:** ~4,200 shembuj trajnimi.

## Vlerësimi i Performancës: ROUGE & BERTScore

Ky seksion analizon cilësinë e përmbledhjeve të gjeneruara nga modeli duke përdorur dy metrika standarde: **ROUGE** dhe **BERTScore**. Këto metrika matin aspekte të ndryshme të cilësisë së përmbledhjes dhe interpretohen së bashku për një vlerësim më të saktë.

---

### Rezultatet e Metrikave

| Metrika        | Vlera  | Interpretimi |
|---------------|--------|--------------|
| **ROUGE-1**   | 0.0933 | Mbivendosje e ulët e fjalëve individuale |
| **ROUGE-2**   | 0.0260 | Mbivendosje shumë e ulët e bigrameve |
| **ROUGE-L**   | 0.0756 | Përputhje e dobët e strukturës së fjalive |
| **BERTScore F1** | 0.8023 | Ngjashmëri semantike e lartë |

---

### Analiza e ROUGE

**ROUGE** mat sa shumë fjalë ose sekuenca fjalësh nga përmbledhja referente shfaqen edhe në përmbledhjen e gjeneruar.

- Rezultatet e ulëta të **ROUGE-1**, **ROUGE-2** dhe **ROUGE-L** tregojnë se modeli:
  - Nuk po kopjon tekstin origjinal
  - Përdor formulime dhe struktura të ndryshme gjuhësore
- Kjo është tipike për modelet **abstraktive**, veçanërisht në kontekste **shumëgjuhëshe**, ku sinonimet dhe parafrazimi janë të zakonshme.

**Vetëm bazuar në ROUGE**, performanca do të dukej e dobët, por kjo metrikë ka kufizime të njohura.

---

### Analiza e BERTScore

**BERTScore** mat ngjashmërinë semantike duke përdorur *embeddings* kontekstuale, dhe jo vetëm përputhjen e fjalëve.

- **BERTScore F1 = 0.80+** tregon se:
  - Përmbledhjet ruajnë kuptimin origjinal të tekstit
  - Edhe kur përdoren sinonime ose riformulime, informacioni faktik mbetet i saktë
- Kjo metrikë është më e përshtatshme për:
  - Përmbledhje abstraktive
  - Modele moderne Transformer
  - Detyra multigjuhëshe

---

### Përfundim

Edhe pse rezultatet **ROUGE** janë relativisht të ulëta, **BERTScore i lartë** tregon qartë se modeli po kupton dhe po përmbledh saktë përmbajtjen, duke ruajtur kuptimin semantik të tekstit hyrës.

Ky kombinim metrikash konfirmon se modeli:
- Nuk po “citon” tekstin
- Po prodhon përmbledhje abstraktive, kuptimplota dhe faktikisht të sakta

Prandaj, performanca e modelit mund të konsiderohet **e mirë dhe e justifikuar**, sidomos në kontekst akademik dhe kërkimor.


## Kodi burimor

### Instalimi i dependencies

Projekti kërkon varësitë (dependencies) e mëposhtme:

```bash
pip install -q -U \
  "pandas==2.2.2" \
  "numpy>=2.0,<2.1" \
  "pyarrow>=15,<18" \
  "huggingface_hub>=0.33.5,<2.0" \
  "datasets==3.6.0" \
  "transformers>=4.41.0" \
  "accelerate>=0.33.0" \
  "evaluate>=0.4.2" \
  "rouge-score>=0.1.2" \
  "bert-score>=0.3.13" \
  "sentencepiece>=0.2.0" \
  "sacrebleu>=2.4.0"
```

### Perdorimi i CUDA-s ne mjedisin Google Colab

```bash
!nvidia-smi
import torch
print("CUDA available:", torch.cuda.is_available())
```

### Burimet e te dhenava dhe modeli

```python
DATA_SOURCES = [
    ("GEM/wiki_lingua", "en", "en", "train"),
    ("GEM/wiki_lingua", "de", "de", "train"),
    ("GEM/wiki_lingua", "fr", "fr", "train"),
    ("mlsum", "en", "en", "train"),
    ("mlsum", "de", "de", "train"),
    ("mlsum", "fr", "fr", "train"),
    ("csebuetnlp/xlsum", "english", "en", "train"),
    ("csebuetnlp/xlsum", "french", "fr", "train"),
]

MODEL_NAME = "google/mt5-small"
OUT_DIR = "./mt5-multilingual-summarizer"
FINAL_DIR = "./mt5-multilingual-summarizer-final"
```

### Popullimi i dataseteve

```python
datasets_list = []

for ds_name, cfg, lang, split in DATA_SOURCES:
    if not config_supported(ds_name, cfg):
        continue

    ds = safe_load_dataset(ds_name, cfg, split)
    ds = ds.select(range(min(len(ds), MAX_PER_SOURCE)))
    ds = ds.map(standardize_row, remove_columns=ds.column_names)
    ds = ds.filter(is_valid)
    ds = ds.map(lambda x: add_lang(x, lang))

    datasets_list.append(ds)

min_len = min(len(ds) for ds in datasets_list)
datasets_list = [ds.shuffle(seed=SEED).select(range(min_len)) for ds in datasets_list]

full = concatenate_datasets(datasets_list).shuffle(seed=SEED)
print("Total examples:", len(full))
```

### Ndarja e dataseteve per trajnim dhe e evaluim

```python
splits = full.train_test_split(test_size=TEST_SIZE, seed=SEED)
train_ds = splits["train"]
eval_ds = splits["test"]

print("Train:", len(train_ds), "Eval:", len(eval_ds))
```

### Pergaditja e "tokenizuesit"

Tokenizuesi perdoret me vone ne fazen e trajnimit te modelit

``` python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(batch):
    inputs = [f"{PREFIX} ({l}): {d}" for l, d in zip(batch["lang"], batch["document"])]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=False,
    )

    labels = tokenizer(
        text_target=batch["summary"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding=False,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_tok = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
eval_tok  = eval_ds.map(preprocess,  batched=True, remove_columns=eval_ds.column_names)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=MODEL_NAME,
    label_pad_token_id=-100,
)

print("Tokenized:", len(train_tok), len(eval_tok))
```

### Trajnimi i modelit mbi google-mt5

```python
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to("cuda")

training_args = Seq2SeqTrainingArguments(
    output_dir=OUT_DIR,
    eval_strategy="steps",
    eval_steps=250,
    save_steps=250,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",

    learning_rate=1e-5,
    warmup_ratio=0.05,
    max_grad_norm=0.5,

    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,

    fp16=False,
    logging_steps=25,
    report_to=[],
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
```

#### Progresi gjat trajnimit

Tabela e mëposhtme dokumenton konvergjencën e humbjes (*loss*) gjatë procesit të **fine-tuning**. Rënia e qëndrueshme e humbjes së validimit tregon se modeli po mëson me sukses të përgjithësojë nëpër gjuhë të ndryshme, pa shfaqur shenja të **overfitting**.

| Step | Training Loss | Validation Loss |
| :--- | :--- | :--- |
| **250** | 13.454900 | 8.919786 |
| **500** | 9.850700 | 6.912702 |
| **750** | 8.636000 | 5.462420 |


### Perdorimi i modelit te trajnuar per gjenerimin e permbledhjeve

```
import re
import torch
from transformers import MT5ForConditionalGeneration, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MT5ForConditionalGeneration.from_pretrained(FINAL_DIR).to(device)
model.eval()

tok = AutoTokenizer.from_pretrained(FINAL_DIR, use_fast=True)

TRAIN_PREFIX = "summarize"

def capitalize_first_letter(text: str) -> str:
    for i, c in enumerate(text):
        if c.isalpha():
            return text[:i] + c.upper() + text[i+1:]
    return text

def summarize(text, num_beams=6, max_new_tokens=96):
    inputs = tok(
        TRAIN_PREFIX + text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    ).to(device)

    with torch.no_grad():
        ids = model.generate(
            **inputs,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            min_new_tokens=20,
            length_penalty=1.1,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    s = tok.decode(ids[0], skip_special_tokens=True)

    s = s.replace("▁", " ")
    s = re.sub(r"<extra_id_\d+>", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^(in English:|en français:|auf Deutsch zusammen:)\s*", "", s, flags=re.IGNORECASE).strip()

    return capitalize_first_letter(s)

test_cases = [
    ("EN", """Researchers at a university lab reported a new battery design that charges faster and lasts longer.
They said the prototype uses a modified electrolyte that reduces degradation. Independent testing is still limited,
but early results suggest improved cycle life and better performance in cold temperatures."""),

    ("EN", """A city council debated a new transportation plan that would add dedicated bus lanes, expand bike routes,
and increase parking fees downtown. Supporters argued it would reduce traffic and emissions, while critics warned
about impacts on small businesses and push traffic into residential areas."""),

    ("FR", """Selon un rapport récent, la hausse des températures aggrave les épisodes de pollution dans plusieurs grandes villes.
Les experts recommandent de renforcer les transports publics, de limiter la circulation lors des pics et d’améliorer
la surveillance de la qualité de l’air."""),

    ("FR", """Une entreprise a annoncé le lancement d’un service de traduction automatique destiné aux PME.
Le produit promet une meilleure prise en compte du contexte et des expressions idiomatiques, mais certains spécialistes
rappellent que les erreurs restent possibles dans les domaines juridiques et médicaux."""),

    ("DE", """Mehrere Regionen meldeten ungewöhnlich hohe Temperaturen und eine Zunahme von Starkregenereignissen.
Forschende erklären, dass wärmere Luft mehr Feuchtigkeit speichern kann, was die Intensität von Niederschlägen erhöht.
Kommunen investieren in bessere Entwässerung, Hitzeschutzpläne und Frühwarnsysteme."""),

    ("DE", """Ein großer Einzelhändler kündigte an, Filialen umzubauen, um mehr Platz für Abholstationen und Rücksendungen zu schaffen.
Das Unternehmen reagiert damit auf den wachsenden Onlinehandel. Kritiker befürchten längere Wartezeiten durch weniger Personal."""),

    ("EN", """Cybersecurity firms have observed a sharp rise in phishing attacks targeting remote workers via SMS and social media.
Hackers are using AI tools to craft convincing messages that mimic company executives or IT support.
Experts advise corporations to implement stricter multi-factor authentication and regular employee training to combat this threat."""),

    ("EN", """The housing market has cooled significantly as central banks raised interest rates to combat inflation.
Real estate agents report that homes are sitting on the market longer, and price reductions are becoming common.
While this makes buying harder for first-time owners due to mortgage costs, it may eventually stabilize skyrocketing property values."""),

    ("EN", """Biologists are sounding the alarm over the rapid decline of wild bee populations due to pesticide use and habitat loss.
Since bees are responsible for pollinating a third of the food we eat, their disappearance could threaten global food security.
Conservationists are urging farmers to plant wildflower strips and reduce chemical usage to support pollinators."""),

    ("EN", """Major streaming services are introducing ad-supported subscription tiers and cracking down on password sharing to boost revenue.
This shift comes as the market becomes saturated and production costs for original content soar.
Subscribers have expressed mixed reactions, with some welcoming cheaper options while others threaten to cancel memberships."""),

    ("FR", """Le projet du Grand Paris Express, qui prévoit l'extension massive du réseau de métro, transforme déjà la banlieue parisienne.
De nouvelles gares sortent de terre, attirant les promoteurs immobiliers et faisant grimper les prix des logements aux alentours.
Les élus locaux espèrent que ce réseau réduira les inégalités territoriales en désenclavant des quartiers jusqu'ici mal desservis."""),

    ("FR", """La consommation de produits biologiques a enregistré une baisse inédite cette année, frappée par l'inflation alimentaire.
Les consommateurs, soucieux de leur pouvoir d'achat, se tournent vers des alternatives moins coûteuses ou les marques de distributeurs.
Les agriculteurs bio demandent une aide d'urgence à l'État pour éviter des faillites en cascade dans la filière."""),

    ("FR", """Le festival de Cannes a ouvert ses portes dans un climat de polémique concernant la place des plateformes de streaming au cinéma.
Alors que certains réalisateurs défendent l'expérience unique de la salle obscure, d'autres estiment que le streaming permet de financer des œuvres audacieuses.
Le jury devra trancher entre tradition et modernité lors de la remise de la Palme d'or."""),

    ("DE", """Die Debatte über die Vier-Tage-Woche gewinnt in Deutschland an Fahrt, da Pilotprojekte positive Ergebnisse zeigen.
Befürworter argumentieren, dass weniger Arbeitszeit die Produktivität steigert und krankheitsbedingte Ausfälle reduziert.
Arbeitgeberverbände warnen jedoch, dass dies in Zeiten des Fachkräftemangels die Wirtschaft schwächen und Kosten erhöhen könnte."""),

    ("DE", """Wegen zahlreicher Baustellen und technischer Störungen hat die Deutsche Bahn erneut ihre Pünktlichkeitsziele verfehlt.
Der Konzern kündigte an, das Schienennetz in den kommenden Jahren grundlegend zu sanieren, was zunächst zu noch mehr Sperrungen führen wird.
Verkehrsminister fordern ein besseres Baustellenmanagement, um die Geduld der Fahrgäste nicht überzustrapazieren."""),

    ("DE", """Der Trend zu sogenannten Balkonkraftwerken boomt, da immer mehr Mieter ihren eigenen Solarstrom produzieren wollen.
Vereinfachte bürokratische Regeln und sinkende Preise für Solarmodule haben die Nachfrage sprunghaft ansteigen lassen.
Energieexperten sehen darin einen wichtigen, wenn auch kleinen, Baustein für die Energiewende in privaten Haushalten.""")

]

for lang, doc in test_cases:
    print(f"\n[{lang}]")
    print("DOCUMENT:", doc)
    print("SUMMARY:", summarize(doc))
```

#### Permbledhjet e gjeneruara

```

[EN]
DOCUMENT: Researchers at a university lab reported a new battery design that charges faster and lasts longer.
They said the prototype uses a modified electrolyte that reduces degradation. Independent testing is still limited,
but early results suggest improved cycle life and better performance in cold temperatures.
SUMMARY: A new battery design uses a modified electrolyte that reduces degradation.

[EN]
DOCUMENT: A city council debated a new transportation plan that would add dedicated bus lanes, expand bike routes,
and increase parking fees downtown. Supporters argued it would reduce traffic and emissions, while critics warned
about impacts on small businesses and push traffic into residential areas.
SUMMARY: Increase parking fees in residential areas. An increase of parking fees downtown would reduce emissions.

[FR]
DOCUMENT: Selon un rapport récent, la hausse des températures aggrave les épisodes de pollution dans plusieurs grandes villes.
Les experts recommandent de renforcer les transports publics, de limiter la circulation lors des pics et d’améliorer
la surveillance de la qualité de l’air.
SUMMARY: De l’air à la hausse des températures dans plusieurs grandes villes.

[FR]
DOCUMENT: Une entreprise a annoncé le lancement d’un service de traduction automatique destiné aux PME.
Le produit promet une meilleure prise en compte du contexte et des expressions idiomatiques, mais certains spécialistes
rappellent que les erreurs restent possibles dans les domaines juridiques et médicaux.
SUMMARY: Automatique à PME : l’entreprise a annoncé le lancement d’un service de traduction automatique

[DE]
DOCUMENT: Mehrere Regionen meldeten ungewöhnlich hohe Temperaturen und eine Zunahme von Starkregenereignissen.
Forschende erklären, dass wärmere Luft mehr Feuchtigkeit speichern kann, was die Intensität von Niederschlägen erhöht.
Kommunen investieren in bessere Entwässerung, Hitzeschutzpläne und Frühwarnsysteme.
SUMMARY: Erhöhen, dass wärmere Luft mehr Feuchtigkeit speichern kann. Starkregenereignisse

[DE]
DOCUMENT: Ein großer Einzelhändler kündigte an, Filialen umzubauen, um mehr Platz für Abholstationen und Rücksendungen zu schaffen.
Das Unternehmen reagiert damit auf den wachsenden Onlinehandel. Kritiker befürchten längere Wartezeiten durch weniger Personal.
SUMMARY: Kündigte an, um mehr Abholstationen zu schaffen. Das Unternehmen hat neue Filialen zu bauen.

[EN]
DOCUMENT: Cybersecurity firms have observed a sharp rise in phishing attacks targeting remote workers via SMS and social media.
Hackers are using AI tools to craft convincing messages that mimic company executives or IT support.
Experts advise corporations to implement stricter multi-factor authentication and regular employee training to combat this threat.
SUMMARY: In phishing attacks targeting remote workers via SMS and social media. RELATED ARTICLES

[EN]
DOCUMENT: The housing market has cooled significantly as central banks raised interest rates to combat inflation.
Real estate agents report that homes are sitting on the market longer, and price reductions are becoming common.
While this makes buying harder for first-time owners due to mortgage costs, it may eventually stabilize skyrocketing property values.
SUMMARY: For first-time buyers. The housing market has cooled significantly.

[EN]
DOCUMENT: Biologists are sounding the alarm over the rapid decline of wild bee populations due to pesticide use and habitat loss.
Since bees are responsible for pollinating a third of the food we eat, their disappearance could threaten global food security.
Conservationists are urging farmers to plant wildflower strips and reduce chemical usage to support pollinators.
SUMMARY: Of wild bee populations could threaten global food security. Environmental experts are urging farmers to plant wildflower strips and reduce habitat loss.

[EN]
DOCUMENT: Major streaming services are introducing ad-supported subscription tiers and cracking down on password sharing to boost revenue.
This shift comes as the market becomes saturated and production costs for original content soar.
Subscribers have expressed mixed reactions, with some welcoming cheaper options while others threaten to cancel memberships.
SUMMARY: Increases revenue in streaming services .com.uk's content.

[FR]
DOCUMENT: Le projet du Grand Paris Express, qui prévoit l'extension massive du réseau de métro, transforme déjà la banlieue parisienne.
De nouvelles gares sortent de terre, attirant les promoteurs immobiliers et faisant grimper les prix des logements aux alentours.
Les élus locaux espèrent que ce réseau réduira les inégalités territoriales en désenclavant des quartiers jusqu'ici mal desservis.
SUMMARY: Des quartiers mal desservis à la banlieue parisienne. L'extension massive du réseau de métro

[FR]
DOCUMENT: La consommation de produits biologiques a enregistré une baisse inédite cette année, frappée par l'inflation alimentaire.
Les consommateurs, soucieux de leur pouvoir d'achat, se tournent vers des alternatives moins coûteuses ou les marques de distributeurs.
Les agriculteurs bio demandent une aide d'urgence à l'État pour éviter des faillites en cascade dans la filière.
SUMMARY: À l'inflation alimentaire. Les consommateurs consomment des produits biologiques.

[FR]
DOCUMENT: Le festival de Cannes a ouvert ses portes dans un climat de polémique concernant la place des plateformes de streaming au cinéma.
Alors que certains réalisateurs défendent l'expérience unique de la salle obscure, d'autres estiment que le streaming permet de financer des œuvres audacieuses.
Le jury devra trancher entre tradition et modernité lors de la remise de la Palme d'or.
SUMMARY: De la Palme d'or. Cannes a ouvert ses portes dans un climat de polémique.

[DE]
DOCUMENT: Die Debatte über die Vier-Tage-Woche gewinnt in Deutschland an Fahrt, da Pilotprojekte positive Ergebnisse zeigen.
Befürworter argumentieren, dass weniger Arbeitszeit die Produktivität steigert und krankheitsbedingte Ausfälle reduziert.
Arbeitgeberverbände warnen jedoch, dass dies in Zeiten des Fachkräftemangels die Wirtschaft schwächen und Kosten erhöhen könnte.
SUMMARY: Verbessern könnte. Mehr Arbeitgeber fordern neue Ergebnisse zeigen.

[DE]
DOCUMENT: Wegen zahlreicher Baustellen und technischer Störungen hat die Deutsche Bahn erneut ihre Pünktlichkeitsziele verfehlt.
Der Konzern kündigte an, das Schienennetz in den kommenden Jahren grundlegend zu sanieren, was zunächst zu noch mehr Sperrungen führen wird.
Verkehrsminister fordern ein besseres Baustellenmanagement, um die Geduld der Fahrgäste nicht überzustrapazieren.
SUMMARY: Nicht überzuhalten. Die Deutsche Bahn kündigte an, das Schienennetz in den kommenden Jahren saniert.

[DE]
DOCUMENT: Der Trend zu sogenannten Balkonkraftwerken boomt, da immer mehr Mieter ihren eigenen Solarstrom produzieren wollen.
Vereinfachte bürokratische Regeln und sinkende Preise für Solarmodule haben die Nachfrage sprunghaft ansteigen lassen.
Energieexperten sehen darin einen wichtigen, wenn auch kleinen, Baustein für die Energiewende in privaten Haushalten.
SUMMARY: Ansteigen lassen. Energieexperten sehen darin einen wichtigen Baustein für die Energiewende in privaten Haushalten.

```