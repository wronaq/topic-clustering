# topic-clustering

Sentence transformer + UMAP + HDBSCAN

## Dataset

Dataset consists of joined train and test set from [Klej-PSC](https://klejbenchmark.com/static/data/klej_psc.zip)

## Repository tree

```zsh
├── data
│   ├── preprocessed
│   │   └── data.txt
│   ├── raw
│   │   ├── test.tsv
│   │   └── train.tsv
│   ├── polish_stopwords.txt
│   └── preprocess.sh
├── run
│   └── experiment.py
├── tasks
│   ├── cluster_data.sh
│   └── config.json
├── topic_clustering
│   ├── data_loader.py
│   ├── __init__.py
│   └── topics.py
├── .gitignore
├── LICENSE
├── Pipfile
├── Pipfile.lock
└── README.md
```

## Setup

```zsh
pipenv sync
```

Add flag -d to also sync development packages.

## Training

* Load and preprocess data
* Create/load sentence embeddings
  * Save it to `output` directory
* Reduce hyperspace dimensions with UMAP
* Cluster data using HDBSCAN
* Print topics keywords
  * Save clustered topics object to `output` directory

```zsh
chmod +x tasks/cluster_data.sh
tasks/cluster_data.sh \
    _PATH_TO_JSON_FILE_WITH_EXPERIMENT_CONFIG_ \
    _NUMBER_OF_TOPICS_TO_FIND_ \
    _NUMBER_OF_WORDS_TO_DESCRIBE_EACH_TOPIC_
```

## Example output

```zsh
INFO:root:Sentence embeddings loaded from file
INFO:root:Dimensionality reduction
INFO:root:Clustering
INFO:root:Calculate class-based TF-IDF
Topics found: 19. Compressing...
Final number of topics: 10
--------------------------------------------------
Topic 0 (78 documents):
        * środki
        * komórek
        * badania
        * przeciwbólowe
        * choroby
        * mózgu
        * alkoholu
--------------------------------------------------
Topic 1 (65 documents):
        * boksu
        * gołota
        * boks
        * tyson
        * zawodowego
        * wimbledon
        * pojedynek
--------------------------------------------------
Topic 2 (113 documents):
        * dzieci
        * szkoły
        * nauczycieli
        * pomocy
        * men
        * szkół
        * szkołach
--------------------------------------------------
Topic 3 (112 documents):
        * żydów
        * jedwabnego
        * niemieckich
        * robotników
        * grossa
        * polnische
        * holokaust
--------------------------------------------------
Topic 4 (89 documents):
        * putin
        * czeczenii
        * putina
        * rosji
        * rosja
        * rosjan
        * władimir
--------------------------------------------------
Topic 5 (79 documents):
        * emerytalnych
        * ppe
        * emerytury
        * systemu
        * zabezpieczenia
        * społecznego
        * emerytalne
--------------------------------------------------
Topic 6 (104 documents):
        * pracy
        * firmy
        * zatrudnienia
        * unido
        * pkp
        * firm
        * drobnej
--------------------------------------------------
Topic 7 (117 documents):
        * unii
        * ue
        * polski
        * europejskiej
        * polska
        * nato
        * unia
--------------------------------------------------
Topic 8 (169 documents):
        * sld
        * andrzej
        * stadion
        * stadionu
        * biernacki
        * polleny
        * polski
--------------------------------------------------
Topic 9 (146 documents):
        * aws
        * partii
        * władzy
        * partia
        * wyborców
        * wyborach
        * sld
--------------------------------------------------
```
