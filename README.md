# Applied-Statistics-Project

## Setup

1. Crea un virtual environment:
   ```bash
   python3 -m venv .venv
   ```

2. Attiva il virtual environment:
   - Su Linux/macOS:
     ```bash
     source .venv/bin/activate
     ```
   - Su Windows:
     ```cmd
     .venv\Scripts\activate
     ```

3. Installa le dipendenze:
   ```bash
   pip install -r requirements.txt
   ```

## Struttura del progetto

```
Applied-Statistics-Project/
├── src/
│   └── main.py            # Entry point della pipeline
├── dataset/
│   ├── preprocess.py      # Fase A: Data Engineering & Wrangling
│   └── *.parquet / *.csv  # Dataset generati (non tracciati da git)
├── requirements.txt
└── README.md
```

## Esecuzione

```bash
python src/main.py
```