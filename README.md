# CERBERUS: Constraing Entity coRruption By Enforcing Restrictions Using Schema


## Avvio del sistema

* **Clona la repository:**
  ```bash
  git clone [https://github.com/ara-t3/tesi-campanozzi.git](https://github.com/ara-t3/tesi-campanozzi.git)
  ```

* **Posizionati nella directory:**
  ```bash
  cd tesi-campanozzi
  ```

* **Crea l'ambiente virtuale e attivalo:**
  ```bash
  python -m venv KGVENV
  ```
  *(Nota: ricordati di eseguire il comando di attivazione specifico per il tuo sistema operativo, ad esempio `source KGVENV/bin/activate` su Linux/macOS oppure `KGVENV\Scripts\activate` su Windows).*

* **Installa le dipendenze:**
  ```bash
  pip install -r requirements.txt
  ```

---

## Utilizzo

* **Unzippa i dati:**
  ```bash
  unzip data.zip
  ```

* **Applica la patch a PyKEEN:**
  ```bash
  python -m src.code.patch_pykeen
  ```

* **Avvia il tuning degli iperparametri:**
  ```bash
  python -m src.code.hyperparameter_optimization --g ARCO_20 --model transe --sampler schema
  ```

* **Avvia il training** *(da eseguire dopo aver completato il tuning)*:
  ```bash
  python -m src.code.train --g ARCO_20 --model transe --sampler schema
  ```

---

### Parametri

Di seguito l'elenco degli argomenti da poter passare agli script di tuning e training:

* `--g`: Nome del dataset di riferimento.
* `--model`: Modello KGE da usare.
  * *Attualmente supportati:* `transe`, `rotate`
* `--sampler`: Tipologia di negative sampler.
  * *Attualmente supportati:* `bernoulli`, `schema`, `schema_ablation`

> **N.B.** L'opzione `schema_ablation` corrisponde al sampler `schema` ma senza il sistema di scoring integrato.
