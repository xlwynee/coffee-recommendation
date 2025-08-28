# Coffee Recommender BERT

A coffee recommendation system using a **BERT-based Arabic model** to analyze Arabic coffee product data and provide recommendations based on tasting notes, product name, and price

The system combines pre-trained models (CAMeL BERT) with manually enriched coffee datasets and embeddings to deliver **personalized coffee suggestions in Arabic**.

---

## Features
- Arabic text normalization and synonym expansion for better matching.
- Embeddings built using `torch` + `transformers` (CAMeL-Lab BERT).
- Recommendation scoring based on:
  - Flavor notes similarity (**70%**)
  - Product name similarity (**20%**)
  - Keyword overlap boosting (**10%**)
- Price filtering.
- Simple Flask interface to input preferences and show results.

---

## Requirements

- Python 3.x
- Flask
- Libraries:
  - torch
  - transformers
  - pandas
  - numpy
  - scikit-learn
  

Install all dependencies with:
```bash
pip install -r requirements.txt
```

---

## How to Run (Locally)

1. **Clone** the repository and navigate into it.
2. Make sure the `data/` and `model/` folders are at the same level as `app/`.
3. Open the terminal and run:
    ```bash
    cd app
    python3 app.py
    ```
4. Open your browser at:
    - http://127.0.0.1:5000

> The application will load the dataset, build/load embeddings, and show the interface.

---
---

## Demo

Here’s a quick preview of the interface:

![Coffee Recommender Demo](images/demo.png)

---

## Deployment Status

I attempted to deploy this app on **Render** (and also tested with **Heroku**).  
The build was successful  
But since the project uses **PyTorch + Transformers**, the memory usage exceeded the limits of the **free tiers**.

A paid plan with higher RAM is required to deploy this app online.

👉 For now, please run the app locally using the steps above.  


---

## Data Collection

The dataset was built using a combination of:
- **Web scraping:** coffee product information from multiple roasters.
- **Manual curation:** adding details like flavor notes, roaster names, and prices.

This hybrid dataset ensures better coverage and quality of recommendations.

---

## Project Structure

```
coffee_recommender_bert/
│
├─ app/       # Flask app code (app.py, templates/, static/)
├─ data/      # Collected dataset (scraped + manual)
├─ model/     # Embeddings (flavor, name)
├─ requirements.txt
└─ README.md
```

---

## Notes
- Ensure that the **Arabic text preprocessing** (normalization, synonyms) is enabled before building embeddings.
- If `model/` contains precomputed embeddings, they will be loaded; otherwise, the app will compute and save them on first run.
- CAMeL BERT is used for Arabic text representations. Please refer to the CAMeL-Lab license and model card for academic use and citation.

---

## Acknowledgements
- [CAMeL-Lab BERT](https://huggingface.co/CAMeL-Lab) for Arabic language modeling.
- Community roasters and public product pages that informed the dataset.

---

## Author
**Leena Alotaibi**  
GitHub: https://github.com/xlwynee
