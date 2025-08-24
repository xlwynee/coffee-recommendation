# Coffee Recommender BERT

A coffee recommendation system using a BERT model to analyze Arabic data and recommend coffee products based on notes. The system combines pre-trained models with curated data to provide personalized coffee suggestions.

---

## Requirements

* Python 3.x
* Flask
* Python libraries:

  * transformers
  * pandas
  * scikit-learn
  * numpy
  * torch

> Install all required libraries with:

```bash
pip install -r requirements.txt
```

---

## How to Run

Open the terminal and navigate to the `app/` folder inside the project:

```bash
cd coffee_recommender_bert/app
```

Run the application:

```bash
python app.py
```

Open your browser at:

```
http://127.0.0.1:5000
```

Make sure that the `models/` and `data/` folders are at the same level as `app/` so the app can access them correctly.

---

## Deployment

Online application link: \[Add your deployment link here]
The app can be deployed easily on Heroku or Vercel. Update the link above after deploying.

---

## Data Collection

The dataset used in this project was collected using a combination of web scraping and manual data entry:

* **Web scraping:** Extracted coffee product information from multiple sources online.
* **Manual entry:** Curated additional details such as flavor notes and ratings to enrich the dataset.

This hybrid approach ensures a high-quality dataset for accurate recommendations.

---

## Project Structure

```
coffee_recommender_bert/
│
├─ app/       # Flask application code, includes app.py, templates/, and static/
├─ models/    # Embeddings (flavor, coffee, name)
├─ data/      # Raw data used for training and analysis (scraped + manually collected)
├─ requirements.txt
└─ README.md
```

---

## Author

Leena Alotaibi
GitHub: [https://github.com/xlwynee](https://github.com/xlwynee)
