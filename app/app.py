from flask import Flask, render_template, request
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import re

app = Flask(__name__)
model_name = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# ---------------- Load and prepare data ----------------
df = pd.read_csv("data/coffee_with_embeddings.csv")

# Ensure "السعر" "price" column is numeric
if df["السعر"].dtype != float and df["السعر"].dtype != int:
    df["السعر"] = pd.to_numeric(df["السعر"], errors="coerce")

# Fill NaN values in text columns
for col in ["الإيحاءات", "اسم المنتج", "اسم المحمصة", "الدولة"]:
    if col in df.columns:
        df[col] = df[col].fillna("")

# ---------------- Arabic text normalization ----------------
ALEF_VARIANTS = "[إأآا]"
TA_MARBUTA = "ة"
HA = "ه"
YEH_VARIANTS = "[يى]"
TATWEEL = "ـ"
DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652]")

def normalize_ar(text: str) -> str:
    """Normalize Arabic text by removing diacritics and unifying characters."""
    text = str(text)
    text = DIACRITICS.sub("", text)
    text = re.sub(TATWEEL, "", text)
    text = re.sub(ALEF_VARIANTS, "ا", text)
    text = re.sub(YEH_VARIANTS, "ي", text)
    text = text.replace(TA_MARBUTA, HA)
    text = text.replace(",", "،")
    return text.strip()

# ---------------- Synonyms expansion ----------------
SYNONYMS = {
    "شوكولاته": ["كاكاو", "بني", "شوكولاتة", "دارك", "حليب"],
    "فواكه": ["فاكهية", "توت", "توتي", "خوخ", "مشمش", "تفاح", "كرز", "مانجو", "أناناس", "فراولة", "رمان"],
    "كراميل": ["توفي", "سكر محروق", "كراميل مملح"],
    "زهري": ["أزهار", "فل", "ياسمين", "ورد", "لافندر", "زهر برتقال"],
    "عسلي": ["عسل", "كهرماني", "شراب", "ميلك"],
    "حمضي": ["حموضة", "ليمون", "حمضيات", "حمضيّة", "برتقال", "جريب فروت", "ليمون حامض"],
    "بهارات": ["قرفة", "قرنفل", "هيل", "زنجبيل", "جوز الطيب", "كزبرة", "شمر"],
    "مكسرات": ["بندق", "لوز", "فستق", "جوز", "كاجو", "بندق محمص"],
    "سكر": ["حلاوة", "سكرية", "مسكر", "سكر بني", "سكر أبيض"],
    "مدخن": ["دخان", "خشب", "مدخنة", "محروق"],
    "ترابي": ["تربة", "أرضي", "أرضية", "طين"],
    "شاي": ["شاي أسود", "شاي أخضر", "أعشاب", "شاي"],
}

def expand_with_synonyms(text: str) -> str:
    """Expand input text with synonyms to improve matching."""
    t = normalize_ar(text)
    tokens = set(re.split(r"\s|،", t))
    added = []
    for base, alts in SYNONYMS.items():
        if base in tokens or any(a in tokens for a in alts):
            added.extend(alts + [base])
    if added:
        t += " " + " ".join(set(added))
    return t

# ---------------- Embeddings functions ----------------
def get_embedding(text):
    """Get BERT embedding for the given text."""
    text = normalize_ar(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    attention_mask = inputs['attention_mask']
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    return (sum_embeddings / sum_mask).squeeze().numpy()

def load_or_build_embeddings(col_name: str, path: str):
    """Load precomputed embeddings or build them if not found."""
    if os.path.exists(path):
        return np.load(path)
    texts = [normalize_ar(x) for x in df[col_name].astype(str).tolist()]
    embs = [get_embedding(t) for t in texts]
    embs = np.vstack(embs)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, embs)
    return embs

# Load or build embeddings
flavor_emb_path = "model/flavor_embeddings.npy"
name_emb_path = "model/name_embeddings.npy"
flavor_embeddings = load_or_build_embeddings("الإيحاءات", flavor_emb_path)
name_embeddings = load_or_build_embeddings("اسم المنتج", name_emb_path)

# ---------------- Scoring helper ----------------
def keyword_boost(user_text: str, product_text: str) -> float:
    """Boost score if exact keywords from user exist in product text."""
    utoks = set([w for w in re.split(r"\s|،", normalize_ar(user_text)) if w])
    ptoks = set([w for w in re.split(r"\s|،", normalize_ar(product_text)) if w])
    if not utoks or not ptoks:
        return 0.0
    overlap = len(utoks & ptoks)
    return min(0.20, overlap * 0.04)  # +0.04 per match up to 0.20

# ---------------- Main route ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []

    if request.method == "POST":
        user_input = request.form.get("preferences", "").strip()
        max_price = request.form.get("max_price", "").strip()

        # Parse max price or set to highest available price in data
        try:
            data_max = float(df["السعر"].max() or 0)
            default_max = data_max if not np.isnan(data_max) else 999999.0
            max_price = float(max_price) if max_price else default_max
        except:
            max_price = 999999.0

        # Filter by max price
        filtered = df[df["السعر"] <= max_price].copy()
        if filtered.empty:
            return render_template("index.html", recommendations=[])

        if user_input:
            # Expand query with synonyms
            user_q = expand_with_synonyms(user_input)
            user_emb = get_embedding(user_q)

            # Match filtered products to embeddings
            idx = filtered.index.values
            flav_emb = flavor_embeddings[idx]
            name_emb = name_embeddings[idx]

            # Similarities
            sim_flavor = cosine_similarity([user_emb], flav_emb)[0]
            sim_name = cosine_similarity([user_emb], name_emb)[0]
            boosts = np.array([keyword_boost(user_q, row["الإيحاءات"]) for _, row in filtered.iterrows()])

            # Weighted score
            score = 0.70 * sim_flavor + 0.20 * sim_name + 0.10 * boosts
            top_indices_local = score.argsort()[::-1][:10]
            filtered = filtered.iloc[top_indices_local]
        else:
            # If no preferences, return cheapest 10 within max price
            filtered = filtered.sort_values("السعر", ascending=True).head(10)

        # Final recommendations
        recommendations = filtered[["اسم المنتج", "اسم المحمصة", "الإيحاءات", "الدولة", "السعر"]].to_dict(orient="records")

    return render_template("index.html", recommendations=recommendations)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
