from flask import Flask, render_template, request, url_for
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

app = Flask(__name__)

OUTPUT_DIR = "static/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def limpar_texto(txt):
    return txt.lower().strip()


def gerar_graficos(tfidf_matrix):
    outputs = {}
    dist = cosine_distances(tfidf_matrix)

    # Heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(dist, cmap="viridis")
    heatmap_path = f"{OUTPUT_DIR}/heatmap.png"
    plt.savefig(heatmap_path)
    plt.close()
    outputs["heatmap"] = "outputs/heatmap.png"

    # Dendrograma
    from scipy.cluster.hierarchy import dendrogram, linkage
    Z = linkage(dist, "ward")
    plt.figure(figsize=(6, 5))
    dendrogram(Z)
    dendro_path = f"{OUTPUT_DIR}/dendrograma.png"
    plt.savefig(dendro_path)
    plt.close()
    outputs["dendrograma"] = "outputs/dendrograma.png"

    return outputs


def gerar_insights(tfidf, feature_names):
    top_terms = []
    for row in tfidf:
        idx = row.argsort()[-5:][::-1]
        termos = [feature_names[i] for i in idx]
        top_terms.append(termos)

    insights = {
        "note": "Análise concluída automaticamente.",
        "top_terms_per_doc": top_terms
    }
    return insights


@app.route("/")
def index():
    return render_template(
        "index.html",
        example_literatura="Dom Casmurro é um romance clássico.\nCapitu tinha olhos de ressaca.",
        example_reviews="Produto ótimo!\nChegou atrasado."
    )


@app.route("/run", methods=["POST"])
def run_pipeline():
    literatura_raw = request.form["literatura_text"].split("\n")
    reviews_raw = request.form["reviews_text"].split("\n")

    literatura_clean = [limpar_texto(t) for t in literatura_raw if t.strip()]
    reviews_clean = [limpar_texto(t) for t in reviews_raw if t.strip()]

    documentos = literatura_clean + reviews_clean

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(documentos).toarray()
    feature_names = vectorizer.get_feature_names_out()

    outputs = gerar_graficos(tfidf)
    insights = gerar_insights(tfidf, feature_names)

    return render_template(
        "resultados.html",
        literatura_clean=literatura_clean,
        reviews_clean=reviews_clean,
        outputs=outputs,
        insights=insights
    )


if __name__ == "__main__":
    app.run(debug=True)
