import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# Loading dataset
df = pd.read_csv("data/processed/hc3_human_ai.csv")

# MIssing rows in text or label columns can cause issues, so I drop them for this visualization. In a real app, you might want to handle this more gracefully.
df = df.dropna(subset=["text", "label"])

# PCA sampling for faster computation
df = df.sample(n=min(500, len(df)), random_state=42)

# Converting text to TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df["text"])

# Converting sparse matrix to dense matrix for PCA
X_dense = X.toarray()

# Reducing to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_dense)

# PCA coordinates to dataframe
df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]

# human.ai classes
human_df = df[df["label"] == "human"]
ai_df = df[df["label"] == "ai"]

# PLot specs
plt.figure(figsize=(10, 6))
plt.scatter(human_df["PC1"], human_df["PC2"], label="Human", alpha=0.7)
plt.scatter(ai_df["PC1"], ai_df["PC2"], label="AI", alpha=0.7)

plt.title("PCA of TF-IDF Vectors: Human vs AI Texts")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.tight_layout()


plt.savefig("pca_tfidf_plot.png", dpi=300)

plt.show()

print("PCA plot saved as pca_tfidf_plot.png")