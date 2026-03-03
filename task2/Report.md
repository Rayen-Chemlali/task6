# Customer Segmentation — Project Report
## Mall Customers Dataset | Task 2

---

## Overview

Unsupervised learning pipeline to segment mall customers into distinct behavioral groups based on annual income and spending score. The project covers full exploratory analysis, optimal cluster selection via elbow and silhouette methods, and a comparison of three clustering algorithms — K-Means, Agglomerative, and DBSCAN.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | Kaggle — `vjchoudhary7/customer-segmentation-tutorial-in-python` |
| Samples | 200 customers |
| Features | CustomerID, Gender, Age, Annual Income (k$), Spending Score (1–100) |
| Missing values | None |
| Class balance | 56% Female, 44% Male |

---

## Key Observations from EDA

- **Income distribution** is roughly normal, centered around 60k$
- **Spending score** is fairly uniform with a slight right skew toward 40–60
- **Age** ranges from 18 to 70, with most customers between 25–50
- **Correlation:** Age has a weak negative correlation with spending (-0.33) and almost no correlation with income (-0.01) — income and spending are essentially independent
- **Gender** has no meaningful impact on income, age, or spending score — clusters are gender-balanced across the board

---

## Methodology

### Feature Selection
Clustering was performed on **Income vs Spending Score** (2D). The 3D variant (Age + Income + Spending) was tested but scored lower on all metrics — Age adds noise without improving segment separation.

### Scaling
StandardScaler applied before clustering to normalize the feature space.

### Optimal K Selection
Both the **Elbow Method** and **Silhouette Score** were used:

| K | Silhouette Score |
|---|-----------------|
| 2 | 0.3268 |
| 3 | 0.4647 |
| 4 | 0.4939 |
| **5** | **0.5547** chosen |
| 6 | 0.5399 |
| 7 | 0.5296 |

The elbow curve shows a clear inflection at K=5. The silhouette peaks at K=5. The Ward dendrogram confirms 5 natural groups. All three methods agree — K=5 is the optimal choice.

---

## Models Compared

| Model | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|-------|-----------|----------------|-------------------|
| **KMeans K=5** | **0.5547** | **0.5722** | **248.6** |
| Agglomerative K=5 | 0.5538 | 0.5779 | 244.4 |
| KMeans K=6 | 0.5399 | 0.6546 | 243.1 |
| KMeans K=4 | 0.4939 | 0.7096 | 174.6 |
| Agglomerative K=4 | 0.4926 | 0.6787 | 169.7 |
| DBSCAN (eps=0.5) | — | — | 2 clusters only |

KMeans K=5 wins on all three metrics. Agglomerative K=5 is a close second but KMeans is more interpretable and efficient. DBSCAN was unable to identify the 5 natural segments due to the uniform density of the data.

---

## Results — 5 Customer Segments

| Cluster | Segment Name | Avg Age | Avg Income | Avg Spending | Count | Female % |
|---------|-------------|---------|------------|--------------|-------|----------|
| 0 | Average Customers | 42.7 | 55.3k$ | 49.5 | 81 | 59.3% |
| 1 | High Income High Spenders | 32.7 | 86.5k$ | 82.1 | 39 | 53.8% |
| 2 | Low Income High Spenders | 25.3 | 25.7k$ | 79.4 | 22 | 59.1% |
| 3 | High Income Low Spenders | 41.1 | 88.2k$ | 17.1 | 35 | 45.7% |
| 4 | Low Income Low Spenders | 45.2 | 26.3k$ | 20.9 | 23 | 60.9% |

---

## Business Interpretation

**Cluster 0 — Average Customers (n=81, largest group)**
The core customer base. Middle income, average spending. Stable but not highly engaged. Loyalty programs or moderate promotions could increase their spending score.

**Cluster 1 — High Income High Spenders (n=39)**
Prime target segment. Young, wealthy, and already spending heavily. Focus on premium products, exclusive offers, and VIP treatment to retain them.

**Cluster 2 — Low Income High Spenders (n=22)**
Interesting outlier. Young customers (avg 25) with low income but high spending — likely driven by lifestyle and trends. Discount bundles and BNPL options could further engage this group.

**Cluster 3 — High Income Low Spenders (n=35)**
High potential, currently underperforming. These customers have money but are not spending it at the mall. Personalized targeting, premium experiences, and exclusive events could unlock their spending.

**Cluster 4 — Low Income Low Spenders (n=23)**
Low priority for marketing spend. Older customers with limited purchasing power. Basic retention strategies are sufficient.

---

## Key Findings

1. **Income and Spending Score are fully independent** — high income does not mean high spending
2. **Age is a weak signal** — it explains some variance but not enough to improve clustering
3. **K=5 is the natural optimum** — confirmed by elbow, silhouette, and dendrogram
4. **DBSCAN is not suitable** for this dataset — uniform density prevents meaningful density-based separation
5. **Gender has no discriminating power** — all segments are roughly gender-balanced
6. **Cluster 1 is the highest-value segment** — young, rich, and willing to spend

---

## Notebook Structure

```
customer_segmentation.ipynb    # Full pipeline notebook (Colab-ready)
split_notebook.py              # Script to split into 5 sub-notebooks

notebooks/
├── 01_setup_eda.ipynb               # Data loading & full EDA
├── 02_optimal_k.ipynb               # Elbow, silhouette, dendrogram
├── 03_final_model.ipynb             # KMeans K=5 + scatter plots
├── 04_profiles_business.ipynb       # Cluster profiles, radar, avg spending
└── 05_comparison_summary.ipynb      # Model comparison, DBSCAN, summary
```

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
kagglehub
```

Install all:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy kagglehub
```
