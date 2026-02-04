#  Persian News Analysis: Deep Learning vs. Classical ML
> **A Comprehensive Study on Classification and Clustering of Persian Corpora using ParsBERT and SVM.**

---

##  Project Overview
This project explores the efficiency of **Transformer-based models (ParsBERT)** compared to **Classical Machine Learning (SVM)** for processing Persian news. The pipeline includes advanced NLP preprocessing, supervised classification, and unsupervised clustering with dimensionality reduction.

---

##  Tech Stack
* **Language:** Python 3.x
* **NLP Tools:** `Hazm` (Normalization & Lemmatization)
* **Transformers:** `HuggingFace` (ParsBERT)
* **ML Framework:** `Scikit-learn` (SVM, Logistic Regression, K-Means)
* **Visualization:** `Seaborn`, `Matplotlib`, `PCA`

---

##  1. Classification Performance Evaluation
We compared a **Baseline (TF-IDF + SVM)** with a **SOTA (ParsBERT + LogReg)** approach.

### Comparative Results
| Model | Feature Extraction | Accuracy | F1-Score (Weighted) |
| :--- | :--- | :--- | :--- |
| **Baseline** | TF-IDF (Sparse Matrix) | 69.28% | 68.43% |
| **ParsBERT** | **[CLS] Token Embedding** | **79.28%** | **79.19%** |

### Key Insight
The **10% performance boost** in ParsBERT is attributed to its ability to capture **contextual semantics**. Unlike TF-IDF, which treats words as isolated tokens, ParsBERT understands polysemy (words with multiple meanings) in Persian, leading to a much more robust decision boundary.

---

## ⚠️ 2. Error Analysis (Confusion Matrix Insights)
Based on the generated Confusion Matrix, I identified two primary error patterns:

1.  **Political-Economic Overlap:** News regarding "Sanctions" or "Budget" often contain high-frequency keywords from both domains, causing the model to oscillate between these two classes.
2.  **The "Universal Keywords" Issue:** Common Persian terms like *'دولت'* (Government) or *'کشور'* (Country) are present in almost 80% of the dataset, creating noise in the latent space for less distinct categories.



---

##  3. Unsupervised Clustering & Discovery
We applied **K-Means** on the 768-dimensional BERT embeddings to discover hidden patterns without using labels.

### Latent Space Visualization (PCA)
To interpret the high-dimensional data, I used **Principal Component Analysis (PCA)** to project embeddings into 2D. 
* **Result:** Distinct "islands" were formed for specialized topics like *Sports* and *Crimes*, while *Social* news showed more dispersion.



### Cluster Profiling (Top Keywords)
By calculating the **Mean TF-IDF** for each cluster, we identified their core themes:
* **Sports (Cluster 15/35):** `فوتبال` | `تیم` | `لیگ` | `پرسپولیس`
* **Legal/Crime (Cluster 37):** `قتل` | `پلیس` | `متهم` | `جسد`
* **International (Cluster 42):** `روسیه` | `اوکراین` | `آمریکا` | `وزیر`

---

## 4. Matching Evaluation
**Does the clustering match reality?**
* **High Alignment:** For technical/domain-specific news, the K-Means clusters had over **85% overlap** with the original labels.
* **Semantic Merging:** The model naturally grouped "Culture" and "Society" together. This isn't necessarily a "mistake" but shows that in an unsupervised setting, these topics share a unified semantic space in Persian journalism.

---
