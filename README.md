---

# Reddit Comment Classification

This project tackles the task of **Reddit comment classification**: predicting whether a comment violates community rules. We experiment with both embedding-based methods and direct fine-tuning of transformer models.

---

## 📊 Current Approaches & Results

1. **Gemma embedding + semantic search** → *0.63 AUC*

   * Used embeddings directly in a similarity-based retrieval/classifier setup.
   * Baseline; weak separation.

2. **Gemma embedding + Logistic Regression** → *0.811 AUC*

   * Encoded comments with Gemma embeddings (dim=768).
   * Trained a linear classifier on top.

3. **Gemma embedding + LightGBM** → *0.833 AUC*

   * Same embeddings, stacked with numerical/categorical features.
   * Tuned with Optuna for regularization and sampling.

4. **DistilBERT fine-tune** → *0.86 AUC*

   * Sequence classification head directly trained on comment text.
   * Best-performing so far, but compute-intensive.

---

## 🚀 Future Directions

* **Full BERT finetune** → Explore larger transformer variants (RoBERTa, DeBERTa, BERTweet).
* **Instruction LLM finetune + logit processing** → Treat classification as constrained instruction following.
* **Similarity search with embeddings** → Explore semantic retrieval + voting/ranking for classification.
* **Ensembling** → Weighted averages or stacking across embedding-based and transformer models.

---

## ⚙️ Tech Stack

* **Models**: Gemma, DistilBERT, LightGBM, Logistic Regression
* **Libraries**: HuggingFace Transformers, SentenceTransformers, LightGBM, Optuna, scikit-learn
* **Metrics**: AUC (ROC-AUC)

---

## 📌 Notes

* Embedding-based methods are fast and scalable, but underperform compared to direct fine-tuning.
* LightGBM with embeddings shows strong results when regularized properly.
* Transformer fine-tuning gives the best raw performance but needs careful hyperparameter control to avoid overfitting.

---
