# Neural Information Retrieval

Capstone project exploring modern neural approaches to Information Retrieval (IR) — from word embeddings to transformer-based ranking and personalized search.

Submitted as part of the B.Tech capstone at **SRM AP University**.

## Topics Covered

### 1. Word Embeddings
Dense vector representations that capture semantic word relationships:
- **Count-based:** GloVe, Hellinger PCA
- **Predictive:** Word2Vec (CBOW & Skip-gram), FastText
- **Character-level:** Subword Skip-gram, Char2Vec
- **Knowledge-enhanced:** ERNIE, Knowledge Graph embeddings

### 2. Passage Ranking
Ranking specific passages (not whole documents) for relevance:
- **Sparse models:** BM25, TF-IDF
- **Dense models:** BERT-based bi-encoders
- **Hybrid:** RocketQA (sparse + dense fusion)
- Two-stage pipeline: first-stage retrieval → second-stage cross-encoder re-ranking

### 3. Personalized Search via Query Expansion
- Kullback-Leibler Divergence (KLD) for query term expansion
- Implements user preference modeling using past query context
- Built on the [PyTerrier](https://github.com/terrier-org/pyterrier) framework

### 4. Text Classification for IR
Categorizing documents to improve retrieval:
- **ML models:** Decision Trees, Gradient Boosting, Random Forests
- **Deep learning:** CNNs, RNNs, BERT, GPT
- Applications: spam filtering, topic classification, intent detection

## Report

Full project report: [`BTech_SRMAP_Project_report_CAPSTONE_B1.pdf`](BTech_SRMAP_Project_report_CAPSTONE_B1.pdf)

## References

- [MS MARCO dataset](https://microsoft.github.io/msmarco/)
- [BEIR benchmark](https://github.com/beir-cellar/beir)
- [PyTerrier](https://github.com/terrier-org/pyterrier)
- Karpukhin et al., *Dense Passage Retrieval for Open-Domain QA* (2020)
- Qu et al., *RocketQA* (2021)
