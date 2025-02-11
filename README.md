# Neural-Information-Retrieval
1. Overview
Neural Information Retrieval (NIR) is a modern approach that integrates neural network techniques into traditional Information Retrieval (IR) systems to enhance search efficiency and relevance. The project explores:

Word Embeddings
Passage Ranking
Personalized Search using Query Expansion
Text Classification
2. Key Topics and Contributions
The project is based on four major components:

1. Word Embedding
Word embeddings map words into dense vector spaces, capturing semantic relationships.
Different models used:
Count-based embeddings (GloVe, Hellinger PCA)
Predictive embeddings (Word2Vec, FastText)
Character-based embeddings (Subword Skipgram, Char2Vec)
Knowledge-enhanced embeddings (ERNIE, Knowledge Graph-based embeddings)
Applications in Information Retrieval:
Improving query expansion
Document representation
Semantic search
2. Passage Ranking
Instead of ranking entire documents, this method ranks specific passages within documents for relevance.
Models used:
Sparse Models (BM25, TF-IDF)
Dense Models (Transformer-based architectures like BERT)
Hybrid Models (RocketQA, which combines sparse and dense models)
A two-stage ranking system is implemented:
First-Stage Ranker: Retrieves candidate passages.
Second-Stage Ranker: Uses deep learning models (like Cross-Encoders) to refine rankings.
3. Personalized Search using Query Expansion
Enhances search results by tailoring them to the user's preferences.
Uses Kullback-Leibler Divergence (KLD) to expand queries with relevant terms.
Implemented using PyTerrier framework.
Personalized search improves user satisfaction by considering past queries and context.
4. Text Classification
Categorizes textual data automatically to streamline retrieval.
Techniques include:
Machine Learning Models: Decision Trees, Gradient Boosting Machines, Random Forests.
Deep Learning Models: CNNs, RNNs, BERT, GPT.
Hybrid Approaches: Combining deep learning with rule-based systems.
Applications:
Spam filtering
Sentiment analysis
Topic categorization
3. Methodologies
Datasets: Various public datasets are used for training and testing models.
Pretrained Models: RoBERTa, DistilBERT, and others are fine-tuned for specific tasks.
Evaluation Metrics:
For Ranking: Mean Reciprocal Rank (MRR), Normalized Discounted Cumulative Gain (NDCG).
For Classification: Accuracy, F1-score.
4. Experiments & Results
Models were trained and tested in different environments.
Performance comparison between traditional and neural-based IR models.
Hybrid models provided the best balance between efficiency and accuracy.
5. Conclusion
NIR significantly improves search effectiveness.
Query expansion enhances personalized search experiences.
Combining different ranking and classification models leads to more accurate and relevant results.
