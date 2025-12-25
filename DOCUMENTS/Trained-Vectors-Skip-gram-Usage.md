#### Skip-gram Trained Vectors Usage

There is an important technical distinction in the Skip-gram model of Word2Vec between the two sets of trained vectors: the input-to-hidden weights (often called $W1$ or the input embedding matrix) and the hidden-to-output weights (often called $W2$ or the output embedding matrix).

**The Core Insight of the Math: Semantic Similarity vs. Contextual Prediction**

1. <u>Finding Synonyms (Semantic Similarity)</u>: The W1 (input-to-hidden) trained vectors are the standard choice for synonym identification, analogies, and semantic similarity tasks. This is the most common and effective use case for Word2Vec embeddings.

- The math is straightforward: Cosine similarity is computed directly between rows of W1 (the input embeddings) $W1 \cdot W1^T$. Words that are interchangeable (synonyms) tend to appear in highly similar contexts, so their input vectors cluster together.

- Example: "**Pain**" ≈ "**Ache**" ≈ "**Soreness**".Using W1 vectors produces strong thesaurus-like results (e.g., a medical bot could suggest "Did you mean ache?" when the user says "pain").

2. <u>Contextual Prediction (Collocations and Functional Relations)</u>: During training, the model uses W1 (input embedding) dotted with W2 (output embeddings) to predict context words.

- The math: $W1 \cdot W2$ (more precisely, input vector dotted with columns of W2) computes prediction scores for likely context words given the center word. 

- This captures collocations, words that frequently co-occur rather than pure synonyms.

- Example: "Antacids" (input) → strongly predicts "relieved" (output context). These are functionally related (treatment → outcome), not synonyms. Similarly: "Abdominal" → "swelling".

- While **W2** (output vectors) can show some similarity structure, they are typically discarded or perform worse on semantic tasks. Some advanced techniques average or concatenate **W1** and **W2** but standard practice relies on **W1** for both semantic and collocational insights.


<U><b>Summary:</b></u>
- Word2Vec embeddings (especially Skip-gram) capture a blend of semantic (synonyms), syntactic, and collocational relations in the same vector space.

- The input embeddings (W1 rows) are almost universally used as the final word vectors for downstream applications.

