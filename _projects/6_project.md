---
layout: page
title: Movie Recommender based on plot similarity
description:  use  NLP (Natural Language Processing) and KMeans to predict the similarity between movies based on the plot from IMDB and Wikipedia
img:
importance: 4
category: fun
---

## Movie Recommender

Utilize tokenization, stemming and vectorize using TfidfVectorizer. Utilize KMeans to create clusters using vectorized data and cosine similarity to find movie plots that are similar to one another.

[Check out the product frontend!](https://movie-recommender-v0wd.onrender.com/)

### Data

[Source](https://www.kaggle.com/datasets/devendra45/movies-similarity)

### Tokenization and Stemming

**Tokenization** is the process of breaking down text into smaller units called tokens, typically words or phrases. This is crucial for NLP as it transforms unstructured text into a format that's easier to analyze.
- Word Tokenization: Splits the text into words. It's the most common form of tokenization.
- Sentence Tokenization: Divides text into sentences. Useful in tasks where sentence-level analysis is required, like sentiment analysis.
- Subword Tokenization: Breaks words into smaller units (like syllables or morphemes). This is helpful in languages where compound words are common.

**Stemming** involves reducing words to their base or root form. For instance, "running", "runs", "ran" are all reduced to the stem "run". This helps in standardizing words for better analysis.
- Porter Stemmer: A widely used, relatively gentle stemmer.
- Lancaster Stemmer: A more aggressive stemmer than Porter.
- Snowball Stemmer: An improvement over Porter, available for several languages.

### Vectorization Methods (TfidfVectorizer)

After tokenization and stemming, the next step is **vectorization**. This converts text data into numerical values which can be processed by machine learning algorithms. **TF-IDF (Term Frequency-Inverse Document Frequency)** is a popular method. It reflects the importance of a word in a document in a corpus.

- Count Vectorization: Represents documents using the frequency of each word. It's simple but can give undue weight to frequent words.
- TF-IDF (Term Frequency-Inverse Document Frequency): Reflects how important a word is to a document in a corpus. It reduces the impact of frequent words across documents.
- Word Embeddings (like Word2Vec, GloVe): These provide more sophisticated representations by capturing semantic meanings and relationships between words.

Absolutely, let's dive into the specifics of each text vectorization method and how they are calculated:

##### 1. Count Vectorization

Count Vectorization, represents text documents as vectors where each dimension corresponds to a unique word in the text corpus. The value in each dimension is the frequency of that word in the document.

**Calculation:**
- **Step 1:** Create a vocabulary of all unique words across all documents in the corpus.
- **Step 2:** For each document, count the number of times each word appears and place that count in the corresponding dimension of the vector.

For example, if your vocabulary is ["apple", "banana", "orange"], and your document is "apple banana apple," the count vector would be [2, 1, 0], indicating "apple" appears twice, "banana" once, and "orange" not at all.

##### 2. TF-IDF (Term Frequency-Inverse Document Frequency)

TF-IDF is a statistical measure used to evaluate the importance of a word in a document, which is part of a corpus. The intuition behind this is that words that appear frequently in a document but not across many documents are likely more significant (ex. the).

**Calculation:**
- **Term Frequency (TF):** The frequency of a word in a document.
  - $$ TF(word) = \frac{\text{Number of times word appears in a document}}{\text{Total number of words in the document}} $$

- **Inverse Document Frequency (IDF):** Measures the importance of the word across the corpus.
  - $$ IDF(word) = \log\left(\frac{\text{Total number of documents}}{\text{Number of documents containing the word}}\right) $$

- **TF-IDF:** The product of TF and IDF scores of a word.
  - $$ TF\text{-}IDF(word) = TF(word) \times IDF(word) $$

A high TF-IDF score for a word in a document indicates it's important and relatively unique to that document.

##### 3. Word Embeddings (Word2Vec, GloVe)

Word Embeddings are techniques that represent words in a continuous vector space where semantically similar words are mapped to nearby points. They capture more nuanced relationships between words.

**Calculation:**
- **Word2Vec:** Uses neural networks to learn word associations from a large corpus of text. There are two main architectures:
  - **CBOW (Continuous Bag of Words):** Predicts a word based on its context.
  - **Skip-gram:** Predicts the context given a word.

- **GloVe (Global Vectors for Word Representation):** Uses matrix factorization techniques on the co-occurrence matrix of words in the corpus. It essentially tries to minimize the difference between the dot product of the embeddings of two words and the logarithm of their co-occurrence probability.

These methods involve complex mathematical operations and neural network training, often requiring large datasets and significant computational resources. The resulting vectors typically have hundreds of dimensions and capture subtle semantic relationships, unlike the simpler count-based methods.

In summary, while Count Vectorization and TF-IDF provide a more basic form of text representation focusing on word frequencies, Word Embeddings offer a more nuanced and deeper understanding of word meanings and relationships.

### Clustering Using KMeans

Finally, you can use **KMeans clustering** to group similar movies based on their plot descriptions. KMeans is an unsupervised learning algorithm that partitions the data into K clusters.

<div class="col-sm mt-3 mt-md-0" >
    {% include figure.html path="assets/img/denogram.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
</div>

### Finding Similar Movies

To find movies similar to a given one, you can compute the cosine similarity between the TF-IDF vectors of the movies. Movies with higher cosine similarity scores are more similar in terms of plot.


### Conclusion

This approach leverages NLP and machine learning to group movies based on the similarity of their plots. W can further enhance this system by incorporating more features like genres, director, and cast. Remember, the quality of recommendations largely depends on the richness of your dataset and the fine-tuning of the model.