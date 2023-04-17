# Semantic Song Search Embeddings
- We allow the user to search for songs based on the inherent meaning in the song's lyrics. This approach goes beyond basic keyword matching to extract semantics independent of specific wording, connecting songs by topics described by the user in arbitrary detail.
- We apply a pre-trained sentence transformer model on an extensive lyrics dataset to create embeddings for each song. These vector representations of natural language are compared in order to find songs similar in meaning to a user-supplied query.


## Talk Outline
- Demo
- Technicals
- Comparisons
- Pro/cons

## Papers / Primary Resources
- [5 Million Song Dataset](https://www.kaggle.com/datasets/nikhilnayak123/5-million-song-lyrics-dataset)
  - Derived from Genius.com
- OpenAI Endpoint
- Sentence Transformers
- GloVe

## Play!
[HuggingFace Space: Semantic Song Search](https://huggingface.co/spaces/sheacon/semantic-song-search)

## Embedding Models

### Sentence Transformers
BERT

### GloVe
Pre-computed embeddings

### OpenAI GPT-3 Endpoint
?

## Architectures

## Question 1: 

## Question 2: 

(## Optional Question: )

## Results
- A few example queries

## Critical Analysis
https://medium.com/@nils_reimers/openai-gpt-3-text-embeddings-really-a-new-state-of-the-art-in-dense-text-embeddings-6571fe3ec9d9
- Embedding dimensions
- Compute on embeddings and downstream application
- Cost comparison

## Additional Resources
- [Previous Semantic Song Search Project](https://github.com/santarabantoosoo/semantic_song_search)

- [GloVe](https://nlp.stanford.edu/projects/glove/)
- Sentence Transformer Model Cards
  - [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
  - [all-distilroberta-v1](https://huggingface.co/sentence-transformers/all-distilroberta-v1)

- [How to Build a Semantic Search | Towards Data Science](https://towardsdatascience.com/how-to-build-a-semantic-search-engine-with-transformers-and-faiss-dcbea307a0e8)
- [Machine Learning with PyTorch and Scikit-Learn](https://learning.oreilly.com/library/view/machine-learning-with/9781801819312/)
- [Natural Language Processing with Transformers](https://learning.oreilly.com/library/view/natural-language-processing/9781098136789/)
- [Sentence-BERT](https://arxiv.org/abs/1908.10084)
- [SentenceTransformers](https://www.sbert.net/index.html)
- [all-MiniLM-L12-v2 Model](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)
- [NLP for Semantic Search](https://www.pinecone.io/learn/fine-tune-sentence-transformers-mnr/)

