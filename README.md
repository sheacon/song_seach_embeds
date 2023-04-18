# Semantic Song Search Embeddings
- Search for songs based on the meaning in the song's lyrics with cosine similarity of embedding vectors

## Question: What is the value in music? Why do you listen to music?

## Motivation
- Songs are enjoyable when they express feelings or situations that have meaning for the listener
- How can we allow listeners to find songs by semantics?

## Solution / Deliverables / Demo
- Compute lyrics embeddings with a variety of models
- Find the similarity of the song by comparing to the user query embedding
- GitHub repo, HuggingFace Space, HuggingFace Dataset

## Embedding Models
- Sentence Transformers
  - [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (256 / 384)
  - [all-distilroberta-v1](https://huggingface.co/sentence-transformers/all-distilroberta-v1) (512 / 768)
- OpenAI: [text-embedding-ada-002](https://platform.openai.com/docs/guides/embeddings/embedding-models) (8191 / 1536)
- [GloVe](https://nlp.stanford.edu/projects/glove/) (no limit / 300)

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

## HuggingFace
- [Space: Semantic Song Search](https://huggingface.co/spaces/sheacon/semantic-song-search)
- [Dataset: Song Lyrics](https://huggingface.co/datasets/sheacon/song_lyrics)

## Embedding Models

### Sentence Transformers
BERT

### GloVe
Pre-computed embeddings

### OpenAI GPT-3 Endpoint
- Took nearly 10 minutes for only 1 percent of my dataset (1,300 songs)!
  - This was the set after >1k views, removing non-roman characters, and removing profanity
  - 3s for minilm, 6s for roberta

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
- [Profanity](https://github.com/surge-ai/profanity) 

- [OpenAI Embedding Paper](https://cdn.openai.com/papers/Text_and_Code_Embeddings_by_Contrastive_Pre_Training.pdf)
- [OpenAI Embedding Guide](https://platform.openai.com/docs/guides/embeddings)

- [How to Build a Semantic Search | Towards Data Science](https://towardsdatascience.com/how-to-build-a-semantic-search-engine-with-transformers-and-faiss-dcbea307a0e8)
- [Machine Learning with PyTorch and Scikit-Learn](https://learning.oreilly.com/library/view/machine-learning-with/9781801819312/)
- [Natural Language Processing with Transformers](https://learning.oreilly.com/library/view/natural-language-processing/9781098136789/)
- [Sentence-BERT](https://arxiv.org/abs/1908.10084)
- [SentenceTransformers](https://www.sbert.net/index.html)
- [all-MiniLM-L12-v2 Model](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)
- [NLP for Semantic Search](https://www.pinecone.io/learn/fine-tune-sentence-transformers-mnr/)

