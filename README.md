# Semantic Song Search Embeddings
- Search for songs based on the meaning in the song's lyrics with cosine similarity of embedding vectors

## Question: What is the value in music? Why do you listen to music?

## Motivation
- Songs are enjoyable when they express feelings or situations that have meaning for the listener
- How can we allow listeners to find songs by semantics?

## Solution / Demo and Deliverables
- Compute lyrics embeddings with a variety of models
- Find the similarity of the song by comparing to the user query embedding
- Demo: [GitHub repo](https://github.com/sheacon/song_search_embeds), [HuggingFace Dataset](https://huggingface.co/datasets/sheacon/song_lyrics), [HuggingFace Space](https://huggingface.co/spaces/sheacon/semantic-song-search)
- Question: example queries

## Embedding Models
- Sentence Transformers
  - [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (256 / 384)
  - [all-distilroberta-v1](https://huggingface.co/sentence-transformers/all-distilroberta-v1) (512 / 768)
- OpenAI: [text-embedding-ada-002](https://platform.openai.com/docs/guides/embeddings/embedding-models) (8191 / 1536)
- [glove.840B.300d](https://nlp.stanford.edu/projects/glove/) (no limit / 300)

![minilm_stats](https://user-images.githubusercontent.com/89158603/232814971-98fb8fcd-29b0-4a86-bc75-a2692ad1cd07.png)

![roberta_stats](https://user-images.githubusercontent.com/89158603/232815015-27dda74e-63f8-4ac3-a8d2-16e3cd2c1400.png)



## Performance Evaluation
- Inherently subjective, examples
- Compute time and cost
  - 1500 song test: GloVe 1s, MiniLM 3s, RoBERTa 6s, OpenAI 10min

## Dataset
- [5 Million Song Dataset](https://www.kaggle.com/datasets/nikhilnayak123/5-million-song-lyrics-dataset)
  - Derived from Genius.com
- Subset to good music
- Remove profanity

## Critical Analysis
- No analytical or review data (scrape from Genius.com and YouTube)
- Cost: $50 Colab Pro+, OpenAI API $1.70

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
- [Sentence-BERT](https://arxiv.org/abs/1908.10084)
- [SentenceTransformers](https://www.sbert.net/index.html)
- [NLP for Semantic Search](https://www.pinecone.io/learn/fine-tune-sentence-transformers-mnr/)

