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

![Screenshot 2023-04-18 at 9 52 14 AM](https://user-images.githubusercontent.com/89158603/232816181-f3b3c07c-6d14-4b56-a7d2-8ec4573807a6.png)

![roberta_stats](https://user-images.githubusercontent.com/89158603/232815347-e4b4e64a-131b-4d36-b0bb-1b00096632af.png)
![minilm_stats2](https://user-images.githubusercontent.com/89158603/232815688-cd2438bc-9486-4419-aa7e-6fd9b33db1ba.png)

![gpt](https://user-images.githubusercontent.com/89158603/232816549-d9e8dac5-567b-4b70-a13a-47e5f44e7c43.png)




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

