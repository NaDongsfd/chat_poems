# RAG FOR CHINESE POEMS GENERATION

The code allows users to upload files with the style of poems they like. 
Users can ask questions or provide input, RAG will generate the poems.

- MindSpore
- MindNLP
- ms2vec
- msimilarities

## Installation
1: Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1.You can change the files as you like in "chat", here we use poems as a example.
we ran it with question:“写一篇柳永风格的古诗？”

   ```bash
   # RAG
   python chat.py
   ```

2: You can also try the demo of generation

   ```bash
   # RAG
   python poems_ui.py
   ```

