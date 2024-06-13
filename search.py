# -*- coding: utf-8 -*-
import hashlib
import os
import re
from typing import Union, List

import jieba
from loguru import logger
from msimilarities import (
    EnsembleSimilarity,
    BertSimilarity,
    BM25Similarity,
)
from msimilarities.similarity import SimilarityABC
from mindnlp.transformers import AutoTokenizer, AutoModelForSequenceClassification

jieba.setLogLevel("ERROR")

class SentenceSplitter:
    def __init__(self, chunk_size: int = 250, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        if self._is_has_chinese(text):
            return self._split_chinese_text(text)
        else:
            return self._split_english_text(text)

    def _split_chinese_text(self, text: str) -> List[str]:
        sentence_endings = {'\n', '。', '！', '？', '；', '…', ',', ' '}
        chunks, current_chunk = [], ''
        for word in jieba.cut(text):
            if len(current_chunk) + len(word) > self.chunk_size:
                # 在这里调整，保留符合长度的部分
                while len(current_chunk) + len(word) > self.chunk_size:
                    chunks.append(current_chunk[:self.chunk_size].strip())
                    current_chunk = current_chunk[self.chunk_size:]
                current_chunk += word
            else:
                current_chunk += word

            if word[-1] in sentence_endings and len(current_chunk) > self.chunk_size - self.chunk_overlap:
                chunks.append(current_chunk.strip())
                current_chunk = ''  # 清空 current_chunk

        if current_chunk:
            chunks.append(current_chunk.strip())

        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._handle_overlap(chunks)

        return chunks

    def _handle_overlap(self, chunks: List[str]) -> List[str]:
        overlapped_chunks = []
        for i in range(len(chunks) - 1):
            overlapped_chunks.append(chunks[i])
            overlap = chunks[i][-self.chunk_overlap:]
            chunks[i + 1] = overlap + chunks[i + 1]

        overlapped_chunks.append(chunks[-1])
        return overlapped_chunks

    def _split_english_text(self, text: str) -> List[str]:
        # 使用正则表达式按句子分割英文文本
        sentences = re.split(r'(?<=[.!?])\s+', text.replace('\n', ' '))
        chunks, current_chunk = [], ''
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size or not current_chunk:
                current_chunk += (' ' if current_chunk else '') + sentence
            else:
                chunks.append(current_chunk)
                current_chunk = sentence
        if current_chunk:  # Add the last chunk
            chunks.append(current_chunk)

        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._handle_overlap(chunks)

        return chunks

    def _is_has_chinese(self, text: str) -> bool:
        # check if contains chinese characters
        if any("\u4e00" <= ch <= "\u9fff" for ch in text):
            return True
        else:
            return False

    def _handle_overlap(self, chunks: List[str]) -> List[str]:
        # 处理块间重叠
        overlapped_chunks = []
        for i in range(len(chunks) - 1):
            chunk = chunks[i] + ' ' + chunks[i + 1][:self.chunk_overlap]
            overlapped_chunks.append(chunk.strip())
        overlapped_chunks.append(chunks[-1])
        return overlapped_chunks


class SearchPDF:
    def __init__(
            self,
            similarity_model: SimilarityABC = None,
            corpus_files: Union[str, List[str]] = None,
            chunk_size: int = 250,
            chunk_overlap: int = 0,
            num_expand_context_chunk: int = 2,
            similarity_top_k: int = 5,
            rerank_top_k: int = 2,
            rerank_model_name_or_path: str = None,
    ):
        if num_expand_context_chunk > 0 and chunk_overlap > 0:
            logger.warning(f" 'num_expand_context_chunk'和 'chunk_overlap'不能同时大于零。"
                           f" 'chunk_overlap'已默认设置为零。")
            chunk_overlap = 0
        self.text_splitter = SentenceSplitter(chunk_size, chunk_overlap)
        if similarity_model is not None:
            self.sim_model = similarity_model
        else:
            m1 = BertSimilarity(model_name_or_path="shibing624/text2vec-base-multilingual")
            m2 = BM25Similarity()
            default_sim_model = EnsembleSimilarity(similarities=[m1, m2], weights=[0.5, 0.5], c=2)
            self.sim_model = default_sim_model
        self.corpus_files = corpus_files
        if corpus_files:
            self.add_corpus(corpus_files)
        self.similarity_top_k = similarity_top_k
        self.num_expand_context_chunk = num_expand_context_chunk
        self.rerank_top_k = rerank_top_k

        if rerank_model_name_or_path:
            self.rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_model_name_or_path)
            self.rerank_model = AutoModelForSequenceClassification.from_pretrained(rerank_model_name_or_path)
            self.rerank_model.eval()
        else:
            self.rerank_model = None
            self.rerank_tokenizer = None

    def add_corpus(self, files: Union[str, List[str]]):
        if isinstance(files, str):
            files = [files]
        for doc_file in files:
            if doc_file.endswith('.pdf'):
                corpus = self.extract_text_from_pdf(doc_file)
            elif doc_file.endswith('.docx'):
                corpus = self.extract_text_from_docx(doc_file)
            elif doc_file.endswith('.md'):
                corpus = self.extract_text_from_markdown(doc_file)
            else:
                corpus = self.extract_text_from_txt(doc_file)
            full_text = '\n'.join(corpus)
            chunks = self.text_splitter.split_text(full_text)
            self.sim_model.add_corpus(chunks)
        self.corpus_files = files
        logger.debug(f"files: {files}, corpus size: {len(self.sim_model.corpus)}, top2: "
                     f"{list(self.sim_model.corpus.values())[:2]}")

    def reset_corpus(self, files: Union[str, List[str]]):
        if isinstance(files, str):
            files = [files]
        for doc_file in files:
            if doc_file.endswith('.pdf'):
                corpus = self.extract_text_from_pdf(doc_file)
            elif doc_file.endswith('.docx'):
                corpus = self.extract_text_from_docx(doc_file)
            elif doc_file.endswith('.md'):
                corpus = self.extract_text_from_markdown(doc_file)
            else:
                corpus = self.extract_text_from_txt(doc_file)
            full_text = '\n'.join(corpus)
            chunks = self.text_splitter.split_text(full_text)
            self.sim_model.reset_corpus(chunks)
        self.corpus_files = files
        logger.debug(f"files: {files}, corpus size: {len(self.sim_model.corpus)}, top3: "
                     f"{list(self.sim_model.corpus.values())[:3]}")

    @staticmethod
    def get_file_hash(fpaths):
        hasher = hashlib.md5()
        target_file_data = bytes()
        if isinstance(fpaths, str):
            fpaths = [fpaths]
        for fpath in fpaths:
            with open(fpath, 'rb') as file:
                chunk = file.read(1024 * 1024)  # read only first 1MB
                hasher.update(chunk)
                target_file_data += chunk

        hash_name = hasher.hexdigest()[:32]
        return hash_name

    @staticmethod
    def extract_text_from_pdf(file_path: str):
        import PyPDF2
        contents = []
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                page_text = page.extract_text().strip()
                raw_text = [text.strip() for text in page_text.splitlines() if text.strip()]
                new_text = ''
                for text in raw_text:
                    new_text += text
                    if text[-1] in ['.', '!', '?', '。', '！', '？', '…', ';', '；', ':', '：', '”', '’', '）', '】', '》', '」',
                                    '』', '〕', '〉', '》', '〗', '〞', '〟', '»', '"', "'", ')', ']', '}']:
                        contents.append(new_text)
                        new_text = ''
                if new_text:
                    contents.append(new_text)
        return contents

    @staticmethod
    def extract_text_from_txt(file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            contents = [text.strip() for text in f.readlines() if text.strip()]
        return contents

    @staticmethod
    def extract_text_from_docx(file_path: str):
        import docx
        document = docx.Document(file_path)
        contents = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
        return contents

    @staticmethod
    def extract_text_from_markdown(file_path: str):
        import markdown
        from bs4 import BeautifulSoup
        with open(file_path, 'r', encoding='utf-8') as f:
            markdown_text = f.read()
        html = markdown.markdown(markdown_text)
        soup = BeautifulSoup(html, 'html.parser')
        contents = [text.strip() for text in soup.get_text().splitlines() if text.strip()]
        return contents

    @staticmethod
    def _add_source_numbers(lst):
        return [f'[{idx + 1}]\t "{item}"' for idx, item in enumerate(lst)]

    def _get_reranker_score(self, query: str, reference_results: List[str]):
        """Get reranker score."""
        pairs = []
        for reference in reference_results:
            pairs.append([query, reference])
        inputs = self.rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        scores = self.rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()
        return scores

    def get_reference_results(self, query: str):
        reference_results = []
        sim_contents = self.sim_model.most_similar(query, topn=self.similarity_top_k)
        hit_chunk_dict = dict()
        for query_id, id_score_dict in sim_contents.items():
            for corpus_id, s in id_score_dict.items():
                hit_chunk = self.sim_model.corpus[corpus_id]
                reference_results.append(hit_chunk)
                hit_chunk_dict[corpus_id] = hit_chunk

        if reference_results:
            if self.rerank_model is not None:
                rerank_scores = self._get_reranker_score(query, reference_results)
                logger.debug(f"rerank_scores: {rerank_scores}")
                reference_results = [reference for reference, score in sorted(
                    zip(reference_results, rerank_scores), key=lambda x: x[1], reverse=True)][:self.rerank_top_k]
                hit_chunk_dict = {corpus_id: hit_chunk for corpus_id, hit_chunk in hit_chunk_dict.items() if
                                  hit_chunk in reference_results}
            if self.num_expand_context_chunk > 0:
                new_reference_results = []
                for corpus_id, hit_chunk in hit_chunk_dict.items():
                    expanded_reference = self.sim_model.corpus.get(corpus_id - 1, '') + hit_chunk
                    for i in range(self.num_expand_context_chunk):
                        expanded_reference += self.sim_model.corpus.get(corpus_id + i + 1, '')
                    new_reference_results.append(expanded_reference)
                reference_results = new_reference_results
        return reference_results

    def save_corpus_emb(self):
        dir_name = self.get_file_hash(self.corpus_files)
        save_dir = os.path.join(self.save_corpus_emb_dir, dir_name)
        if hasattr(self.sim_model, 'save_corpus_embeddings'):
            self.sim_model.save_corpus_embeddings(save_dir)
            logger.debug(f"Saving corpus embeddings to {save_dir}")
        return save_dir

    def load_corpus_emb(self, emb_dir: str):
        if hasattr(self.sim_model, 'load_corpus_embeddings'):
            logger.debug(f"Loading corpus embeddings from {emb_dir}")
            self.sim_model.load_corpus_embeddings(emb_dir)

