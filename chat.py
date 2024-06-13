
import argparse
from loguru import logger
from msimilarities import BertSimilarity

from search import SearchPDF
from generation import GeneratePDF, PROMPT_TEMPLATE

class ChatPDF:
    def __init__(self, similarity_model_name: str, generate_model_type: str, generate_model_name_or_path: str,
                 lora_model_name_or_path: str = None, corpus_files: str = None, chunk_size: int = 250,
                 chunk_overlap: int = 0, rerank_model_name_or_path: str = None, num_expand_context_chunk: int = 2):
        self.search_model = SearchPDF(similarity_model=BertSimilarity(model_name_or_path=similarity_model_name),
                                        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.generate_model = GeneratePDF(generate_model_type=generate_model_type,
                                            generate_model_name_or_path=generate_model_name_or_path,
                                            lora_model_name_or_path=lora_model_name_or_path)
        if corpus_files:
            self.search_model.add_corpus(corpus_files.split(','))
        self.num_expand_context_chunk = num_expand_context_chunk

    def predict(self, query: str, max_length: int = 512, context_len: int = 2048, temperature: float = 0.7):
        reference_results = self.search_model.get_reference_results(query)
        if not reference_results:
            return '没有提供足够的相关信息', reference_results
        reference_results = self.search_model.get_reference_results(reference_results)
        context_str = '\n'.join(reference_results)[:(context_len - len(PROMPT_TEMPLATE))]
        prompt = PROMPT_TEMPLATE.format(context_str=context_str, query_str=query)
        logger.debug(f"prompt: {prompt}")
        self.generate_model.history.append([prompt, ''])
        response = self.generate_model.predict(query, max_length=max_length, context_len=context_len, temperature=temperature)
        self.generate_model.history[-1][1] = response
        return response, reference_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_model_name", type=str, default="shibing624/text2vec-base-multilingual")
    parser.add_argument("--gen_model_type", type=str, default="auto")
    parser.add_argument("--gen_model_name", type=str, default="01-ai/Yi-6B-Chat")
    parser.add_argument("--lora_model", type=str, default=None)
    parser.add_argument("--corpus_files", type=str, default="poems.pdf")
    parser.add_argument("--chunk_size", type=int, default=220)
    parser.add_argument("--chunk_overlap", type=int, default=0)
    parser.add_argument("--num_expand_context_chunk", type=int, default=1)
    args = parser.parse_args()
    print(args)
    chat_pdf = ChatPDF(
        similarity_model_name=args.sim_model_name,
        generate_model_type=args.gen_model_type,
        generate_model_name_or_path=args.gen_model_name,
        lora_model_name_or_path=args.lora_model,
        corpus_files=args.corpus_files,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        num_expand_context_chunk=args.num_expand_context_chunk,
    )
    r, refs = chat_pdf.predict('写一篇柳永风格的古诗？')
    print(r)
    print(refs)