# logic.py
import os
from search import SearchPDF
from generation import GeneratePDF
from msimilarities import BertSimilarity

# Initialize models
sim_model = BertSimilarity(model_name_or_path="shibing624/text2vec-base-multilingual")
search = SearchPDF(
    similarity_model=sim_model,
    chunk_size=220,
    chunk_overlap=0,
    num_expand_context_chunk=1,
    rerank_top_k=3,
)
generate = GeneratePDF(
    generate_model_type="auto",
    generate_model_name_or_path="01-ai/Yi-6B-Chat",
)

PROMPT_TEMPLATE = """基于以下已知信息，简洁和专业地来回答用户的问题。
如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。

已知内容:
{context_str}

问题:
{query_str}
"""


def add_text(chatbot, text):
    chatbot.append(("user", text))
    return chatbot


def generate_response(chatbot, text, pdf):
    if pdf is not None:
        search.add_corpus([pdf.name])

    question = text
    reference_results = search.get_reference_results(question)
    if not reference_results:
        response = '没有提供足够的相关信息'
    else:
        reference_results = search._add_source_numbers(reference_results)
        context_str = '\n'.join(reference_results)[:(2048 - len(PROMPT_TEMPLATE))]
        prompt = PROMPT_TEMPLATE.format(context_str=context_str, query_str=question)
        response = generate.generate(prompt)

    chatbot.append(("bot", response))
    return chatbot, ""


def render_file(pdf):
    # Returns the path of the uploaded PDF file for rendering
    return pdf


def clear_chatbot():
    # Clears the chatbot history
    return []