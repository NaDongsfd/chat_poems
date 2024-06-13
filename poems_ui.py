# main.py
from search import SearchPDF
from generation import GeneratePDF
import gradio as gr
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


def chat_with_pdf(pdf, question):
    if pdf is not None:
        search.add_corpus([pdf.name])

    reference_results = search.get_reference_results(question)
    if not reference_results:
        return [[("没有提供足够的相关信息", "bot")]], None

    reference_results = search._add_source_numbers(reference_results)
    context_str = '\n'.join(reference_results)[:(2048 - len(PROMPT_TEMPLATE))]
    prompt = PROMPT_TEMPLATE.format(context_str=context_str, query_str=question)

    response = generate.predict(prompt)
    return [[(response, "bot")]], pdf.name  # 返回一个包含响应和 PDF 文件名的元组列表


# Create Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot(label="Chatbot")
        with gr.Column():
            pdf_preview = gr.File(label="PDF Preview")

    with gr.Row():
        input_pdf = gr.File(label="Upload PDF")
        input_text = gr.Textbox(label="Ask your pdf?")

        examples = gr.Examples(
            examples=[
                ["poems.pdf", "写一篇柳永风格诗歌？"],
                ["poems.pdf", "描写春天的宋词？"],
                ["poems.pdf", "写一首悲伤的诗？"],
                ["poems.pdf", "最常见的古诗事物？"],
            ],
            inputs=[input_pdf, input_text],
        )

    with gr.Row():
        submit_btn = gr.Button("Send")

    submit_btn.click(
        chat_with_pdf,
        inputs=[input_pdf, input_text],
        outputs=[chatbot, pdf_preview]
    )

# Launch the demo
demo.launch()