# -*- coding: utf-8 -*-
from threading import Thread

import _queue
from loguru import logger
from mindnlp.peft import PeftModel
from mindnlp.transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BloomForCausalLM,
    BloomTokenizerFast,
    LlamaTokenizer,
    LlamaForCausalLM,
    TextIteratorStreamer,
    GenerationConfig,
    AutoModelForSequenceClassification,
)

MODEL_CLASSES = {
    "bloom": (BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoModel, AutoTokenizer),
    "llama": (LlamaForCausalLM, LlamaTokenizer),
    "baichuan": (AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
}

PROMPT_TEMPLATE = """基于以下已知信息，简洁和专业的来回答用户的问题。
如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。

已知内容:
{context_str}

问题:
{query_str}
"""

class GeneratePDF:
    def __init__(
            self,
            generate_model_type: str = "auto",
            generate_model_name_or_path: str = "01-ai/Yi-6B-Chat",
            lora_model_name_or_path: str = None,
            int8: bool = False,
            int4: bool = False,
            enable_history: bool = False,
    ):
        self.gen_model, self.tokenizer = self._init_gen_model(
            generate_model_type,
            generate_model_name_or_path,
            peft_name=lora_model_name_or_path,
            int8=int8,
            int4=int4,
        )
        self.history = []
        self.enable_history = enable_history

    def _init_gen_model(
            self,
            gen_model_type: str,
            gen_model_name_or_path: str,
            peft_name: str = None,
            int8: bool = False,
            int4: bool = False,
    ):
        """Init generate model."""
        model_class, tokenizer_class = MODEL_CLASSES[gen_model_type]
        tokenizer = tokenizer_class.from_pretrained(gen_model_name_or_path, trust_remote_code=True)
        model = model_class.from_pretrained(
            gen_model_name_or_path,
        )
        try:
            model.generation_config = GenerationConfig.from_pretrained(gen_model_name_or_path, trust_remote_code=True)
        except Exception as e:
            logger.warning(f"Failed to load generation config from {gen_model_name_or_path}, {e}")
        if peft_name:
            model = PeftModel.from_pretrained(
                model,
                peft_name,
                torch_dtype="auto",
            )
            logger.info(f"Loaded peft model from {peft_name}")
        model.set_train(False)
        logger.debug(f"Initialized model: {model}")  # 调试输出
        logger.debug(f"Initialized tokenizer: {tokenizer}")  # 调试输出
        return model, tokenizer

    def _get_chat_input(self):
        messages = []
        for conv in self.history:
            if conv and len(conv) > 0 and conv[0]:
                messages.append({'role': 'user', 'content': conv[0]})
            if conv and len(conv) > 1 and conv[1]:
                messages.append({'role': 'assistant', 'content': conv[1]})
        input_ids = self.tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors='ms'
        )
        # 确保 input_ids 的长度不会超过模型的最大输入长度
        max_input_length = self.gen_model.config.max_position_embeddings - self.gen_model.config.max_length
        input_ids = input_ids[:, -max_input_length:]
        return input_ids

    def stream_generate_answer(
            self,
            max_new_tokens=512,
            temperature=0.7,
            repetition_penalty=1.0,
            context_len=2048
    ):
        streamer = TextIteratorStreamer(self.tokenizer, timeout=1200.0, skip_prompt=True, skip_special_tokens=True)
        input_ids = self._get_chat_input()
        # 确保 input_ids 的长度不会超过 context_len - max_new_tokens - 8
        max_src_len = context_len - max_new_tokens - 8
        input_ids = input_ids[:, -max_src_len:]
        generation_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            repetition_penalty=repetition_penalty,
            streamer=streamer,
        )
        thread = Thread(target=self.gen_model.generate, kwargs=generation_kwargs)
        thread.start()

        try:
            yield from streamer
        except _queue.Empty:
            raise RuntimeError("The text queue is empty. No text was generated.")

    def predict_stream(self, query: str, max_length: int = 512, context_len: int = 2048, temperature: float = 0.7):
        """Generate predictions stream."""
        stop_str = self.tokenizer.eos_token if self.tokenizer.eos_token else "</s>"
        response = ""
        for new_text in self.stream_generate_answer(max_new_tokens=max_length, temperature=temperature, context_len=context_len):
            if new_text != stop_str:
                response += new_text
                yield response

    def predict(self, query: str, max_length: int = 512, context_len: int = 2048, temperature: float = 0.7):
        """Generate predictions."""
        response = ""
        for new_text in self.stream_generate_answer(max_new_tokens=max_length, temperature=temperature, context_len=context_len):
            response += new_text
        return response.strip()

