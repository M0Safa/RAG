from typing import List, Dict, Any, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models import MinimalSource, MinimalSearchResults, MinimalAnswer, UnansweredQuestion, AnsweredQuestion
from src.retriever import BM25Retriever


class RAGGenerator:

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        processed_dir: str = "data/processed",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        print(f"--> Initializing Retriever from {processed_dir}...")
        self.retriever = BM25Retriever(processed_dir=processed_dir)

        print(f"--> Loading model '{model_name}' on device '{device}'...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=self.device if self.device == "cuda" else None,
        )

    def _format_context(self, sources: List[MinimalSource]) -> str:
        context_blocks: List[str] = []
        for i, source in enumerate(sources, start=1):
            try:
                with open(source.file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                snippet = content[source.first_character_index:source.last_character_index]
                context_blocks.append(f"--- Context Snippet [{i}] ({source.file_path}) ---\n{snippet.strip()}")
            except Exception:
                continue

        return "\n\n".join(context_blocks)

    def generate_answer(
        self, search_results: MinimalSearchResults, max_new_tokens: int = 512
    ) -> MinimalAnswer:
        context_str = self._format_context(search_results.retrieved_sources)

        system_prompt = (
            "You are a helpful coding assistant. Answer the question accurately using ONLY "
            "the provided context snippets. If the answer cannot be found in the context, "
            "state clearly that the information is insufficient."
        )

        user_content = f"Context:\n{context_str}\n\nQuestion:\n{search_results.question}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]


        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        model_inputs = self.tokenizer([prompt_text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        output_ids = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        generated_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        return MinimalAnswer(
            question_id=search_results.question_id,
            question=search_results.question,
            retrieved_sources=search_results.retrieved_sources,
            answer=generated_text.strip(),
        )

    def answer_question(
        self, question: Union[UnansweredQuestion, AnsweredQuestion], k: int = 5
    ) -> MinimalAnswer:
        search_results = self.retriever.search_question(question, k)
        return self.generate_answer(search_results)