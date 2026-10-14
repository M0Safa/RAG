"""Generate grounded answers from retrieved source snippets."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models import (AnsweredQuestion, MinimalAnswer, MinimalSearchResults,
                        MinimalSource, UnansweredQuestion)
from src.retriever import BM25Retriever


class RAGGenerator:
    """Use a local Qwen model to answer questions using retrieved context."""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B",
                 processed_dir: str = "data/processed",
                 device: str | None = None) -> None:
        """Load the index and Qwen model."""
        self.retriever = BM25Retriever(processed_dir)
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto"
        ).to(self.device)  # type: ignore[arg-type]
        self.model.eval()

    def _format_context(self, sources: list[MinimalSource]) -> str:
        """Read only the spans selected by retrieval."""
        blocks = []
        for i, source in enumerate(sources, 1):
            try:
                with open(source.file_path, "r", encoding="utf-8",
                          errors="ignore") as f:
                    content = f.read()
                snippet = content[source.first_character_index:
                                  source.last_character_index]
                blocks.append(
                    f"--- Source {i}: {source.file_path} ---\n{snippet}"
                )
            except OSError as exc:
                print(f"Cannot read {source.file_path}: {exc}")
        return "\n\n".join(blocks)

    def generate_answer(self, results: MinimalSearchResults,
                        max_new_tokens: int = 256) -> MinimalAnswer:
        """Generate an answer from the retrieved evidence."""
        context = self._format_context(results.retrieved_sources)
        messages = [
            {"role": "system", "content": (
                "Answer the question using only the supplied source snippets. "
                "Be concise. If the snippets are insufficient, say so."
            )},
            {"role": "user", "content": (
                f"Context:\n{context}\n\nQuestion:\n{results.question}"
            )},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        answer_ids = generated[:, inputs.input_ids.shape[1]:]
        answer = self.tokenizer.batch_decode(
            answer_ids, skip_special_tokens=True
        )[0].strip()
        return MinimalAnswer(
            question_id=results.question_id, question=results.question,
            retrieved_sources=results.retrieved_sources, answer=answer,
        )

    def answer_question(self, question: UnansweredQuestion | AnsweredQuestion,
                        k: int = 5) -> MinimalAnswer:
        """Retrieve evidence and generate one answer."""
        results = self.retriever.search_question(question, k)
        return self.generate_answer(results)
