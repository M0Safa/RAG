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
        """Initialize the retriever, tokenizer, and language model.

        Args:
            model_name (str): Name of the Hugging Face model.
            processed_dir (str): Directory containing the BM25 index.
            device (str | None): Device used to run the model.
                Uses CUDA when available, otherwise CPU.
        """
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
        """Extract text from the retrieved source locations.

        Reads the original files and extracts the character ranges
        selected by the retriever.

        Args:
            sources (list[MinimalSource]): Retrieved source locations.

        Returns:
            str: Retrieved snippets combined into one context string.
        """
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
        """Generate an answer using the retrieved context.

        Builds a prompt containing the question and source snippets,
        then uses the Qwen model to generate an answer.

        Args:
            results (MinimalSearchResults): Question and retrieved sources.
            max_new_tokens (int): Maximum number of tokens to generate.

        Returns:
            MinimalAnswer: Generated answer with question information
            and retrieved sources.
        """
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
        """Retrieve relevant sources and generate an answer.

        Args:
            question (UnansweredQuestion | AnsweredQuestion):
                Question to answer.
            k (int): Number of source chunks to retrieve.

        Returns:
            MinimalAnswer: Generated answer and its source locations.
        """
        results = self.retriever.search_question(question, k)
        return self.generate_answer(results)
