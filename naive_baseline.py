"""
Basic RAG Baseline — Chạy TRƯỚC để có scores so sánh.
=====================================================
Basic = paragraph chunking + dense-only search (không hybrid, không rerank, không enrichment).
Đây là RAG đã học ở buổi trước — hôm nay sẽ cải thiện từng bước.
"""

import sys, os, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.m1_chunking import load_documents, chunk_basic
from src.m2_search import DenseSearch
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from config import NAIVE_COLLECTION


def _generate_answer(client, question: str, contexts: list[str]) -> str:
    from config import GENERATION_MODEL

    context_str = "\n\n".join(contexts)
    response = client.chat.completions.create(model=GENERATION_MODEL, temperature=0, messages=[
        {"role": "system", "content": "Trả lời CHỈ dựa trên context. Nếu không có → nói 'Không tìm thấy.'"},
        {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {question}"},
    ])
    answer = response.choices[0].message.content
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Generation returned an empty answer")
    return answer.strip()


def main():
    print("=" * 60)
    print("BASIC RAG BASELINE")
    print("(paragraph chunking + dense-only, no rerank, no enrichment)")
    print("=" * 60)

    docs = load_documents()
    chunks = []
    for doc in docs:
        for c in chunk_basic(doc["text"], metadata=doc["metadata"]):
            chunks.append({"text": c.text, "metadata": c.metadata})
    print(f"  {len(chunks)} basic paragraph chunks")

    search = DenseSearch()
    search.index(chunks, collection=NAIVE_COLLECTION)

    test_set = load_test_set()
    questions, answers, all_contexts, ground_truths = [], [], [], []

    from config import OPENAI_API_KEY, OPENAI_BASE_URL, GENERATION_MODEL, EMBEDDING_MODEL
    llm_client = None
    if OPENAI_API_KEY:
        from openai import OpenAI
        llm_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, timeout=60, max_retries=0)

    generation_modes, generation_errors, retrieval_sources = [], [], []

    for i, item in enumerate(test_set):
        results = search.search(item["question"], top_k=3, collection=NAIVE_COLLECTION)
        contexts = [r.text for r in results]
        retrieval_sources.append([{"source": r.metadata.get("source"), "score": r.score,
                                   "chunk_index": r.metadata.get("chunk_index")} for r in results])

        if llm_client and contexts:
            try:
                answer = _generate_answer(llm_client, item["question"], contexts)
                generation_modes.append("llm")
            except Exception as exc:
                answer = contexts[0]
                generation_modes.append("extractive_fallback")
                generation_errors.append({"row_index": i, "type": type(exc).__name__,
                                           "http_status": getattr(exc, "status_code", None)})
        else:
            answer = contexts[0] if contexts else "Không tìm thấy."
            generation_modes.append("extractive_fallback" if contexts else "empty_context")

        answers.append(answer)
        questions.append(item["question"])
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    results["run_metadata"] = {"pipeline": "naive", "generation_model": GENERATION_MODEL,
        "dense_model": EMBEDDING_MODEL, "dense_backend": search.backend,
        "generation_modes": generation_modes, "generation_errors": generation_errors,
        "generation_success_count": generation_modes.count("llm"), "retrieval_sources": retrieval_sources}
    from main import dataset_fingerprint, baseline_fingerprint
    results["run_metadata"]["dataset_fingerprint"] = dataset_fingerprint()
    results["run_metadata"]["baseline_fingerprint"] = baseline_fingerprint()
    print("\nBASIC BASELINE SCORES")
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        print(f"  {m}: {results.get(m, 0):.4f}")
    save_report(results, failure_analysis(results["per_question"], bottom_n=5), path="reports/naive_baseline_report.json")
    print(f"Evaluation status: {results['eval_status']}; LLM answers: {generation_modes.count('llm')}/{len(test_set)}")
    print("\nDone! Now implement advanced modules and run: python main.py")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Total: {time.time() - start:.1f}s")
