"""第5回 宿題（発展）: 正解ページ付きの質問で、検索方式ごとの取得率を比べる。

使い方:
    uv run python -m scripts.eval_retrieval              # BM25・ベクトル・ハイブリッドを比べる（要 GEMINI_API_KEY）
    uv run python -m scripts.eval_retrieval --no-vector  # APIを使わず、BM25だけを測る

data/eval/lec05_retrieval.jsonl の各質問について、上位5件に正解ページ（gold）が
いくつ入ったか（recall@5）と、最初の正解の順位の逆数（MRR）を出す。
回答の正しさではなく「根拠を取れたか」だけを測る。
"""

import argparse
import json
from pathlib import Path

from app.config import HYBRID_CANDIDATES, RAG_TOP_K
from app.retrieval import hybrid_rank, lexical_search, search

EVAL_SET = Path(__file__).resolve().parent.parent / "data/eval/lec05_retrieval.jsonl"


def score(hits: list[str], gold: list[str]) -> tuple[float, float]:
    """recall@k（正解のうち取れた割合）と、最初の正解の順位の逆数を返す。"""
    recall = sum(page in hits for page in gold) / len(gold)
    reciprocal = next((1 / rank for rank, page in enumerate(hits, 1) if page in gold), 0.0)
    return recall, reciprocal


def main() -> None:
    parser = argparse.ArgumentParser(description="検索方式ごとに、正解ページを取れた割合を比べる")
    parser.add_argument("--no-vector", action="store_true", help="APIを使わず、BM25だけを測る")
    args = parser.parse_args()

    cases = [json.loads(line) for line in EVAL_SET.read_text(encoding="utf-8").splitlines() if line.strip()]
    methods = ["bm25"] if args.no_vector else ["vector", "bm25", "hybrid_nofilter", "hybrid"]
    totals = {method: [0.0, 0.0] for method in methods}
    print(f"質問 {len(cases)} 件・上位 {RAG_TOP_K} 件で比較（数字は recall@{RAG_TOP_K} / 最初の正解の順位）\n")
    print("| id | 種類 | " + " | ".join(methods) + " |")
    print("|---|---|" + "---|" * len(methods))
    for case in cases:
        ranked = {"bm25": [hit["page_id"] for hit in lexical_search(case["question"], top_k=RAG_TOP_K)]}
        if not args.no_vector:
            vector_hits = search(case["question"], top_k=HYBRID_CANDIDATES)
            ranked["vector"] = [hit["page_id"] for hit in vector_hits[:RAG_TOP_K]]
            for name, use_filter in (("hybrid_nofilter", False), ("hybrid", True)):
                fused = hybrid_rank(case["question"], vector_hits, use_lecture_filter=use_filter)
                ranked[name] = [hit["page_id"] for hit in fused["results"]]
        cells = []
        for method in methods:
            recall, reciprocal = score(ranked[method], case["gold"])
            totals[method][0] += recall
            totals[method][1] += reciprocal
            first = round(1 / reciprocal) if reciprocal else "-"
            cells.append(f"{recall:.2f} / {first}")
        print(f"| {case['id']} | {case['type']} | " + " | ".join(cells) + " |")
    print("| **平均** | | " + " | ".join(
        f"**{recall / len(cases):.2f}** / MRR {mrr / len(cases):.2f}" for recall, mrr in totals.values()) + " |")


if __name__ == "__main__":
    main()
