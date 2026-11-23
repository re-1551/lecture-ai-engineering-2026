"""第5回 宿題（発展）: ハイブリッド検索のテスト（APIキー不要。ベクトル検索は差し替える）。"""

import json

from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.test import TestModel

from app import tools
from app.agent import HYBRID_INSTRUCTIONS, agent
from app.retrieval import (
    expand_query, fuse_rankings, hybrid_rank, lecture_hints, lexical_search, tokenize,
)


def test_tokenize_splits_ascii_words_and_japanese_bigrams():
    assert tokenize("QLoRAは4bit") == ["qlora", "は", "4bit"]
    assert tokenize("日々のメモ") == ["日々", "々の", "のメ", "メモ"]


def test_expand_query_adds_aliases_from_graph_terms():
    _, added = expand_query("LoRAとは")
    assert "low-rank adaptation" in added


def test_lexical_search_finds_exact_terms_without_api():
    hits = lexical_search("QLoRA 4bit 圧縮", top_k=3)
    assert hits[0]["page_id"] == "lec02-main-p052"


def test_lecture_hints_ignore_lectures_outside_target():
    assert lecture_hints("第２回のLoRA") == {2}
    assert lecture_hints("第5回のRAG") == set()


def test_rrf_prefers_pages_found_by_both_rankings():
    a, b, c = ({"page_id": pid} for pid in ("a", "b", "c"))
    fused = fuse_rankings({"vector": [a, b], "lexical": [c, b]}, top_k=3)
    assert fused[0]["page_id"] == "b"
    assert fused[0]["ranks"] == {"vector": 2, "lexical": 2}


def test_hybrid_rank_filters_by_lecture_and_adds_neighbors():
    vector_hits = [{"page_id": "lec01-main-p049", "lecture": 1, "distance": 0.1},
                   {"page_id": "lec02-main-p051", "lecture": 2, "distance": 0.2}]
    result = hybrid_rank("第2回の説明で、LoRAは何を固定しますか", vector_hits)
    assert result["lecture_filter"] == [2]
    assert {hit["lecture"] for hit in result["results"]} == {2}
    p052 = next(hit for hit in result["results"] if hit["page_id"] == "lec02-main-p052")
    assert p052["neighbors"]["prev"]["page_id"] == "lec02-main-p051"
    assert p052["neighbors"]["next"]["page_id"] == "lec02-main-p053"


def test_hybrid_mode_offers_search_and_read_page(monkeypatch):
    monkeypatch.setattr("app.retrieval.search", lambda *args, **kwargs: [])
    model = TestModel(call_tools=["search_hybrid"])
    with agent.override(instructions=HYBRID_INSTRUCTIONS, tools=[tools.search_hybrid, tools.read_page]):
        result = agent.run_sync("QLoRAとは？", model=model)
    offered = {t.name for t in model.last_model_request_parameters.function_tools}
    assert {"search_hybrid", "read_page"} <= offered
    returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]
    assert returns and "results" in json.loads(returns[0].content)
