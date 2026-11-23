"""第5回: ページ単位のベクトル検索。"""

import argparse
import json
import math
import re
import sqlite3
import unicodedata
from contextlib import contextmanager
from collections import Counter, defaultdict, deque
from functools import cache
from pathlib import Path

from google import genai
from google.genai import types
from pydantic_ai.exceptions import ModelHTTPError
import sqlite_vec

from app.config import (
    EMBEDDING_DIMENSIONS, EMBEDDING_MAX_TOKENS, EMBEDDING_MODEL,
    RAG_DB, RAG_DOCUMENT_IDS, RAG_SOURCE, RAG_TOP_K, RAG_TREE, get_api_key,
    RAG_GRAPH, GRAPH_START_K, GRAPH_HOPS, GRAPH_MAX_RESULTS,
    HYBRID_CANDIDATES, HYBRID_RRF_K, HYBRID_TOP_K,
)


def get_all_materials() -> str:
    """全資料を、講義・章・ページの情報付きで1つの文字列にまとめる。"""
    documents = json.loads(RAG_SOURCE.read_text(encoding="utf-8"))
    return "\n\n".join(
        f"資料: {document['id']} / {document['title']}\n"
        f"章: {section['section']}\n"
        f"ページ: PDF p.{page['page']} / {page['title']}\n{page['content']}"
        for document in documents
        for section in document["content"]
        for page in section["content"]
    )

def load_pages(source: Path = RAG_SOURCE) -> list[dict]:
    """資料→章→ページのJSONを、対象資料のページ一覧に変換する。"""
    documents = json.loads(source.read_text(encoding="utf-8"))
    pages = []
    for document in documents:
        if document["id"] not in RAG_DOCUMENT_IDS:
            continue
        for section in document["content"]:
            for page in section["content"]:
                pages.append({"page_id": page["id"], "document_id": document["id"],
                              "lecture": int(document["id"][3:5]), "document_title": document["title"],
                              "section": section["section"], "title": page["title"],
                              "page": page["page"], "content": page["content"]})
    return pages


def browse_tree(node_id: str | None = None) -> dict:
    """直下のページは先頭文、指定したページ自身は全文を返す。"""
    tree = json.loads(RAG_TREE.read_text(encoding="utf-8"))
    tree["children"] = [node for node in tree["children"] if node["id"] in RAG_DOCUMENT_IDS]
    node_id = node_id or tree["id"]
    pages = {page["page_id"]: page for page in load_pages(RAG_SOURCE)}

    def describe(node, full=False):
        info = {key: node[key] for key in ("id", "kind", "title")}
        if node["kind"] == "page":
            info.update({key: value for key, value in pages[node["id"]].items() if key != "page_id"})
            if not full:
                info["content"] = re.split(r"(?<=[。！？])|\n", info["content"].strip(), maxsplit=1)[0]
        return info

    remaining = [tree]
    while remaining:
        node = remaining.pop()
        if node["id"] == node_id:
            return {
                "node": describe(node, full=True),
                "children": [describe(child) for child in node.get("children", [])],
            }
        remaining.extend(node.get("children", []))
    raise ValueError("対象の目次ノードが見つかりません。")


def read_page(page_id: str) -> dict:
    """第1〜3回の本講義から、指定した1ページの本文と引用情報を返す。"""
    for page in load_pages(RAG_SOURCE):
        if page["page_id"] == page_id:
            return page
    raise ValueError("対象のページが見つかりません。")


def normalized_vector(values: list[float]) -> list[float]:
    """ベクトルの長さを1に揃える。"""
    norm = math.hypot(*values)
    if norm == 0:
        raise ValueError("埋め込みがゼロベクトルです。")
    return [value / norm for value in values]


def embed_text(text: str, task_type: str) -> list[float]:
    """Geminiで文章を埋め込む。入力上限を超えた場合は切り捨てず停止する。"""
    with genai.Client(api_key=get_api_key()) as client:
        count = client.models.count_tokens(model=EMBEDDING_MODEL, contents=text).total_tokens
        if count is None or count > EMBEDDING_MAX_TOKENS:
            raise ValueError(f"埋め込み入力が上限を超えるか計測不能です: {count} tokens。")
        response = client.models.embed_content(
            model=EMBEDDING_MODEL, contents=text,
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBEDDING_DIMENSIONS),
        )
    embedding = response.embeddings[0]
    if embedding.statistics and embedding.statistics.truncated:
        raise ValueError("埋め込みAPIが本文を切り捨てました。")
    return normalized_vector(embedding.values)


@contextmanager
def open_index(path: Path):
    """既存の検索DBを読み取り専用で開き、終了時に閉じる。"""
    if not path.is_file():
        raise FileNotFoundError("検索DBがありません。")
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        db.row_factory = sqlite3.Row
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        yield db
    finally:
        db.close()


def search(query: str, top_k: int = RAG_TOP_K) -> list[dict]:
    """質問を埋め込み、近いページを最大top_k件（既定はRAG_TOP_K件）返す。"""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("検索語が空です。")
    with open_index(RAG_DB) as db:
        vector = embed_text(query.strip(), "RETRIEVAL_QUERY")
        rows = db.execute("""
            SELECT page_id, document_id, lecture, document_title, section, title, page, content,
                   vec_distance_cosine(embedding, ?) AS distance
            FROM chunks ORDER BY distance, page_id LIMIT ?
        """, (sqlite_vec.serialize_float32(vector), top_k))
        return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# 第5回 宿題（発展）: ハイブリッド検索
#   ベクトル検索は言い換えに強いが、「QLoRA」「日々のメモ」「4bit」のような
#   固有の語・数値の一致を取りこぼすことがある。語の一致（BM25）を別に数え、
#   2つの順位をRRF（Reciprocal Rank Fusion）で融合する。
#   さらに graph.json の用語ノードの別名（例: LoRA ⇔ Low-Rank Adaptation）で
#   語の検索だけを言い換えて広げる。埋め込み・索引は作り直さない。
# ---------------------------------------------------------------------------

BM25_K1 = 1.2
BM25_B = 0.75
TITLE_WEIGHT = 2  # タイトルの語は本文の2回分として数える


def tokenize(text: str) -> list[str]:
    """英数字は単語ごと、日本語などは文字bigram（1文字だけの塊はその文字）に分ける。"""
    text = unicodedata.normalize("NFKC", text).casefold()
    tokens = []
    for run in re.findall(r"[a-z0-9]+|[^\W_a-z0-9]+", text):
        if run.isascii() or len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


@cache
def _bm25_index() -> tuple[list[dict], list[Counter], dict[str, int], float]:
    """対象ページの語の出現回数と、語ごとの出現ページ数（df）を一度だけ数える。"""
    pages = load_pages(RAG_SOURCE)
    term_counts = [Counter(tokenize(page["title"]) * TITLE_WEIGHT + tokenize(page["content"]))
                   for page in pages]
    df = Counter(term for counts in term_counts for term in counts)
    average_length = sum(sum(counts.values()) for counts in term_counts) / len(term_counts)
    return pages, term_counts, dict(df), average_length


@cache
def _term_aliases() -> list[list[str]]:
    """graph.json の用語ノードごとに、表記ゆれ（タイトルと別名）をまとめる。"""
    graph = json.loads(RAG_GRAPH.read_text(encoding="utf-8"))
    groups = []
    for node in graph["nodes"]:
        if node["kind"] == "term":
            names = {unicodedata.normalize("NFKC", name).casefold()
                     for name in [node["title"], *node.get("aliases", [])]}
            groups.append(sorted(names, key=len, reverse=True))
    return groups


def expand_query(query: str) -> tuple[str, list[str]]:
    """質問に用語の別名が含まれていれば、同じ用語の他の表記を語の検索用に足す。"""
    normalized = unicodedata.normalize("NFKC", query).casefold()
    added = []
    for names in _term_aliases():
        # 「r」のような短い別名が何にでも当たらないよう、2文字以下の英字は使わない
        if any(name in normalized for name in names if not (name.isascii() and len(name) <= 2)):
            added.extend(name for name in names if name not in normalized and name not in added)
    return " ".join([query, *added]), added


def lexical_search(query: str, top_k: int = HYBRID_CANDIDATES) -> list[dict]:
    """BM25で語の一致を数え、スコアの高いページを最大top_k件返す。APIは使わない。"""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("検索語が空です。")
    pages, term_counts, df, average_length = _bm25_index()
    expanded, _ = expand_query(query)
    query_terms = set(tokenize(expanded))
    n = len(pages)
    scored = []
    for page, counts in zip(pages, term_counts):
        length = sum(counts.values())
        score = 0.0
        for term in query_terms & counts.keys():
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            tf = counts[term]
            score += idf * tf * (BM25_K1 + 1) / (tf + BM25_K1 * (1 - BM25_B + BM25_B * length / average_length))
        if score > 0:
            scored.append((score, page))
    scored.sort(key=lambda item: (-item[0], item[1]["page_id"]))
    return [{**page, "bm25": round(score, 3)} for score, page in scored[:top_k]]


def fuse_rankings(rankings: dict[str, list[dict]], top_k: int = HYBRID_TOP_K,
                  rrf_k: int = HYBRID_RRF_K) -> list[dict]:
    """複数の検索結果の順位をRRFで融合する。スコアの尺度が違っても順位だけで比べられる。"""
    fused: dict[str, dict] = {}
    for name, results in rankings.items():
        for rank, hit in enumerate(results, 1):
            entry = fused.setdefault(hit["page_id"], {**hit, "rrf": 0.0, "ranks": {}})
            entry["rrf"] += 1 / (rrf_k + rank)
            entry["ranks"][name] = rank
            entry.update({key: value for key, value in hit.items() if key in ("distance", "bm25")})
    ordered = sorted(fused.values(), key=lambda hit: (-hit["rrf"], hit["page_id"]))[:top_k]
    for hit in ordered:
        hit["rrf"] = round(hit["rrf"], 5)
    return ordered


def _neighbors(page_id: str) -> dict:
    """同じ資料の前後ページのIDとタイトル（本文は付けない）。"""
    pages = _bm25_index()[0]
    index = next(i for i, page in enumerate(pages) if page["page_id"] == page_id)
    result = {}
    for key, j in (("prev", index - 1), ("next", index + 1)):
        if 0 <= j < len(pages) and pages[j]["document_id"] == pages[index]["document_id"]:
            result[key] = {"page_id": pages[j]["page_id"], "page": pages[j]["page"],
                           "title": pages[j]["title"]}
    return result


def lecture_hints(query: str) -> set[int]:
    """「第2回」のように回が明示されていれば、その回番号を返す（対象外の回は除く）。"""
    targets = {int(document_id[3:5]) for document_id in RAG_DOCUMENT_IDS}
    numbers = {int(n) for n in re.findall(r"第\s*(\d+)\s*回", unicodedata.normalize("NFKC", query))}
    return numbers & targets


def hybrid_rank(query: str, vector_hits: list[dict], use_lecture_filter: bool = True) -> dict:
    """ベクトル検索の結果にBM25の結果を融合する。ベクトル側は呼び出し元が渡す（評価で再利用するため）。

    質問で回が明示されていれば、両方の候補をその回のページに絞ってから融合する。
    """
    _, added = expand_query(query)
    hints = lecture_hints(query) if use_lecture_filter else set()
    rankings = {"vector": vector_hits, "lexical": lexical_search(query, top_k=HYBRID_CANDIDATES)}
    if hints:
        rankings = {name: [hit for hit in hits if hit["lecture"] in hints]
                    for name, hits in rankings.items()}
    results = fuse_rankings(rankings)
    for hit in results:
        hit["neighbors"] = _neighbors(hit["page_id"])
    return {"results": results, "expanded_terms": added, "lecture_filter": sorted(hints)}


def search_hybrid(query: str) -> dict:
    """ベクトル検索とBM25の上位候補をRRFで融合し、前後ページの見出しを付けて返す。"""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("検索語が空です。")
    return hybrid_rank(query, search(query, top_k=HYBRID_CANDIDATES))


def search_graph(query: str) -> dict:
    """ベクトル検索を入口に、前後ページと共通用語で候補を広げる。"""
    pages = {page["page_id"]: page for page in load_pages(RAG_SOURCE)}
    graph = json.loads(RAG_GRAPH.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in graph["nodes"]
             if node["kind"] == "term" or node["id"] in pages}
    adjacent = defaultdict(set)
    for edge in graph["edges"]:
        a, b = edge["source"], edge["target"]
        if edge["relation"] in ("next_page", "mentions") and a in nodes and b in nodes:
            adjacent[a].add(b)
            adjacent[b].add(a)

    seeds = [hit for hit in search(query) if hit["page_id"] in pages][:min(GRAPH_START_K, GRAPH_MAX_RESULTS)]
    seed_ids = {seed["page_id"] for seed in seeds}
    results, candidates = [], {}
    for rank, seed in enumerate(seeds):
        root = seed["page_id"]
        results.append({**pages[root], "distance": seed["distance"], "hops": 0,
                        "path": [{"id": root, "title": pages[root]["title"]}]})
        queue = deque([(root, [root])])
        visited = {root}
        while queue:
            current, path = queue.popleft()
            if len(path) - 1 >= GRAPH_HOPS:
                continue
            # 同じ深さなら、接続先が少ない用語からたどる。
            neighbors = sorted(adjacent[current], key=lambda n: (len(adjacent[n]), n))
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                route = path + [neighbor]
                queue.append((neighbor, route))
                if neighbor not in pages or neighbor in seed_ids:
                    continue
                frequency = max((len(adjacent[n]) for n in route if nodes[n]["kind"] == "term"), default=0)
                priority = (len(route) - 1, frequency, rank, neighbor)
                candidate = {**pages[neighbor], "hops": len(route) - 1,
                             "path": [{"id": n, "title": nodes[n]["title"]} for n in route]}
                if neighbor not in candidates or priority < candidates[neighbor][0]:
                    candidates[neighbor] = (priority, candidate)
    expanded = [candidate for _, candidate in sorted(candidates.values())]
    return {"seeds": results, "expanded": expanded[:max(0, GRAPH_MAX_RESULTS - len(results))],
            "max_hops": GRAPH_HOPS}



def run_full_context(agent, question: str, model, instructions: str, prompt_template: str):
    """全資料を一括投入し、検索ツールを使わずに回答する。"""
    prompt = prompt_template.format(materials=get_all_materials(), question=question)
    print(f"[full-context] 入力{len(prompt):,}文字")
    try:
        with agent.override(tools=[], instructions=instructions):
            return agent.run_sync(prompt, model=model)
    except ModelHTTPError as e:
        print(f"[full-context] APIエラー HTTP {e.status_code}: {e.body}")
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="エージェントを使わず、講義資料の検索だけを実行する")
    parser.add_argument("query", help="調べたい内容")
    args = parser.parse_args()
    for i, result in enumerate(search(args.query), 1):
        print(f"\n{'=' * 60}\n検索結果 {i}\n{'=' * 60}")
        print(f"講義: {result['document_title']}")
        print(f"章: {result['section']}")
        print(f"タイトル: {result['title']}")
        print(f"ページ: {result['page']}")
        print(f"\n{result['content']}")
