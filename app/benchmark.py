"""複数の質問・モデルで応答時間とトークン数を比較するベンチマーク。"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai
from google.genai import errors
from google.genai import types

from app.config import api_error_message, get_api_key

DEFAULT_CONFIG = Path(__file__).parent.parent / "lectures" / "01" / "question.json"
DEFAULT_MARKDOWN_OUTPUT = (
    Path(__file__).parent.parent / "lectures" / "01" / "benchmark_results.md"
)
REQUEST_TIMEOUT_MS = 120_000


def load_config(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    """ベンチマーク設定を読み込み、質問とモデル設定を検証する。"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"設定ファイルを読み込めません: {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("設定ファイルのトップレベルは JSON オブジェクトにしてください")

    questions = data.get("questions")
    models = data.get("models")
    if not _non_empty_string_list(questions):
        raise ValueError("questions は空でない文字列の配列にしてください")
    if not isinstance(models, list) or not models:
        raise ValueError("models は空でない配列にしてください")
    model_specs = []
    for item in models:
        if isinstance(item, str) and item.strip():
            model_specs.append({"model": item, "thinking_level": None})
            continue
        if not isinstance(item, dict):
            raise ValueError(
                "models の各要素はモデル名の文字列または設定オブジェクトにしてください"
            )
        model = item.get("model")
        thinking_level = item.get("thinking_level")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("モデル設定の model は空でない文字列にしてください")
        if thinking_level not in (None, "minimal", "low", "medium", "high"):
            raise ValueError(
                "thinking_level は minimal、low、medium、high のいずれかにしてください"
            )
        model_specs.append(
            {"model": model, "thinking_level": thinking_level}
        )
    return questions, model_specs


def _non_empty_string_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(
        isinstance(item, str) and item.strip() for item in value
    )


def run_benchmark(
    questions: list[str],
    models: list[dict[str, Any]],
    *,
    pause_seconds: float = 0.0,
) -> dict[str, Any]:
    """質問とモデルの全組み合わせを実行し、比較可能な結果を返す。"""
    client = genai.Client(
        api_key=get_api_key(),
        http_options={"timeout": REQUEST_TIMEOUT_MS},
    )
    records: list[dict[str, Any]] = []
    request_index = 0

    for model_spec in models:
        model = model_spec["model"]
        thinking_level = model_spec["thinking_level"]
        for question in questions:
            request_index += 1
            started = time.perf_counter()
            record: dict[str, Any] = {
                "request_index": request_index,
                "model": model,
                "thinking_level": thinking_level,
                "question": question,
            }
            total_requests = len(models) * len(questions)
            print(
                f"[{request_index}/{total_requests}] {model}: リクエスト中...",
                file=sys.stderr,
                flush=True,
            )
            try:
                config = (
                    types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(
                            thinking_level=thinking_level
                        )
                    )
                    if thinking_level
                    else None
                )
                response = client.models.generate_content(
                    model=model,
                    contents=question,
                    config=config,
                )
                elapsed_ms = (time.perf_counter() - started) * 1000
                usage = response.usage_metadata
                record.update(
                    {
                        "status": "ok",
                        "elapsed_ms": round(elapsed_ms, 1),
                        "input_tokens": getattr(usage, "prompt_token_count", None),
                        "output_tokens": getattr(
                            usage, "candidates_token_count", None
                        ),
                        "thinking_tokens": getattr(
                            usage, "thoughts_token_count", None
                        ),
                        "response_chars": len(response.text or ""),
                        "response": response.text or "",
                    }
                )
            except errors.APIError as exc:
                record.update(
                    {
                        "status": "error",
                        "elapsed_ms": round(
                            (time.perf_counter() - started) * 1000, 1
                        ),
                        "error_code": exc.code,
                        "error_message": api_error_message(exc.code, exc.message),
                        "rate_limit_error": exc.code == 429,
                    }
                )
            records.append(record)
            status = "完了" if record["status"] == "ok" else "エラー"
            print(
                f"[{request_index}/{total_requests}] {status} "
                f"({record['elapsed_ms']} ms)",
                file=sys.stderr,
                flush=True,
            )
            if pause_seconds > 0 and request_index < len(models) * len(questions):
                time.sleep(pause_seconds)

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "question_count": len(questions),
        "model_count": len(models),
        "request_count": len(records),
        "rate_limit_errors": sum(
            record.get("rate_limit_error", False) for record in records
        ),
        "records": records,
        "model_summary": summarize_by_model(records),
        "question_summary": summarize_by_question(records),
    }


def summarize_by_model(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """成功した試行をモデルごとに集計する。"""
    models = list(
        dict.fromkeys(
            (record["model"], record.get("thinking_level")) for record in records
        )
    )
    summaries = []
    for model, thinking_level in models:
        model_records = [
            record
            for record in records
            if (record["model"], record.get("thinking_level"))
            == (model, thinking_level)
        ]
        successful = [record for record in model_records if record["status"] == "ok"]
        summaries.append(
            {
                "model": model,
                "thinking_level": thinking_level,
                "requests": len(model_records),
                "successes": len(successful),
                "errors": len(model_records) - len(successful),
                "rate_limit_errors": sum(
                    record.get("rate_limit_error", False) for record in model_records
                ),
                "average_elapsed_ms": _average(successful, "elapsed_ms"),
                "average_input_tokens": _average(successful, "input_tokens"),
                "average_output_tokens": _average(successful, "output_tokens"),
                "average_thinking_tokens": _average(successful, "thinking_tokens"),
                "average_response_chars": _average(successful, "response_chars"),
            }
        )
    return summaries


def summarize_by_question(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """成功した試行を質問ごとに集計する。"""
    questions = list(dict.fromkeys(record["question"] for record in records))
    summaries = []
    for question in questions:
        question_records = [
            record for record in records if record["question"] == question
        ]
        successful = [
            record for record in question_records if record["status"] == "ok"
        ]
        summaries.append(
            {
                "question": question,
                "requests": len(question_records),
                "successes": len(successful),
                "errors": len(question_records) - len(successful),
                "rate_limit_errors": sum(
                    record.get("rate_limit_error", False)
                    for record in question_records
                ),
                "average_elapsed_ms": _average(successful, "elapsed_ms"),
                "average_input_tokens": _average(successful, "input_tokens"),
                "average_output_tokens": _average(successful, "output_tokens"),
                "average_response_chars": _average(successful, "response_chars"),
            }
        )
    return summaries


def _average(records: list[dict[str, Any]], key: str) -> float | None:
    values = [record[key] for record in records if record.get(key) is not None]
    return round(sum(values) / len(values), 1) if values else None


def render_markdown(result: dict[str, Any]) -> str:
    """計測結果を質問別・モデル別に比較できる Markdown に変換する。"""
    lines = [
        "# ベンチマーク結果",
        "",
        f"- 実行日時: `{result['created_at']}`",
        f"- 質問数: {result['question_count']}",
        f"- モデル数: {result['model_count']}",
        f"- リクエスト数: {result['request_count']}",
        f"- レート制限（429）: {result['rate_limit_errors']} 件",
        "",
        "## モデル別比較",
        "",
        "| モデル | thinking | 成功/試行 | 平均応答時間 (ms) | 平均入力トークン | 平均出力トークン | 平均思考トークン | 平均応答文字数 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in result["model_summary"]:
        lines.append(
            "| {} | {} | {}/{} | {} | {} | {} | {} | {} |".format(
                _markdown_cell(summary["model"]),
                _markdown_cell(summary.get("thinking_level") or "default"),
                summary["successes"],
                summary["requests"],
                summary["average_elapsed_ms"],
                summary["average_input_tokens"],
                summary["average_output_tokens"],
                summary["average_thinking_tokens"],
                summary["average_response_chars"],
            )
        )

    lines.extend(
        [
            "",
            "## 質問別比較",
            "",
            "| 質問 | 成功/試行 | 平均応答時間 (ms) | 平均入力トークン | 平均出力トークン | 平均応答文字数 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for summary in result["question_summary"]:
        lines.append(
            "| {question} | {successes}/{requests} | "
            "{average_elapsed_ms} | {average_input_tokens} | "
            "{average_output_tokens} | {average_response_chars} |".format(
                **{
                    **summary,
                    "question": _markdown_cell(summary["question"]),
                }
            )
        )

    lines.extend(
        [
            "",
            "## 試行別詳細",
            "",
            "| # | モデル | thinking | 質問 | 結果 | 応答時間 (ms) | 入力トークン | 出力トークン | 思考トークン | 応答文字数 | エラー |",
            "|---:|---|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for record in result["records"]:
        if record["status"] == "ok":
            values = (
                "成功",
                record["elapsed_ms"],
                record["input_tokens"],
                record["output_tokens"],
                record.get("thinking_tokens", "-"),
                record["response_chars"],
                "",
            )
        else:
            values = (
                "失敗",
                record["elapsed_ms"],
                "-",
                "-",
                "-",
                "-",
                record["error_message"],
            )
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                record["request_index"],
                _markdown_cell(record["model"]),
                _markdown_cell(record.get("thinking_level") or "default"),
                _markdown_cell(record["question"]),
                *values,
            )
        )

    lines.extend(["", "## 回答本文", ""])
    for record in result["records"]:
        response = record.get("response", record.get("error_message", ""))
        fence = "```"
        while fence in response:
            fence += "`"
        lines.extend(
            [
                "",
                f"### 試行 {record['request_index']}: "
                f"{_markdown_cell(record['model'])} / "
                f"{_markdown_cell(record.get('thinking_level') or 'default')}",
                "",
                f"質問: {_markdown_cell(record['question'])}",
                "",
                "#### 回答",
                "",
                f"{fence}text",
                response,
                fence,
            ]
        )
    return "\n".join(lines) + "\n"


def _markdown_cell(value: Any) -> str:
    """Markdown 表のセルを壊さないように整形する。"""
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def main() -> None:
    parser = argparse.ArgumentParser(description="質問・モデル別の応答性能を比較する")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=DEFAULT_MARKDOWN_OUTPUT,
        help="比較用 Markdown レポートの保存先",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.0,
        help="リクエスト間に入れる待ち時間（秒）。レート制限対策に使う",
    )
    args = parser.parse_args()
    if args.pause_seconds < 0:
        parser.error("--pause-seconds は 0 以上にしてください")

    try:
        questions, models = load_config(args.config)
        result = run_benchmark(
            questions,
            models,
            pause_seconds=args.pause_seconds,
        )
    except ValueError as exc:
        parser.error(str(exc))
    except errors.APIError as exc:
        parser.error(api_error_message(exc.code, exc.message))

    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(
        render_markdown(result),
        encoding="utf-8",
    )
    print(f"Markdown結果を保存しました: {args.markdown_output}")


if __name__ == "__main__":
    main()
