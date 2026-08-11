"""Phase 8 JSON and Markdown evaluation artifact writers."""

import json
from pathlib import Path


def _metric(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def render_markdown_summary(artifact: dict) -> str:
    totals = artifact["totals"]
    aggregate = artifact["aggregate_metrics"]
    quality_gate = artifact["quality_gate"]
    lines = [
        "# Medical RAG demo evaluation",
        "",
        "> Synthetic/de-identified data only — not for clinical use.",
        "",
        f"- Generated: {artifact['generated_at']}",
        f"- Model: `{artifact['environment']['model']}`",
        f"- Collection: `{artifact['environment']['collection']}`",
        f"- Cases: {totals['case_count']} (pass {totals['pass_count']}, fail {totals['fail_count']}, error {totals['error_count']})",
        f"- Hit@K: {_metric(aggregate['retrieval_hit_rate_at_k'])}",
        f"- Recall@K: {_metric(aggregate['mean_recall_at_k'])}",
        f"- MRR: {_metric(aggregate['mean_reciprocal_rank'])}",
        f"- Citation validity: {_metric(aggregate['citation_validity_rate'])}",
        f"- Reference-token-F1 demo gate: >= {quality_gate['reference_token_f1_min']:.2f} when a reference answer is present",
        "",
        "The demo pass gate combines expected retrieval evidence, citation presence, "
        "citation validity, citation relevance, and the lexical F1 gate above. "
        "Reference token F1 measures lexical overlap only; it is not semantic "
        "equivalence, medical correctness, or clinical safety validation.",
        "",
        "## Cases",
        "",
        "| Case | Status | Ref F1 | Answer | Sources |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in artifact["cases"]:
        answer = (row["answer"] or row["error"] or "").replace("|", "\\|").replace("\n", " ")
        sources = ", ".join(source["chunk_id"] for source in row["sources"]) or "—"
        reference_f1 = row["quality_gate"].get("reference_token_f1")
        lines.append(
            f"| {row['case_id']} | {row['status']} | {_metric(reference_f1)} | "
            f"{answer} | {sources} |"
        )
    lines.extend(["", "## Known limitations", ""])
    lines.extend(f"- {item}" for item in artifact["limitations"])
    return "\n".join(lines) + "\n"


def write_evaluation_artifacts(artifact: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = artifact["generated_at"].replace("-", "").replace(":", "").replace("+00:00", "Z")
    json_path = output_dir / f"evaluation-{run_id}.json"
    markdown_path = output_dir / f"evaluation-{run_id}.md"
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError(f"evaluation artifacts already exist for run {run_id}")
    json_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown_summary(artifact), encoding="utf-8")
    return json_path, markdown_path
