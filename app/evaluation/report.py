"""Machine-readable report output and concise terminal summaries."""

from pathlib import Path

from app.evaluation.models import EvaluationReport


def write_json_report(report: EvaluationReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")


def format_summary(report: EvaluationReport) -> str:
    metrics = report.aggregate

    def display(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    lines = [
        f"mode={report.mode} cases={metrics.case_count} ",
        f"hit@k={display(metrics.retrieval_hit_rate_at_k)} ",
        f"recall@k={display(metrics.mean_recall_at_k)} ",
        f"mrr={display(metrics.mean_reciprocal_rank)}",
    ]
    if report.mode == "rag":
        lines.extend(
            [
                f" citation_presence={display(metrics.citation_presence_rate)}",
                f" citation_validity={display(metrics.citation_validity_rate)}",
                f" citation_relevance={display(metrics.citation_relevance_rate)}",
                f" abstention_accuracy={display(metrics.abstention_accuracy)}",
                f" reference_token_f1={display(metrics.mean_reference_token_f1)}",
            ]
        )
    return "".join(lines)
