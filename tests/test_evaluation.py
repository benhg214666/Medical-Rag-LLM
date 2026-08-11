"""Phase 6 deterministic evaluation tests; no model server or GPU required."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.evaluation.answer_metrics import (
    citation_relevance,
    citation_validity,
    extract_citations,
    reference_token_f1,
)
from app.evaluation.cli import build_parser, main
from app.evaluation.dataset import EvaluationDatasetError, load_jsonl_dataset
from app.evaluation.evaluator import Evaluator, aggregate_results
from app.evaluation.models import (
    EvaluationCase,
    EvaluationCaseResult,
    RetrievalMetrics,
)
from app.evaluation.report import format_summary, write_json_report
from app.rag.constants import INSUFFICIENT_CONTEXT_ANSWER
from app.rag.models import RAGAnswer, RAGSource
from app.retrieval.models import RetrievalResult


def case(**overrides) -> EvaluationCase:
    values = {
        "id": "case-1",
        "query": "What medication is documented?",
        "answerable": True,
        "top_k": 5,
        "expected_chunk_ids": ["c-relevant"],
        "reference_answer": "Patient takes metformin.",
    }
    values.update(overrides)
    return EvaluationCase(**values)


def result(chunk_id: str, document_id: str = "doc-1") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=document_id,
        text="Synthetic medical record text.",
        distance=0.1,
        score=0.9,
        distance_metric="cosine",
        metadata={"source": "synthetic.txt"},
    )


def source(number: int, chunk_id: str, document_id: str = "doc-1") -> RAGSource:
    return RAGSource(
        source_number=number,
        **result(chunk_id, document_id).model_dump(),
    )


class FakeRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, int | None]] = []

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        self.calls.append((query, top_k))
        return self.results[:top_k]


class FakeRAG:
    def __init__(self, answer: RAGAnswer) -> None:
        self.result = answer
        self.calls: list[tuple[str, int | None]] = []

    def answer(self, query: str, top_k: int | None = None) -> RAGAnswer:
        self.calls.append((query, top_k))
        return self.result


def write_cases(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


class TestDataset:
    def test_valid_jsonl_loads(self, tmp_path: Path) -> None:
        path = tmp_path / "cases.jsonl"
        write_cases(path, [case().model_dump()])
        loaded = load_jsonl_dataset(path)
        assert loaded[0].id == "case-1"
        assert loaded[0].answerable is True

    def test_duplicate_ids_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "cases.jsonl"
        row = case().model_dump()
        write_cases(path, [row, row])
        with pytest.raises(EvaluationDatasetError, match="duplicate"):
            load_jsonl_dataset(path)

    @pytest.mark.parametrize("query", ["", "   ", "\n"])
    def test_blank_query_rejected(self, query: str) -> None:
        with pytest.raises(ValidationError):
            case(query=query)

    @pytest.mark.parametrize("top_k", [0, -1, 51, True, "5"])
    def test_invalid_top_k_rejected(self, top_k) -> None:
        with pytest.raises(ValidationError):
            case(top_k=top_k)


class TestRetrievalMetrics:
    def evaluate(self, evaluation_case: EvaluationCase, results: list[RetrievalResult]):
        return Evaluator(retriever=FakeRetriever(results)).evaluate_dataset(
            [evaluation_case], "retrieval"
        ).cases[0].retrieval

    def test_hit_recall_and_rank_one(self) -> None:
        metrics = self.evaluate(
            case(expected_chunk_ids=["c-relevant", "c-other"]),
            [result("c-relevant"), result("noise")],
        )
        assert metrics.hit_at_k == 1.0
        assert metrics.recall_at_k == 0.5
        assert metrics.reciprocal_rank == 1.0

    def test_later_rank_and_absent(self) -> None:
        later = self.evaluate(case(), [result("noise"), result("c-relevant")])
        absent = self.evaluate(case(), [result("noise")])
        assert later.reciprocal_rank == 0.5
        assert absent.hit_at_k == 0.0
        assert absent.recall_at_k == 0.0
        assert absent.reciprocal_rank == 0.0

    def test_document_matching_counts_document_once(self) -> None:
        metrics = self.evaluate(
            case(expected_chunk_ids=[], expected_document_ids=["doc-a", "doc-b"]),
            [result("a1", "doc-a"), result("a2", "doc-a")],
        )
        assert metrics.hit_at_k == 1.0
        assert metrics.recall_at_k == 0.5

    def test_chunk_labels_take_precedence_without_double_counting(self) -> None:
        metrics = self.evaluate(
            case(expected_chunk_ids=["wanted"], expected_document_ids=["doc-noise"]),
            [result("noise", "doc-noise")],
        )
        assert metrics.hit_at_k == 0.0

    def test_top_k_is_respected(self) -> None:
        metrics = self.evaluate(
            case(top_k=1), [result("noise"), result("c-relevant")]
        )
        assert metrics.hit_at_k == 0.0


class TestAnswerMetrics:
    def test_citation_extraction_and_validity(self) -> None:
        citations = extract_citations("Fact [1], repeated [1], plus [2] and [10].")
        assert citations == [1, 2, 10]
        assert citation_validity(citations, [source(1, "a"), source(2, "b")]) == pytest.approx(2 / 3)

    def test_citation_relevance_maps_returned_source_number(self) -> None:
        sources = [source(1, "noise"), source(2, "c-relevant")]
        assert citation_relevance(case(), [2], sources) == 1.0
        assert citation_relevance(case(), [1, 3], sources) == 0.0

    def test_citation_checks_use_explicit_source_numbers(self) -> None:
        sources = [source(7, "c-relevant"), source(2, "noise")]
        assert citation_validity([7, 1], sources) == 0.5
        assert citation_relevance(case(), [7], sources) == 1.0

    def test_document_citation_relevance(self) -> None:
        evaluation_case = case(
            expected_chunk_ids=[], expected_document_ids=["doc-relevant"]
        )
        assert citation_relevance(
            evaluation_case, [1], [source(1, "c", "doc-relevant")]
        ) == 1.0

    def test_token_f1_and_duplicate_counts(self) -> None:
        assert reference_token_f1("Metformin 500 MG", "metformin 500 mg") == 1.0
        # overlap is two tokens, not three: precision=2/3, recall=1.
        assert reference_token_f1("drug drug drug", "drug drug") == pytest.approx(0.8)
        assert reference_token_f1("alpha", "beta") == 0.0

    def test_cjk_character_tokens_produce_meaningful_overlap(self) -> None:
        assert reference_token_f1(
            "患者目前服用二甲雙胍500毫克。", "二甲雙胍500毫克"
        ) > 0.0
        assert reference_token_f1("高血壓", "糖尿病") == 0.0


class TestEvaluator:
    def test_runner_rejects_empty_cases_and_unknown_mode_before_services(self) -> None:
        rag = FakeRAG(RAGAnswer(answer="must not run", model="fake", sources=[]))
        evaluator = Evaluator(rag_service=rag)
        with pytest.raises(ValueError, match="no cases"):
            evaluator.evaluate_dataset([], "rag")
        with pytest.raises(ValueError, match="unsupported"):
            evaluator.evaluate_dataset([case()], "invalid")  # type: ignore[arg-type]
        assert rag.calls == []

    def test_retrieval_only_never_calls_rag(self) -> None:
        retriever = FakeRetriever([result("c-relevant")])
        rag = FakeRAG(RAGAnswer(answer="must not run", model="fake", sources=[]))
        report = Evaluator(retriever=retriever, rag_service=rag).evaluate_dataset(
            [case()], "retrieval"
        )
        assert report.cases[0].retrieval.hit_at_k == 1.0
        assert rag.calls == []

    def test_rag_mode_reuses_returned_sources_and_metrics(self) -> None:
        rag = FakeRAG(
            RAGAnswer(
                answer="The patient takes metformin [2].",
                model="fake",
                sources=[source(1, "noise"), source(2, "c-relevant")],
            )
        )
        report = Evaluator(rag_service=rag).evaluate_dataset([case()], "rag")
        evaluated = report.cases[0]
        assert rag.calls == [(case().query, 5)]
        assert evaluated.retrieval.hit_at_k == 1.0
        assert evaluated.answer is not None
        assert evaluated.answer.citation_presence is True
        assert evaluated.answer.citation_validity == 1.0
        assert evaluated.answer.citation_relevance == 1.0
        assert evaluated.answer.reference_token_f1 is not None

    @pytest.mark.parametrize(
        "rag_answer,expected",
        [
            (
                RAGAnswer(
                    answer=INSUFFICIENT_CONTEXT_ANSWER, model="fake", sources=[]
                ),
                True,
            ),
            (
                RAGAnswer(
                    answer=f"  {INSUFFICIENT_CONTEXT_ANSWER.upper()}  ",
                    model="fake",
                    sources=[source(1, "unrelated")],
                ),
                True,
            ),
            (
                RAGAnswer(
                    answer="The patient has diabetes [1].",
                    model="fake",
                    sources=[source(1, "unrelated")],
                ),
                False,
            ),
        ],
    )
    def test_unanswerable_abstention_contract(
        self, rag_answer: RAGAnswer, expected: bool
    ) -> None:
        evaluation_case = case(
            answerable=False, expected_chunk_ids=[], reference_answer=None
        )
        answer = Evaluator(rag_service=FakeRAG(rag_answer)).evaluate_dataset(
            [evaluation_case], "rag"
        ).cases[0].answer
        assert answer is not None
        assert answer.abstained is expected
        assert answer.citation_presence is None

    def test_abstention_aggregate_uses_only_unanswerable_cases(self) -> None:
        unanswerable = case(
            id="no-answer", answerable=False, expected_chunk_ids=[], reference_answer=None
        )
        answerable = case(id="has-answer")
        rag = FakeRAG(
            RAGAnswer(answer=INSUFFICIENT_CONTEXT_ANSWER, model="fake", sources=[])
        )
        aggregate = Evaluator(rag_service=rag).evaluate_dataset(
            [unanswerable, answerable], "rag"
        ).aggregate
        assert aggregate.abstention_applicable_count == 1
        assert aggregate.abstention_accuracy == 1.0

    def test_aggregate_ignores_non_applicable_values(self) -> None:
        results = [
            EvaluationCaseResult(
                case_id="labeled", query="q", answerable=True, top_k=5,
                retrieval=RetrievalMetrics(hit_at_k=1, recall_at_k=0.5, reciprocal_rank=1),
            ),
            EvaluationCaseResult(
                case_id="unlabeled", query="q", answerable=False, top_k=5,
                retrieval=RetrievalMetrics(),
            ),
        ]
        aggregate = aggregate_results(results)
        assert aggregate.retrieval_applicable_count == 1
        assert aggregate.retrieval_hit_rate_at_k == 1.0
        assert aggregate.mean_recall_at_k == 0.5

    def test_json_report_serialization(self, tmp_path: Path) -> None:
        report = Evaluator(retriever=FakeRetriever([result("c-relevant")])).evaluate_dataset(
            [case()], "retrieval"
        )
        output = tmp_path / "nested" / "report.json"
        write_json_report(report, output)
        loaded = json.loads(output.read_text(encoding="utf-8"))
        assert loaded["cases"][0]["case_id"] == "case-1"
        assert "hit@k=1.000" in format_summary(report)


class TestCLI:
    def test_parser_rejects_invalid_mode(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--dataset", "cases.jsonl", "--mode", "bad"])

    def test_missing_dataset_returns_nonzero(self, tmp_path: Path, capsys) -> None:
        exit_code = main(
            ["--dataset", str(tmp_path / "missing.jsonl"), "--mode", "retrieval"]
        )
        assert exit_code == 1
        assert "does not exist" in capsys.readouterr().err
