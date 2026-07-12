from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional

from .dataset import prepare_dataset
from .evaluation import (
    build_report,
    compare_reports,
    run_qa_judge_run,
    run_retrieval_run,
)
from .extractor import extract_run
from .extractor_v1 import extract_run_v1
from .graph_store import build_run_graphs
from .llm_client import OpenAICompatibleClient
from .query_parser_v2 import parse_run


def split_csv(value: str) -> List[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def client_from_args(args: argparse.Namespace, prefix: str = "") -> Optional[OpenAICompatibleClient]:
    key = f"{prefix}_" if prefix else ""
    base_url = getattr(args, key + "base_url", "") or os.environ.get("OPENAI_BASE_URL", "")
    api_key = getattr(args, key + "api_key", "") or os.environ.get("OPENAI_API_KEY", "")
    model = getattr(args, key + "model", "") or os.environ.get("MODEL_NAME", "")
    if not (base_url and api_key and model):
        return None
    return OpenAICompatibleClient(
        base_url=base_url,
        api_key=api_key,
        model_name=model,
        timeout=int(getattr(args, "timeout", 300)),
        retries=int(getattr(args, "retries", 3)),
    )


def add_common_client(parser: argparse.ArgumentParser, prefix: str = "") -> None:
    label = f"{prefix}-" if prefix else ""
    dest = f"{prefix}_" if prefix else ""
    parser.add_argument(f"--{label}base-url", dest=dest + "base_url", default="")
    parser.add_argument(f"--{label}api-key", dest=dest + "api_key", default="")
    parser.add_argument(f"--{label}model", dest=dest + "model", default="")


def add_run_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--question-types", default="", help="comma-separated LongMemEval types")
    parser.add_argument("--force", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="From-scratch LongMemEval validation for enhanced DimMem graph memory."
    )
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=3)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="raw LongMemEval -> overlap-aware sample windows")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--output-root", required=True)
    prepare.add_argument("--run-name", required=True)
    prepare.add_argument("--window-size", type=int, default=15)
    prepare.add_argument("--overlap", type=int, default=3)
    prepare.add_argument("--truncate-threshold", type=int, default=8000)
    prepare.add_argument("--max-items", type=int, default=0)
    add_run_filters(prepare)

    extract = sub.add_parser("extract", help="extract enhanced dimensional memories")
    extract.add_argument("--run-root", required=True)
    extract.add_argument("--heuristic", action="store_true", help="smoke test only")
    extract.add_argument("--schema", choices=["v1", "v2", "both"], default="v2")
    extract.add_argument("--max-tokens", type=int, default=5000)
    add_common_client(extract)
    add_run_filters(extract)

    graph = sub.add_parser("build-graph", help="build typed graph and update chains")
    graph.add_argument("--run-root", required=True)
    add_run_filters(graph)

    query = sub.add_parser("parse-query", help="parse query into revisable dimensional hypotheses")
    query.add_argument("--run-root", required=True)
    query.add_argument("--heuristic", action="store_true", help="smoke test only")
    add_common_client(query)
    add_run_filters(query)

    retrieval = sub.add_parser("retrieve", help="run one retrieval variant")
    retrieval.add_argument("--run-root", required=True)
    retrieval.add_argument("--mode", required=True, choices=["dimmem_v1", "dimension_v2", "graph_static", "graph_active"])
    retrieval.add_argument("--output-name", required=True)
    retrieval.add_argument("--route-k", type=int, default=20)
    retrieval.add_argument("--initial-k", type=int, default=12)
    retrieval.add_argument("--final-k", type=int, default=15)
    retrieval.add_argument("--max-rounds", type=int, default=3)
    retrieval.add_argument("--router-mode", choices=["llm", "heuristic"], default="llm")
    retrieval.add_argument("--embedder", choices=["hash", "sentence-transformers"], default="hash")
    retrieval.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    retrieval.add_argument("--device", default="cpu")
    add_common_client(retrieval, "active")
    add_run_filters(retrieval)

    qa = sub.add_parser("qa-judge", help="answer and judge a retrieval run")
    qa.add_argument("--run-root", required=True)
    qa.add_argument("--retrieval-name", required=True)
    qa.add_argument("--heuristic-qa", action="store_true", help="smoke test only")
    qa.add_argument("--exact-judge", action="store_true", help="smoke test only")
    add_common_client(qa, "qa")
    add_common_client(qa, "judge")
    add_run_filters(qa)

    report = sub.add_parser("report")
    report.add_argument("--run-root", required=True)
    report.add_argument("--retrieval-name", required=True)

    compare = sub.add_parser("compare")
    compare.add_argument("--run-root", required=True)
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)

    all_cmd = sub.add_parser("all", help="raw dataset -> extraction -> graph -> query -> retrieval -> QA/judge")
    all_cmd.add_argument("--input", required=True)
    all_cmd.add_argument("--output-root", required=True)
    all_cmd.add_argument("--run-name", required=True)
    all_cmd.add_argument("--window-size", type=int, default=15)
    all_cmd.add_argument("--overlap", type=int, default=3)
    all_cmd.add_argument("--truncate-threshold", type=int, default=8000)
    all_cmd.add_argument("--max-items", type=int, default=0)
    all_cmd.add_argument("--mode", choices=["dimmem_v1", "dimension_v2", "graph_static", "graph_active"], default="graph_active")
    all_cmd.add_argument("--output-name", default="graph_active")
    all_cmd.add_argument("--route-k", type=int, default=20)
    all_cmd.add_argument("--initial-k", type=int, default=12)
    all_cmd.add_argument("--final-k", type=int, default=15)
    all_cmd.add_argument("--max-rounds", type=int, default=3)
    all_cmd.add_argument("--router-mode", choices=["llm", "heuristic"], default="llm")
    all_cmd.add_argument("--embedder", choices=["hash", "sentence-transformers"], default="hash")
    all_cmd.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    all_cmd.add_argument("--device", default="cpu")
    all_cmd.add_argument("--smoke-test", action="store_true", help="use heuristic extraction/query/QA and exact judge")
    add_common_client(all_cmd, "extract")
    add_common_client(all_cmd, "query")
    add_common_client(all_cmd, "active")
    add_common_client(all_cmd, "qa")
    add_common_client(all_cmd, "judge")
    add_run_filters(all_cmd)

    ablation = sub.add_parser("ablation", help="run four controlled variants on one prepared run")
    ablation.add_argument("--run-root", required=True)
    ablation.add_argument("--route-k", type=int, default=20)
    ablation.add_argument("--initial-k", type=int, default=12)
    ablation.add_argument("--final-k", type=int, default=15)
    ablation.add_argument("--max-rounds", type=int, default=3)
    ablation.add_argument("--router-mode", choices=["llm", "heuristic"], default="llm")
    ablation.add_argument("--embedder", choices=["hash", "sentence-transformers"], default="hash")
    ablation.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ablation.add_argument("--device", default="cpu")
    ablation.add_argument("--heuristic-qa", action="store_true")
    ablation.add_argument("--exact-judge", action="store_true")
    add_common_client(ablation, "active")
    add_common_client(ablation, "qa")
    add_common_client(ablation, "judge")
    add_run_filters(ablation)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    qtypes = split_csv(getattr(args, "question_types", ""))

    if args.command == "prepare":
        result = prepare_dataset(
            input_path=args.input,
            output_root=args.output_root,
            run_name=args.run_name,
            window_size=args.window_size,
            overlap=args.overlap,
            truncate_threshold=args.truncate_threshold,
            max_items=args.max_items,
            question_types=qtypes,
            force=args.force,
        )
    elif args.command == "extract":
        client = client_from_args(args)
        outputs = {}
        if args.schema in {"v1", "both"}:
            outputs["v1"] = extract_run_v1(
                args.run_root, client=client, force=args.force, heuristic=args.heuristic,
                max_tokens=min(args.max_tokens, 4200), question_types=qtypes,
            )
        if args.schema in {"v2", "both"}:
            outputs["v2"] = extract_run(
                args.run_root, client=client, force=args.force, heuristic=args.heuristic,
                max_tokens=args.max_tokens, question_types=qtypes,
            )
        result = outputs
    elif args.command == "build-graph":
        result = build_run_graphs(args.run_root, force=args.force, question_types=qtypes)
    elif args.command == "parse-query":
        result = parse_run(
            args.run_root,
            client=client_from_args(args),
            heuristic=args.heuristic,
            force=args.force,
            question_types=qtypes,
        )
    elif args.command == "retrieve":
        result = run_retrieval_run(
            args.run_root,
            mode=args.mode,
            output_name=args.output_name,
            route_k=args.route_k,
            initial_k=args.initial_k,
            final_k=args.final_k,
            max_rounds=args.max_rounds,
            active_client=client_from_args(args, "active"),
            router_mode=args.router_mode,
            embedder_kind=args.embedder,
            embedding_model=args.embedding_model,
            device=args.device,
            force=args.force,
            question_types=qtypes,
        )
    elif args.command == "qa-judge":
        result = run_qa_judge_run(
            args.run_root,
            retrieval_name=args.retrieval_name,
            qa_client=client_from_args(args, "qa"),
            judge_client=client_from_args(args, "judge"),
            heuristic_qa=args.heuristic_qa,
            exact_judge_mode=args.exact_judge,
            force=args.force,
            question_types=qtypes,
        )
    elif args.command == "report":
        result = build_report(args.run_root, args.retrieval_name)
    elif args.command == "compare":
        result = compare_reports(args.run_root, args.baseline, args.candidate)
    elif args.command == "all":
        prepared = prepare_dataset(
            input_path=args.input,
            output_root=args.output_root,
            run_name=args.run_name,
            window_size=args.window_size,
            overlap=args.overlap,
            truncate_threshold=args.truncate_threshold,
            max_items=args.max_items,
            question_types=qtypes,
            force=args.force,
        )
        run_root = prepared["run_root"]
        extraction_client = client_from_args(args, "extract")
        extract_run_v1(
            run_root, client=extraction_client, force=args.force,
            heuristic=args.smoke_test, question_types=qtypes,
        )
        extract_run(
            run_root, client=extraction_client, force=args.force,
            heuristic=args.smoke_test, question_types=qtypes,
        )
        build_run_graphs(run_root, force=args.force, question_types=qtypes)
        parse_run(
            run_root,
            client=client_from_args(args, "query"),
            heuristic=args.smoke_test,
            force=args.force,
            question_types=qtypes,
        )
        run_retrieval_run(
            run_root,
            mode=args.mode,
            output_name=args.output_name,
            route_k=args.route_k,
            initial_k=args.initial_k,
            final_k=args.final_k,
            max_rounds=args.max_rounds,
            active_client=client_from_args(args, "active"),
            router_mode="heuristic" if args.smoke_test else args.router_mode,
            embedder_kind=args.embedder,
            embedding_model=args.embedding_model,
            device=args.device,
            force=args.force,
            question_types=qtypes,
        )
        run_qa_judge_run(
            run_root,
            retrieval_name=args.output_name,
            qa_client=client_from_args(args, "qa"),
            judge_client=client_from_args(args, "judge"),
            heuristic_qa=args.smoke_test,
            exact_judge_mode=args.smoke_test,
            force=args.force,
            question_types=qtypes,
        )
        result = build_report(run_root, args.output_name)
    elif args.command == "ablation":
        active_client = client_from_args(args, "active")
        qa_client = client_from_args(args, "qa")
        judge_client = client_from_args(args, "judge")
        variants = [
            ("dimmem_v1", "dimmem_v1"),
            ("dimension_v2", "dimension_v2"),
            ("graph_static", "graph_static"),
            ("graph_active", "graph_active"),
        ]
        reports = {}
        for mode, output_name in variants:
            run_retrieval_run(
                args.run_root,
                mode=mode,
                output_name=output_name,
                route_k=args.route_k,
                initial_k=args.initial_k,
                final_k=args.final_k,
                max_rounds=args.max_rounds,
                active_client=active_client,
                router_mode=args.router_mode,
                embedder_kind=args.embedder,
                embedding_model=args.embedding_model,
                device=args.device,
                force=args.force,
                question_types=qtypes,
            )
            run_qa_judge_run(
                args.run_root,
                retrieval_name=output_name,
                qa_client=qa_client,
                judge_client=judge_client,
                heuristic_qa=args.heuristic_qa,
                exact_judge_mode=args.exact_judge,
                force=args.force,
                question_types=qtypes,
            )
            reports[output_name] = build_report(args.run_root, output_name)
        comparisons = {
            "v1_vs_v2": compare_reports(args.run_root, "dimmem_v1", "dimension_v2"),
            "v2_vs_static_graph": compare_reports(args.run_root, "dimension_v2", "graph_static"),
            "static_vs_active_graph": compare_reports(args.run_root, "graph_static", "graph_active"),
            "v1_vs_active_graph": compare_reports(args.run_root, "dimmem_v1", "graph_active"),
        }
        result = {"reports": reports, "comparisons": comparisons}
    else:
        parser.error("unhandled command")
        return 2

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
