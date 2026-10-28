"""Command line interface for the real-kana bank pipeline.

Examples::

    python -m manga_ocr_dev.real_kana harvest --limit 2000
    python -m manga_ocr_dev.real_kana segment --limit 200
    python -m manga_ocr_dev.real_kana clean
    python -m manga_ocr_dev.real_kana upscale --scale 2
    python -m manga_ocr_dev.real_kana verify --limit 50
    python -m manga_ocr_dev.real_kana upright
    python -m manga_ocr_dev.real_kana build
    python -m manga_ocr_dev.real_kana audit
    python -m manga_ocr_dev.real_kana render-demo --text おはよう
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from manga_ocr_dev.real_kana.bank import (
    KanaBank,
    audit_bank,
    build_manifest,
    export_review_sheets,
)
from manga_ocr_dev.real_kana.clean import clean_candidates
from manga_ocr_dev.real_kana.config import DEFAULT_ROOT, KanaConfig
from manga_ocr_dev.real_kana.dataset import harvest_candidates
from manga_ocr_dev.real_kana.handwriting import train_head, write_heuristic_labels
from manga_ocr_dev.real_kana.segment import review_candidates, segment_candidates, verify_candidates
from manga_ocr_dev.real_kana.upright import upright_candidates
from manga_ocr_dev.real_kana.upscale import upscale_candidates


def _cfg(args: argparse.Namespace) -> KanaConfig:
    cfg = KanaConfig(root=Path(getattr(args, "root", None) or DEFAULT_ROOT).expanduser())
    if getattr(args, "model", None):
        cfg.vision_model = args.model
    if getattr(args, "provider", None):
        cfg.vision_provider = args.provider
    if getattr(args, "upscale_scale", None) is not None:
        cfg.upscale = float(args.upscale_scale)
    if getattr(args, "upscale_min_side", None) is not None:
        cfg.upscale_min_side = int(args.upscale_min_side)
    if getattr(args, "stroke_tolerance", None) is not None:
        cfg.stroke_tolerance = float(args.stroke_tolerance)
    if getattr(args, "local_model", None):
        cfg.local_model_dir = str(args.local_model)
    if getattr(args, "local_backend", None):
        cfg.local_backend = str(args.local_backend)
    if getattr(args, "local_min_confidence", None) is not None:
        cfg.local_min_confidence = float(args.local_min_confidence)
    if getattr(args, "local_threads", None) is not None:
        cfg.local_threads = int(args.local_threads)
    if getattr(args, "verify_confidence", None) is not None:
        cfg.verify_confidence = float(args.verify_confidence)
    if getattr(args, "mercury_model", None):
        cfg.mercury_model = str(args.mercury_model)
    if getattr(args, "mercury_url", None):
        cfg.mercury_url = str(args.mercury_url)
    if getattr(args, "mercury_timeout", None) is not None:
        cfg.mercury_timeout_s = float(args.mercury_timeout)
    if getattr(args, "mercury_retries", None) is not None:
        cfg.mercury_retries = int(args.mercury_retries)
    if getattr(args, "mercury_max_side", None) is not None:
        cfg.mercury_max_side = int(args.mercury_max_side)
    cfg.validate()
    return cfg


def _vision(cfg: KanaConfig, args: argparse.Namespace):
    from manga_ocr_dev.real_kana.vision import build_vision

    kwargs: dict[str, Any] = {"provider": cfg.vision_provider, "model": cfg.vision_model}
    if getattr(args, "api_key", None):
        kwargs["api_key"] = args.api_key
    if getattr(args, "base_url", None):
        kwargs["base_url"] = args.base_url
    return build_vision(cfg, use_cache=not getattr(args, "no_cache", False), **kwargs)


def _mercury(cfg: KanaConfig, args: argparse.Namespace):
    from manga_ocr_dev.real_kana.mercury import MercuryDecisionClient

    return MercuryDecisionClient(
        api_key=getattr(args, "openrouter_api_key", None),
        model=cfg.mercury_model,
        url=cfg.mercury_url,
        timeout_s=cfg.mercury_timeout_s,
        retries=cfg.mercury_retries,
        max_side=cfg.mercury_max_side,
        cache_dir=None if getattr(args, "no_cache", False) else cfg.root / "vision_cache" / "mercury",
    )


def cmd_harvest(args: argparse.Namespace) -> None:
    cfg = _cfg(args)
    backend = getattr(args, "handwriting_backend", "auto")
    mercury_client = _mercury(cfg, args) if backend == "mercury" else None
    try:
        stats = harvest_candidates(
            cfg,
            limit=args.limit,
            max_scan=args.max_scan,
            score_threshold=args.score_threshold,
            head_path=args.head,
            handwriting_backend=backend,
            mercury_client=mercury_client,
            resume=not args.no_resume,
        )
    finally:
        if mercury_client is not None:
            mercury_client.close()
    print(json.dumps(stats.as_dict(), ensure_ascii=False, indent=2))


def cmd_label_handwriting(args: argparse.Namespace) -> None:
    from manga_ocr_dev.real_kana.handwriting import heuristic_handwriting_score
    from manga_ocr_dev.real_kana.utils import iter_jsonl, load_image

    cfg = _cfg(args)
    backend = "heuristic" if getattr(args, "heuristic_only", False) else getattr(args, "backend", "vision")
    threshold = 0.5 if getattr(args, "threshold", None) is None else float(args.threshold)
    client = None
    if backend == "vision":
        from manga_ocr_dev.real_kana.vision import label_handwriting

        vision_client = _vision(cfg, args)
    elif backend == "mercury":
        client = _mercury(cfg, args)
    elif backend != "heuristic":
        raise SystemExit(f"unknown handwriting backend {backend!r}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, int] = {}
    if out.is_file() and not args.overwrite:
        for line in out.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("path,"):
                continue
            path, _, raw_label = line.rpartition(",")
            try:
                existing[path.strip()] = int(raw_label)
            except ValueError:
                continue
    rows: list[tuple[str, int]] = []
    done = 0
    try:
        for rec in iter_jsonl(cfg.root / "candidates" / "candidates.jsonl"):
            path = str(cfg.root / rec["image_path"])
            if path in existing:
                rows.append((path, existing[path]))
                continue
            if args.limit is not None and done >= args.limit:
                break
            image = load_image(path, "RGB")
            if backend == "heuristic":
                score = heuristic_handwriting_score(image)
                result = {
                    "handwritten": score >= threshold,
                    "confidence": score,
                    "reason": "heuristic",
                }
            elif backend == "mercury":
                result = client.label(image, threshold=threshold)
            else:
                result = label_handwriting(vision_client, image)
            rows.append((path, int(bool(result["handwritten"]))))
            done += 1
    finally:
        if client is not None:
            client.close()
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("path,label\n")
        for path, label in rows:
            fh.write(f"{path},{label}\n")
    print(f"wrote {len(rows)} labels to {out} ({done} newly labelled with {backend})")


def cmd_heuristic_labels(args: argparse.Namespace) -> None:
    cfg = _cfg(args)
    n = write_heuristic_labels(cfg.root / "candidates" / "candidates.jsonl", args.output, root=cfg.root)
    print(f"wrote {n} heuristic labels to {args.output}")


def cmd_train_head(args: argparse.Namespace) -> None:
    result = train_head(
        args.labels,
        args.output,
        model_id=args.dino_model,
        threshold=args.threshold,
        embedding_cache=args.embedding_cache,
    )
    print(json.dumps(result, indent=2))


def cmd_train_cnn(args: argparse.Namespace) -> None:
    from manga_ocr_dev.real_kana.cnn_filter import train_cnn

    result = train_cnn(
        args.labels,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        workers=args.workers,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_filter_cnn(args: argparse.Namespace) -> None:
    from manga_ocr_dev.real_kana.cnn_filter import (
        mercury_refine_uncertain,
        prune_unselected_stages,
        score_candidates,
        write_selected_bank,
    )
    from manga_ocr_dev.real_kana.utils import write_jsonl

    cfg = _cfg(args)
    out_root = Path(args.output_root).expanduser()
    if out_root.resolve() == cfg.root.resolve():
        raise SystemExit("--output-root must differ from --root")
    scored = score_candidates(
        cfg.root / "candidates" / "candidates.jsonl",
        args.checkpoint,
        batch_size=args.batch_size,
        workers=args.workers,
        cache_path=cfg.root / "candidates" / "cnn_score_cache.jsonl",
    )
    refine = None
    if getattr(args, "mercury_uncertain", False):
        limit = args.limit if args.limit is not None else 100_000
        ranked = sorted(
            (float(row.get("cnn_score", 0.0)) for row in scored), reverse=True
        )
        k = min(max(0, int(limit) - 1), max(0, len(ranked) - 1))
        cutoff = ranked[k] if ranked else 0.5
        low = args.mercury_low if args.mercury_low is not None else max(0.0, cutoff - args.mercury_band)
        high = args.mercury_high if args.mercury_high is not None else min(1.0, cutoff + args.mercury_band)
        client = _mercury(cfg, args)
        try:
            refine = mercury_refine_uncertain(
                scored,
                cfg.root,
                client,
                low=low,
                high=high,
                max_calls=args.mercury_max_calls,
                threshold=0.5,
                center=cutoff,
            )
            refine["cutoff"] = cutoff
        finally:
            client.close()
    scores_path = cfg.root / "candidates" / "cnn_scores.jsonl"
    write_jsonl(scores_path, scored)
    result = write_selected_bank(
        scored,
        cfg.root,
        out_root,
        limit=args.limit if args.limit is not None else 100_000,
        threshold=args.threshold,
    )
    result["scores"] = str(scores_path)
    if getattr(args, "prune_stale", False):
        from manga_ocr_dev.real_kana.utils import iter_jsonl

        keep = [
            str(record.get("crop_id") or "")
            for record in iter_jsonl(out_root / "candidates" / "candidates.jsonl")
        ]
        result["pruned"] = prune_unselected_stages(out_root, keep)
    if refine is not None:
        result["mercury"] = refine
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_segment(args: argparse.Namespace) -> None:
    cfg = _cfg(args)
    stats = segment_candidates(
        cfg,
        limit=args.limit,
        resume=not args.no_resume,
        review=args.review,
        shard=getattr(args, "shard", 0),
        shards=getattr(args, "shards", 1),
    )
    print(json.dumps(stats, indent=2))


def cmd_verify(args: argparse.Namespace) -> None:
    cfg = _cfg(args)
    client = _vision(cfg, args)
    stats = verify_candidates(
        cfg,
        client,
        stage=args.stage,
        limit=args.limit,
        resume=not args.no_resume,
        uncertain_only=not args.all,
    )
    print(json.dumps(stats, indent=2))


def cmd_review(args: argparse.Namespace) -> None:
    cfg = _cfg(args)
    client = _vision(cfg, args)
    stats = review_candidates(cfg, client, limit=args.limit, resume=not args.no_resume)
    print(json.dumps(stats, indent=2))


def cmd_clean(args: argparse.Namespace) -> None:
    cfg = _cfg(args)
    stats = clean_candidates(
        cfg,
        limit=args.limit,
        resume=not args.no_resume,
        shard=getattr(args, "shard", 0),
        shards=getattr(args, "shards", 1),
    )
    print(json.dumps(stats, indent=2))


def cmd_upscale(args: argparse.Namespace) -> None:
    cfg = _cfg(args)
    stats = upscale_candidates(
        cfg,
        limit=args.limit,
        resume=not args.no_resume,
        realesrgan_model=args.realesrgan_model,
        shard=getattr(args, "shard", 0),
        shards=getattr(args, "shards", 1),
    )
    print(json.dumps(stats, indent=2))


def cmd_upright(args: argparse.Namespace) -> None:
    cfg = _cfg(args)
    stats = upright_candidates(
        cfg,
        limit=args.limit,
        resume=not args.no_resume,
        use_vision_tilt=not args.estimate_tilt,
        angles_csv=args.angles,
        shard=getattr(args, "shard", 0),
        shards=getattr(args, "shards", 1),
    )
    print(json.dumps(stats, indent=2))


def cmd_build(args: argparse.Namespace) -> None:
    cfg = _cfg(args)
    rows = build_manifest(cfg.root, cfg)
    report = audit_bank(cfg.root, cfg)
    print(json.dumps({"manifest_rows": len(rows), "audit": report}, ensure_ascii=False, indent=2))


def cmd_audit(args: argparse.Namespace) -> None:
    cfg = _cfg(args)
    print(json.dumps(audit_bank(cfg.root, cfg), ensure_ascii=False, indent=2))


def cmd_sheets(args: argparse.Namespace) -> None:
    cfg = _cfg(args)
    n = export_review_sheets(cfg.root, args.output, per_sheet=args.per_sheet)
    print(f"wrote {n} review sheets")


def cmd_stats(args: argparse.Namespace) -> None:
    cfg = _cfg(args)
    bank = KanaBank.load(cfg.root, cfg)
    print(json.dumps(bank.stats(), ensure_ascii=False, indent=2))


def cmd_run(args: argparse.Namespace) -> None:
    cfg = _cfg(args)
    backend = getattr(args, "handwriting_backend", "auto")
    mercury_client = _mercury(cfg, args) if backend == "mercury" else None
    try:
        print(
            json.dumps(
                {
                    "harvest": harvest_candidates(
                        cfg,
                        limit=args.limit,
                        max_scan=args.max_scan,
                        handwriting_backend=backend,
                        mercury_client=mercury_client,
                    ).as_dict()
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        if mercury_client is not None:
            mercury_client.close()
    print(json.dumps({"segment": segment_candidates(cfg, limit=args.limit)}, indent=2))
    print(json.dumps({"clean": clean_candidates(cfg, limit=args.limit)}, indent=2))
    print(
        json.dumps(
            {"upscale": upscale_candidates(cfg, limit=args.limit, realesrgan_model=args.realesrgan_model)},
            indent=2,
        )
    )
    if getattr(args, "verify", True):
        client = _vision(cfg, args)
        print(
            json.dumps(
                {
                    "verify": verify_candidates(
                        cfg,
                        client,
                        stage="upscaled",
                        limit=args.limit,
                        uncertain_only=not args.verify_all,
                    )
                },
                indent=2,
            )
        )
    print(json.dumps({"upright": upright_candidates(cfg, limit=args.limit)}, indent=2))
    rows = build_manifest(cfg.root, cfg)
    print(json.dumps({"manifest_rows": len(rows), "audit": audit_bank(cfg.root, cfg)}, ensure_ascii=False, indent=2))


def cmd_render_demo(args: argparse.Namespace) -> None:
    from PIL import Image

    from manga_ocr_dev.synthetic_data_generator.renderer import Renderer

    cfg = _cfg(args)
    bank = KanaBank.load(cfg.root, cfg)
    if len(bank) == 0:
        raise SystemExit(f"no glyphs in {cfg.root}; run the pipeline first")
    renderer = Renderer.__new__(Renderer)
    renderer.max_size = 600
    renderer.kana_bank = bank
    css = {
        "font_path": args.font,
        "font_size": args.font_size,
        "vertical": not args.horizontal,
        "irregular": True,
        "stroke_size": 0,
        "text_color": "black",
        "color_mode": "solid",
        "line_height": 0.45,
        "real_kana": True,
        "real_kana_prob": 1.0,
        "real_kana_seed": args.seed,
        "rotate": 3.0,
        "wobble": 0.05,
    }
    img, params = renderer.render_text([args.text], css)
    rgba = Image.fromarray(img[:, :, [2, 1, 0, 3]], mode="RGBA")
    canvas = Image.new("RGB", rgba.size, (255, 255, 255))
    canvas.paste(rgba.convert("RGB"), mask=rgba.getchannel("A"))
    if args.scale != 1:
        canvas = canvas.resize((canvas.width * args.scale, canvas.height * args.scale), Image.LANCZOS)
    canvas.save(args.output)
    print(json.dumps({"output": str(args.output), "stats": params.get("real_kana_stats")}, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m manga_ocr_dev.real_kana", description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="bank root (default: assets/handwriting)")
    parser.add_argument("--model", default=None, help="vision model name (default: gemini-3.5-flash-lite)")
    parser.add_argument("--provider", default=None, choices=["gemini", "openai", "local", "compatible"])
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--local-model", default=None, help="path to the manga-ocr-charspot-small checkout")
    parser.add_argument("--local-backend", default=None, choices=["auto", "onnx", "torch"])
    parser.add_argument("--local-min-confidence", type=float, default=None)
    parser.add_argument("--verify-confidence", type=float, default=None)
    parser.add_argument("--openrouter-api-key", default=None, help="OpenRouter key for Mercury (default: $OPENROUTER_API_KEY)")
    parser.add_argument("--mercury-model", default=None, help="Mercury model id (default: inception/mercury-decide:free)")
    parser.add_argument("--mercury-url", default=None, help="Mercury decisions endpoint")
    parser.add_argument("--mercury-max-side", type=int, default=None, help="downscale crops above this many pixels")
    parser.add_argument("--mercury-timeout", type=float, default=None, help="per-request timeout in seconds")
    parser.add_argument("--mercury-retries", type=int, default=None, help="retries for transient Mercury failures")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--log", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("harvest", help="stream AnimeText-OCR and save candidates")
    p.add_argument("--max-scan", type=int, default=250_000)
    p.add_argument("--score-threshold", type=float, default=None)
    p.add_argument("--head", default=None, help="trained DINO logistic head (.npz)")
    p.add_argument(
        "--handwriting-backend",
        default="auto",
        choices=["auto", "heuristic", "dino", "mercury"],
        help="auto uses the trained head if given, else heuristic",
    )
    p.add_argument("--openrouter-api-key", default=argparse.SUPPRESS)
    p.add_argument("--mercury-model", default=argparse.SUPPRESS)
    p.add_argument("--mercury-url", default=argparse.SUPPRESS)
    p.add_argument("--mercury-max-side", type=int, default=argparse.SUPPRESS)
    p.add_argument("--mercury-timeout", type=float, default=argparse.SUPPRESS)
    p.add_argument("--mercury-retries", type=int, default=argparse.SUPPRESS)
    p.add_argument("--limit", type=int, default=argparse.SUPPRESS)
    p.set_defaults(func=cmd_harvest)

    p = sub.add_parser("label-handwriting", help="weak-label candidates with vision, Mercury, or the heuristic")
    p.add_argument("--output", required=True)
    p.add_argument("--backend", default="vision", choices=["vision", "mercury", "heuristic"])
    p.add_argument("--threshold", type=float, default=None, help="decision threshold for mercury/heuristic (default: 0.5)")
    p.add_argument("--heuristic-only", action="store_true", help="deprecated: use --backend heuristic")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--openrouter-api-key", default=argparse.SUPPRESS)
    p.add_argument("--mercury-model", default=argparse.SUPPRESS)
    p.add_argument("--mercury-url", default=argparse.SUPPRESS)
    p.add_argument("--mercury-max-side", type=int, default=argparse.SUPPRESS)
    p.add_argument("--mercury-timeout", type=float, default=argparse.SUPPRESS)
    p.add_argument("--mercury-retries", type=int, default=argparse.SUPPRESS)
    p.add_argument("--limit", type=int, default=argparse.SUPPRESS)
    p.set_defaults(func=cmd_label_handwriting)

    p = sub.add_parser("heuristic-labels", help="write a starting labels CSV locally")
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_heuristic_labels)

    p = sub.add_parser("train-head", help="train a DINO logistic handwriting head")
    p.add_argument("--labels", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dino-model", default="facebook/dinov2-small")
    p.add_argument("--threshold", type=float, default=0.95)
    p.add_argument("--embedding-cache", default=None)
    p.set_defaults(func=cmd_train_head)

    p = sub.add_parser("train-cnn", help="train the fast Mercury-distilled handwriting CNN")
    p.add_argument("--labels", required=True, help="CSV with path,label,image_id")
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--crop-size", type=int, default=96)
    p.add_argument("--workers", type=int, default=3)
    p.set_defaults(func=cmd_train_cnn)

    p = sub.add_parser("filter-cnn", help="score candidates and link the top handwriting crops")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--limit", type=int, default=100_000)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--mercury-uncertain", action="store_true", help="ask Mercury about boundary crops")
    p.add_argument("--mercury-band", type=float, default=0.1, help="band half-width around the selection cutoff")
    p.add_argument("--mercury-low", type=float, default=None)
    p.add_argument("--mercury-high", type=float, default=None)
    p.add_argument("--mercury-max-calls", type=int, default=2000)
    p.add_argument("--prune-stale", action="store_true", help="delete stage dirs for crops outside the selection")
    p.add_argument("--openrouter-api-key", default=argparse.SUPPRESS)
    p.add_argument("--mercury-model", default=argparse.SUPPRESS)
    p.add_argument("--mercury-url", default=argparse.SUPPRESS)
    p.add_argument("--mercury-max-side", type=int, default=argparse.SUPPRESS)
    p.add_argument("--mercury-timeout", type=float, default=argparse.SUPPRESS)
    p.add_argument("--mercury-retries", type=int, default=argparse.SUPPRESS)
    p.set_defaults(func=cmd_filter_cnn)

    p = sub.add_parser("segment", help="cut kana with the local charspot model")
    p.add_argument("--review", action="store_true", help="legacy: immediately verify raw crops with Gemini")
    p.add_argument("--local-model", default=None)
    p.add_argument("--local-backend", default=None, choices=["auto", "onnx", "torch"])
    p.add_argument("--local-min-confidence", type=float, default=None)
    p.add_argument("--threads", dest="local_threads", type=int, default=None, help="ONNX threads per process")
    p.add_argument("--limit", type=int, default=argparse.SUPPRESS)
    p.add_argument("--shard", type=int, default=0, help="segment only candidates where index %% shards == shard")
    p.add_argument("--shards", type=int, default=1)
    p.set_defaults(func=cmd_segment)

    p = sub.add_parser("verify", help="final Gemini verification of cut glyphs")
    p.add_argument("--stage", default="upscaled", choices=["raw", "cleaned", "upscaled"])
    p.add_argument("--all", action="store_true", help="verify every glyph, not only doubtful ones")
    p.add_argument("--limit", type=int, default=argparse.SUPPRESS)
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("review", help="deprecated: alias for `verify --stage raw --all`")
    p.add_argument("--limit", type=int, default=argparse.SUPPRESS)
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("clean", help="alpha extraction and component cleaning")
    p.add_argument("--limit", type=int, default=argparse.SUPPRESS)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.set_defaults(func=cmd_clean)

    p = sub.add_parser("upscale", help="mild upscaling and artefact checks")
    p.add_argument("--scale", dest="upscale_scale", type=float, default=None)
    p.add_argument("--min-side", dest="upscale_min_side", type=int, default=None)
    p.add_argument("--realesrgan-model", default=None)
    p.add_argument("--limit", type=int, default=argparse.SUPPRESS)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.set_defaults(func=cmd_upscale)

    p = sub.add_parser("upright", help="rotate each line's glyphs together")
    p.add_argument("--estimate-tilt", action="store_true", help="PCA estimate instead of vision tilt")
    p.add_argument("--angles", default=None, help="CSV of glyph_id,angle or crop_id,line,angle overrides")
    p.add_argument("--limit", type=int, default=argparse.SUPPRESS)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.set_defaults(func=cmd_upright)

    p = sub.add_parser("build", help="write the manifest and audit")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("audit", help="check the bank")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("sheets", help="write per-label human review sheets")
    p.add_argument("--output", default=None)
    p.add_argument("--per-sheet", type=int, default=64)
    p.set_defaults(func=cmd_sheets)

    p = sub.add_parser("stats", help="print bank statistics")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("run", help="harvest -> segment -> clean -> upscale -> verify -> upright -> build")
    p.add_argument("--max-scan", type=int, default=50_000)
    p.add_argument("--review", action="store_true", help="deprecated: verification is on by default")
    p.add_argument("--verify", action="store_true", default=True, help="run the final Gemini verifier (default)")
    p.add_argument("--no-verify", dest="verify", action="store_false", help="skip the Gemini verifier")
    p.add_argument("--verify-all", action="store_true", help="verify every glyph, not only doubtful ones")
    p.add_argument("--realesrgan-model", default=None)
    p.add_argument("--local-model", default=None)
    p.add_argument("--local-backend", default=None, choices=["auto", "onnx", "torch"])
    p.add_argument("--local-min-confidence", type=float, default=None)
    p.add_argument("--threads", dest="local_threads", type=int, default=None, help="ONNX threads per process")
    p.add_argument(
        "--handwriting-backend",
        default="auto",
        choices=["auto", "heuristic", "dino", "mercury"],
        help="handwriting filter backend for the harvest step",
    )
    p.add_argument("--openrouter-api-key", default=argparse.SUPPRESS)
    p.add_argument("--mercury-model", default=argparse.SUPPRESS)
    p.add_argument("--mercury-url", default=argparse.SUPPRESS)
    p.add_argument("--mercury-max-side", type=int, default=argparse.SUPPRESS)
    p.add_argument("--mercury-timeout", type=float, default=argparse.SUPPRESS)
    p.add_argument("--mercury-retries", type=int, default=argparse.SUPPRESS)
    p.add_argument("--limit", type=int, default=argparse.SUPPRESS)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("render-demo", help="render a sample with the installed bank")
    p.add_argument("--text", default="おはよう")
    p.add_argument("--font", default="/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc")
    p.add_argument("--font-size", type=int, default=48)
    p.add_argument("--horizontal", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--output", default="real_kana_demo.png")
    p.set_defaults(func=cmd_render_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.func(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
