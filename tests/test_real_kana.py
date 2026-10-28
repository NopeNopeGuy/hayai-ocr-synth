"""Tests for the real handwritten-kana bank (no network, no API keys)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from manga_ocr_dev.real_kana.bank import (
    KanaBank,
    StrokeBudget,
    audit_bank,
    build_manifest,
)
from manga_ocr_dev.real_kana.clean import clean_candidates, clean_crop
from manga_ocr_dev.real_kana.config import KanaConfig, ensure_bank_dirs
from manga_ocr_dev.real_kana.segment import segment_candidates
from manga_ocr_dev.real_kana.upright import estimate_tilt_from_centres, upright_candidates
from manga_ocr_dev.real_kana.upscale import upscale_candidates
from manga_ocr_dev.real_kana.utils import (
    estimate_stroke_width,
    is_kana,
    is_likely_sfx,
    kana_sequence,
    normalize_kana_label,
    write_jsonl,
)
from manga_ocr_dev.real_kana.vision import apply_review, normalise_segmentation

FONT = "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"
pytestmark = pytest.mark.skipif(not Path(FONT).exists(), reason="test font not available")


def _font(size=80):
    return ImageFont.truetype(FONT, size)


def _glyph_image(ch: str, size: int = 80) -> Image.Image:
    font = _font(size)
    image = Image.new("RGBA", (size * 2, size * 2), (0, 0, 0, 0))
    ImageDraw.Draw(image).text((size, size), ch, font=font, fill=(20, 18, 16, 255), anchor="mm")
    bbox = image.getchannel("A").getbbox()
    assert bbox is not None
    return image.crop(bbox)


def _make_bank(tmp_path: Path, chars="あいうえおかっ", cfg=None) -> KanaBank:
    cfg = cfg or KanaConfig(root=tmp_path)
    for ch in chars:
        out = tmp_path / "glyphs" / ch
        out.mkdir(parents=True, exist_ok=True)
        _glyph_image(ch).save(out / f"{ch}_0.png")
    build_manifest(tmp_path, cfg)
    return KanaBank.load(tmp_path, cfg)


def _candidate(tmp_path: Path, text: str = "あいうえお") -> tuple[list[list[int]], Path]:
    ensure_bank_dirs(tmp_path)
    font = _font(40)
    image = Image.new("RGB", (320, 90), "white")
    draw = ImageDraw.Draw(image)
    boxes = []
    for i, ch in enumerate(text):
        x = 20 + i * 55
        draw.text((x, 45), ch, font=font, fill="black", anchor="lm")
        b = draw.textbbox((x, 45), ch, font=font, anchor="lm")
        boxes.append([max(0, b[0] - 2), max(0, b[1] - 2), min(image.width, b[2] + 2), min(image.height, b[3] + 2)])
    image_dir = tmp_path / "candidates" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image.save(image_dir / "c0.png")
    write_jsonl(
        tmp_path / "candidates" / "candidates.jsonl",
        [
            {
                "crop_id": "c0",
                "image_id": "i0",
                "transcription": text,
                "image_path": "candidates/images/c0.png",
                "dataset": "test",
                "revision": "r0",
            }
        ],
    )
    return boxes, image


class FakeVision:
    model = "fake-vision"

    def __init__(self, boxes, text):
        self.boxes = boxes
        self.text = text
        self.calls = 0

    def generate_json(self, prompt, images, schema):
        self.calls += 1
        if "auditing" in prompt:
            return {
                "glyphs": [
                    {"index": i, "remove": False, "confidence": 0.99, "reason": "clean"}
                    for i in range(len(self.text))
                ]
            }
        return {
            "is_handwritten": True,
            "is_sound_effect": False,
            "lines": [
                {
                    "tilt_deg": 2.0,
                    "glyphs": [
                        {
                            "label": ch,
                            "x0": b[0],
                            "y0": b[1],
                            "x1": b[2],
                            "y1": b[3],
                            "confidence": 0.99,
                        }
                        for ch, b in zip(self.text, self.boxes)
                    ],
                }
            ],
        }


class FakeMercuryClient:
    method = "mercury_noul"

    def __init__(self, score=0.97):
        self.score = score
        self.calls = 0
        self.closed = False

    def label(self, image, *, threshold=0.5):
        self.calls += 1
        return {
            "handwritten": self.score >= threshold,
            "confidence": self.score,
            "noul": self.score,
            "method": self.method,
            "model": "fake-mercury",
        }

    def score_handwriting(self, image):
        self.calls += 1
        return self.score

    def close(self):
        self.closed = True


class _FakeHTTPResponse:
    status_code = 200
    text = ""

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _FakeHTTPSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0
        self.last_json = None
        self.closed = False

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls += 1
        self.last_json = json
        return _FakeHTTPResponse(self.payload)

    def close(self):
        self.closed = True


def test_kana_helpers():
    assert is_kana("あ") and is_kana("ア")
    assert not is_kana("ー") and not is_kana("漢") and not is_kana("a")
    assert kana_sequence("あア1漢か") == ["あ", "ア", "か"]
    assert is_likely_sfx("ドドド")
    assert not is_likely_sfx("おはよう")
    assert normalize_kana_label("「ぁ」") == "ぁ"
    assert normalize_kana_label("漢") is None


def test_stroke_width_estimate():
    for thickness, expected in [(2, 4.0), (3, 5.0), (5, 7.0)]:
        alpha = np.zeros((80, 80), np.uint8)
        import cv2

        cv2.line(alpha, (15, 40), (65, 40), 255, thickness)
        value = estimate_stroke_width(alpha)
        assert abs(value - expected) <= 1.5


def test_colour_alpha_and_clean_drops_coloured_dirt():
    rgb = np.full((80, 80, 3), 250, np.uint8)
    rgb[25:55, 10:30] = 20  # black stroke
    rgb[3:8, 3:8] = (230, 10, 10)  # red dirt
    rgba, report = clean_crop(rgb)
    arr = np.asarray(rgba)
    assert arr.shape[2] == 4
    assert (arr[:, :, 3] > 0).sum() > 100
    # The red dot is not on the black->white colour axis, so it is not inked.
    red = (arr[:, :, 0] > 150) & (arr[:, :, 1] < 80)
    assert not red.any()
    assert report["components"] >= 1
    # A solid block is correctly flagged: it is not a valid pen stroke.
    assert "too_much_ink" in report["flags"]


def test_segmentation_validation_clamps_and_drops():
    cfg = KanaConfig()
    response = {
        "is_handwritten": True,
        "is_sound_effect": False,
        "lines": [
            {
                "tilt_deg": 90,
                "glyphs": [
                    {"label": "あ", "x0": -5, "y0": 0, "x1": 20, "y1": 20, "confidence": 0.9},
                    {"label": "漢", "x0": 25, "y0": 0, "x1": 45, "y1": 20, "confidence": 0.9},
                    {"label": "い", "x0": 30, "y0": 0, "x1": 50, "y1": 20, "confidence": 0.9},
                    {"label": "う", "x0": 40, "y0": 0, "x1": 60, "y1": 20, "confidence": 0.1},
                ],
            }
        ],
    }
    out = normalise_segmentation(response, width=100, height=100, transcription="あいう", cfg=cfg)
    assert out["status"] == "ok"
    assert out["glyph_count"] == 2  # あ + い, low confidence dropped
    assert out["lines"][0]["tilt_deg"] == 45
    assert out["lines"][0]["glyphs"][0]["box"][0] == 0
    assert "非kana" not in " ".join(out["warnings"])
    assert any("non-kana" in w for w in out["warnings"])


def test_segment_maps_boxes_from_prepared_image(tmp_path):
    """The model sees an enlarged crop; boxes must map back to source pixels."""
    from manga_ocr_dev.real_kana.vision import prepare_for_vision

    ensure_bank_dirs(tmp_path)
    original = Image.new("RGB", (40, 30), "white")
    image_dir = tmp_path / "candidates" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    original.save(image_dir / "tiny.png")
    write_jsonl(
        tmp_path / "candidates" / "candidates.jsonl",
        [{"crop_id": "tiny", "transcription": "あ", "image_path": "candidates/images/tiny.png"}],
    )

    class DimensionVision:
        model = "fake"

        def __init__(self):
            self.size = None

        def generate_json(self, prompt, images, schema):
            image = images[0]
            self.size = image.size
            assert f"{image.width} x {image.height}" in prompt
            return {
                "is_handwritten": True,
                "is_sound_effect": False,
                "lines": [
                    {
                        "tilt_deg": 0,
                        "glyphs": [
                            {
                                "label": "あ",
                                "x0": image.width // 4,
                                "y0": image.height // 4,
                                "x1": (3 * image.width) // 4,
                                "y1": (3 * image.height) // 4,
                                "confidence": 0.99,
                            }
                        ],
                    }
                ],
            }

    client = DimensionVision()
    cfg = KanaConfig(root=tmp_path)
    stats = segment_candidates(cfg, client)
    assert stats["glyphs"] == 1
    assert client.size is not None and client.size != original.size
    assert client.size == prepare_for_vision(original, cfg.vision_upscale_max_side).size
    records = [
        record
        for path in (tmp_path / "raw").glob("*/records.json")
        for record in json.loads(path.read_text(encoding="utf-8"))
    ]
    assert len(records) == 1
    box = records[0]["box"]
    assert 0 <= box[0] < box[2] <= original.width
    assert 0 <= box[1] < box[3] <= original.height
    assert records[0]["model_size"] == list(client.size)
    with Image.open(tmp_path / records[0]["raw_path"]) as crop:
        assert crop.size == (box[2] - box[0], box[3] - box[1])


def test_local_segment_with_charspot(tmp_path):
    """Default segmentation is local: no vision client, no prompt, no upscale."""
    from manga_ocr_dev.real_kana.local_ocr import group_chars

    _candidate(tmp_path, "あいう")
    cfg = KanaConfig(root=tmp_path)
    ensure_bank_dirs(tmp_path)

    class FakeCharspot:
        model = "fake-charspot"

        def __init__(self):
            self.sizes = []

        def read(self, image):
            self.sizes.append(image.size)
            return {
                "text": "あ漢い",
                "chars": [
                    {"char": "あ", "box": [10, 10, 50, 60], "score": 0.9},
                    {"char": "漢", "box": [60, 10, 100, 60], "score": 0.9},
                    {"char": "い", "box": [110, 10, 150, 60], "score": 0.1},
                    {"char": "う", "box": [160, 10, 200, 60], "score": 0.4},
                ],
            }

    segmenter = FakeCharspot()
    stats = segment_candidates(cfg, segmenter=segmenter)
    assert stats["glyphs"] == 2  # kanji and the low-score kana are dropped
    assert segmenter.sizes == [(320, 90)]  # source pixels, not a prepared copy

    path = next((tmp_path / "raw").glob("*/records.json"))
    records = json.loads(path.read_text(encoding="utf-8"))
    assert [r["label"] for r in records] == ["あ", "う"]
    assert all(r["segmenter"] == "charspot" for r in records)
    assert all(r["model_size"] == [320, 90] for r in records)
    assert records[0]["box"] == [10, 10, 50, 60]

    # Vertical columns group together and keep a tilt estimate.
    lines = group_chars(
        [
            {"char": "あ", "box": [10, 0, 40, 30], "score": 0.9},
            {"char": "い", "box": [12, 32, 42, 62], "score": 0.9},
            {"char": "う", "box": [120, 80, 160, 110], "score": 0.9},
        ],
        200,
        150,
        vertical=True,
    )
    assert [len(line["glyphs"]) for line in lines] == [2, 1]
    assert all(g["label"] != "" for line in lines for g in line["glyphs"])


def test_verify_preserves_upscaled_status(tmp_path):
    from manga_ocr_dev.real_kana.segment import verify_candidates

    cfg = KanaConfig(root=tmp_path)
    ensure_bank_dirs(tmp_path)
    group = tmp_path / "upscaled" / "g"
    group.mkdir(parents=True)
    records = []
    for i, label in enumerate("あいう"):
        path = group / f"g{i}.png"
        _glyph_image(label).save(path)
        records.append(
            {
                "glyph_id": f"g{i}",
                "crop_id": "g",
                "transcription": "あいう",
                "label": label,
                "confidence": 0.3,
                "status": "upscaled",
                "reviewed": False,
                "upscaled_path": str(path.relative_to(tmp_path)),
            }
        )
    (group / "records.json").write_text(json.dumps(records), encoding="utf-8")

    class FakeVerifier:
        model = "fake-verifier"

        def __init__(self):
            self.calls = 0

        def generate_json(self, prompt, images, schema):
            self.calls += 1
            assert "auditing" in prompt
            return {
                "glyphs": [
                    {"index": 0, "label": "お", "remove": False, "confidence": 0.99, "reason": "misread"},
                    {"index": 1, "remove": True, "confidence": 0.99, "reason": "neighbour"},
                    {"index": 2, "label": "う", "remove": False, "confidence": 0.99, "reason": "clean"},
                ]
            }

    verifier = FakeVerifier()
    stats = verify_candidates(cfg, verifier, stage="upscaled")
    assert stats["reviewed"] == 1
    assert stats["removed"] == 1
    assert stats["changed"] == 1
    assert verifier.calls == 1

    updated = json.loads((group / "records.json").read_text(encoding="utf-8"))
    assert [r["glyph_id"] for r in updated] == ["g0", "g2"]
    assert updated[0]["label"] == "お"
    assert all(r["status"] == "upscaled" and r["reviewed"] for r in updated)
    assert not (group / "g1.png").exists()
    assert (group / "verify_upscaled.json").is_file()

    # Resume skips a group that was already verified.
    assert verify_candidates(cfg, verifier, stage="upscaled")["skipped"] == 1
    assert verifier.calls == 1


def test_verify_only_spends_calls_on_doubtful_glyphs(tmp_path):
    from manga_ocr_dev.real_kana.segment import verify_candidates

    cfg = KanaConfig(root=tmp_path, verify_confidence=0.5)
    ensure_bank_dirs(tmp_path)
    group = tmp_path / "upscaled" / "g"
    group.mkdir(parents=True)
    path = group / "g0.png"
    _glyph_image("あ").save(path)
    (group / "records.json").write_text(
        json.dumps(
            [
                {
                    "glyph_id": "g0",
                    "crop_id": "g",
                    "transcription": "あ",
                    "label": "あ",
                    "confidence": 0.99,
                    "status": "upscaled",
                    "reviewed": False,
                    "upscaled_path": str(path.relative_to(tmp_path)),
                }
            ]
        ),
        encoding="utf-8",
    )

    class NeverCalled:
        model = "never"

        def generate_json(self, *args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("verifier should not be called")

    stats = verify_candidates(cfg, NeverCalled(), stage="upscaled")
    assert stats["no_targets"] == 1
    assert stats["reviewed"] == 0

    class KeepAll:
        model = "keep"

        def __init__(self):
            self.calls = 0

        def generate_json(self, prompt, images, schema):
            self.calls += 1
            return {
                "glyphs": [
                    {"index": 0, "label": "あ", "remove": False, "confidence": 0.99, "reason": "clean"},
                ]
            }

    # An explicit `--all` audit is not blocked by the earlier doubtful-only
    # report that found nothing.
    keeper = KeepAll()
    all_stats = verify_candidates(cfg, keeper, stage="upscaled", uncertain_only=False)
    assert all_stats["reviewed"] == 1
    assert keeper.calls == 1
    assert verify_candidates(cfg, keeper, stage="upscaled", uncertain_only=False)["skipped"] == 1
    assert keeper.calls == 1


def test_apply_review_changes_and_removes():
    records = [
        {"glyph_id": "a", "label": "ん"},
        {"glyph_id": "b", "label": "せ"},
        {"glyph_id": "c", "label": "あ"},
    ]
    response = {
        "glyphs": [
            {"index": 0, "label": "人", "remove": False, "confidence": 0.9, "reason": "wrong"},
            {"index": 1, "remove": True, "confidence": 0.9, "reason": "neighbour"},
        ]
    }
    out, warnings = apply_review(response, records)
    assert [r["glyph_id"] for r in out] == ["a", "c"]
    assert out[0]["label"] == "ん"  # 人 is not kana, so the label is unchanged
    assert out[0].get("review_changed") in (None, True)
    assert any("removed" in w for w in warnings)


def test_estimate_tilt():
    vertical = estimate_tilt_from_centres([(0, 0), (1, 10), (2, 20)], vertical=True)
    assert -8 < vertical < -2
    horizontal = estimate_tilt_from_centres([(0, 0), (10, 1), (20, 2)], vertical=False)
    assert 2 < horizontal < 8


def test_bank_stroke_filter_and_derived_small_kana(tmp_path):
    bank = _make_bank(tmp_path, chars="あい")
    assert "ぃ" in bank.labels()
    derived = bank.labels()["ぃ"]
    assert derived >= 1
    assert bank.pick("ぃ", target_stroke=4.0, target_height=40) is not None
    # An impossible target stroke returns nothing.
    assert bank.pick("あ", target_stroke=80.0, target_height=40) is None
    budget = StrokeBudget(1.2)
    budget.add(2.0)
    assert budget.allows(2.2)
    assert not budget.allows(3.0)


def test_bank_manifest_and_audit(tmp_path):
    bank = _make_bank(tmp_path)
    assert len(bank) >= 5
    report = audit_bank(tmp_path, bank.cfg)
    assert report["glyphs"] >= 5
    assert report["problems"] == []
    assert Path(tmp_path / "manifest.jsonl").is_file()


def test_offline_pipeline_and_renderer(tmp_path):
    boxes, _ = _candidate(tmp_path)
    cfg = KanaConfig(root=tmp_path)
    ensure_bank_dirs(tmp_path)
    client = FakeVision(boxes, "あいうえお")
    assert segment_candidates(cfg, client, review=False)["glyphs"] == 5
    assert clean_candidates(cfg)["glyphs"] == 5
    assert upscale_candidates(cfg)["glyphs"] == 5
    assert upright_candidates(cfg)["glyphs"] == 5
    build_manifest(tmp_path, cfg)
    bank = KanaBank.load(tmp_path, cfg)
    assert len(bank) >= 5

    from manga_ocr_dev.synthetic_data_generator.renderer import Renderer

    css = {
        "font_path": FONT,
        "font_size": 48,
        "vertical": True,
        "stroke_size": 0,
        "irregular": True,
        "text_color": "black",
        "color_mode": "solid",
        "real_kana": True,
        "real_kana_prob": 1.0,
        "real_kana_seed": 1,
        "rotate": 0.0,
        "wobble": 0.0,
    }
    np.random.seed(0)
    renderer = Renderer.__new__(Renderer)
    renderer.max_size = 600
    renderer.kana_bank = bank
    img, params = renderer.render_text(["あいうえお"], css)
    assert params["real_kana_stats"]["attached"] > 0
    assert (img[:, :, 3] > 0).sum() > 100

    np.random.seed(0)
    plain_renderer = Renderer.__new__(Renderer)
    plain_renderer.max_size = 600
    plain_img, _ = plain_renderer.render_text(["あいうえお"], {**css, "real_kana": False})
    assert plain_img.shape == img.shape
    assert np.abs(img[:, :, 3].astype(int) - plain_img[:, :, 3].astype(int)).mean() > 0.01


def test_renderer_real_kana_is_opt_in(tmp_path):
    """The default font path stays untouched; real kana is a switch."""

    from manga_ocr_dev.synthetic_data_generator.renderer import Renderer

    bank = _make_bank(tmp_path)
    css = {"font_size": 48, "vertical": True, "irregular": False, "text_color": "black"}

    np.random.seed(0)
    default_renderer = Renderer(kana_bank=bank)
    _, default_params = default_renderer.render_text(["あいうえお"], css)
    assert "real_kana_stats" not in default_params

    np.random.seed(0)
    enabled_renderer = Renderer(kana_bank=bank, real_kana=True)
    _, enabled_params = enabled_renderer.render_text(["あいうえお"], css)
    assert enabled_params["real_kana_stats"]["attached"] > 0

    # Per-sample override wins over the constructor default.
    np.random.seed(0)
    _, per_sample = default_renderer.render_text(["あいうえお"], {**css, "real_kana": True})
    assert per_sample["real_kana_stats"] == enabled_params["real_kana_stats"]


def test_small_kana_use_smaller_target_height(tmp_path):
    from PIL import ImageFont

    from manga_ocr_dev.real_kana.bank import attach_real_kana

    # The synthetic fixture glyphs are bold; relax the stroke matching so both
    # characters attach and only the size rule is under test.
    cfg = KanaConfig(root=tmp_path)
    cfg.stroke_tolerance = 6.0
    cfg.image_stroke_ratio = 12.0
    bank = _make_bank(tmp_path, cfg=cfg)
    font = ImageFont.truetype(FONT, 48)
    ops = [{"kind": "char", "ch": ch, "cx": 0.0, "cy": 0.0, "font": font} for ch in "かっ"]
    stats = attach_real_kana(ops, bank, cfg=bank.cfg, rng=np.random.default_rng(0), probability=1.0)
    assert stats["attached"] == 2
    heights = {op["ch"]: op["kana"]["target_height"] for op in ops}
    assert heights["っ"] < heights["か"] * 0.7


def test_identity_aspect_is_eligible_but_warped_ops_are_not(tmp_path):
    """SFX sets ``aspect=(1, 1)``/``persp=0``; that is a no-op, unlike a warp."""

    from PIL import ImageFont

    from manga_ocr_dev.real_kana.bank import attach_real_kana

    cfg = KanaConfig(root=tmp_path)
    cfg.stroke_tolerance = 6.0
    cfg.image_stroke_ratio = 12.0
    bank = _make_bank(tmp_path, "あい", cfg=cfg)
    font = ImageFont.truetype(FONT, 48)
    base = {"kind": "char", "cx": 0.0, "cy": 0.0, "font": font, "cell": 48}

    flat = [{**base, "ch": "あ", "persp": 0.0, "aspect": (1.0, 1.0)}]
    warped = [{**base, "ch": "い", "persp": 0.0, "aspect": (1.4, 0.8)}]
    perspectives = [{**base, "ch": "い", "persp": 0.2}]
    for op in (*warped, *perspectives):
        stats = attach_real_kana([op], bank, cfg=bank.cfg, rng=np.random.default_rng(0), probability=1.0)
        assert stats["attached"] == 0
        assert "kana" not in op

    stats = attach_real_kana(flat, bank, cfg=bank.cfg, rng=np.random.default_rng(0), probability=1.0)
    assert stats["attached"] == 1
    assert "kana" in flat[0]


def test_heuristic_labels_write_absolute_paths(tmp_path):
    from manga_ocr_dev.real_kana.handwriting import write_heuristic_labels

    _candidate(tmp_path, "あいう")
    out = tmp_path / "labels.csv"
    n = write_heuristic_labels(tmp_path / "candidates" / "candidates.jsonl", out, root=tmp_path)
    assert n == 1
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "path,label"
    path_text = lines[1].rsplit(",", 1)[0]
    assert Path(path_text).is_absolute()
    assert Path(path_text).is_file()
    assert lines[1].rsplit(",", 1)[1] in {"0", "1"}


def test_upright_cleans_stale_and_relabelled_glyphs(tmp_path):
    cfg = KanaConfig(root=tmp_path)
    ensure_bank_dirs(tmp_path)
    up = tmp_path / "upscaled" / "g"
    up.mkdir(parents=True)
    records = []
    for i, label in enumerate(["あ", "い"]):
        path = up / f"g{i}.png"
        _glyph_image(label).save(path)
        records.append(
            {
                "glyph_id": f"g{i}",
                "crop_id": "g",
                "label": label,
                "box": [0, 0, 40, 40],
                "line_index": 0,
                "line_tilt_deg": 0.0,
                "status": "upscaled",
                "upscaled_path": str(path.relative_to(tmp_path)),
            }
        )
    (up / "records.json").write_text(json.dumps(records), encoding="utf-8")

    # A glyph removed by the verifier, and g0's old copy under a stale label.
    stale = tmp_path / "glyphs" / "ら" / "g_stale.png"
    stale.parent.mkdir(parents=True)
    _glyph_image("ら").save(stale)
    stale.with_suffix(".json").write_text(
        json.dumps({"glyph_id": "g_stale", "crop_id": "g", "label": "ら"}), encoding="utf-8"
    )
    old = tmp_path / "glyphs" / "か" / "g0.png"
    old.parent.mkdir(parents=True)
    _glyph_image("か").save(old)
    old.with_suffix(".json").write_text(json.dumps({"glyph_id": "g0", "crop_id": "g", "label": "か"}), encoding="utf-8")

    stats = upright_candidates(cfg)
    assert stats["glyphs"] == 2
    assert not stale.exists() and not stale.with_suffix(".json").exists()
    assert not old.exists() and not old.with_suffix(".json").exists()
    assert (tmp_path / "glyphs" / "あ" / "g0.png").is_file()
    assert (tmp_path / "glyphs" / "い" / "g1.png").is_file()


def test_review_pass(tmp_path):
    boxes, _ = _candidate(tmp_path)
    cfg = KanaConfig(root=tmp_path)
    ensure_bank_dirs(tmp_path)
    client = FakeVision(boxes, "あいうえお")
    segment_candidates(cfg, client, review=False)
    from manga_ocr_dev.real_kana.segment import review_candidates

    stats = review_candidates(cfg, client)
    assert stats["reviewed"] == 1
    assert stats["removed"] == 0


def test_mercury_client_parses_and_caches(tmp_path):
    from manga_ocr_dev.real_kana.mercury import MercuryDecisionClient

    session = _FakeHTTPSession(
        {
            "model": "inception/mercury-decide-test",
            "answers": {"is_handwritten": {"type": "noul", "noul": 0.87}},
            "usage": {"cost": 0},
        }
    )
    client = MercuryDecisionClient(
        api_key="test-key",
        session=session,
        cache_dir=tmp_path / "cache",
        max_side=256,
    )
    image = _glyph_image("あ")
    first = client.label(image, threshold=0.8)
    second = client.label(image, threshold=0.8)
    assert first == second
    assert first["handwritten"] is True
    assert first["noul"] == pytest.approx(0.87)
    assert first["method"] == "mercury_noul"
    assert session.calls == 1
    assert session.last_json["model"] == "inception/mercury-decide:free"
    assert session.last_json["state"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    client.close()
    assert session.closed is True


def test_mercury_client_rejects_bad_answer(tmp_path):
    from manga_ocr_dev.real_kana.mercury import MercuryDecisionClient, MercuryUnavailable

    session = _FakeHTTPSession({"answers": {}})
    client = MercuryDecisionClient(api_key="test-key", session=session, cache_dir=None)
    with pytest.raises(MercuryUnavailable):
        client.label(_glyph_image("い"))
    client.close()


def test_handwriting_scorer_dispatches_to_mercury():
    from manga_ocr_dev.real_kana.handwriting import HandwritingScorer

    fake = FakeMercuryClient(0.75)
    scorer = HandwritingScorer(backend="mercury", mercury_client=fake)
    assert scorer.backend == "mercury"
    assert scorer.method == "mercury_noul"
    assert scorer.score_image(_glyph_image("う")) == pytest.approx(0.75)
    assert fake.calls == 1


def test_harvest_uses_mercury_backend(tmp_path, monkeypatch):
    from manga_ocr_dev.real_kana import dataset

    image = _glyph_image("あ").convert("RGB")
    row = {
        "crop_id": "m0",
        "image_id": "i0",
        "transcription": "あいうえお",
        "language": "ja",
        "image": image,
        "seq_confidence": 0.99,
        "min_token_confidence": 0.99,
        "source": "test",
    }
    monkeypatch.setattr(dataset, "_iter_stream", lambda revision: iter([row]))
    fake = FakeMercuryClient(0.9)
    cfg = KanaConfig(root=tmp_path)
    stats = dataset.harvest_candidates(
        cfg,
        limit=1,
        max_scan=10,
        handwriting_backend="mercury",
        mercury_client=fake,
    )
    assert stats.saved == 1
    assert fake.calls == 1
    rec = json.loads((tmp_path / "candidates" / "candidates.jsonl").read_text(encoding="utf-8"))
    assert rec["handwriting_method"] == "mercury_noul"
    assert rec["handwriting_score"] == pytest.approx(0.9)


def test_label_handwriting_mercury_keeps_resumed_csv_label(tmp_path, monkeypatch):
    from manga_ocr_dev.real_kana import cli

    _candidate(tmp_path, "あいう")
    index = tmp_path / "candidates" / "candidates.jsonl"
    rec = json.loads(index.read_text(encoding="utf-8"))
    rec["handwriting_score"] = 0.0
    index.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
    image_path = tmp_path / "candidates" / "images" / "c0.png"
    out = tmp_path / "labels.csv"
    out.write_text(f"path,label\n{image_path},1\n", encoding="utf-8")
    fake = FakeMercuryClient(0.99)
    monkeypatch.setattr(cli, "_mercury", lambda cfg, args: fake)

    cli.main(
        [
            "--root",
            str(tmp_path),
            "label-handwriting",
            "--output",
            str(out),
            "--backend",
            "mercury",
        ]
    )

    assert fake.calls == 0
    assert fake.closed is True
    assert out.read_text(encoding="utf-8").splitlines()[1] == f"{image_path},1"


def test_label_handwriting_mercury_labels_new_rows(tmp_path, monkeypatch):
    from manga_ocr_dev.real_kana import cli

    _candidate(tmp_path, "あいう")
    image_path = tmp_path / "candidates" / "images" / "c0.png"
    out = tmp_path / "labels.csv"
    fake = FakeMercuryClient(0.99)
    monkeypatch.setattr(cli, "_mercury", lambda cfg, args: fake)

    cli.main(
        [
            "--root",
            str(tmp_path),
            "label-handwriting",
            "--output",
            str(out),
            "--backend",
            "mercury",
            "--threshold",
            "0.9",
        ]
    )

    assert fake.calls == 1
    assert out.read_text(encoding="utf-8").splitlines()[1] == f"{image_path},1"


def test_cnn_crop_preparation_and_label_rows(tmp_path):
    from manga_ocr_dev.real_kana.cnn_filter import prepare_crop, read_label_rows

    image = Image.new("RGBA", (20, 50), (0, 0, 0, 0))
    ImageDraw.Draw(image).rectangle([5, 5, 14, 44], fill=(0, 0, 0, 255))
    crop = prepare_crop(image, 96)
    assert crop.size == (96, 96)
    assert crop.mode == "RGB"
    assert crop.getpixel((0, 0)) == (255, 255, 255)

    labels = tmp_path / "labels.csv"
    labels.write_text("path,label\n/a/1000410_4.png,1\n/b/1000271_8.png,0\n", encoding="utf-8")
    rows = read_label_rows(labels)
    assert rows == [
        ("/a/1000410_4.png", 1, "1000410"),
        ("/b/1000271_8.png", 0, "1000271"),
    ]


def test_cnn_selected_bank_links_images(tmp_path):
    from manga_ocr_dev.real_kana.cnn_filter import write_selected_bank

    source = tmp_path / "source"
    image = source / "candidates" / "images" / "a_1.png"
    image.parent.mkdir(parents=True)
    _glyph_image("あ").save(image)
    target = tmp_path / "target"
    rows = [
        {"image_path": "candidates/images/a_1.png", "crop_id": "a_1", "cnn_score": 0.91},
        {"image_path": "candidates/images/missing.png", "crop_id": "x", "cnn_score": 0.99},
    ]
    result = write_selected_bank(rows, source, target, limit=10, threshold=0.5)
    assert result["selected"] == 1
    record = json.loads((target / "candidates" / "candidates.jsonl").read_text(encoding="utf-8"))
    assert record["selection_method"] == "mercury_distilled_cnn"
    assert (target / record["image_path"]).is_file()


def test_cnn_mercury_refinement_uses_mercury_for_boundary(tmp_path):
    from manga_ocr_dev.real_kana.cnn_filter import mercury_refine_uncertain

    image = tmp_path / "img.png"
    _glyph_image("う").save(image)
    rows = [
        {"image_path": "img.png", "cnn_score": 0.9},
        {"image_path": "img.png", "cnn_score": 0.5},
        {"image_path": "img.png", "cnn_score": 0.1},
    ]

    class Client:
        calls = 0

        def label(self, path, *, threshold=0.5):
            self.calls += 1
            return {"noul": 0.8, "handwritten": True}

    client = Client()
    stats = mercury_refine_uncertain(rows, tmp_path, client, max_calls=10)
    assert stats["uncertain"] == 1
    assert stats["labelled"] == 1
    assert client.calls == 1
    assert rows[1]["mercury_noul"] == 0.8
    assert "mercury_noul" not in rows[0]


def test_cnn_score_cache_reuses_previous_scores(tmp_path):
    from manga_ocr_dev.real_kana.cnn_filter import score_candidates

    index = tmp_path / "candidates" / "candidates.jsonl"
    index.parent.mkdir(parents=True)
    rows = [
        {"image_path": "candidates/images/a_1.png", "crop_id": "a_1"},
        {"image_path": "candidates/images/b_2.png", "crop_id": "b_2"},
    ]
    index.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    cache = tmp_path / "candidates" / "scores.jsonl"
    cache.write_text(
        json.dumps({"image_path": "candidates/images/a_1.png", "cnn_score": 0.9}) + "\n"
        + json.dumps({"image_path": "candidates/images/b_2.png", "cnn_score": 0.1}) + "\n",
        encoding="utf-8",
    )
    scored = score_candidates(index, tmp_path / "missing.pt", cache_path=cache)
    assert [row["cnn_score"] for row in scored] == [0.9, 0.1]
    assert scored[0]["crop_id"] == "a_1"


def test_segment_shards_partition_candidates(tmp_path):
    ensure_bank_dirs(tmp_path)
    image_dir = tmp_path / "candidates" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(3):
        name = f"c{i}.png"
        Image.new("RGB", (60, 40), "white").save(image_dir / name)
        rows.append(
            {
                "crop_id": f"c{i}",
                "image_id": f"i{i}",
                "transcription": "あ",
                "image_path": f"candidates/images/{name}",
            }
        )
    write_jsonl(tmp_path / "candidates" / "candidates.jsonl", rows)

    class Fake:
        model = "fake"

        def read(self, image):
            return {"text": "", "chars": []}

    cfg = KanaConfig(root=tmp_path)
    stats = [segment_candidates(cfg, segmenter=Fake(), shard=s, shards=3) for s in range(3)]
    assert sum(item["seen"] for item in stats) == 3
    assert len(list((tmp_path / "raw").glob("*/meta.json"))) == 3
