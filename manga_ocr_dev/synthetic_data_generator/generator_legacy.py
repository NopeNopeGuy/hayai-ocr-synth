import re

import budoux
import numpy as np
import pandas as pd

from manga_ocr_dev.env import ASSETS_PATH
from manga_ocr_dev.synthetic_data_generator.lang import (
    COMMON_HAN_FREQ,
    DIALOGUE_PUNCT,
    EN_WORDS,
    LANG_CORE_CHARS,
    SFX_BASES,
    SFX_ELONG_TAILS,
    SFX_MARK_TAILS,
    detect_lang,
    normalize_lang,
)
from manga_ocr_dev.synthetic_data_generator.renderer import Renderer
from manga_ocr_dev.synthetic_data_generator.sfx_text import (
    ChineseSFXComposer,
    EnglishSFXComposer,
    KoreanSFXComposer,
    SFXComposer,
)
from manga_ocr_dev.synthetic_data_generator.utils import (
    font_supports_text_cmap,
    get_charsets,
    get_font_cmap_chars,
    get_font_meta,
    get_font_profile,
    is_ascii,
    is_drawable,
    is_hangul,
    is_hiragana,
    is_kanji,
    is_katakana,
)

_OBSOLETE_KANA = set("ゐゑゔゕゖヰヱヷヸヹヺヵヶゎヮ")
_SMALL_KANA = set("ぁぃぅぇぉっゃゅょァィゥェォッャュョ")
_WS_SPLIT_RE = re.compile(r"\S+\s*")


def _kana_weight(c):
    if c in _OBSOLETE_KANA:
        return 0.02
    if c in _SMALL_KANA:
        return 0.25
    return 1.0


class SyntheticDataGenerator:
    def __init__(
        self,
        background_df=None,
        kana_bank=None,
        real_kana=False,
        real_kana_prob=None,
    ):
        self.vocab, self.hiragana, self.katakana = get_charsets()
        self.vocab_set = set(self.vocab.tolist())
        self.len_to_p = pd.read_csv(ASSETS_PATH / "len_to_p.csv")
        self._len_vals = self.len_to_p.len.to_numpy()
        self._len_probs = self.len_to_p.p.to_numpy()

        self.parser = budoux.load_default_japanese_parser()
        self.parsers = {"ja": self.parser}
        try:
            self.parsers["zh_hans"] = budoux.load_default_simplified_chinese_parser()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.parsers["zh_hant"] = budoux.load_default_traditional_chinese_parser()
        except Exception:  # noqa: BLE001
            pass

        self.fonts_df, self.font_map = get_font_meta()
        # Precompute NumPy boolean masks once instead of rebuilding per sample
        labels_arr = self.fonts_df.label.to_numpy()
        self._is_manga_mask = labels_arr != "omit"
        self._label_masks = {lbl: (labels_arr == lbl) for lbl in self.fonts_df.label.unique()}
        self._num_chars_arr = self.fonts_df.num_chars.to_numpy()

        self.font_labels, self.font_p = self.get_font_labels_prob()
        self.renderer = Renderer(
            background_df=background_df,
            kana_bank=kana_bank,
            real_kana=real_kana,
            real_kana_prob=real_kana_prob,
        )
        self._pool_cache = {}
        self._lang_font_cache: dict = {}
        self._sfx_font_cache: dict = {}
        self.sfx_fonts, self.sfx_font_p = self.get_sfx_fonts("ja")
        self.sfx_composer = SFXComposer(SFX_BASES["ja"], SFX_ELONG_TAILS["ja"], SFX_MARK_TAILS["ja"], max_len=12)
        self.sfx_composers = {
            "ja": self.sfx_composer,
            "ko": KoreanSFXComposer(SFX_BASES["ko"], SFX_ELONG_TAILS["ko"], SFX_MARK_TAILS["ko"], max_len=10),
            "zh": ChineseSFXComposer(SFX_BASES["zh"], SFX_ELONG_TAILS["zh"], SFX_MARK_TAILS["zh"], max_len=10),
            "en": EnglishSFXComposer(SFX_BASES["en"], SFX_ELONG_TAILS["en"], SFX_MARK_TAILS["en"], max_len=16),
        }

    def process(self, text=None, override_css_params=None, sfx=False, lang=None):
        if override_css_params is None:
            override_css_params = {}
        sfx = bool(sfx or override_css_params.get("sfx", False))

        if lang is None:
            lang = override_css_params.get("lang", None)
        if lang is None and text is not None:
            lang = detect_lang(text)
        lang = normalize_lang(lang)

        if sfx:
            return self.process_sfx(text=text, override_css_params=override_css_params, lang=lang)

        user_font = "font_path" in override_css_params
        for _ in range(5):
            params_in = dict(override_css_params)
            params_in["lang"] = lang
            lines, vocab = self._prepare_dialogue_lines(text, params_in, lang=lang)
            if lines or user_font:
                break
        if not lines:
            raise ValueError("no renderable text after filtering by font support")
        override_css_params = params_in
        text_gt = "\n".join(lines)

        if lang == "ja" and np.random.random() < 0.5:
            word_prob = np.random.choice([0.33, 1.0], p=[0.3, 0.7])
            font_path = override_css_params["font_path"]
            lines = [self.add_random_furigana(line, word_prob, vocab, font_path=font_path) for line in lines]

        img, params = self.renderer.render(lines, override_css_params)
        params["lang"] = lang
        return img, text_gt, params

    def _prepare_dialogue_lines(self, text, override_css_params, lang="ja"):
        if text is None:
            if "font_path" not in override_css_params:
                override_css_params["font_path"] = self.get_random_font(lang=lang)
            vocab = self.font_map[override_css_params["font_path"]]
            words = self.get_random_words(vocab, override_css_params["font_path"], lang=lang)
        else:
            text = text.replace("\u3000", " ").replace("\u2026", "...")
            words = self.split_into_words(text, lang=lang)

        lines = self.words_to_lines(words, lang=lang)

        if "font_path" not in override_css_params:
            override_css_params["font_path"] = self.get_random_font("".join(lines), lang=lang)
        font_path = override_css_params["font_path"]
        prof = get_font_profile(font_path)
        if prof and prof.get("is_manga", True):
            if "font_size" not in override_css_params:
                d_min = prof.get("dialogue_size_min", 28)
                d_max = prof.get("dialogue_size_max", 52)
                override_css_params["font_size"] = int(np.random.uniform(d_min, d_max))
            if "stroke_size" not in override_css_params:
                if np.random.rand() < prof.get("stroke_prob", 0.35):
                    st_min = prof.get("stroke_min", 0)
                    st_max = prof.get("stroke_max", 3)
                    ratio = prof.get("stroke_ratio", 0.03)
                    calc_st = int(round(override_css_params["font_size"] * ratio))
                    override_css_params["stroke_size"] = max(st_min, min(st_max, calc_st))
                else:
                    override_css_params["stroke_size"] = 0
            if "double_outline" not in override_css_params:
                override_css_params["double_outline"] = bool(np.random.rand() < prof.get("double_outline_prob", 0.03))
        vocab = self.font_map.get(font_path)
        vocab_check = vocab if vocab is not None else ()

        lines = ["".join(c for c in line if c not in "<>&") for line in lines]
        lines = [
            "".join(c for c in line if (c in vocab_check or c.isspace() or is_drawable(font_path, c)))
            for line in lines
        ]
        lines = [line for line in lines if line.strip()]
        return lines, vocab

    def get_sfx_fonts(self, lang="ja"):
        lang = normalize_lang(lang)
        cached = self._sfx_font_cache.get(lang)
        if cached is not None:
            return cached
        if lang == "ja":
            result = self._get_sfx_fonts_ja()
        elif lang == "en":
            result = self._get_sfx_fonts_en()
        elif lang in ("ko", "zh"):
            result = self._get_sfx_fonts_cjk(lang)
        else:
            result = self._get_sfx_fonts_ja()
        self._sfx_font_cache[lang] = result
        if lang == "ja":
            self.sfx_fonts, self.sfx_font_p = result
        return result

    def _sfx_weights(self, paths):
        weights = []
        for p in paths:
            prof = get_font_profile(p)
            if prof.get("priority") == "omit" or not prof.get("is_manga", True):
                w = 0.0
            else:
                w = float(prof.get("sfx_weight", 1.0))
            weights.append(w)
        w = np.array(weights, dtype=float)
        if w.sum() == 0:
            return np.ones(len(paths)) / len(paths)
        return w / w.sum()

    def _vocab_probe(self, lang):
        if lang == "ko":
            return [c for c in self.vocab if is_hangul(c)]
        if lang == "zh":
            return [c for c in self.vocab if is_kanji(c)]
        if lang == "en":
            return list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
        return []

    def _fonts_covering(self, need, thresh=0.98, extra_check=None):
        if not len(self.font_map):
            raise RuntimeError("font_map is empty: run scan_fonts.py first")
        cover = self.font_map.coverage(need)
        paths = self.font_map.paths
        good = [paths[i] for i in np.flatnonzero(cover >= thresh)]
        if extra_check is not None:
            good = [p for p in good if extra_check(p)]
        if not good:
            best = cover.max()
            good = [paths[i] for i in np.flatnonzero(cover >= best - 1e-9)]
            if extra_check is not None:
                checked = [p for p in good if extra_check(p)]
                good = checked or good
        return good, self._sfx_weights(good)

    def _cmap_hit(self, font_path, need):
        try:
            cmap = get_font_cmap_chars(font_path)
        except Exception:  # noqa: BLE001
            return None
        return sum(ord(c) in cmap for c in need) / max(1, len(need))

    def _get_sfx_fonts_ja(self):
        need = set("".join(SFX_BASES["ja"])) | set("ッーっ")
        core_kana = "ドンバガッー"
        return self._fonts_covering(
            need,
            thresh=0.98,
            extra_check=lambda p: all(self.font_map.has_char(p, c) for c in core_kana),
        )

    def _get_sfx_fonts_en(self):
        return self._fonts_covering(set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"), thresh=0.98)

    def _get_sfx_fonts_cjk(self, lang):
        bases = SFX_BASES[lang]
        need = set("".join(bases))
        core = LANG_CORE_CHARS[lang]
        probe = self._vocab_probe(lang)
        pre_cover = self.font_map.coverage(probe) if probe else np.zeros(len(self.font_map.paths))
        order = np.argsort(-pre_cover)
        paths = self.font_map.paths
        good = []
        for idx in order:
            if len(good) >= 80:
                break
            if pre_cover[idx] < 0.30:
                break
            p = paths[int(idx)]
            try:
                cmap = get_font_cmap_chars(p)
            except Exception:  # noqa: BLE001
                continue
            if not all(ord(c) in cmap for c in core):
                continue
            hit = sum(ord(c) in cmap for c in need) / max(1, len(need))
            if hit >= 0.85:
                good.append(p)
        if not good:
            scored = []
            for idx in order[:200]:
                p = paths[int(idx)]
                hit = self._cmap_hit(p, need)
                if hit is None:
                    continue
                scored.append((hit, p))
            if scored:
                best = max(h for h, _ in scored)
                good = [p for h, p in scored if h >= best - 1e-9]
        if not good:
            return self._get_sfx_fonts_ja()
        return good, self._sfx_weights(good)

    def get_random_sfx_font(self, lang="ja"):
        fonts, p = self.get_sfx_fonts(lang)
        return str(np.random.choice(fonts, p=p))

    def process_sfx(self, text=None, override_css_params=None, lang="ja"):
        lang = normalize_lang(lang)
        explicit_keys = set((override_css_params or {}).keys())
        merged = Renderer.get_random_sfx_params(lang=lang)
        merged.update(override_css_params or {})
        merged["sfx"] = True
        merged["irregular"] = False
        merged["lang"] = lang
        override_css_params = merged

        user_font = "font_path" in explicit_keys
        sfx_text = None
        for _ in range(10):
            if not user_font:
                override_css_params["font_path"] = self.get_random_sfx_font(lang)
            font_path = override_css_params["font_path"]
            prof = get_font_profile(font_path)
            if prof and prof.get("is_manga", True):
                if "font_size" not in explicit_keys:
                    s_min = prof.get("sfx_size_min", 45)
                    s_max = prof.get("sfx_size_max", 160)
                    override_css_params["font_size"] = int(np.random.uniform(s_min, s_max))
                if "stroke_size" not in explicit_keys:
                    if np.random.rand() < prof.get("stroke_prob", 0.80):
                        st_min = prof.get("stroke_min", 2)
                        st_max = prof.get("stroke_max", 12)
                        ratio = prof.get("stroke_ratio", 0.07)
                        calc_st = int(round(override_css_params["font_size"] * ratio))
                        override_css_params["stroke_size"] = max(st_min, min(st_max, calc_st))
                        if "stroke_color" not in explicit_keys:
                            text_col = override_css_params.get("text_color", "black")
                            override_css_params["stroke_color"] = "white" if text_col != "white" else "black"
                    else:
                        override_css_params["stroke_size"] = 0
                if "double_outline" not in explicit_keys:
                    override_css_params["double_outline"] = bool(np.random.rand() < prof.get("double_outline_prob", 0.30))

            vocab = self.font_map.get(font_path, set())

            if text is None:
                sfx_text = self.get_random_sfx_text(vocab, font_path, lang=lang)
            else:
                cand = text.replace("\u3000", " ").replace("\u2026", "...")
                cand = "".join(
                    c for c in cand if (c in vocab or c.isspace() or is_drawable(font_path, c)) and c not in "<>&"
                )
                sfx_text = cand.strip() or self.get_random_sfx_text(vocab, font_path, lang=lang)
            if sfx_text and not all((c in vocab or is_drawable(font_path, c)) for c in sfx_text):
                sfx_text = None
            if sfx_text:
                break
        if not sfx_text:
            raise ValueError("no SFX text could be rendered with the available fonts")

        lines = [sfx_text]
        if " " in sfx_text and lang == "en":
            lines = [sfx_text]
        elif len(sfx_text) > 6 and np.random.rand() < 0.45:
            cut = len(sfx_text) // 2
            lines = [sfx_text[:cut], sfx_text[cut:]]
            if len(sfx_text) > 9 and np.random.rand() < 0.25:
                a = len(sfx_text) // 3
                lines = [sfx_text[:a], sfx_text[a : 2 * a], sfx_text[2 * a :]]
        elif len(sfx_text) > 4 and np.random.rand() < 0.3:
            cut = len(sfx_text) // 2
            lines = [sfx_text[:cut], sfx_text[cut:]]
        text_gt = "\n".join(lines)

        img, params = self.renderer.render(lines, override_css_params)
        params["lang"] = lang
        return img, text_gt, params

    def get_random_sfx_text(self, vocab, font_path=None, lang="ja"):
        lang = normalize_lang(lang)
        composer = self.sfx_composers.get(lang, self.sfx_composer)
        vocab = vocab if isinstance(vocab, (set, frozenset)) else (set(vocab) if vocab is not None else self.vocab_set)
        if lang == "ja":
            for _ in range(20):
                s = composer.sample(vocab)
                if s:
                    return s
            cands = [b for b in SFX_BASES["ja"] if all(c in vocab for c in b)]
            if cands:
                return str(np.random.choice(cands))
            kana = [c for c in self.katakana if c in vocab]
            if len(kana) >= 2:
                return "".join(np.random.choice(kana, int(np.random.randint(2, 5))).tolist())
            return None
        for _ in range(20):
            s = composer.sample(vocab, font_path=font_path)
            if s:
                return s
        cands = [b for b in SFX_BASES[lang] if all(c in vocab or (font_path and is_drawable(font_path, c)) for c in b)]
        if cands:
            return str(np.random.choice(cands))
        return None

    def _pools(self, vocab_key, vocab):
        pools = self._pool_cache.get(vocab_key)
        if pools is None:
            pools = {k: [] for k in ("hira", "kata", "kanji", "hangul", "digit", "latin", "punct", "other")}
            for c in vocab:
                if c.isspace():
                    continue
                if is_hiragana(c):
                    pools["hira"].append(c)
                elif is_katakana(c):
                    pools["kata"].append(c)
                elif is_kanji(c):
                    pools["kanji"].append(c)
                elif is_hangul(c):
                    if "\uac00" <= c <= "\ud7a3":
                        pools["hangul"].append(c)
                    else:
                        pools["other"].append(c)
                elif is_ascii(c) and c.isdigit():
                    pools["digit"].append(c)
                elif is_ascii(c) and c.isalpha():
                    pools["latin"].append(c)
                else:
                    pools["other"].append(c)
            for key in ("hira", "kata"):
                w = np.array([_kana_weight(c) for c in pools[key]], dtype=float)
                pools[key + "_p"] = w / w.sum() if w.size else None
            self._pool_cache[vocab_key] = pools
            while len(self._pool_cache) > 16:
                self._pool_cache.pop(next(iter(self._pool_cache)))
        return pools

    @staticmethod
    def _pick(pool, lo, hi, p=None):
        n = int(np.random.randint(lo, hi + 1))
        return "".join(np.random.choice(pool, n, p=p).tolist())

    @staticmethod
    def _uniform_fallback(vocab, default="あ"):
        vocab_list = [c for c in vocab if not c.isspace()] or [default]
        return ["".join(np.random.choice(vocab_list, np.random.randint(1, 4))) for _ in range(3)]

    @staticmethod
    def _punct_word(punct):
        chars, w = zip(*punct)
        w = np.array(w) / sum(w)
        return str(np.random.choice(chars, p=w))

    def get_random_words(self, vocab, font_path=None, lang="ja"):
        lang = normalize_lang(lang)
        if lang == "ko":
            return self._get_random_words_ko(vocab, font_path)
        if lang == "zh":
            return self._get_random_words_zh(vocab, font_path)
        if lang == "en":
            return self._get_random_words_en(vocab, font_path)
        return self._get_random_words_ja(vocab, font_path)

    def _get_random_words_ja(self, vocab, font_path=None):
        pools = self._pools(font_path if font_path is not None else id(vocab), vocab)
        if len(pools["hira"]) < 10 or len(pools["kanji"]) < 10:
            return self._uniform_fallback(vocab, default="あ")
        punct = self._punct_for_lang(vocab, "ja")
        if "." in vocab and "..." not in [p for p, _ in punct]:
            punct.append(("...", 0.06))

        pick = self._pick
        max_text_len = np.random.choice(self._len_vals, p=self._len_probs)
        words, text_len = [], 0
        kinds = ["kana", "kanji", "kata", "punct", "num", "hangul"]
        kp = np.array([0.40, 0.28, 0.10, 0.12, 0.05, 0.05])
        kp /= kp.sum()
        while True:
            kind = np.random.choice(kinds, p=kp)
            if kind == "kana":
                word = pick(pools["hira"], 1, 4, pools["hira_p"])
            elif kind == "kanji":
                word = pick(pools["kanji"], 1, 3)
                if np.random.rand() < 0.6:
                    word += pick(pools["hira"], 1, 3, pools["hira_p"])
            elif kind == "kata":
                word = pick(pools["kata"], 2, 5, pools["kata_p"])
                if "ー" in vocab and np.random.rand() < 0.3:
                    word += "ー"
            elif kind == "hangul" and pools["hangul"]:
                word = pick(pools["hangul"], 1, 3)
            elif kind == "punct" and punct:
                word = self._punct_word(punct)
            elif kind == "num" and (pools["digit"] or pools["latin"]):
                use_digit = pools["digit"] and (not pools["latin"] or np.random.rand() < 0.6)
                word = pick(pools["digit"] if use_digit else pools["latin"], 1, 3)
            else:
                word = pick(pools["hira"], 1, 3, pools["hira_p"])
            words.append(word)
            text_len += len(word)
            if text_len + len(word) >= max_text_len:
                break

        return words

    def _punct_for_lang(self, vocab, lang):
        punct, weights = DIALOGUE_PUNCT[lang]
        pairs = [(p, w) for p, w in zip(punct, weights) if p in vocab]
        if lang == "en" and "." in vocab and "..." not in [p for p, _ in pairs]:
            pairs.append(("...", 0.06))
        return pairs

    def _get_random_words_ko(self, vocab, font_path=None):
        from manga_ocr_dev.synthetic_data_generator.sfx_text import _COMMON_HANGUL_FREQ

        pools = self._pools(font_path if font_path is not None else id(vocab), vocab)
        hangul = pools["hangul"]
        if len(hangul) < 10:
            return self._uniform_fallback(vocab, default="가")
        freq_set = set(_COMMON_HANGUL_FREQ)
        core = [c for c in hangul if c in freq_set]
        rest = [c for c in hangul if c not in freq_set]
        if not core:
            core = hangul
        punct = self._punct_for_lang(vocab, "ko")
        pick = self._pick

        def pick_hangul(lo, hi):
            n = int(np.random.randint(lo, hi + 1))
            out = []
            for _ in range(n):
                pool = core if np.random.rand() < 0.85 else (rest or core)
                out.append(str(np.random.choice(pool)))
            return "".join(out)

        max_text_len = np.random.choice(self._len_vals, p=self._len_probs)
        words, text_len = [], 0
        kinds = ["hangul", "punct", "num", "hanja", "latin"]
        kp = np.array([0.80, 0.10, 0.05, 0.02, 0.03])
        kp /= kp.sum()
        while True:
            kind = str(np.random.choice(kinds, p=kp))
            if kind == "hangul":
                word = pick_hangul(1, 4)
            elif kind == "hanja" and pools["kanji"]:
                word = pick(pools["kanji"], 1, 2)
            elif kind == "punct" and punct:
                word = self._punct_word(punct)
            elif kind in ("num", "latin") and (pools["digit"] or pools["latin"]):
                pool = pools["digit"] if kind == "num" and pools["digit"] else pools["latin"]
                if not pool:
                    pool = pools["digit"] or pools["latin"]
                word = pick(pool, 1, 3)
            else:
                word = pick_hangul(1, 3)
            words.append(word)
            text_len += len(word) + 1
            if text_len + len(word) >= max_text_len:
                break
        return words

    def _get_random_words_zh(self, vocab, font_path=None):
        pools = self._pools(font_path if font_path is not None else id(vocab), vocab)
        han = pools["kanji"]
        if len(han) < 10:
            return self._uniform_fallback(vocab, default="的")
        freq_set = set(COMMON_HAN_FREQ)
        core = [c for c in han if c in freq_set]
        rest = [c for c in han if c not in freq_set]
        if not core:
            core = han
        punct = self._punct_for_lang(vocab, "zh")
        pick = self._pick

        def pick_han(lo, hi):
            n = int(np.random.randint(lo, hi + 1))
            out = []
            for _ in range(n):
                pool = core if np.random.rand() < 0.85 else (rest or core)
                out.append(str(np.random.choice(pool)))
            return "".join(out)

        max_text_len = np.random.choice(self._len_vals, p=self._len_probs)
        words, text_len = [], 0
        kinds = ["han", "punct", "num", "latin"]
        kp = np.array([0.72, 0.15, 0.07, 0.06])
        kp /= kp.sum()
        while True:
            kind = str(np.random.choice(kinds, p=kp))
            if kind == "han":
                word = pick_han(1, 4)
            elif kind == "punct" and punct:
                word = self._punct_word(punct)
            elif kind in ("num", "latin") and (pools["digit"] or pools["latin"]):
                pool = pools["digit"] if kind == "num" and pools["digit"] else pools["latin"]
                if not pool:
                    pool = pools["digit"] or pools["latin"]
                word = pick(pool, 1, 3)
            else:
                word = pick_han(1, 3)
            words.append(word)
            text_len += len(word)
            if text_len + len(word) >= max_text_len:
                break
        return words

    def _get_random_words_en(self, vocab, font_path=None):
        pools = self._pools(font_path if font_path is not None else id(vocab), vocab)
        if not pools["latin"]:
            return self._uniform_fallback(vocab, default="a")
        vocab_set = vocab if isinstance(vocab, (set, frozenset)) else set(vocab)
        words_pool = [w for w in EN_WORDS if all(c in vocab_set for c in w)] or ["hey", "wow", "oh"]
        punct = self._punct_for_lang(vocab, "en")
        upper_p = float(np.random.uniform(0.3, 0.7))

        def shape(w):
            r = np.random.rand()
            if r < upper_p:
                return w.upper()
            if r < upper_p + 0.15:
                return w.capitalize()
            return w

        max_text_len = int(max(8, np.random.choice(self._len_vals, p=self._len_probs)))
        words, text_len = [], 0
        while True:
            r = np.random.rand()
            if r < 0.82:
                word = shape(str(np.random.choice(words_pool)))
            elif r < 0.90 and punct:
                word = self._punct_word(punct)
            else:
                pool = pools["digit"] if (pools["digit"] and np.random.rand() < 0.6) else pools["latin"]
                word = self._pick(pool, 1, 3)
                if np.random.rand() < upper_p:
                    word = word.upper()
            words.append(word)
            text_len += len(word) + 1
            if text_len + len(word) >= max_text_len:
                break
        return words

    def split_into_words(self, text, lang="ja"):
        lang = normalize_lang(lang)
        max_text_len = np.random.choice(self._len_vals, p=self._len_probs)

        if lang in ("ko", "en"):
            words = _WS_SPLIT_RE.findall(text) or [text]
        elif lang == "zh":
            parser = self.parsers.get("zh_hans") or self.parsers.get("zh_hant") or self.parser
            try:
                words = parser.parse(text)
            except Exception:  # noqa: BLE001
                words = list(text)
        else:
            words = self.parser.parse(text)

        out, text_len = [], 0
        for word in words:
            out.append(word)
            text_len += len(word)
            if text_len + len(word) >= max_text_len:
                break

        return out

    def words_to_lines(self, words, lang="ja"):
        lang = normalize_lang(lang)
        if lang in ("ko", "en"):
            return self._words_to_lines_spaced(words)
        text = "".join(words)

        max_num_lines = 10
        min_line_len = len(text) // max_num_lines
        max_line_len = 20
        max_line_len = np.clip(np.random.poisson(6), max(1, min_line_len), max_line_len)
        lines = []
        line = ""
        for word in words:
            line += word
            if len(line) >= max_line_len:
                lines.append(line)
                line = ""
        if line:
            lines.append(line)

        return lines

    @staticmethod
    def _words_to_lines_spaced(words):
        tokens: list[str] = []
        for w in words:
            parts = str(w).split("\n")
            for i, part in enumerate(parts):
                for tok in part.split():
                    tokens.append(tok)
                if i < len(parts) - 1:
                    tokens.append("\n")
        max_line_len = int(np.clip(np.random.poisson(14), 8, 28))
        lines, line = [], ""
        for tok in tokens:
            if tok == "\n":
                if line.strip():
                    lines.append(line.strip())
                line = ""
                continue
            candidate = tok if not line.strip() else line + " " + tok
            if len(candidate) > max_line_len and line.strip():
                lines.append(line.strip())
                while len(tok) > 24:
                    lines.append(tok[:24])
                    tok = tok[24:]
                line = tok
            else:
                line = candidate
            while len(line) > 24:
                lines.append(line[:24].strip())
                line = line[24:].lstrip()
        if line.strip():
            lines.append(line.strip())
        glued = [re.sub(r"\s+([.,!?;:…%‰\)\]\}’”\"»])", r"\1", ln) for ln in lines]
        glued = [re.sub(r"([\(\[\{‘“\"«])\s+", r"\1", ln) for ln in glued]
        return [ln for ln in glued if ln.strip()]

    def add_random_furigana(self, line, word_prob=1.0, vocab=None, font_path=None):
        if vocab is None:
            vocab = self.vocab
            hiragana, katakana = self.hiragana, self.katakana
            all_non_kanji = None
        else:
            vocab_set = vocab if isinstance(vocab, (set, frozenset)) else set(vocab)
            # Characters in vocab_set are already verified drawable via fonts.csv!
            hiragana = [c for c in self.hiragana if c in vocab_set]
            katakana = [c for c in self.katakana if c in vocab_set]
            if not hiragana:
                return line
            all_non_kanji = None

        processed = ""
        kanji_group = ""
        ascii_group = ""
        for i, c in enumerate(line):
            if is_kanji(c):
                c_type = "kanji"
                kanji_group += c
            elif is_ascii(c):
                c_type = "ascii"
                ascii_group += c
            else:
                c_type = "other"

            if (c_type != "kanji" or i == len(line) - 1) and kanji_group:
                if np.random.uniform() < word_prob:
                    furigana_len = int(np.clip(np.random.normal(1.5, 0.5), 1, 4) * len(kanji_group))
                    src_key = np.random.choice(["hiragana", "katakana", "all"], p=[0.8, 0.15, 0.05])
                    if src_key == "hiragana":
                        char_source = hiragana
                    elif src_key == "katakana":
                        char_source = katakana if len(katakana) else hiragana
                    else:
                        if all_non_kanji is None:
                            all_non_kanji = [
                                ch for ch in vocab if not is_kanji(ch) and not ch.isspace() and ch not in "<>&"
                            ] or hiragana
                        char_source = all_non_kanji
                    furigana = "".join(np.random.choice(char_source, furigana_len))
                    if furigana:
                        processed += f"<ruby>{kanji_group}<rt>{furigana}</rt></ruby>"
                    else:
                        processed += kanji_group
                else:
                    processed += kanji_group
                kanji_group = ""

            if (c_type != "ascii" or i == len(line) - 1) and ascii_group:
                if len(ascii_group) <= 3 and np.random.uniform() < 0.7:
                    processed += f'<span style="text-combine-upright: all">{ascii_group}</span>'
                else:
                    processed += ascii_group
                ascii_group = ""

            if c_type == "other":
                processed += c

        return processed

    def is_font_supporting_text(self, font_path, text):
        chars = self.font_map.get(font_path, set())
        for c in text:
            if c.isspace() or c in chars:
                continue
            if not is_drawable(font_path, c):
                return False
        return True

    is_font_supporting_text_any = is_font_supporting_text

    def get_font_labels_prob(self):
        labels = {
            "special": 0.50,
            "common": 0.35,
            "regular": 0.15,
        }
        avail = [k for k in self.fonts_df.label.unique() if k in labels]
        if not avail:
            return ["regular"], np.array([1.0])
        p = np.array([labels[k] for k in avail], dtype=float)
        p = p / p.sum()
        return avail, p

    def _dialogue_font_candidates(self, lang):
        cached = self._lang_font_cache.get(lang)
        if cached is not None:
            return cached
        n = len(self.font_map.paths)
        thresholds = {"en": 0.98, "ko": 0.30, "zh": 0.50}
        if lang == "ja":
            idx = np.arange(n)
        else:
            probe = self._vocab_probe(lang)
            cover = self.font_map.coverage(probe) if probe else np.zeros(n)
            idx = np.flatnonzero(cover >= thresholds.get(lang, 0.50))
            if not len(idx):
                idx = np.flatnonzero(cover >= cover.max() - 1e-9) if len(cover) else np.arange(n)
        self._lang_font_cache[lang] = idx
        return idx

    def _sample_valid_font(self, valid_mask: np.ndarray) -> str:
        indices = np.flatnonzero(valid_mask)
        chosen = int(indices[np.random.randint(len(indices))])
        return self.font_map.paths[chosen]

    def get_random_font(self, text=None, lang="ja"):
        lang = normalize_lang(lang)
        label = np.random.choice(self.font_labels, p=self.font_p)
        is_label = self._label_masks[label]
        is_manga = self._is_manga_mask

        if text is None:
            idx = self._dialogue_font_candidates(lang)
            lang_mask = np.zeros(len(self.font_map.paths), dtype=bool)
            lang_mask[idx] = True
            valid = lang_mask & is_label & is_manga
            if not valid.any():
                valid = lang_mask & is_manga
            if not valid.any():
                valid = is_label & is_manga
            if not valid.any():
                valid = is_manga
            return self._sample_valid_font(valid)

        supports = self.font_map.support_mask(text)
        valid = is_label & supports & is_manga
        if not valid.any():
            valid = supports & is_manga
            if valid.any():
                return self._sample_valid_font(valid)
            found = self._find_font_by_cmap(text, is_label & is_manga, lang=lang)
            if found is not None:
                return found
            n = self._num_chars_arr
            valid = is_label & is_manga & (n >= 4000)
            if not valid.any():
                manga_idx = np.flatnonzero(is_manga)
                if len(manga_idx):
                    max_n = n[manga_idx].max()
                    valid = is_manga & (n >= max_n)
                else:
                    valid = is_label

        return self._sample_valid_font(valid)

    def _find_font_by_cmap(self, text, is_label, lang="ja", limit=15):
        chars = [c for c in set(text) if not c.isspace()]
        if not chars:
            return None
        # If all chars are in vocab.csv, FontMap.support_mask already checked every font!
        if all(c in self.vocab_set for c in chars):
            return None
        pool = self._dialogue_font_candidates(lang)
        is_manga = self._is_manga_mask
        pool_labels = np.zeros(len(self.font_map.paths), dtype=bool)
        pool_labels[pool] = True
        pool_labels &= is_label & is_manga
        if not pool_labels.any():
            pool_labels = np.zeros(len(self.font_map.paths), dtype=bool)
            pool_labels[pool] = True
            pool_labels &= is_manga
        order = np.flatnonzero(pool_labels)
        np.random.shuffle(order)
        checked = 0
        for i in order:
            p = self.font_map.paths[int(i)]
            if not font_supports_text_cmap(p, text):
                checked += 1
                if checked >= limit:
                    break
                continue
            if all(is_drawable(p, c) for c in chars):
                return p
            checked += 1
            if checked >= limit:
                break
        return None
