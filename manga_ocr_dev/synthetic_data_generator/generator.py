import unicodedata

import budoux
import numpy as np
import pandas as pd

from manga_ocr_dev.env import ASSETS_PATH, FONTS_ROOT
from manga_ocr_dev.synthetic_data_generator.renderer import Renderer
from manga_ocr_dev.synthetic_data_generator.utils import (
    get_font_meta,
    get_charsets,
    is_ascii,
    is_hiragana,
    is_katakana,
    is_kanji,
    is_drawable,
    MissingGlyphError,
)

# SFX (sound-effect) lexicon for the separate SFX split. Impact, motion,
# emotion and ecchi onomatopoeia; decorative tails (hearts/stars/punct) are
# appended at generation time and survive only when the font supports them,
# so GT always matches the rendered font glyphs.
SFX_BASES = [
    "ドン", "バン", "ガン", "ドカン", "ズドン", "ドゴォン", "バゴォン",
    "ガシャン", "バリン", "パリン", "メキ", "メキメキ", "バキ", "ボキ",
    "グシャ", "ズシャ", "ドス", "ズシン", "ゴン", "ガツン", "ボゴン",
    "ザッ", "サッ", "シュッ", "ビュン", "ヒュン", "ゴゴゴ", "ザワザワ",
    "ザザッ", "ダダダ", "タタタ", "ドドド", "ガガガ", "ビリビリ", "グラグラ",
    "ユラ", "フワ", "パッ", "バッ", "ワー", "キャー", "ギャー", "ウワァン",
    "エーン", "クスクス", "アハハ", "ゲラゲラ", "ヒソヒソ", "ドキドキ",
    "ドキン", "ズキュン", "キュン", "ハッ", "ホッ", "フゥ", "ハァ", "ヒィッ",
    "あん", "あっ", "んっ", "はぁ", "ふぁ", "ひゃ", "くちゅ", "ぐちゅ",
    "ぬちゅ", "びちゃ", "ぷるん", "ぷるぷる", "ぞく", "ぞくぞく", "ビク",
    "ビクン", "ハァハァ", "どん", "ばん", "がしゃん", "どきどき",
]
# More families: ambient / silence, liquids & fire, body reactions, facial
# expressions, sparkle, and hiragana "soft" SFX. Kept at the same tasteful
# level as the base list above.
SFX_BASES += [
    "シーン", "ゴォォ", "ザーッ", "ポツポツ", "ザァァ", "ヒュゥゥ", "ビュオオ", "ジジジ",
    "カチッ", "ピッ", "ピピッ", "ブォォン", "キィィ", "ガタン", "ゴトン", "ガタガタ",
    "コツコツ", "カツカツ", "ペタペタ", "ジャー", "ポチャン", "ボコボコ", "グツグツ",
    "ジュウウ", "パチパチ", "メラメラ", "ボォォ", "ドロドロ", "ヌルッ", "ヌルヌル",
    "ゾクッ", "ゾワッ", "ビクッ", "ガクッ", "ガクガク", "ブルブル", "ワナワナ",
    "ソワソワ", "イライラ", "モヤモヤ", "ズーン", "ガーン", "ピキーン", "キラーン",
    "キラキラ", "ピカッ", "ピカピカ", "ニコニコ", "ニヤリ", "ニヤニヤ", "ムスッ",
    "プンプン", "ムッ", "ジロッ", "ギロッ", "チラッ", "ジー", "ぎゅっ", "ぎゅう",
    "ぱふ", "ぽよん", "ぷにぷに", "ちゅっ", "ぴちゃ", "どくん", "ぞくり", "はっ",
    "ひぅ", "やっ", "ふぅ", "へへ", "えへ", "ぽっ", "かぁぁ", "もじもじ", "ふらふら",
    "ぐるぐる", "ぱたぱた", "とことこ", "ごくり", "ごくん", "すやすや", "ぐぅ", "ぐー",
    "ドカーン", "ズガーン", "バシッ", "バチン", "ビシッ", "スパッ", "ザクッ", "ドスン",
    "ガラガラ", "ゴロゴロ", "ドドドド", "ババババ", "ゴッ", "ガッ", "ボコッ", "ベシッ",
    "シュババ", "ダッ", "タッ", "ピョン", "ピョンピョン", "トコトコ", "スタスタ",
    "ガバッ", "バサッ", "ドサッ", "ボフッ", "ボン", "ポン", "ポカ", "ポカポカ",
    "ワイワイ", "ガヤガヤ", "ザワ", "ゴクリ", "ヒィィ", "ギャァァ", "ウオォォ", "オォォ",
]
SFX_BASES = list(dict.fromkeys(SFX_BASES))  # de-duplicate, keep order

# Tails are literal characters that get rendered by the font (never shapes).
# The ellipsis is spelled "..." so the ground truth follows the same
# normalisation as the dialogue path (``…`` -> ``...``).
SFX_ELONG_TAILS = ["ー", "ーー", "ーーー", "ッ", "ッッ", "〜", "～", "〜〜", "..."]
SFX_MARK_TAILS = [
    "!", "!!", "！", "！！", "!?", "?!", "！？", "？！", "？", "♡", "♥", "★", "☆", "っ", "ん", "...",
]
SMALL_VOWELS_KATA = {"A": "ァ", "I": "ィ", "U": "ゥ", "E": "ェ", "O": "ォ"}
SMALL_VOWELS_HIRA = {"A": "ぁ", "I": "ぃ", "U": "ぅ", "E": "ぇ", "O": "ぉ"}
# Punctuation that shows up in real dialogue; only those the font supports are used.
DIALOGUE_PUNCT = ["、", "。", "！", "？", "ー", "〜", "「", "」", "!", "?", ",", "."]
DIALOGUE_PUNCT_P = [0.16, 0.14, 0.2, 0.12, 0.1, 0.05, 0.05, 0.05, 0.03, 0.03, 0.03, 0.04]


_OBSOLETE_KANA = set("ゐゑゔゕゖヰヱヷヸヹヺヵヶゎヮ")
_SMALL_KANA = set("ぁぃぅぇぉっゃゅょァィゥェォッャュョ")


def _kana_weight(c):
    """Relative sampling weight: everyday kana >> small kana >> obsolete kana."""
    if c in _OBSOLETE_KANA:
        return 0.02
    if c in _SMALL_KANA:
        return 0.25
    return 1.0


def _romaji_vowel(ch):
    """Vowel (A/I/U/E/O) of a kana, from its Unicode name; None when not applicable."""
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return None
    if "SMALL" in name or "LETTER" not in name:
        return None
    v = name.split()[-1][-1]
    return v if v in "AIUEO" else None


def _to_hiragana(s):
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)


def _to_katakana(s):
    return "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in s)


class SyntheticDataGenerator:
    def __init__(self):
        self.vocab, self.hiragana, self.katakana = get_charsets()
        self.len_to_p = pd.read_csv(ASSETS_PATH / "len_to_p.csv")
        self.parser = budoux.load_default_japanese_parser()
        self.fonts_df, self.font_map = get_font_meta()
        self.font_labels, self.font_p = self.get_font_labels_prob()
        self.renderer = Renderer()
        self._pool_cache = {}
        self.sfx_fonts, self.sfx_font_p = self.get_sfx_fonts()

    # ------------------------------------------------------------------ dialogue
    def process(self, text=None, override_css_params=None, sfx=False):
        """
        Generate image, text pair. Use source text if provided, otherwise generate random text.

        SFX mode (``sfx=True`` or ``override_css_params={"sfx": True}``) is a
        separate dataset split: short onomatopoeia rendered huge with curved
        baselines, per-glyph perspective, speedlines and burst balloons. No
        furigana is added and only font-rendered glyphs appear in the GT.
        """

        if override_css_params is None:
            override_css_params = {}
        sfx = bool(sfx or override_css_params.get("sfx", False))

        if sfx:
            return self.process_sfx(text=text, override_css_params=override_css_params)

        # Retry a few times: a font/text pair can end up with nothing renderable
        # after unsupported characters are stripped, and an empty label is noise.
        user_font = "font_path" in override_css_params
        for attempt in range(5):
            params_in = dict(override_css_params)
            lines, vocab = self._prepare_dialogue_lines(text, params_in)
            if lines:
                break
            if user_font:
                break
        if not lines:
            raise ValueError("no renderable text after filtering by font support")
        override_css_params = params_in
        text_gt = "\n".join(lines)

        if np.random.random() < 0.5:
            word_prob = np.random.choice([0.33, 1.0], p=[0.3, 0.7])
            font_path = override_css_params["font_path"]
            lines = [self.add_random_furigana(line, word_prob, vocab, font_path=font_path) for line in lines]

        # The renderer refuses to draw a tofu box (MissingGlyphError). It should
        # never fire after the filtering above; if it ever does, the sample is
        # thrown away rather than shipped with a wrong label.
        img, params = self.renderer.render(lines, override_css_params)
        return img, text_gt, params

    def _prepare_dialogue_lines(self, text, override_css_params):
        """Pick font + text, return (clean lines, font vocab). Mutates override_css_params."""
        if text is None:
            # Random text: choose the font first, then draw only characters it supports.
            if "font_path" not in override_css_params:
                override_css_params["font_path"] = self.get_random_font()
            vocab = self.font_map[override_css_params["font_path"]]
            words = self.get_random_words(vocab)
        else:
            text = text.replace("\u3000", " ")
            text = text.replace("\u2026", "...")
            words = self.split_into_words(text)

        lines = self.words_to_lines(words)

        if "font_path" not in override_css_params:
            override_css_params["font_path"] = self.get_random_font("".join(lines))
        font_path = override_css_params["font_path"]
        vocab = self.font_map.get(font_path)

        # Markup characters would be eaten by the tag parser in the renderer and
        # desync the label from the picture, so they never reach it.
        lines = ["".join(c for c in line if c not in "<>&") for line in lines]
        if vocab is not None:
            # remove unsupported characters so GT == what is actually drawn
            lines = ["".join(c for c in line if c in vocab) for line in lines]
        # Belt and braces against a stale/incomplete fonts.csv: also drop anything
        # the font renders as blank or as a tofu box (checked on the real raster).
        lines = ["".join(c for c in line if is_drawable(font_path, c)) for line in lines]
        # drop lines that became empty / whitespace only (they'd add blank GT rows)
        lines = [line for line in lines if line.strip()]
        return lines, vocab

    # ----------------------------------------------------------------------- SFX
    def get_sfx_fonts(self):
        """Fonts that can draw (nearly) all kana used by the SFX lexicon.

        A font missing kana would render tofu boxes while the label still says
        ``ドン``, so SFX only samples from kana-capable fonts (weighted towards
        ``special``/``common`` display fonts, which are the ones used for
        lettering in manga).
        """
        need = set("".join(SFX_BASES)) | set("ッーっ")
        cover = {}
        for path, chars in self.font_map.items():
            cover[path] = len(need & chars) / len(need)
        if not cover:
            raise RuntimeError("font_map is empty: run scan_fonts.py first")
        good = [p for p, c in cover.items() if c >= 0.98]
        if not good:  # degrade gracefully: best available coverage
            best = max(cover.values(), default=0.0)
            good = [p for p, c in cover.items() if c >= best - 1e-9]
        weight_by_label = {"special": 3.0, "common": 2.0, "regular": 1.0}
        label_of = dict(zip(self.fonts_df.font_path, self.fonts_df.label))
        w = np.array([weight_by_label.get(label_of.get(p, "regular"), 1.0) for p in good], dtype=float)
        return good, w / w.sum()

    def get_random_sfx_font(self):
        return str(np.random.choice(self.sfx_fonts, p=self.sfx_font_p))

    def process_sfx(self, text=None, override_css_params=None):
        """Generate one SFX image/text pair (separate dataset split)."""
        if override_css_params is None:
            override_css_params = {}
        # SFX render defaults; explicit overrides win.
        merged = Renderer.get_random_sfx_params()
        merged.update(override_css_params)
        merged["sfx"] = True
        merged["irregular"] = False
        override_css_params = merged

        user_font = "font_path" in override_css_params
        sfx_text = None
        for _ in range(10):
            if not user_font:
                override_css_params["font_path"] = self.get_random_sfx_font()
            font_path = override_css_params["font_path"]
            vocab = self.font_map.get(font_path, set())

            if text is None:
                sfx_text = self.get_random_sfx_text(vocab)
            else:
                cand = text.replace("\u3000", " ").replace("\u2026", "...")
                cand = "".join(c for c in cand if (c in vocab or c.isspace()) and c not in "<>&")
                sfx_text = cand.strip() or self.get_random_sfx_text(vocab)
            if sfx_text and not all(is_drawable(font_path, c) for c in sfx_text):
                sfx_text = None  # tofu risk: pick another font/text
            if sfx_text:
                break
        if not sfx_text:
            raise ValueError("no SFX text could be rendered with the available fonts")

        # SFX is 1 line, rarely 2 for longer bursts (first line = first column
        # in vertical text; the renderer places it rightmost to match).
        lines = [sfx_text]
        if len(sfx_text) > 4 and np.random.rand() < 0.3:
            cut = len(sfx_text) // 2
            lines = [sfx_text[:cut], sfx_text[cut:]]
        text_gt = "\n".join(lines)

        img, params = self.renderer.render(lines, override_css_params)
        return img, text_gt, params

    def _compose_sfx(self, tails_elong, tails_mark):
        """Build one candidate onomatopoeia (not yet checked against a font)."""
        s = str(np.random.choice(SFX_BASES))
        r = np.random.rand()
        if r < 0.22 and len(s) >= 2:
            # stretched vowel: ドン -> ドォン, ゴゴ -> ゴォォゴ ...
            v = _romaji_vowel(s[0])
            if v:
                small = (SMALL_VOWELS_HIRA if "ぁ" <= s[0] <= "ん" else SMALL_VOWELS_KATA)[v]
                s = s[0] + small * int(np.random.randint(1, 4)) + s[1:]
        elif r < 0.42 and len(s) <= 3:
            unit = s[: int(np.random.randint(1, min(3, len(s)) + 1))]
            s = unit * int(np.random.randint(2, 5))
        elif r < 0.50 and len(s) <= 3:
            s = s[0] * int(np.random.randint(2, 4)) + s  # stutter: ドドドン
        elif r < 0.62 and len(s) <= 4:
            extra = str(np.random.choice(SFX_BASES))
            if len(s + extra) <= 8:
                s += extra
        # script variants: real SFX freely mixes kana scripts
        r = np.random.rand()
        if r < 0.10:
            s = _to_hiragana(s)
        elif r < 0.14:
            s = _to_katakana(s)
        if tails_elong and np.random.rand() < 0.38:
            s += str(np.random.choice(tails_elong))
        if tails_mark and np.random.rand() < 0.32:
            s += str(np.random.choice(tails_mark))
        return s

    def get_random_sfx_text(self, vocab):
        """Sample an onomatopoeia that ``vocab`` can draw in full, or None.

        Every character of the result is guaranteed to be in ``vocab``: samples
        are rejected (never silently truncated), so the label always equals the
        rendered glyphs and words are not mangled into fragments like ``ド``.
        """
        vocab = set(vocab) if vocab is not None else set(self.vocab)
        tails_elong = [t for t in SFX_ELONG_TAILS if all(c in vocab for c in t)]
        tails_mark = [t for t in SFX_MARK_TAILS if all(c in vocab for c in t)]
        for _ in range(60):
            s = self._compose_sfx(tails_elong, tails_mark)
            if 1 <= len(s) <= 10 and all(c in vocab for c in s):
                return s
        # Fallback 1: a plain lexicon word this font can draw fully.
        cands = [b for b in SFX_BASES if all(c in vocab for c in b)]
        if cands:
            return str(np.random.choice(cands))
        # Fallback 2: random kana burst from supported kana only.
        kana = [c for c in self.katakana if c in vocab]
        if len(kana) >= 2:
            return "".join(np.random.choice(kana, int(np.random.randint(2, 5))).tolist())
        return None

    # ------------------------------------------------------------ random dialogue
    def _pools(self, vocab_key, vocab):
        """Per-font character pools by class (cached; classification is slow)."""
        pools = self._pool_cache.get(vocab_key)
        if pools is None:
            pools = {"hira": [], "kata": [], "kanji": [], "digit": [], "latin": [], "punct": [], "other": []}
            for c in vocab:
                if c.isspace():
                    continue
                if is_hiragana(c):
                    pools["hira"].append(c)
                elif is_katakana(c):
                    pools["kata"].append(c)
                elif is_kanji(c):
                    pools["kanji"].append(c)
                elif c.isdigit() and is_ascii(c):
                    pools["digit"].append(c)
                elif is_ascii(c) and c.isalpha():
                    pools["latin"].append(c)
                else:
                    pools["other"].append(c)
            for key in ("hira", "kata"):
                w = np.array([_kana_weight(c) for c in pools[key]], dtype=float)
                pools[key + "_p"] = w / w.sum() if w.size else None
            self._pool_cache[vocab_key] = pools
        return pools

    def get_random_words(self, vocab):
        """Random text shaped like real Japanese instead of uniform noise.

        Words follow simple patterns (kana runs, kanji + okurigana, katakana
        loanwords, punctuation, digits/latin) with class frequencies close to
        manga dialogue, so the model sees kana-dominated text with kanji
        clusters and sentence punctuation rather than rare kanji everywhere.
        """
        pools = self._pools(id(vocab), vocab)
        if len(pools["hira"]) < 10 or len(pools["kanji"]) < 10:
            # Symbol-poor / kana-less fonts: fall back to plain uniform sampling.
            vocab_list = [c for c in vocab if not c.isspace()] or ["あ"]
            return ["".join(np.random.choice(vocab_list, np.random.randint(1, 4))) for _ in range(3)]
        punct = [(p, w) for p, w in zip(DIALOGUE_PUNCT, DIALOGUE_PUNCT_P) if p in vocab]
        if "." in vocab:
            punct.append(("...", 0.06))  # dialogue GT spells the ellipsis "..."

        def pick(pool, lo, hi, p=None):
            n = int(np.random.randint(lo, hi + 1))
            return "".join(np.random.choice(pool, n, p=p).tolist())

        max_text_len = np.random.choice(self.len_to_p.len, p=self.len_to_p.p)
        words, text_len = [], 0
        kinds = ["kana", "kanji", "kata", "punct", "num"]
        kp = np.array([0.42, 0.30, 0.10, 0.12, 0.06])
        while True:
            kind = np.random.choice(kinds, p=kp / kp.sum())
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
            elif kind == "punct" and punct:
                chars, w = zip(*punct)
                w = np.array(w) / sum(w)
                word = str(np.random.choice(chars, p=w))
            elif kind == "num" and (pools["digit"] or pools["latin"]):
                pool = pools["digit"] if (pools["digit"] and (not pools["latin"] or np.random.rand() < 0.6)) else pools["latin"]
                word = pick(pool, 1, 3)
            else:
                word = pick(pools["hira"], 1, 3, pools["hira_p"])
            words.append(word)
            text_len += len(word)
            if text_len + len(word) >= max_text_len:
                break

        return words

    def split_into_words(self, text):
        max_text_len = np.random.choice(self.len_to_p.len, p=self.len_to_p.p)

        words = []
        text_len = 0
        for word in self.parser.parse(text):
            words.append(word)
            text_len += len(word)
            if text_len + len(word) >= max_text_len:
                break

        return words

    def words_to_lines(self, words):
        text = "".join(words)

        max_num_lines = 10
        min_line_len = len(text) // max_num_lines
        # lower bound 1: poisson(6) can draw 0, which would put every word on its own line
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

    def add_random_furigana(self, line, word_prob=1.0, vocab=None, font_path=None):
        if vocab is None:
            vocab = self.vocab
            hiragana, katakana = self.hiragana, self.katakana
        else:
            # Furigana glyphs must exist in the font too, otherwise the ruby
            # row is a strip of tofu boxes (not in the label, but pure noise).
            vocab_set = set(vocab)
            hiragana = [c for c in self.hiragana if c in vocab_set]
            katakana = [c for c in self.katakana if c in vocab_set]
            if font_path is not None:
                hiragana = [c for c in hiragana if is_drawable(font_path, c)]
                katakana = [c for c in katakana if is_drawable(font_path, c)]
            vocab = list(vocab_set)
            if not hiragana:
                return line

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

            if c_type != "kanji" or i == len(line) - 1:
                if kanji_group:
                    if np.random.uniform() < word_prob:
                        furigana_len = int(np.clip(np.random.normal(1.5, 0.5), 1, 4) * len(kanji_group))
                        char_source = np.random.choice(["hiragana", "katakana", "all"], p=[0.8, 0.15, 0.05])
                        char_source = {
                            "hiragana": hiragana,
                            "katakana": katakana if len(katakana) else hiragana,
                            "all": [c for c in vocab if not is_kanji(c) and not c.isspace() and c not in "<>&"] or hiragana,
                        }[char_source]
                        furigana = "".join(np.random.choice(char_source, furigana_len))
                        if font_path is not None:
                            furigana = "".join(c for c in furigana if is_drawable(font_path, c))
                        if furigana:
                            processed += f"<ruby>{kanji_group}<rt>{furigana}</rt></ruby>"
                        else:
                            processed += kanji_group
                    else:
                        processed += kanji_group
                    kanji_group = ""

            if c_type != "ascii" or i == len(line) - 1:
                if ascii_group:
                    if len(ascii_group) <= 3 and np.random.uniform() < 0.7:
                        processed += f'<span style="text-combine-upright: all">{ascii_group}</span>'
                    else:
                        processed += ascii_group
                    ascii_group = ""

            if c_type == "other":
                processed += c

        return processed

    def is_font_supporting_text(self, font_path, text):
        chars = self.font_map[font_path]
        for c in text:
            if c.isspace():
                continue
            if c not in chars:
                return False
        return True

    def get_font_labels_prob(self):
        labels = {
            "common": 0.2,
            "regular": 0.75,
            "special": 0.05,
        }
        labels = {k: labels[k] for k in self.fonts_df.label.unique()}
        p = np.array(list(labels.values()))
        p = p / p.sum()
        labels = list(labels.keys())
        return labels, p

    def get_random_font(self, text=None):
        label = np.random.choice(self.font_labels, p=self.font_p)
        df = self.fonts_df[self.fonts_df.label == label]

        if text is None:
            return str(df.sample(1).iloc[0].font_path)

        valid_mask = df.font_path.apply(lambda x: self.is_font_supporting_text(x, text))
        if not valid_mask.any():
            # try every label before giving up on full support
            valid_all = self.fonts_df.font_path.apply(lambda x: self.is_font_supporting_text(x, text))
            if valid_all.any():
                return str(self.fonts_df[valid_all].sample(1).iloc[0].font_path)
            # text contains characters not supported by any font: pick one of the more capable fonts
            valid_mask = df.num_chars >= 4000
            if not valid_mask.any():
                valid_mask = df.num_chars >= df.num_chars.max()

        return str(df[valid_mask].sample(1).iloc[0].font_path)
