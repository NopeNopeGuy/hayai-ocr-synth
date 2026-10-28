"""Procedural Japanese onomatopoeia (SFX) text generator.

Instead of picking from a fixed lexicon (a few hundred words), this builds
words from a mora grammar, so the number of distinct outputs is effectively
unbounded while every result still *looks* like real SFX:

  * "grammar" path  - assembles words from moras with real SFX shapes
                      (バッ, ドォン, ガシャーン, ドキドキ, ニヤリ, ドッカーン ...)
  * "mutation" path - takes a lexicon word and mutates it (voicing, stretching,
                      repetition, ending swaps, compounding)

Every character is checked against the font vocab, and the mora pool is
built per (script, supported-kana) so unsupported glyphs are never produced.
"""

import unicodedata

import numpy as np

# consonant row -> (a, i, u, e, o); "" = bare vowels. None = does not exist.
_ROWS = {
    "": "アイウエオ",
    "k": "カキクケコ",
    "g": "ガギグゲゴ",
    "s": "サシスセソ",
    "z": "ザジズゼゾ",
    "t": "タチツテト",
    "d": "ダ..デド",
    "n": "ナニヌネノ",
    "h": "ハヒフヘホ",
    "b": "バビブベボ",
    "p": "パピプペポ",
    "m": "マミムメモ",
    "r": "ラリルレロ",
    "y": "ヤ.ユ.ヨ",
    "w": "ワ...",
}
# How typical each onset is in SFX (voiced/plosive onsets dominate impact SFX).
_ONSET_W = {
    "": 0.12, "k": 1.2, "g": 1.4, "s": 0.9, "z": 1.1, "t": 0.9, "d": 1.5,
    "n": 0.6, "h": 0.9, "b": 1.4, "p": 1.1, "m": 0.7, "r": 0.5, "y": 0.4, "w": 0.4,
}
# a, i, u, e, o: a/o/u dominate in SFX (ドン, バン, ガン, ゴゴ).
_VOWEL_W = [1.3, 0.8, 1.0, 0.6, 1.3]
_YOON_STEMS = "キギシジチヒビピリ"
_YOON_SMALL = "ャュョ"
_FOREIGN = ["ティ", "ディ", "ファ", "フィ", "フォ", "シェ", "ジェ", "チェ"]  # rare flavour
_SMALL_VOWELS = "ァィゥェォ"


def _to_hira(s):
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)


def _to_kata(s):
    return "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in s)


def _vowel_of(ch):
    """A/I/U/E/O of a (non-small) kana, else None."""
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return None
    if "SMALL" in name or "LETTER" not in name:
        return None
    v = name.split()[-1][-1]
    return v if v in "AIUEO" else None


def _voice(c, mark="\u3099"):
    """Add a (semi-)voiced mark if the kana supports it, else return c unchanged."""
    out = unicodedata.normalize("NFC", c + mark)
    return out if len(out) == 1 else c


def _devoice(c):
    d = unicodedata.normalize("NFD", c)
    return unicodedata.normalize("NFC", d[0]) if len(d) == 2 else c


def _rand(a, b):
    """Uniform int in [a, b]."""
    return int(np.random.randint(a, b + 1))


class _Pool:
    """Moras and specials usable in one script with one font's kana support."""

    def __init__(self, hira, vocab):
        conv = _to_hira if hira else (lambda s: s)
        ok = lambda s: all(c in vocab for c in s)  # noqa: E731
        self.hira = hira
        moras, w = [], []
        for cons, row in _ROWS.items():
            for vi, k in enumerate(row):
                if k == "." or not ok(conv(k)):
                    continue
                moras.append(conv(k))
                w.append(_ONSET_W[cons] * _VOWEL_W[vi])
        yoon = []
        for st in _YOON_STEMS:
            for sm in _YOON_SMALL:
                if ok(conv(st + sm)):
                    yoon.append(conv(st + sm))
        foreign = [] if hira else [f for f in _FOREIGN if ok(f)]
        self.plain = moras
        self.yoon = yoon
        self.foreign = foreign
        # Final sampling table; yoon ~14%, foreign ~2% of moras.
        allm = moras + yoon + foreign
        weights = np.array(w + [0.0] * (len(yoon) + len(foreign)), dtype=float)
        if weights.sum() == 0:
            self.moras, self.p = [], None
            return
        base = weights.sum()
        if yoon:
            weights[len(moras) : len(moras) + len(yoon)] = base * 0.14 / len(yoon)
        if foreign:
            weights[len(moras) + len(yoon) :] = base * 0.02 / len(foreign)
        self.moras = allm
        self.p = weights / weights.sum()
        self.tsu = conv("ッ") if ok(conv("ッ")) else None
        self.n = conv("ン") if ok(conv("ン")) else None
        self.long = "ー" if "ー" in vocab else None
        self.ri = conv("リ") if ok(conv("リ")) else None
        self.smalls = {}
        for v, sm in zip("AIUEO", _SMALL_VOWELS):
            if ok(conv(sm)):
                self.smalls[v] = conv(sm)

    def usable(self):
        return len(self.moras) >= 10 and self.n is not None and self.tsu is not None

    def m(self):
        return self.moras[np.random.choice(len(self.moras), p=self.p)]

    def small_after(self, mora):
        """Small vowel matching the mora's final vowel (ド -> ォ), else any."""
        v = _vowel_of(mora[-1]) if mora[-1] not in _YOON_SMALL and _to_kata(mora[-1]) not in _YOON_SMALL else None
        if v and v in self.smalls:
            return self.smalls[v]
        return next(iter(self.smalls.values()), None) if self.smalls else None


class SFXComposer:
    """Unbounded SFX text sampler. Call ``sample(vocab)``; returns str or None."""

    def __init__(self, bases, elong_tails, mark_tails, lexicon_p=0.3, max_len=10):
        self.bases = list(bases)
        self.elong_tails = list(elong_tails)
        self.mark_tails = list(mark_tails)
        self.lexicon_p = lexicon_p
        self.max_len = max_len
        self._pools = {}
        self._kana = frozenset("".join(_ROWS.values()) + _YOON_STEMS + _YOON_SMALL + _SMALL_VOWELS + "ッンー") - {"."}
        self._patterns = [
            (self._burst, 1.2),      # バッ ガッ
            (self._impact, 1.6),     # ドン ドォン ズドォォン
            (self._crash, 1.0),      # ガシャーン バリーン
            (self._repeat, 2.0),     # ドキドキ ガタガタ ダダダ
            (self._alt_repeat, 0.8),  # ドカドカ ゴロゴロ (voicing alternation)
            (self._geminate, 1.0),   # ドッカーン ガッシャン
            (self._tail_ri, 0.7),    # ニヤリ ゴクリ
            (self._stretch, 1.0),    # ゴォォ ザァァ ヒュゥゥ
            (self._breath, 0.5),     # ハァハァ フゥ
            (self._stutter, 0.6),    # ドドドン ババババ
        ]
        w = np.array([w for _, w in self._patterns])
        self._pattern_p = w / w.sum()

    # ------------------------------------------------------------------ public
    def sample(self, vocab, font_path=None, max_len=None):
        # font_path accepted (ignored) for a uniform composer interface.
        max_len = max_len or self.max_len
        vocab = vocab if isinstance(vocab, (set, frozenset)) else set(vocab)
        hira = np.random.rand() < 0.12
        pool = self._get_pool(hira, vocab)
        if pool is None:
            hira = not hira
            pool = self._get_pool(hira, vocab)
            if pool is None:
                return None
        conv = _to_hira if hira else _to_kata

        if np.random.rand() < self.lexicon_p and self.bases:
            s = self._mutate(conv(str(np.random.choice(self.bases))), pool)
        else:
            fn = self._patterns[np.random.choice(len(self._patterns), p=self._pattern_p)][0]
            s = fn(pool)
            if np.random.rand() < 0.12:  # compound of two generated words
                s += fn(pool) if np.random.rand() < 0.5 else self._patterns[_rand(0, 2)][0](pool)

        s = self._add_tails(s, vocab)
        if 1 <= len(s) <= max_len and all(c in vocab for c in s):
            return s
        return None

    # ------------------------------------------------------------------ pools
    def _get_pool(self, hira, vocab):
        key = (hira, frozenset(c for c in self._kana if c in vocab))
        if key not in self._pools:
            pool = _Pool(hira, vocab)
            self._pools[key] = pool if pool.usable() else None
            while len(self._pools) > 64:
                self._pools.pop(next(iter(self._pools)))
        return self._pools[key]

    # --------------------------------------------------------------- patterns
    def _long_or(self, p, default=""):
        return p.long if p.long and np.random.rand() < 0.5 else default

    def _burst(self, p):
        return p.m() + p.tsu

    def _impact(self, p):
        m = p.m()
        s = m
        sm = p.small_after(m)
        if sm and np.random.rand() < 0.35:
            s += sm * _rand(1, 3)
        elif p.long and np.random.rand() < 0.2:
            s += p.long
        return s + p.n

    def _crash(self, p):
        a = p.m()
        b = p.yoon[_rand(0, len(p.yoon) - 1)] if p.yoon else p.m()
        mid = p.long if p.long and np.random.rand() < 0.6 else ""
        return a + b + mid + p.n

    def _repeat(self, p):
        unit = "".join(p.m() for _ in range(_rand(1, 2)))
        n = _rand(2, 4) if len(unit) == 1 else _rand(2, 3)
        s = unit * n
        r = np.random.rand()
        if r < 0.2:
            s = s[: -len(unit)] + unit + p.tsu
        elif r < 0.3 and p.long:
            s += p.long
        return s

    def _alt_repeat(self, p):
        a, b = p.m(), p.m()
        if np.random.rand() < 0.5:
            b = _devoice(b[0]) + b[1:]
        n = _rand(2, 3)
        return (a + b) * n if np.random.rand() < 0.5 else (a + b) * (n - 1) + a

    def _geminate(self, p):
        s = p.m() + p.tsu + p.m()
        if p.long and np.random.rand() < 0.6:
            s += p.long
        return s + (p.n if np.random.rand() < 0.7 else p.tsu)

    def _tail_ri(self, p):
        if not p.ri:
            return p.m() + p.m() + p.tsu
        return p.m() + p.m() + p.ri

    def _stretch(self, p):
        m = p.m()
        sm = p.small_after(m)
        if not sm:
            return m + (p.long or p.tsu)
        s = m + sm * _rand(1, 4)
        if np.random.rand() < 0.35:
            s = m + sm * _rand(1, 2) + p.m() + sm * _rand(1, 2)
        return s

    def _breath(self, p):
        a = p.m()
        sm = p.small_after(a)
        unit = a + (sm if sm and np.random.rand() < 0.7 else "")
        return unit * _rand(1, 3)

    def _stutter(self, p):
        m = p.m()
        s = m * _rand(2, 4)
        return s + (p.n if np.random.rand() < 0.6 else p.tsu)

    # --------------------------------------------------------------- mutation
    def _mutate(self, s, pool):
        """Apply 1-3 random edits to a lexicon word."""
        for _ in range(_rand(1, 3)):
            r = np.random.rand()
            if r < 0.18:  # flip (de)voicing of the first kana
                c = s[0]
                new = _devoice(c) if _devoice(c) != c else _voice(c)
                s = new + s[1:]
            elif r < 0.36 and len(s) >= 2:  # stretch first mora
                sm = pool.small_after(s[0])
                if sm:
                    s = s[0] + sm * _rand(1, 3) + s[1:]
            elif r < 0.52 and len(s) <= 4:  # repeat the word / a unit
                unit = s[: _rand(1, min(3, len(s)))] if np.random.rand() < 0.5 else s
                s = s + unit * _rand(1, 2)
            elif r < 0.66:  # swap the ending
                end = [e for e in (pool.tsu, pool.n, pool.long, pool.ri) if e]
                s = (s[:-1] if len(s) > 1 else s) + str(np.random.choice(end))
            elif r < 0.78 and len(s) >= 2:  # geminate after first mora
                s = s[0] + pool.tsu + s[1:]
            elif r < 0.9:  # compound with another generated chunk
                s += self._patterns[_rand(0, len(self._patterns) - 1)][0](pool)
            elif len(s) > 2:  # truncate
                s = s[: _rand(2, len(s) - 1)]
            if len(s) > self.max_len:
                s = s[: self.max_len]
        return s

    # ------------------------------------------------------------------ tails
    def _add_tails(self, s, vocab):
        elong = [t for t in self.elong_tails if all(c in vocab for c in t)]
        marks = [t for t in self.mark_tails if all(c in vocab for c in t)]
        if elong and np.random.rand() < 0.3 and s[-1] not in "ーッっ":
            s += str(np.random.choice(elong))
        if marks and np.random.rand() < 0.3:
            s += str(np.random.choice(marks))
        return s


# ---------------------------------------------------------------------------
# Multilingual SFX composers (ko/zh/en).
#
# Same contract as ``SFXComposer``: ``sample(vocab, font_path=None)`` returns a
# string or None. ``vocab`` is the FontMap set (vocab.csv-limited); characters
# outside it are still usable when the font really draws them (checked via
# ``is_drawable``), because fonts.csv only records vocab.csv coverage.
# ---------------------------------------------------------------------------

def _ok_checker(vocab, font_path=None):
    """Return predicate ``ok(s)``: every char of ``s`` is renderable."""
    vocab = vocab if isinstance(vocab, (set, frozenset)) else set(vocab)
    if font_path is None:
        return lambda s: all(c in vocab for c in s)
    from manga_ocr_dev.synthetic_data_generator.utils import is_drawable

    def ok(s):
        for c in s:
            if c in vocab:
                continue
            if c.isspace():
                continue
            if not is_drawable(font_path, c):
                return False
        return True

    return ok


def _filter_tails(tails, ok):
    return [t for t in tails if ok(t)]


class _BaseSFXComposer:
    """Shared init / pattern-pick / tail logic for ko/zh/en SFX composers."""

    def __init__(self, bases, elong_tails, mark_tails, lexicon_p, max_len):
        self.bases = list(bases)
        self.elong_tails = list(elong_tails)
        self.mark_tails = list(mark_tails)
        self.lexicon_p = lexicon_p
        self.max_len = max_len
        self._patterns = []
        self._pattern_p = None

    def _init_patterns(self, patterns):
        self._patterns = list(patterns)
        w = np.array([w for _, w in self._patterns], dtype=float)
        self._pattern_p = w / w.sum()

    def _random_pattern(self):
        return self._patterns[int(np.random.choice(len(self._patterns), p=self._pattern_p))][0]

    def _lexicon_cands(self, ok):
        return [b for b in self.bases if ok(b)] if self.bases else []

    def _lexicon_fallback(self, ok, default="BOOM"):
        cands = self._lexicon_cands(ok)
        return str(np.random.choice(cands)) if cands else default

    def _add_tails(self, s, ok, elong_p=0.25, mark_p=0.35):
        elong = _filter_tails(self.elong_tails, ok)
        marks = _filter_tails(self.mark_tails, ok)
        if elong and np.random.rand() < elong_p:
            s += str(np.random.choice(elong))
        if marks and np.random.rand() < mark_p:
            s += str(np.random.choice(marks))
        return s

    def _mutate_generic(self, s, pool, repeat_p=0.35, swap_p=0.55, compound_p=0.70):
        """Shared lexicon mutation: repeat / swap ending / compound / truncate."""
        for _ in range(_rand(1, 2)):
            r = np.random.rand()
            if r < repeat_p and len(s) <= 4:  # repeat a unit
                s += s[: _rand(1, min(3, len(s)))]
            elif r < swap_p and pool:  # swap ending
                s = (s[:-1] if len(s) > 1 else s) + str(np.random.choice(pool))
            elif r < compound_p:  # compound with another generated chunk
                s += self._random_pattern()(pool)
            elif len(s) > 2:  # truncate
                s = s[: _rand(2, len(s) - 1)]
            if len(s) > self.max_len:
                s = s[: self.max_len]
        return s


# Frequent Hangul syllables for the procedural Korean pool (every entry is
# precomposed U+AC00-U+D7A3).
_COMMON_HANGUL_FREQ = (
    "가나다는라마바사아자차카타파하거너더러머버서어저처커터퍼허"
    "고노도로모보소오조초코토포호구누두루무부수우주추쿠투푸후"
    "기니디리미비시이지치키티피히그느드르므브스으즈츠크트프흐"
    "개내대래매배새애재채캐태패해게네데레메베세에제체케테페헤"
    "과놔돠봐솨와좌콰톼퐈화괴뇌되뢰뫼뵈쇠외죄최쾨퇴푀회교뇨료묘"
    "쇼요죠쿄툐표효뉴듀드뷰슈유츄큐튜퓨휴근는든른믄븐슨은"
    "즌큰튼픈흔글늘들믈블슬을즐클틀플흘금님딤림밈빔심임짐침"
    "킴팀힘갑답랩맵샙앱잽챕캡탭팹합것넛덧멋벗섯엇젓쳇컷텃"
    "펏헛공농동롱몽봉송옹종총콩통퐁홍광낭당랑망방상앙장창캉탕팡"
    "항국눈둔룬문분순운준춘쿤툰푼훈굴눌둘룰물불술울줄출쿨툴풀훌"
    "궁둥뭉숭웅중충쿵풍훙권넌던런먼번선언전천컨턴펀헌걸널"
    "덜럴멀벌설얼절철컬털펄헐검넘덤럼멈범섬엄점첨컴템펌험겁덥"
    "럽멉법섭업접첩컵텁펍헙겹봅솁엽쳡협경녕덩렁멍"
    "벙성엉정청컹텅펑헝룡봉송용종총콩통퐁홍밖값삶앉않읽없었있했됐"
)


class KoreanSFXComposer(_BaseSFXComposer):
    """Korean onomatopoeia sampler (Hangul syllables + ASCII tails)."""

    def __init__(self, bases, elong_tails, mark_tails, lexicon_p=0.45, max_len=10):
        super().__init__(bases, elong_tails, mark_tails, lexicon_p, max_len)
        # Static frequent-syllable pool (kept Hangul-only, de-duplicated).
        freq = "".join(c for c in _COMMON_HANGUL_FREQ if "\uac00" <= c <= "\ud7a3")
        self._freq = list(dict.fromkeys(freq))
        self._init_patterns(
            [
                (self._burst, 1.4),  # 쿵 쾅
                (self._repeat, 1.8),  # 두근두근 콩닥콩닥
                (self._stutter, 1.0),  # 다다다 두두두
                (self._stretch, 1.0),  # 우우우 아아아
                (self._double, 0.8),  # 쿵쾅 벌벌
            ]
        )

    def sample(self, vocab, font_path=None, max_len=None):
        max_len = max_len or self.max_len
        ok = _ok_checker(vocab, font_path)
        pool = self._pool(vocab, font_path, ok)
        if np.random.rand() < self.lexicon_p and self.bases:
            cands = self._lexicon_cands(ok)
            if cands:
                s = self._mutate(str(np.random.choice(cands)), pool, ok)
            else:
                s = self._random_pattern()(pool)
        else:
            s = self._random_pattern()(pool)
            if np.random.rand() < 0.10 and pool:
                s += self._random_pattern()(pool)
        if not s:
            return None
        s = self._add_tails(s, ok)
        if 1 <= len(s) <= max_len and ok(s):
            return s
        return None

    def _pool(self, vocab, font_path, ok):
        vocab = vocab if isinstance(vocab, (set, frozenset)) else set(vocab)
        from manga_ocr_dev.synthetic_data_generator.utils import is_hangul

        # Prefer frequent syllables + lexicon characters (natural Korean);
        # fall back to the full vocab Hangul pool only if too few survive.
        freq = [c for c in self._freq if ok(c)]
        lex = list({c for b in self.bases for c in b if is_hangul(c) and ok(c)})
        pool = list(dict.fromkeys(freq + lex))
        if len(pool) < 20:
            extra = [c for c in vocab if is_hangul(c) and ok(c)]
            pool = list(dict.fromkeys(pool + extra))
        return pool or [c for c in vocab if ok(c) and not c.isspace()] or ["가", "나"]

    def _burst(self, pool):
        return str(np.random.choice(pool))

    def _repeat(self, pool):
        unit = "".join(np.random.choice(pool, _rand(1, 2)).tolist())
        return unit * _rand(2, 3)

    def _stutter(self, pool):
        m = str(np.random.choice(pool))
        return m * _rand(2, 4)

    def _stretch(self, pool):
        m = str(np.random.choice(pool))
        return m + m * _rand(1, 3)

    def _double(self, pool):
        a, b = np.random.choice(pool, 2).tolist()
        return a + b if np.random.rand() < 0.5 else a + b + a + b

    def _mutate(self, s, pool, _ok):
        # Korean thresholds preserved; shared implementation in base.
        return self._mutate_generic(s, pool, repeat_p=0.30, swap_p=0.50, compound_p=0.65)


class ChineseSFXComposer(_BaseSFXComposer):
    """Chinese onomatopoeia sampler (Han characters + CJK punctuation)."""

    # Han characters that actually occur in onomatopoeia / interjections.
    # The procedural pool is drawn from these (plus lexicon chars), never from
    # the full 6k Han vocab (which would give rare characters like 鮫/姾).
    _SFX_HAN = (
        "哈哈呵嘿哼啊呀哇呜呼轰砰啪嘭咚嗖飕隆哗啦劈啪滴答嘀嗒叮咚"
        "咔嚓哐当乒乓噼啪扑哧嘿咻冲杀嗷汪喵咩哞喔咯嘎呱吱嗡"
        "沙淅簌咕噜嘟锵当啷叮铃哔唔嗯哦恩"
        "唉哎哟喂吼叫嚎啸呼呼飕飕嗖嗖沙沙淅淅"
    )

    def __init__(self, bases, elong_tails, mark_tails, lexicon_p=0.5, max_len=10):
        super().__init__(bases, elong_tails, mark_tails, lexicon_p, max_len)
        self._init_patterns(
            [
                (self._single, 1.2),  # 砰 轰
                (self._repeat, 1.8),  # 哈哈哈 砰砰砰
                (self._aabb, 1.0),  # 哗啦哗啦
                (self._abab, 0.8),  # 轰隆轰隆
            ]
        )

    def sample(self, vocab, font_path=None, max_len=None):
        max_len = max_len or self.max_len
        ok = _ok_checker(vocab, font_path)
        pool = self._pool(vocab, font_path, ok)
        if not pool:
            return None
        if np.random.rand() < self.lexicon_p and self.bases:
            cands = self._lexicon_cands(ok)
            s = self._mutate(str(np.random.choice(cands)), pool, ok) if cands else self._random_pattern()(pool)
        else:
            s = self._random_pattern()(pool)
        if not s:
            return None
        s = self._add_tails(s, ok)
        if 1 <= len(s) <= max_len and ok(s):
            return s
        return None

    def _pool(self, vocab, font_path, ok):
        vocab = vocab if isinstance(vocab, (set, frozenset)) else set(vocab)
        from manga_ocr_dev.synthetic_data_generator.utils import is_kanji

        # Onomatopoeia Han first (natural SFX); fall back to vocab Han only
        # when the font draws almost none of them.
        prefer = list(dict.fromkeys([c for c in self._SFX_HAN if is_kanji(c)] + [c for b in self.bases for c in b if is_kanji(c)]))
        pool = [c for c in prefer if ok(c)]
        if len(pool) < 10:
            extra = [c for c in vocab if is_kanji(c) and ok(c)]
            pool = list(dict.fromkeys(pool + extra))
        return pool

    def _single(self, pool):
        return str(np.random.choice(pool))

    def _repeat(self, pool):
        unit = "".join(np.random.choice(pool, _rand(1, 2)).tolist())
        return unit * _rand(2, 4)

    def _aabb(self, pool):
        # Prefer real reduplicated bigrams from the lexicon (哗啦, 轰隆);
        # random Han pairings often read as nonsense words.
        if self.bases and np.random.rand() < 0.7:
            cands = [b for b in self.bases if len(b) >= 2]
            if cands:
                b = str(np.random.choice(cands))[:2]
                return b[0] + b[0] + b[1] + b[1] if np.random.rand() < 0.5 else (b[0] + b[1]) * 2
        a, b = np.random.choice(pool, 2).tolist()
        return a + a + b + b if np.random.rand() < 0.5 else (a + b) * 2

    def _abab(self, pool):
        if self.bases and np.random.rand() < 0.7:
            cands = [b for b in self.bases if len(b) >= 2]
            if cands:
                b = str(np.random.choice(cands))[:2]
                return (b[0] + b[1]) * _rand(2, 3)
        a, b = np.random.choice(pool, 2).tolist()
        return (a + b) * _rand(2, 3)

    def _mutate(self, s, pool, _ok):
        return self._mutate_generic(s, pool)


class EnglishSFXComposer(_BaseSFXComposer):
    """English onomatopoeia sampler (Latin comic lettering, mostly UPPERCASE)."""

    _ONSETS = ["B", "P", "K", "T", "D", "S", "Z", "W", "CR", "BL", "SPL", "THW", "SM", "GR", "CL", "WH", "Z", "V", "F", "H"]
    _NUCLEI = ["OO", "OW", "A", "E", "I", "O", "U", "EE", "AA", "OU"]
    _CODAS = ["M", "NG", "SH", "CK", "P", "T", "D", "G", "ZZ", "SHH", "NG", "P", "FF"]

    def __init__(self, bases, elong_tails, mark_tails, lexicon_p=0.55, max_len=16):
        super().__init__(bases, elong_tails, mark_tails, lexicon_p, max_len)

    def sample(self, vocab, font_path=None, max_len=None):
        max_len = max_len or self.max_len
        ok = _ok_checker(vocab, font_path)
        if np.random.rand() < self.lexicon_p and self.bases:
            cands = self._lexicon_cands(ok)
            if cands:
                s = self._mutate(str(np.random.choice(cands)), ok)
            else:
                s = self._generate(ok)
        else:
            s = self._generate(ok)
            if np.random.rand() < 0.12 and self.bases:
                cands = self._lexicon_cands(ok)
                if cands:
                    s += " " + str(np.random.choice(cands)) if np.random.rand() < 0.3 else str(np.random.choice(cands))
        if not s:
            return None
        s = self._case(s)
        s = self._add_tails(s, ok)
        if 1 <= len(s) <= max_len and ok(s):
            return s
        return None

    def _generate(self, ok):
        for _ in range(10):
            o = str(np.random.choice(self._ONSETS))
            n = str(np.random.choice(self._NUCLEI))
            c = str(np.random.choice(self._CODAS)) if np.random.rand() < 0.8 else ""
            if np.random.rand() < 0.35:  # vowel stretch: BOOOOM
                n = n[0] * _rand(2, 4) if len(n) == 1 else n + n[-1] * _rand(1, 3)
            s = o + n + c
            if np.random.rand() < 0.25:  # repeat: HA HA HA
                s = " ".join([s] * _rand(2, 3))
            elif np.random.rand() < 0.2:  # double: CRASHCRASH
                s = s * 2
            if ok(s.replace(" ", "")) and len(s.replace(" ", "")) <= self.max_len:
                return s
        return self._lexicon_fallback(ok)

    def _mutate(self, s, ok):
        base = s.replace(" ", "")
        for _ in range(_rand(1, 2)):
            r = np.random.rand()
            if r < 0.30:  # vowel stretch
                for v in "AEIOU":
                    if v in base and np.random.rand() < 0.5:
                        base = base.replace(v, v * _rand(2, 4), 1)
                        break
            elif r < 0.50 and len(base) <= 5:  # repeat
                base = base + base[: _rand(1, min(3, len(base)))]
            elif r < 0.65:  # exclamation compound
                base += str(np.random.choice(["BOOM", "POW", "BANG", "CRASH"]))
            elif len(base) > 3:
                base = base[: _rand(2, len(base) - 1)]
            if len(base) > self.max_len:
                base = base[: self.max_len]
        return base if ok(base) else s

    def _case(self, s):
        # Comic SFX is mostly UPPERCASE; keep some variety for training.
        r = np.random.rand()
        if r < 0.75:
            return s.upper()
        if r < 0.90:
            return s.title() if " " in s else s.capitalize()
        return s

    def _add_tails(self, s, ok):
        elong = _filter_tails(self.elong_tails, ok)
        marks = _filter_tails(self.mark_tails, ok)
        nospace = s.replace(" ", "")
        if elong and np.random.rand() < 0.20 and (not nospace or nospace[-1] not in "OMH"):
            s += str(np.random.choice(elong))
        if marks and np.random.rand() < 0.45:
            s += str(np.random.choice(marks))
        return s
