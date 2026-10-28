"""Language configuration for multilingual synthetic OCR data (ja/ko/zh/en).

Japanese (``ja``) is the original pipeline: vertical text, furigana/ruby,
tate-chu-yoko, kana mora SFX. This module adds the typographic defaults for:

* ``ko`` - Korean (Hangul syllables, spaces between words, horizontal only)
* ``zh`` - Chinese (Han characters, no spaces, mostly horizontal)
* ``en`` - English (Latin words, spaces, horizontal only)

``zh`` covers both simplified and traditional Han: the generator samples Han
characters from the font's own coverage, so the script variant follows the
fonts installed (SourceHan-style CJK fonts draw both).
"""

from __future__ import annotations

SUPPORTED_LANGS = ("ja", "ko", "zh", "en")

# Vertical-text probability per language for dialogue. Japanese manga mixes
# vertical and horizontal; Chinese manhua is mostly horizontal with occasional
# vertical; Korean webtoons and English comics are (near-)always horizontal.
DIALOGUE_VERTICAL_PROB = {
    "ja": 0.70,
    "zh": 0.12,
    "ko": 0.02,
    "en": 0.0,
}

# Vertical-text probability per language for SFX lettering.
SFX_VERTICAL_PROB = {
    "ja": 0.40,
    "zh": 0.15,
    "ko": 0.05,
    "en": 0.0,
}

# OpenType ``language`` tag used for vertical (ttb) shaping in Pillow.
TTB_LANGUAGE = {
    "ja": "ja",
    "zh": "zh",
    "ko": "ko",
    "en": "en",
}

# Dialogue punctuation per language (weights in generator.py). Only entries
# the font can draw are used.
DIALOGUE_PUNCT = {
    "ja": (["、", "。", "！", "？", "ー", "〜", "「", "」", "!", "?", ",", "."], [0.16, 0.14, 0.2, 0.12, 0.1, 0.05, 0.05, 0.05, 0.03, 0.03, 0.03, 0.04]),
    "ko": ([".", ",", "!", "?", "~", "…", "。", "！", "？", "“", "”", "‘", "’"], [0.18, 0.16, 0.18, 0.12, 0.08, 0.05, 0.04, 0.06, 0.04, 0.03, 0.03, 0.015, 0.015]),
    "zh": (["，", "。", "！", "？", "、", "：", "；", "“", "”", "‘", "’", "…", "—", "!", "?", ",", "."], [0.16, 0.14, 0.14, 0.10, 0.08, 0.05, 0.04, 0.04, 0.04, 0.025, 0.025, 0.05, 0.03, 0.03, 0.02, 0.02, 0.03]),
    "en": ([".", ",", "!", "?", "...", '"', "'", "—", ":", ";", "(", ")"], [0.22, 0.20, 0.16, 0.12, 0.06, 0.05, 0.05, 0.03, 0.04, 0.03, 0.02, 0.02]),
}

# SFX seed lexicons per language (procedural composers use these for the
# "mutation" path and as a fallback; most SFX text is generated procedurally).
# Seed lexicon for Japanese (impact, motion, emotion, ambient, liquids/fire,
# body reactions, facial expressions, sparkle, soft hiragana SFX, ecchi).
_JA_SFX_BASES = """
ドン バン ガン ドカン ズドン ドゴォン バゴォン ガシャン バリン パリン メキ メキメキ
バキ ボキ グシャ ズシャ ドス ズシン ゴン ガツン ボゴン ザッ サッ シュッ ビュン ヒュン
ゴゴゴ ザワザワ ザザッ ダダダ タタタ ドドド ガガガ ビリビリ グラグラ ユラ フワ パッ バッ
ワー キャー ギャー ウワァン エーン クスクス アハハ ゲラゲラ ヒソヒソ ドキドキ ドキン
ズキュン キュン ハッ ホッ フゥ ハァ ヒィッ あん あっ んっ はぁ ふぁ ひゃ くちゅ ぐちゅ
ぬちゅ びちゃ ぷるん ぷるぷる ぞく ぞくぞく ビク ビクン ハァハァ どん ばん がしゃん
どきどき
シーン ゴォォ ザーッ ポツポツ ザァァ ヒュゥゥ ビュオオ ジジジ カチッ ピッ ピピッ
ブォォン キィィ ガタン ゴトン ガタガタ コツコツ カツカツ ペタペタ ジャー ポチャン
ボコボコ グツグツ ジュウウ パチパチ メラメラ ボォォ ドロドロ ヌルッ ヌルヌル ゾクッ
ゾワッ ビクッ ガクッ ガクガク ブルブル ワナワナ ソワソワ イライラ モヤモヤ ズーン
ガーン ピキーン キラーン キラキラ ピカッ ピカピカ ニコニコ ニヤリ ニヤニヤ ムスッ
プンプン ムッ ジロッ ギロッ チラッ ジー ぎゅっ ぎゅう ぱふ ぽよん ぷにぷに ちゅっ
ぴちゃ どくん ぞくり はっ ひぅ やっ ふぅ へへ えへ ぽっ かぁぁ もじもじ ふらふら
ぐるぐる ぱたぱた とことこ ごくり ごくん すやすや ぐぅ ぐー ドカーン ズガーン バシッ
バチン ビシッ スパッ ザクッ ドスン ガラガラ ゴロゴロ ドドドド ババババ ゴッ ガッ ボコッ
ベシッ シュババ ダッ タッ ピョン ピョンピョン トコトコ スタスタ ガバッ バサッ ドサッ
ボフッ ボン ポン ポカ ポカポカ ワイワイ ガヤガヤ ザワ ゴクリ ヒィィ ギャァァ ウオォォ
オォォ
""".split()
_JA_SFX_BASES = list(dict.fromkeys(_JA_SFX_BASES))  # de-duplicate, keep order

SFX_BASES = {
    "ja": _JA_SFX_BASES,
    "ko": """
쿵 쾅 펑 팡 콰광 두두두 다다다 타타타 파파파 바바바 삐용 슈웅 쌩
콩닥콩닥 두근두근 쿵쾅 벌벌덜덜 후덜덜
와아아 으아아 꺄아아 흐흑 흑흑 엉엉 하하 호호 히히 킥킥 쿨쿨 드르렁
쿵 쾅 펑 팡 콰광 두두두 다다다 타타타 파파파 바바바 삐용 슈웅 쌩
철썩 쿵쿵 두근 쿵쿵콩닥 아앗 앗앗 헉 휴우 후우 쉬익 스윽 스슥 촤악
빠밤 두둥 따단 짜잔 뿅 뿅뿅 샤랄랄라 반짝반짝 번쩍 콰과광 우르릉
덜컹 덜컹덜컹 쿨럭 지글지글 보글보글 치직 지지직 파직 탁 탁탁 톡톡
두드득 바스락 사락사락 바삭 와삭 와그작 꿀꺽 꼴깍 츄릅 쪽 쭈욱 냠냠
오물오물 바들바들 오들오들 부들부들 힐끗 흘끗 빤히 지긋이 스윽
""".split(),
    "zh": """
砰 轰 哗 咚 啪 嘭 嗖 呼 哈 呜 啊 呀 嘿 哼 咔 嚓 哐 隆 咕 嘟
哈哈 呵呵 嘿嘿 呜呜 哇哇 呀呀 哼哼 轰隆 哗啦 咕咚 扑通 砰砰 轰轰
啪啪 咚咚 哈哈哈 呜呜呜 哗哗 隆隆 咕噜 咕噜咕噜 滴答 嘀嗒 叮咚
咚咚咚 啪啪啪 轰轰轰 哗啦啦 呼呼 飕飕 嗖嗖 沙沙 淅淅 簌簌 咔嚓
哐当 乒乓 噼啪 噼里啪啦 咕嘟 扑哧 嘿咻 加油 冲 杀 嗷嗷 汪汪
喵喵 咩咩 哞哞 喔喔 咯咯 嘎嘎 呱呱 吱吱 嗡嗡 轰鸣 巨响 爆炸
""".split(),
    "en": """
BOOM POW BANG CRASH WHAM ZAP BAM BLAM KABOOM BOOM
THUD THUMP SMASH SPLAT SPLASH WHOOSH SWOOSH VROOM
BEEP HONK RING DING DONG KNOCK TAP TAP RAP
HA HA HEE HEE HOHO HIC SOB WAAH GASP GULP
MUNCH CRUNCH SLURP SIP GRRR ROAR HISSS BUZZ
TICK TOCK DRIP DROP PLOP PLIP CLANG CLANK CLINK
THWACK WHACK SLAP STOMP STAMP CREAK SQUEAK RATTLE
ZOOM ZIP ZING FIZZ POP SIZZLE CRACKLE RUMBLE
ARGH OOF OUCH OW EEEK YIKES WOW WHOA HEY
YAY HOORAY BRAVO SHH PSST Ahem
""".split(),
}

# SFX tails per language (literal characters rendered by the font).
SFX_ELONG_TAILS = {
    "ja": ["ー", "ーー", "ーーー", "ッ", "ッッ", "〜", "～", "〜〜", "..."],
    "ko": ["—", "——", "ㅋ", "ㅋㅋ", "ㅎ", "ㅎㅎ", "~~~", "～～", "...", "!!"],
    "zh": ["—", "——", "啊", "呀", "啦", "～～", "……", "...", "!!!"],
    "en": ["O", "OO", "M", "MM", "H", "!!!", "...", "??", "!?"],
}

SFX_MARK_TAILS = {
    "ja": ["!", "!!", "！", "！！", "!?", "?!", "！？", "？！", "？", "♡", "♥", "★", "☆", "っ", "ん", "..."],
    "ko": ["!", "!!", "!!!", "!?", "?!", "?", "??", "~~~", "...", "♥", "★", "☆", "ㅋㅋ", "ㅎㅎ"],
    "zh": ["!", "!!", "!!!", "！", "！！", "！？", "？！", "？", "……", "...", "★", "☆", "♥"],
    "en": ["!", "!!", "!!!", "?", "??", "???", "!?", "?!", "...", "★", "♥"],
}

# Common English dialogue words for random-text generation (comic-flavoured:
# pronouns, verbs, interjections). Sampled to form word-like sentences.
EN_WORDS = """
I you he she we they it this that these those here there where when why
what who how not no yes yeah yep nope hey hi hello thanks thank please
sorry wow oh ah huh hmm well so but and or if then just really very
quite too also only even ever never always sometimes often look see
watch listen hear say tell talk speak ask answer know think feel want
need like love hate go come run walk stop wait stay leave take give
get got help let make do did done have had has are was were is be
been will would could should shall may might must can time day night
morning man woman boy girl kid guy friend home school work world
life heart mind hand eye head back up down out in on off over under
again still yet already soon now today tonight tomorrow yesterday
hey wow oops ouch argh hooray hurrah shh psst bravo encore hello
goodbye please sorry excuse pardon indeed alright okay ok nope yep
""".split()

EN_WORDS = list(dict.fromkeys(w for w in EN_WORDS if w))

# Core characters used to test whether a font can draw a language at all
# (fast cmap check, not a full coverage requirement).
LANG_CORE_CHARS = {
    "ja": set("ドンバガッーあア漢"),
    "ko": set("가나다는고에의은는을를한"),
    "zh": set("的一是不了我有个人好这中大来上"),
    "en": set("HelloWorld"),
}

# Frequent Han characters for natural Chinese dialogue (simplified +
# traditional). Random Chinese text samples ~80% from these and ~20% from the
# full font Han pool (so rare characters are still covered for robustness).
COMMON_HAN_FREQ = (
    "的一是不了我有个人在上大中和国到说时要就出也得里后自行年过能对生事用作方多经么去法学如都同现当没动面起看定天分还进好小些主样理心学文"
    "的一是不了我有個人在上大中和國到說時要就出也得裡後自行年過能對生事用作方多經麼去法學如同現當沒動面起看定天分還進好小些主樣理心學文"
    "你好谢他她它这那哪谁什怎为以于及即既使令从由自向往行用发成事工开分合作者家园地方产品经现理然因果所为好想意感看法点线角光影声色味触梦"
    "你好謝他她它這那哪誰什怎為以於及即既使令從由自向往行用發成事工開分合作者家園地方產品經現理然因果所為好想意感看法點線角光影聲色味觸夢"
)


def normalize_lang(lang: str | None) -> str:
    """Validate a language code, defaulting to ``ja`` (backward compat)."""
    if lang is None:
        return "ja"
    lang = str(lang).strip().lower()
    # Accept common variants: 'kr'/'korean', 'cn'/'chinese', 'jp'/'japanese'.
    aliases = {
        "jp": "ja", "jpn": "ja", "japanese": "ja",
        "kr": "ko", "kor": "ko", "korean": "ko",
        "cn": "zh", "chs": "zh", "cht": "zh", "chinese": "zh",
        "eng": "en", "english": "en",
    }
    lang = aliases.get(lang, lang)
    if lang not in SUPPORTED_LANGS:
        raise ValueError(f"unsupported lang {lang!r}: choose from {SUPPORTED_LANGS}")
    return lang


def detect_lang(text: str) -> str:
    """Heuristic script-based language guess for a line of text.

    Counts Hiragana/Katakana (ja), Hangul (ko), Han (zh/ja), Latin (en) and
    picks the dominant script. Han alone is ambiguous (ja vs zh): kana
    presence decides Japanese, otherwise Chinese.
    """
    from manga_ocr_dev.synthetic_data_generator.utils import (
        is_hangul,
        is_hiragana,
        is_kanji,
        is_katakana,
    )

    ja_kana = ko = han = latin = 0
    for c in text:
        if c.isspace():
            continue
        if is_hiragana(c) or is_katakana(c):
            ja_kana += 1
        elif is_hangul(c):
            ko += 1
        elif is_kanji(c):
            han += 1
        elif c.isascii() and c.isalpha():
            latin += 1
    if ja_kana > 0 and ja_kana + han > 0:
        # Any kana -> Japanese (even with heavy kanji/latin mixing).
        if ja_kana >= 1 and (ja_kana + han) > latin and (ja_kana + han) > ko:
            return "ja"
    if ko > 0 and ko >= han and ko >= latin and ko >= ja_kana:
        return "ko"
    if han > 0 and ja_kana == 0:
        if han >= latin and han >= ko:
            return "zh"
    if latin > 0 and latin >= han and latin >= ko and latin >= ja_kana:
        return "en"
    # Fallbacks: kana -> ja, hangul -> ko, han -> zh, else en for pure latin.
    if ja_kana:
        return "ja"
    if ko:
        return "ko"
    if han:
        return "zh"
    return "en"
