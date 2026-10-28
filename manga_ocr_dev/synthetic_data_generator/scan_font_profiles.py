"""Comprehensive font scanner and profiler for Manga OCR Synth Generator.

Scans all fonts under FONTS_ROOT (all 19,000+ fonts), classifies them into manga/comic
typographic archetypes, omits non-manga / office / easy / system fonts, assigns
recommended sizing and border parameters based on type, and writes:
  - assets/font_profiles.json (fast runtime lookup)
  - assets/font_profiles.csv (metadata report)
  - assets/fonts.csv (updated with priority labels & new fonts)
"""

from __future__ import annotations

import csv
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from fontTools.ttLib import TTFont
from tqdm import tqdm

from manga_ocr_dev.env import ASSETS_PATH, FONTS_ROOT
from manga_ocr_dev.synthetic_data_generator.scan_fonts import process as scan_font_chars

# Archetype defaults: sizes, strokes, weights
ARCHETYPE_CONFIGS = {
    "dialogue_gothic": {
        "priority": "special",
        "dialogue_weight": 4.5,
        "sfx_weight": 0.8,
        "dialogue_size": [28, 52],
        "sfx_size": [45, 110],
        "stroke_range": [0, 3],
        "stroke_ratio": 0.03,
        "stroke_prob": 0.35,
        "double_outline_prob": 0.03,
    },
    "dialogue_round": {
        "priority": "special",
        "dialogue_weight": 4.5,
        "sfx_weight": 1.2,
        "dialogue_size": [30, 54],
        "sfx_size": [50, 120],
        "stroke_range": [0, 3],
        "stroke_ratio": 0.03,
        "stroke_prob": 0.40,
        "double_outline_prob": 0.05,
    },
    "narration_serif": {
        "priority": "common",
        "dialogue_weight": 3.0,
        "sfx_weight": 0.3,
        "dialogue_size": [26, 48],
        "sfx_size": [40, 95],
        "stroke_range": [0, 2],
        "stroke_ratio": 0.02,
        "stroke_prob": 0.20,
        "double_outline_prob": 0.0,
    },
    "sfx_impact": {
        "priority": "special",
        "dialogue_weight": 0.2,
        "sfx_weight": 5.0,
        "dialogue_size": [36, 52],
        "sfx_size": [70, 195],
        "stroke_range": [4, 14],
        "stroke_ratio": 0.08,
        "stroke_prob": 0.85,
        "double_outline_prob": 0.45,
    },
    "sfx_brush": {
        "priority": "special",
        "dialogue_weight": 0.2,
        "sfx_weight": 4.5,
        "dialogue_size": [34, 50],
        "sfx_size": [55, 160],
        "stroke_range": [3, 8],
        "stroke_ratio": 0.05,
        "stroke_prob": 0.70,
        "double_outline_prob": 0.25,
    },
    "sfx_cute": {
        "priority": "special",
        "dialogue_weight": 0.8,
        "sfx_weight": 4.5,
        "dialogue_size": [30, 50],
        "sfx_size": [45, 130],
        "stroke_range": [3, 9],
        "stroke_ratio": 0.06,
        "stroke_prob": 0.75,
        "double_outline_prob": 0.35,
    },
    "sfx_horror": {
        "priority": "special",
        "dialogue_weight": 0.3,
        "sfx_weight": 3.5,
        "dialogue_size": [28, 46],
        "sfx_size": [42, 120],
        "stroke_range": [2, 6],
        "stroke_ratio": 0.04,
        "stroke_prob": 0.60,
        "double_outline_prob": 0.15,
    },
    "handwritten": {
        "priority": "common",
        "dialogue_weight": 2.0,
        "sfx_weight": 1.5,
        "dialogue_size": [22, 40],
        "sfx_size": [35, 80],
        "stroke_range": [0, 2],
        "stroke_ratio": 0.02,
        "stroke_prob": 0.20,
        "double_outline_prob": 0.0,
    },
    "display_headline": {
        "priority": "common",
        "dialogue_weight": 1.5,
        "sfx_weight": 2.5,
        "dialogue_size": [36, 65],
        "sfx_size": [50, 140],
        "stroke_range": [2, 6],
        "stroke_ratio": 0.05,
        "stroke_prob": 0.60,
        "double_outline_prob": 0.20,
    },
    "comic_latin": {
        "priority": "special",
        "dialogue_weight": 4.5,
        "sfx_weight": 3.0,
        "dialogue_size": [26, 48],
        "sfx_size": [50, 140],
        "stroke_range": [0, 4],
        "stroke_ratio": 0.04,
        "stroke_prob": 0.40,
        "double_outline_prob": 0.10,
    },
    "regular_cjk": {
        "priority": "regular",
        "dialogue_weight": 1.0,
        "sfx_weight": 0.8,
        "dialogue_size": [28, 50],
        "sfx_size": [45, 110],
        "stroke_range": [1, 3],
        "stroke_ratio": 0.03,
        "stroke_prob": 0.35,
        "double_outline_prob": 0.05,
    },
    "omit": {
        "priority": "omit",
        "dialogue_weight": 0.0,
        "sfx_weight": 0.0,
        "dialogue_size": [32, 48],
        "sfx_size": [48, 96],
        "stroke_range": [0, 0],
        "stroke_ratio": 0.0,
        "stroke_prob": 0.0,
        "double_outline_prob": 0.0,
    },
}

BANNED_OFFICE_FONTS = [
    "yahei", "微软雅黑", "iyahei", "simsun", "nsimsun", "simhei", "新宋体", "中易",
    "gulim", "dotum", "굴림", "돋움", "batangche", "droid sans",
    "arial", "helvetica", "calibri", "cambria", "courier", "consolas", "georgia",
    "verdana", "tahoma", "trebuchet", "segoe", "lucida", "dejavu", "liberation",
    "freesans", "freeserif", "freemono", "roboto", "myriad", "minion", "baskerville",
    "garamond", "caslon", "bodoni", "didot", "palatino", "optima", "univers",
]

COMIC_MAYO_WHITELIST = [
    "bb", "comic", "manga", "cartoon", "action", "secret agent",
    "madlines", "makinas", "ohisama", "soviet", "smackattack", "trashhand",
    "wild", "ccwild", "ames", "belizarius", "betty", "bottix", "bottlerocket",
    "boxer", "craterface", "knuckle", "anime", "digistrip", "shorcy", "zombie",
    "bada", "armor", "ashcan", "atland", "back issues", "blackhole", "blam",
    "blowhole", "brain freezer", "brush script", "chiller", "creeping",
    "eurocomic", "fanboy", "holy mack", "humbucker", "jackarmstrong", "marspolice",
    "misfit", "miskatonic", "mouthbre", "orangefizz", "pythia", "radgranny",
    "roswreck", "spiritmedium", "umberto", "vampiress", "vanhelsing", "wizardspeak",
    "yoshi", "zapraygun", "komik", "kmk", "vtc letterer",
]


def check_omit(rel_path: str) -> tuple[bool, str]:
    """Determine if a font has no business in manga (office, system, symbols, vendor non-CJK)."""
    p = rel_path.lower().replace("\\", "/")
    fn = Path(rel_path).name.lower()

    # 1. Non-CJK vendor directories (Latin/Arabic/Hebrew/Digits/Symbols packages)
    if any(k in p for k in [
        "fontworks/英文", "fontworks/其他",
        "morisawa（森泽）/其他", "morisawa（森泽）/数字", "morisawa（森泽）/符号",
        "adobe/英文",
    ]):
        return True, "vendor_latin_or_symbols"

    # 2. Mayo folder: keep authentic comic lettering or CJK fonts
    if p.startswith("mayo/"):
        has_cjk_name = any(ord(c) > 127 for c in fn)
        is_comic = any(k in fn for k in COMIC_MAYO_WHITELIST)
        if not (has_cjk_name or is_comic):
            return True, "mayo_non_comic"

    # 3. Explicitly banned office / desktop / system fonts (banned in manga scanlation)
    for b in BANNED_OFFICE_FONTS:
        if b in fn or f"/{b}" in p:
            return True, "office_system_font"

    # 4. Coding / Monospace fonts
    if any(k in fn for k in ["coding", "inconsolata", "monaco", "menlo", "fixedsys", "sourcecodepro"]):
        return True, "coding_font"

    # 5. Non-target script fonts
    if any(k in fn for k in ["arabic", "hebrew", "devanagari", "thai", "cyrillic", "greek", "vietnamese", "armenian"]):
        return True, "non_cjk_script"

    return False, ""


def classify_font(rel_path: str) -> str:
    """Classify an authentic manga font into its specific typographic archetype."""
    p = rel_path.lower().replace("\\", "/")
    fn = Path(rel_path).name.lower()

    # SFX - Horror / Chills / Eerie Tension
    if any(k in p for k in [
        "mystery", "comicmystery", "으스스", "euseuseu", "흑백영화", "bombaram",
        "黄草", "残圆", "黑魔法", "霹雳", "破损", "血滴", "怨灵",
        "creepster", "nosifer", "eater", "rubikbeastly", "rubikglitch", "vampiroone",
        "zombieguts", "creepingevil", "bucketoblood", "chills", "spookytooth",
        "voodoo", "miskatonic", "vanhelsing", "vampiress", "chiller",
    ]):
        return "sfx_horror"

    # SFX - Dynamic Brush / Slashes / Martial Arts
    if any(k in p for k in [
        "ginryu", "kokuryu", "touryu", "raijin", "samurai", "ninja", "kanteiryu",
        "gyosho", "sosho", "무풍지대", "도원결의", "조각도", "forgedblade", "dokdo",
        "mashanzheng", "longcang", "liujianmaocao", "尚巍", "狂劫", "剑体", "刀锋",
        "榜书", "勘亭流", "金文", "流隶", "reisho", "chosungs", "nanumbrush",
        "cezannebokutoh", "bokutohn",
        "permanentmarker", "rocksalt", "lacquer", "wetpaint", "brush script",
        "spills", "splashdown",
    ]):
        return "sfx_brush"

    # SFX - Cute / Pop / Comedic Gag
    if any(k in p for k in [
        "popjoy", "pophappiness", "babypop", "palramune", "yuruka", "주간만화",
        "리드미컬", "kiranghaerang", "bazzi", "bagelfat", "syongsyong", "dongdong",
        "supermagic", "喵呜", "胖娃", "胖头鱼", "少儿", "卡通", "哈哈", "嘟嘟",
        "糖糖", "跳跳", "泡泡", "豆豆", "娃娃", "少女文字", "kuaile", "moyamoya",
        "kkubulim", "lotteria", "samliphopangche", "skip", "slump", "stick", "comet",
        "carat", "budo", "aokane", "aralet", "lyra", "macaroni",
        "sniglet", "chicle", "cherrycream", "londrina", "carterone", "chango",
        "slackey", "spicyrice", "12tonsushi", "12tonfishstick", "mummy loves you",
        "misfit", "orangefizz", "kidkosmic", "sf toontime", "sf cartoonist",
        "fanboyhc", "yoshi",
    ]):
        return "sfx_cute"

    # SFX - Impact / Heavy Action
    if any(k in p for k in [
        "raglan", "rocknroll", "rowdy", "kurokane", "punch", "midashigo", "futogo",
        "pop1", "popcorn", "dohyeon", "tmonmonsori", "jalnan", "cookierun",
        "blackhansans", "sebang", "huangyou", "longzhu", "活龙", "粗活龙", "水黑",
        "琥珀", "力量黑", "大黑", "超粗黑", "咆哮", "呐喊", "chab", "ddag", "hopangche",
        "balsa", "aggro", "heavy-labourer", "reggae", "comicreggae", "tmontium",
        "bangers", "luckiestguy", "boogaloo", "bungee", "titanone", "fasterone",
        "russoone", "blackopsone", "fugazone", "sigmar", "knewave", "rampartone",
        "whiz bang", "clobberintime", "flame on", "flameon", "meltdown", "zoinks",
        "craterface", "humbucker", "zapraygun", "holymackerel", "knucklesandwich",
        "adamantium", "astro city", "bronto", "carry on screaming", "dutchcourage",
        "monster mash", "phases on stun", "pulp fiction", "stormtrooper", "thrills",
        "battlelines", "marspolice",
    ]):
        return "sfx_impact"

    # Handwritten / Muttering / Casual Doodles
    if any(k in p for k in [
        "klee", "kafu", "marker", "penji", "tegaki", "미생", "그날오후", "ownglyph",
        "gaegu", "gamjaflower", "cutefont", "himelody", "nanumpen", "静蕾", "显仁",
        "黑板报", "下午茶", "手札", "火柴", "doodle", "handwrite", "손글씨",
        "caveat", "architectsdaughter", "gochihand", "kalam", "indieflower",
        "patrickhand", "stanhand", "sketchpad", "shorthand", "flat earth scribe",
    ]):
        return "handwritten"

    # Display / 8-bit / Status Screens / Stylized Headline
    if any(k in p for k in [
        "dotgothic", "dotmincho", "dunggeunmo", "bitbit", "pixel",
        "pressstart2p", "silkscreen", "monoton", "metalmania", "pirataone",
        "njnaruto", "androidnation", "shojumaru", "shrikhand", "zendots", "trainone"
    ]):
        return "display_headline"

    # Comic Latin (for romaji / English comic dialogue)
    if any(k in fn for k in [
        "bb.", "bb ", "comic", "secret agent", "madlines", "makinas",
        "ohisama", "smackattack", "action man", "trashhand",
        "mangat", "wild words", "wildwords", "animeace", "anime ace", "digistrip",
        "digitalstrip", "komika", "kmk", "komtx", "wbxkomik", "comic book",
        "comic strip", "comicgeek", "comix", "eurocomic", "jackarmstrong",
        "blambot", "ames", "shann", "detectivesinc", "vtc letterer", "workingman",
        "obelixpro", "rooftoprun", "threearrows", "wizardspeak", "comicrazy",
        "divineright", "double back", "jim lee", "jimlee", "stand by4 action",
        "thestorysofar", "to be continued", "comicneue", "fredoka"
    ]) or p.startswith("latin_comic/"):
        return "comic_latin"

    # Dialogue - Round (Maru-Gothic / 漫画圆体)
    if any(k in p for k in [
        "seurat", "tsukuard", "tsukubrd", "marufo", "marutoday", "maruanti",
        "dfmarugothic", "hsmarugothic", "round-gothic", "准圆", "粗圆", "兰亭圆",
        "嘉丽圆", "正圆", "润圆", "jua", "danjunghae", "ssurround", "yozai",
        "marugothic", "marugo", "gmarugo", "subyuan", "zhunyuan"
    ]):
        return "dialogue_round"

    # Narration - Serif / Mincho / Batang / Song / Kai
    if any(k in p for k in [
        "tsukumin", "ryumin", "a1mincho", "shueimin", "telopmin", "modemin",
        "udmincho", "dfmincho", "hsmincho", "kaimin", "kakumin", "kyoka",
        "宋", "楷", "仿宋", "batang", "myeongjo", "centennial", "chosun",
        "gowunbatang", "songmyung", "maru buri", "kopubbatang", "serif", "subserif"
    ]):
        return "narration_serif"

    # Dialogue - Gothic / Clean Sans
    if any(k in p for k in [
        "newrodin", "shingo", "gothicmb101", "rodin", "tsukugo", "chiaro",
        "humming", "udkakugo", "ntoday", "jun", "dfgothic", "hsgothic",
        "arphic gothic", "兰亭黑", "旗黑", "文黑", "睿黑", "智黑", "盈黑",
        "hei prc", "lineseed", "nexonlv1", "nexonlv2", "applesdgothic",
        "rix고딕", "sourcehansans", "notosans", "gothic", "kopubdotum",
        "subsans", "sublan"
    ]):
        return "dialogue_gothic"

    # Display / 8-bit / Status Screens
    if any(k in p for k in ["dotgothic", "dotmincho", "dunggeunmo", "bitbit", "pixel"]):
        return "display_headline"

    return "regular_cjk"


def detect_lang_hint(rel_path: str) -> str:
    p = rel_path.lower().replace("\\", "/")
    fn = Path(rel_path).name.lower()
    if p.startswith("korean/") or "/韩文/" in p or "ydbackjae" in p:
        return "ko"
    if p.startswith("chinese_additions/") or any(k in p for k in ["/简体/", "/繁体/", "/简繁/", "founder", "hanyi"]):
        return "zh"
    if "/日文/" in p or "fot-" in p or "a-otf-" in p or "epson" in p:
        return "ja"
    if p.startswith("latin_comic/"):
        return "en"
    if p.startswith("mayo/"):
        if any(ord(c) > 127 for c in fn):
            return "cjk"
        return "en"
    if "/英文/" in p:
        return "en"
    return "cjk"


def profile_single_font(rel_path: str) -> dict:
    """Generate complete profile dictionary for a single font."""
    omit, reason = check_omit(rel_path)
    lang_hint = detect_lang_hint(rel_path)

    if omit:
        font_type = "omit"
        cfg = ARCHETYPE_CONFIGS["omit"]
        is_manga = False
    else:
        font_type = classify_font(rel_path)
        cfg = ARCHETYPE_CONFIGS[font_type]
        is_manga = True

    return {
        "font_path": rel_path,
        "font_type": font_type,
        "priority": cfg["priority"],
        "is_manga": is_manga,
        "omit_reason": reason,
        "dialogue_weight": cfg["dialogue_weight"],
        "sfx_weight": cfg["sfx_weight"],
        "dialogue_size_min": cfg["dialogue_size"][0],
        "dialogue_size_max": cfg["dialogue_size"][1],
        "sfx_size_min": cfg["sfx_size"][0],
        "sfx_size_max": cfg["sfx_size"][1],
        "stroke_min": cfg["stroke_range"][0],
        "stroke_max": cfg["stroke_range"][1],
        "stroke_ratio": cfg["stroke_ratio"],
        "stroke_prob": cfg["stroke_prob"],
        "double_outline_prob": cfg["double_outline_prob"],
        "lang_hint": lang_hint,
    }


def update_fonts_csv_with_profiles(profiles: dict[str, dict], out_csv: Path):
    """Update assets/fonts.csv so label reflects priority and new fonts are scanned."""
    print("Reading existing fonts.csv...")
    existing = pd.read_csv(out_csv)
    existing_paths = set(existing.font_path)

    # Check which fonts on disk are not yet in fonts.csv
    missing = [p for p in profiles if p not in existing_paths]
    if missing:
        print(f"Scanning character coverage for {len(missing)} newly downloaded fonts...")
        new_rows = []
        with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 4, 16)) as pool:
            results = list(pool.map(scan_font_chars, [str(FONTS_ROOT / p) for p in missing]))
            for rel, chars in zip(missing, results):
                prof = profiles[rel]
                new_rows.append({
                    "font_path": rel,
                    "supported_chars": chars,
                    "num_chars": len(chars),
                    "label": prof["priority"],
                })
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            existing = pd.concat([existing, new_df], ignore_index=True)
            print(f"Added {len(new_rows)} new fonts to dataframe.")

    # Update label column for all fonts based on scanned priority
    labels = [profiles.get(p, {}).get("priority", "regular") for p in existing.font_path]
    existing["label"] = labels

    tmp_path = out_csv.with_suffix(".tmp.csv")
    existing.to_csv(tmp_path, index=False)
    os.replace(tmp_path, out_csv)
    print(f"Successfully updated {out_csv} ({len(existing)} rows).")


def main():
    print(f"Scanning all fonts in FONTS_ROOT: {FONTS_ROOT}")
    suffixes = {".ttf", ".otf", ".ttc", ".otc"}
    all_fonts = sorted(
        str(Path(root).relative_to(FONTS_ROOT) / f)
        for root, _, files in os.walk(FONTS_ROOT, followlinks=True)
        for f in files
        if Path(f).suffix.lower() in suffixes
    )
    print(f"Total font files found on disk: {len(all_fonts)}")

    profiles = {}
    rows = []
    type_counts = {}
    priority_counts = {}

    for rel in all_fonts:
        prof = profile_single_font(rel)
        profiles[rel] = prof
        rows.append(prof)
        ft = prof["font_type"]
        pr = prof["priority"]
        type_counts[ft] = type_counts.get(ft, 0) + 1
        priority_counts[pr] = priority_counts.get(pr, 0) + 1

    print("\n--- Font Archetype Breakdown ---")
    for k, v in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:20s}: {v:5d}")

    print("\n--- Font Priority Breakdown ---")
    for k, v in sorted(priority_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:20s}: {v:5d}")

    # 1. Write assets/font_profiles.json
    json_path = ASSETS_PATH / "font_profiles.json"
    print(f"\nWriting {json_path}...")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)

    # 2. Write assets/font_profiles.csv
    csv_path = ASSETS_PATH / "font_profiles.csv"
    print(f"Writing {csv_path}...")
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    # 3. Update assets/fonts.csv
    fonts_csv_path = ASSETS_PATH / "fonts.csv"
    update_fonts_csv_with_profiles(profiles, fonts_csv_path)

    print("\nFont profiling and classification completed successfully!")


run_profiling = main

if __name__ == "__main__":
    main()
