"""Upload More_JP fonts (61 files) and curated hard & high-value Japanese fonts (44 files) to Hugging Face dataset."""

import os
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd

REPO_ID = "JustANormalTinkerer/manga-ocr-multilingual-fonts"
FONTS_ROOT = Path("/mnt/data/jp_fonts")
api = HfApi()

print(f"Target repository: {REPO_ID}")

# ---------------------------------------------------------------------------
# 1. More_JP Fonts (all 61 fonts extracted from /mnt/more jp fonts)
# ---------------------------------------------------------------------------
more_jp_dir = FONTS_ROOT / "More_JP"
more_jp_files = sorted(
    list(more_jp_dir.rglob("*.ttf"))
    + list(more_jp_dir.rglob("*.otf"))
    + list(more_jp_dir.rglob("*.ttc"))
)
print(f"Found {len(more_jp_files)} font files in More_JP ({sum(f.stat().st_size for f in more_jp_files) / (1024*1024):.1f} MB)")

more_jp_ops = []
for f in more_jp_files:
    rel = f.relative_to(FONTS_ROOT).as_posix()
    more_jp_ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(f)))

print(f"Uploading {len(more_jp_ops)} More_JP fonts via commit...")
api.create_commit(
    repo_id=REPO_ID,
    repo_type="dataset",
    operations=more_jp_ops,
    commit_message=f"feat: upload More_JP Japanese manga & handwritten fonts ({len(more_jp_ops)} fonts)",
)
print("More_JP upload complete!")

# ---------------------------------------------------------------------------
# 2. Curated Hard & High-Value JP Fonts (44 fonts)
# ---------------------------------------------------------------------------
curated_rel = [
    # Calligraphy & Battle Brush (Demon Slayer / Action SFX)
    "完整包/其他/日文/Showa（昭和）/A_KsoGinryu.otf",
    "完整包/其他/日文/Showa（昭和）/A_KsoKokuryu.otf",
    "完整包/其他/日文/Showa（昭和）/A_KsoTouryu.otf",
    "完整包/其他/日文/Showa（昭和）/A_KsoKagerou.otf",
    "完整包/其他/日文/KsoRaijin.otf",
    "完整包/其他/日文/Hakusyu Fonts（白舟書体）/HOT-NinjaStd-R.otf",
    "完整包/其他/日文/Hakusyu Fonts（白舟書体）/HOT-SamuraiStd-R.otf",
    "精简包/日文/DynaFont（华康）/DFKanTeiRyu-XB & DFPKanTeiRyu-XB & DFGKanTeiRyu-XB.ttc",
    "完整包/Morisawa（森泽）/日文/MorisawaAOTF/日文/デザイン書体（创意字体）/A-OTF-KanteiryuStd-Ultra.otf",
    # Aggressive Action & Impact Display SFX
    "精简包/日文/Fontworks/FOT-RaglanPunch Std UB.otf",
    "精简包/日文/Fontworks/FOT-Raglan Std UB.otf",
    "精简包/日文/Fontworks/FOT-RocknRoll Std DB.otf",
    "精简包/日文/Fontworks/FOT-ComicReggae Std B.otf",
    "精简包/日文/Fontworks/FOT-Reggae Std B.otf",
    "精简包/日文/Fontworks/FOT-Rowdy Std EB.otf",
    "精简包/日文/Fontworks/FOT-Kurokane Std EB.otf",
    "完整包/Fontworks/日文/デザイン書体（创意字体）/FOT-RampartStd-EB.otf",
    "完整包/Fontworks/日文/デザイン書体（创意字体）/FOT-RampartTLStd-EB.otf",
    "精简包/日文/Fontworks/FOT-Matisse Pro EB.otf",
    # Horror & Mystery
    "精简包/日文/Fontworks/FOT-ComicMystery Std DB.otf",
    "精简包/日文/Fontworks/FOT-Mystery Std DB.otf",
    # Comic POP, Gag & Irregular
    "精简包/日文/Fontworks/FOT-PopJoy Std B.otf",
    "精简包/日文/Fontworks/FOT-PopHappiness Std EB.otf",
    "精简包/日文/Fontworks/FOT-PopFury Std B.otf",
    "精简包/日文/Fontworks/FOT-BabyPop Std EB.otf",
    "精简包/日文/DynaFont（华康）/DFPOP1-W9 & DFPPOP1-W9 & DFGPOP1-W9.ttc",
    "精简包/日文/DynaFont（华康）/DFPOPCorn-W12 & DFPPOPCorn-W12 & DFGPOPCorn-W12.ttc",
    "精简包/日文/DynaFont（华康）/DFPOPMix-W5 & DFPPOPMix-W5 & DFGPOPMix-W5.ttc",
    "精简包/日文/DynaFont（华康）/DFCraftYu-W5 & DFPCraftYu-W5 & DFGCraftYu-W5.ttc",
    "精简包/日文/DynaFont（华康）/DFTegakiKaku-W4 & DFPTegakiKaku-W4 & DFGTegakiKaku-W4.ttc",
    "精简包/日文/DynaFont（华康）/DFTegakiRaku-W3 & DFPTegakiRaku-W3 & DFGTegakiRaku-W3.ttc",
    "精简包/日文/Morisawa（森泽）/A-OTF TakaHand Std B.otf",
    # Retro Dot / Pixel
    "精简包/日文/Fontworks/FOT-DotGothic12 Std M.otf",
    "精简包/日文/Fontworks/FOT-DotGothic16 Std M.otf",
    # Essential Workhorses (Dialogue / Narration)
    "精简包/日文/Morisawa（森泽）/A-OTF Shin Go Pr6N B.otf",
    "精简包/日文/Morisawa（森泽）/A-OTF Shin Go Pr6N M.otf",
    "精简包/日文/Morisawa（森泽）/A-OTF Shin Maru Go Pr6N B.otf",
    "精简包/日文/Fontworks/FOT-NewRodin ProN B.otf",
    "精简包/日文/Fontworks/FOT-Seurat Pro B.otf",
    "精简包/日文/Fontworks/FOT-Klee Pro DB.otf",
    "精简包/日文/Morisawa（森泽）/A-OTF Ryumin Pr6N B-KL.otf",
    "精简包/日文/Fontworks/FOT-TsukuMin Pr5N B.otf",
    "精简包/日文/Morisawa（森泽）/A-OTF Gothic MB101 Pr6N B.otf",
    "精简包/日文/Morisawa（森泽）/A-OTF Futo Go B101 Pr6N Bold.otf",
]

curated_ops = []
for r in curated_rel:
    p = FONTS_ROOT / r
    if p.is_file():
        curated_ops.append(CommitOperationAdd(path_in_repo=r, path_or_fileobj=str(p)))
    else:
        print(f"Warning: file not found: {r}")

# Also include the updated fonts.csv in this commit
csv_path = Path("assets/fonts.csv")
if csv_path.is_file():
    curated_ops.append(CommitOperationAdd(path_in_repo="assets/fonts.csv", path_or_fileobj=str(csv_path)))

print(f"Uploading {len(curated_ops)} curated hard fonts + fonts.csv via commit...")
api.create_commit(
    repo_id=REPO_ID,
    repo_type="dataset",
    operations=curated_ops,
    commit_message=f"feat: upload curated hard & high-value Japanese manga fonts ({len(curated_ops)-1} fonts + fonts.csv)",
)
print("Curated hard fonts upload complete!")

print("\nAll requested Japanese fonts successfully uploaded to Hugging Face!")

