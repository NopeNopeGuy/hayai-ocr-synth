import tempfile
from pathlib import Path

import pandas as pd

from manga_ocr_dev.data.process_text import (
    RollingDedup,
    clean_dialogue_line,
    clean_novel_text,
    export_lines,
    iter_mixed_stream,
    split_novel_chapter,
)


def test_clean_dialogue_line_subtitles():
    # ASS override tags and styling
    assert clean_dialogue_line(r"{\pos(192,240)\c&H000000&}Hello world!") == "Hello world!"
    # ASS newlines to space
    assert clean_dialogue_line(r"Line one\NLine two") == "Line one Line two"
    assert clean_dialogue_line(r"Line one\nLine two") == "Line one Line two"
    # HTML tags
    assert clean_dialogue_line("<i>Italic dialogue</i>") == "Italic dialogue"
    assert clean_dialogue_line('<font color="#ff0000">Red</font> text') == "Red text"
    # Leading speaker prefixes
    assert clean_dialogue_line("（ジェイ）私から見ても") == "私から見ても"
    assert clean_dialogue_line("(Speaker) English line") == "English line"
    assert clean_dialogue_line("[Narrator] Meanwhile...") == "Meanwhile..."
    # Music notes
    assert clean_dialogue_line("♪ ラララ ♪") == "ラララ"
    # Outer quotes stripped
    assert clean_dialogue_line('"できない"では つまらない') == '"できない"では つまらない'
    assert clean_dialogue_line('"はい、分かりました"') == "はい、分かりました"
    assert clean_dialogue_line("“Yes, sir!”") == "Yes, sir!"
    assert clean_dialogue_line("'Okay'") == "Okay"
    # Japanese fullwidth space normalization
    assert clean_dialogue_line("0.3秒の照射で\u3000網膜を損傷させ") == "0.3秒の照射で 網膜を損傷させ"
    # English & Japanese mixed
    assert clean_dialogue_line("OK、了解した！") == "OK、了解した！"
    # None or empty
    assert clean_dialogue_line(None) == ""
    assert clean_dialogue_line("") == ""


def test_clean_novel_text_aozora():
    aozora_text = "深いおどろき《驚き》にうたれて、［＃改ページ］名高いウェストミンスター"
    cleaned = clean_novel_text(aozora_text)
    assert "おどろき" in cleaned
    assert "《驚き》" not in cleaned
    assert "［＃改ページ］" not in cleaned


def test_split_novel_chapter():
    chapter = """
【第1章　出会い】
-------------------
来世があるなら平穏に生きたいと願っていたのだけど、どうやら今世もダメみたい。
「――汝らは永遠に幸せであることをここに誓いますか？」
「はい、誓います」
「Yes, I do!」
儀式を執り行うために着せられた、ウエディングドレスとはまた少し違う真っ白な衣装に身を包み、神父らしき男の退屈でよく分からない言葉を聞かされた。
"""
    lines = split_novel_chapter(chapter, min_len=2, max_len=100)
    assert len(lines) >= 4
    # Dialogues extracted
    assert any("汝らは永遠に幸せであることをここに誓いますか？" in l for l in lines)
    assert any("はい、誓います" in l for l in lines)
    assert any("Yes, I do!" in l for l in lines)
    # Separator '-------------------' should be skipped
    assert not any("---" in l for l in lines)
    # Narrative sentence included
    assert any("来世があるなら平穏に生きたい" in l for l in lines)


def test_rolling_dedup():
    dedup = RollingDedup(window_size=3)
    assert not dedup.is_duplicate("line1")
    assert not dedup.is_duplicate("line2")
    assert not dedup.is_duplicate("line3")
    # Duplicate within window
    assert dedup.is_duplicate("line1")
    # Push new items to evict oldest
    assert not dedup.is_duplicate("line4")
    assert not dedup.is_duplicate("line5")
    # Now line2 should have been evicted
    assert not dedup.is_duplicate("line2")


def test_iter_mixed_stream():
    dialogues = iter([{"source": "dialogue_test", "id": f"d_{i}", "line": f"dialogue {i}"} for i in range(5)])
    novels = iter([{"source": "novel_test", "id": f"n_{i}", "line": f"novel {i}"} for i in range(5)])

    mixed = list(iter_mixed_stream(dialogues, novels, dialogue_ratio=0.5, seed=123))
    assert len(mixed) == 10
    sources = {item["source"] for item in mixed}
    assert "dialogue_test" in sources
    assert "novel_test" in sources


def test_export_lines_local_mock():
    # Test export_lines with local dummy text file
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        dialogue_file = tmp_path / "dummy_dialogue.txt"
        dialogue_file.write_text(
            'Hello world!\nこんにちは、世界！\n「大丈夫？」\n"Yes, absolutely!"\nこれはテストです。\n',
            encoding="utf-8",
        )

        out_lines_dir = tmp_path / "lines"
        created_files = export_lines(
            num_lines_in_each_package=3,
            num_packages=1,
            start_package=0,
            source_type="dialogue",
            dialogue_data_dir=str(dialogue_file),
            out_dir=out_lines_dir,
            streaming=False,
            min_len=2,
            dedup_window=10,
        )

        assert len(created_files) == 1
        pkg_file = created_files[0]
        assert pkg_file.is_file()
        assert pkg_file.name == "0000.csv"

        df = pd.read_csv(pkg_file)
        assert list(df.columns) == ["source", "id", "line"]
        assert len(df) == 3
        # Check that lines contain cleaned text
        assert any("こんにちは、世界！" in str(l) for l in df["line"])
