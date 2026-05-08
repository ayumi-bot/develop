"""
voronoi_checker.py
------------------
PDFまたはPPTXファイルからボロノイ図を検出し、セル数（発生点数）を数える。
指定した閾値以下のファイルをリストアップする。

使い方:
    python voronoi_checker.py <フォルダまたはファイル> [--threshold 10] [--out result.xlsx]
"""

import sys, os, argparse, glob
from pathlib import Path

import cv2
import numpy as np
import fitz  # PyMuPDF

try:
    from pptx import Presentation
    from pptx.util import Inches
    HAS_PPTX = True
except ImportError:
    HAS_PPTX = False


# ─── PDF → ページ画像リスト ───────────────────────────────────────────────────

def pdf_to_images(pdf_path: str, scale: float = 2.0) -> list[np.ndarray]:
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        images.append(arr)
    doc.close()
    return images


# ─── PPTX → ページ画像リスト ─────────────────────────────────────────────────
# LibreOfficeがない場合は一時PDFに変換できないため、
# 各スライドをPNGへエクスポートする簡易手段を使う

def pptx_to_images(pptx_path: str) -> list[np.ndarray]:
    """PPTXをPDFに変換してから読み込む（LibreOffice必要）。
    なければ None を返す。"""
    import subprocess, tempfile, shutil

    lo = shutil.which("libreoffice") or shutil.which("soffice")
    if lo is None:
        return []

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [lo, "--headless", "--convert-to", "pdf", "--outdir", tmp, pptx_path],
            check=True, capture_output=True
        )
        pdfs = glob.glob(os.path.join(tmp, "*.pdf"))
        if not pdfs:
            return []
        return pdf_to_images(pdfs[0])


# ─── ボロノイ図ページの検出 ───────────────────────────────────────────────────

def find_voronoi_page(images: list[np.ndarray]) -> tuple[int, np.ndarray] | tuple[None, None]:
    """非白ピクセル数が最大のページをボロノイ図候補とする。"""
    if not images:
        return None, None
    scores = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        scores.append(int(np.sum(gray < 230)))
    best = int(np.argmax(scores))
    return best, images[best]


# ─── セル数カウント ───────────────────────────────────────────────────────────

def count_cells(img: np.ndarray, debug_path: str | None = None) -> int:
    """
    ボロノイ図画像から発生点（ジェネレータ）の数を推定する。
    複数の色範囲で試みて、最も妥当な数を返す。
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]
    img_area = h * w

    best_count = 0

    # ── 試す色マスク（典型的な発生点の色）──
    # ── 試す色マスク（典型的な発生点の色）──
    mask_red1 = cv2.inRange(hsv, np.array([0,   100, 100]), np.array([10,  255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
    masks_to_try = [
        ("red",    cv2.bitwise_or(mask_red1, mask_red2)),
        ("blue",   cv2.inRange(hsv, np.array([100, 80, 80]),  np.array([130, 255, 255]))),
        ("cyan",   cv2.inRange(hsv, np.array([85,  80, 80]),  np.array([100, 255, 255]))),
        ("green",  cv2.inRange(hsv, np.array([40,  80, 80]),  np.array([80,  255, 255]))),
        ("purple", cv2.inRange(hsv, np.array([130, 80, 80]),  np.array([160, 255, 255]))),
        ("yellow", cv2.inRange(hsv, np.array([20,  80, 80]),  np.array([40,  255, 255]))),
    ]

    debug_img = img.copy()
    candidate_counts = {}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for name, mask in masks_to_try:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        min_area = max(10, img_area * 0.0001)
        max_area = img_area * 0.02
        valid = [i for i in range(1, n_labels)
                 if min_area < stats[i, cv2.CC_STAT_AREA] < max_area]

        candidate_counts[name] = len(valid)

        if debug_path and len(valid) > 0:
            for i in valid:
                cx = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] // 2
                cy = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] // 2
                cv2.circle(debug_img, (cx, cy), 8, (0, 255, 0), 2)

    # 最も多いカラーを採用（多すぎる場合はマップのノイズなので上限100）
    valid_counts = {k: v for k, v in candidate_counts.items() if 3 <= v <= 150}
    if valid_counts:
        best_color = max(valid_counts, key=lambda k: valid_counts[k])
        best_count = valid_counts[best_color]
    else:
        # フォールバック: エッジベースの閉領域カウント
        best_count = count_by_regions(img)

    if debug_path:
        cv2.imwrite(debug_path, debug_img)

    return best_count


def count_markers(img: np.ndarray) -> int:
    """
    発生点マーカー（円・ピン・十字）を検出して数える。
    モルフォロジー的クローズで輪郭円・十字の腕をつなげてから
    アスペクト比と面積でライン（ボロノイエッジ）を除外する。
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]
    img_area = h * w

    mask_red1 = cv2.inRange(hsv, np.array([0,   100, 100]), np.array([10,  255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
    masks = {
        "red":    cv2.bitwise_or(mask_red1, mask_red2),
        "blue":   cv2.inRange(hsv, np.array([100, 80,  80]),  np.array([130, 255, 255])),
        "cyan":   cv2.inRange(hsv, np.array([85,  80,  80]),  np.array([100, 255, 255])),
        "orange": cv2.inRange(hsv, np.array([10,  100, 100]), np.array([30,  255, 255])),
        "yellow": cv2.inRange(hsv, np.array([20,  80,  80]),  np.array([40,  255, 255])),
        "purple": cv2.inRange(hsv, np.array([130, 80,  80]),  np.array([160, 255, 255])),
        "pink":   cv2.inRange(hsv, np.array([140, 50,  150]), np.array([175, 180, 255])),
    }

    # 大きめのクローズカーネル: 輪郭円の隙間・十字の腕をつなぐ
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    open_k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    best = 0
    for cname, mask in masks.items():
        if mask.sum() < 200:
            continue
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k, iterations=2)
        cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_k, iterations=1)

        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)

        valid = []
        for i in range(1, n_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            bw   = stats[i, cv2.CC_STAT_WIDTH]
            bh   = stats[i, cv2.CC_STAT_HEIGHT]
            # サイズ: マーカーとして妥当な範囲
            if not (img_area * 0.00015 < area < img_area * 0.025):
                continue
            # アスペクト比: 縦横どちらかに長すぎない（ライン除外）
            ratio = max(bw, bh) / max(min(bw, bh), 1)
            if ratio > 3.5:
                continue
            valid.append(i)

        if len(valid) > best:
            best = len(valid)

    return best


def count_by_regions(img: np.ndarray) -> int:
    """エッジを検出して閉領域数を数えるフォールバック手法。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # プロットエリアを大まかに中央部分に絞る
    h, w = gray.shape
    crop = gray[int(h*0.1):int(h*0.9), int(w*0.05):int(w*0.95)]

    blurred = cv2.GaussianBlur(crop, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 100)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # 輪郭を閉領域として数える
    contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    img_area = h * w
    cells = [c for c in contours
             if img_area * 0.001 < cv2.contourArea(c) < img_area * 0.3]
    return len(cells)


# ─── ファイル処理 ─────────────────────────────────────────────────────────────

# ─── PDFテキストから発生点数を抽出 ────────────────────────────────────────────

def extract_count_from_text(filepath: str) -> int | None:
    """PDFのテキストから「N個の発生点/施設/AED...」という記述を探す。"""
    import re
    doc = fitz.open(filepath)
    text = " ".join(page.get_text() for page in doc)
    doc.close()

    patterns = [
        # 英語: "set/use/collected N generators/locations/AEDs/shelters/facilities..."
        r'(?:set|use|using|used|total|chose|select\w*|collect\w*|compil\w*|gather\w*|identif\w*|chose|pick\w*)\s+(\d+)\s+\w*\s*(?:generator|AED|facilit|location|site|shelter|clinic|hospital|park|super|point|center)',
        r'(?:generator|AED|facilit|location|site|shelter|clinic|hospital|park|super|point|center)\w*[^.!?\n]*?(\d+)\s+(?:generator|AED|facilit|location|site|shelter|clinic|hospital)',
        r'(\d+)\s+(?:AED|evacuation|facilit|location|generator|shelter|clinic|hospital|park|super)\w*\s+(?:location|site|point|as generator)',
        # 日本語: "N個の施設/スーパー/公園..."
        r'(\d+)\s*(?:個|か所|カ所|箇所|店舗|施設|避難所|公園|スーパー|病院|クリニック)',
        # "n = N" 形式
        r'\bn\s*=\s*(\d+)',
    ]
    candidates = []
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            n = int(m.group(1))
            if 5 <= n <= 500:
                candidates.append(n)

    if not candidates:
        return None
    # 最頻値を返す（複数ヒットした場合）
    from collections import Counter
    freq = Counter(candidates)
    return freq.most_common(1)[0][0]


def process_file(filepath: str, debug_dir: str | None = None) -> dict:
    ext = Path(filepath).suffix.lower()
    name = Path(filepath).name

    if ext == ".pdf":
        images = pdf_to_images(filepath)
    elif ext in (".pptx", ".ppt"):
        if not HAS_PPTX:
            return {"file": name, "path": filepath, "page": None,
                    "count": -1, "error": "python-pptx not installed"}
        images = pptx_to_images(filepath)
        if not images:
            return {"file": name, "path": filepath, "page": None,
                    "count": -1, "error": "LibreOffice not available for PPTX"}
    else:
        return {"file": name, "path": filepath, "page": None,
                "count": -1, "error": "unsupported format"}

    if not images:
        return {"file": name, "path": filepath, "page": None,
                "count": -1, "error": "no pages"}

    page_idx, voronoi_img = find_voronoi_page(images)
    if voronoi_img is None:
        return {"file": name, "path": filepath, "page": None,
                "count": -1, "error": "page not found"}

    debug_path = None
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        debug_path = os.path.join(debug_dir, f"{Path(filepath).stem}_p{page_idx+1}_debug.png")

    visual_count  = count_cells(voronoi_img, debug_path=debug_path)
    marker_count  = count_markers(voronoi_img)

    # テキストから発生点数を取得（PDFのみ）
    text_count = None
    if Path(filepath).suffix.lower() == ".pdf":
        text_count = extract_count_from_text(filepath)

    # 採用するカウント: テキスト優先、なければ視覚
    count  = text_count if text_count is not None else visual_count
    source = "text" if text_count is not None else "visual"

    # 要確認: テキスト未検出
    uncertain = (text_count is None)

    return {
        "file":    name,
        "path":    filepath,
        "page":    page_idx + 1,
        "count":   count,
        "source":  source,
        "visual":  visual_count,
        "markers": marker_count,
        "uncertain": uncertain,
        "error":   None,
    }


# ─── メイン ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ボロノイ図セル数チェッカー")
    parser.add_argument("target", help="PDFファイルまたはフォルダ")
    parser.add_argument("--threshold", type=int, default=30,
                        help="この数以下のセル数をリストアップ (default: 30)")
    parser.add_argument("--out", default=None,
                        help="結果をExcelに保存するファイル名")
    parser.add_argument("--debug", action="store_true",
                        help="検出結果を画像で保存 (./debug/ フォルダ)")
    args = parser.parse_args()

    # ファイル収集
    target = args.target
    if os.path.isfile(target):
        files = [target]
    elif os.path.isdir(target):
        # glob の [] をエスケープ（パスに角括弧が含まれる場合の対策）
        safe = glob.escape(target)
        files = (glob.glob(os.path.join(safe, "**/*.pdf"), recursive=True) +
                 glob.glob(os.path.join(safe, "**/*.pptx"), recursive=True))
    else:
        print(f"[ERROR] {target} が見つかりません")
        sys.exit(1)

    if not files:
        print("対象ファイルが見つかりません")
        sys.exit(1)

    print(f"対象ファイル: {len(files)}件  閾値: {args.threshold}セル以下\n")

    debug_dir = "./debug" if args.debug else None
    results = []

    for i, f in enumerate(sorted(files), 1):
        print(f"[{i:02d}/{len(files)}] {Path(f).name} ... ", end="", flush=True)
        r = process_file(f, debug_dir=debug_dir)
        results.append(r)
        if r["error"]:
            print(f"ERROR: {r['error']}")
        else:
            src  = r.get("source", "?")
            cell = r["count"]
            mrk  = r.get("markers", "-")
            both_low = (cell < args.threshold and isinstance(mrk, int) and mrk < args.threshold)
            flag = " ★両方不足" if both_low else (" ⚠セルのみ" if cell < args.threshold else "")
            unc  = " [要確認]" if r.get("uncertain") else ""
            print(f"page{r['page']}  セル:{cell}({src}) マーカー:{mrk}{flag}{unc}")

    T = args.threshold

    def both_low(r):
        return (r["error"] is None
                and r["count"] < T
                and isinstance(r.get("markers"), int)
                and r["markers"] < T)

    # 二重判定で両方不足かつテキスト確認済み → 確定
    definite = [r for r in results if both_low(r) and r.get("source") == "text"]
    # 二重判定で両方不足だが視覚のみ → 要確認
    visual_both = [r for r in results if both_low(r) and r.get("source") == "visual"]
    # セルのみ不足（マーカーはOK） → 検出誤差の可能性高い
    cell_only = [r for r in results
                 if r["error"] is None and r["count"] < T
                 and not both_low(r)]
    errors = [r for r in results if r["error"] is not None]

    print(f"\n{'─'*60}")
    print(f"【二重判定：両方不足確定（テキスト確認済み）】: {len(definite)}件")
    for r in sorted(definite, key=lambda x: x["count"]):
        print(f"  ✗ {r['file']}  セル:{r['count']} マーカー:{r['markers']} (page{r['page']})")

    print(f"\n【二重判定：両方不足（要手動確認）】: {len(visual_both)}件")
    for r in sorted(visual_both, key=lambda x: x["count"]):
        print(f"  ? {r['file']}  セル:{r['count']} マーカー:{r['markers']} (page{r['page']})")

    print(f"\n【セルのみ不足・マーカーOK（検出誤差の可能性）】: {len(cell_only)}件")
    for r in sorted(cell_only, key=lambda x: x["count"]):
        mrk = r.get("markers", "-")
        print(f"  - {r['file']}  セル:{r['count']} マーカー:{mrk} (page{r['page']})")

    if errors:
        print(f"\n処理エラー: {len(errors)}件")
        for r in errors:
            print(f"  ✗ {r['file']}  →  {r['error']}")

    # Excel出力
    if args.out:
        try:
            import openpyxl
            from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "ボロノイチェック"

            headers = ["ファイル名", "検出ページ", "セル数", "マーカー数", "判定", "備考"]
            widths   = [45, 12, 10, 12, 10, 30]
            hfill = PatternFill("solid", fgColor="4472C4")
            hfont = Font(bold=True, color="FFFFFF")
            border = Border(
                left=Side(style='thin'), right=Side(style='thin'),
                top=Side(style='thin'), bottom=Side(style='thin')
            )

            for col, (h, w) in enumerate(zip(headers, widths), 1):
                cell = ws.cell(row=1, column=col, value=h)
                cell.font = hfont; cell.fill = hfill
                cell.alignment = Alignment(horizontal="center")
                cell.border = border
                ws.column_dimensions[
                    openpyxl.utils.get_column_letter(col)].width = w

            for row_i, r in enumerate(sorted(results, key=lambda x: x["count"] if x["count"] >= 0 else 999), 2):
                warn = (r["error"] is None and r["count"] <= args.threshold)
                fill = PatternFill("solid", fgColor="FFE0E0" if warn else "FFFFFF")
                b_low = both_low(r)
                note = ""
                if r.get("uncertain") and b_low:
                    note = "要確認（両方不足）"
                elif r.get("uncertain"):
                    note = "視覚検出のみ"
                elif r.get("error"):
                    note = r["error"]
                vals = [r["file"],
                        r["page"] if r["page"] else "",
                        r["count"] if r["count"] >= 0 else "",
                        r.get("markers", ""),
                        "★" if b_low else ("⚠" if warn else ""),
                        note]
                for col, v in enumerate(vals, 1):
                    cell = ws.cell(row=row_i, column=col, value=v)
                    cell.fill = fill
                    cell.border = border
                    cell.alignment = Alignment(horizontal="center" if col in (2,3,4) else "left")

            wb.save(args.out)
            print(f"\nExcel保存: {args.out}")
        except ImportError:
            print("\n[WARNING] openpyxl がないためExcel出力をスキップ")

    return 0 if not (definite or visual_both) else 1


if __name__ == "__main__":
    sys.exit(main())
