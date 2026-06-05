from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path


DEFAULT_BASE_DIR = Path("demo_videos")
DEFAULT_OUTPUT_NAME = "merged_outputs"
IMAGE_PAGE_DPI = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge each frame PNG with the PDF that has the same filename stem. "
            "The output PDF starts with the image page, followed by the original PDF pages."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Base folder that contains frames/ and answers/. Default: demo_videos",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder. Default: <base-dir>/merged_outputs",
    )
    parser.add_argument(
        "--category",
        action="append",
        dest="categories",
        help="Only process this category folder. Can be used multiple times.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print matching/missing files; do not create PDFs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output PDFs.",
    )
    return parser.parse_args()


def load_pdf_dependencies():
    try:
        from PIL import Image
        from pypdf import PdfReader, PdfWriter
    except ModuleNotFoundError as exc:
        missing_name = exc.name or "required package"
        raise SystemExit(
            f"Missing dependency: {missing_name}\n"
            "Install dependencies with:\n"
            "  .\\.venv\\Scripts\\python.exe -m pip install pillow pypdf"
        ) from exc

    return Image, PdfReader, PdfWriter


def discover_categories(frames_dir: Path, answers_dir: Path) -> list[str]:
    frame_categories = {path.name for path in frames_dir.iterdir() if path.is_dir()}
    answer_categories = {path.name for path in answers_dir.iterdir() if path.is_dir()}
    return sorted(frame_categories & answer_categories)


def iter_png_files(folder: Path) -> list[Path]:
    files = [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == ".png"]
    return sorted(files, key=lambda path: path.name.lower())


def png_to_single_page_pdf(
    image_module,
    png_path: Path,
    temp_pdf_path: Path,
    page_width_pt: float,
    page_height_pt: float,
) -> None:
    canvas_width = round(page_width_pt / 72 * IMAGE_PAGE_DPI)
    canvas_height = round(page_height_pt / 72 * IMAGE_PAGE_DPI)

    with image_module.open(png_path) as image:
        image = image.convert("RGB")
        image_ratio = image.width / image.height
        canvas_ratio = canvas_width / canvas_height

        if image_ratio > canvas_ratio:
            resized_width = canvas_width
            resized_height = round(canvas_width / image_ratio)
        else:
            resized_height = canvas_height
            resized_width = round(canvas_height * image_ratio)

        resized = image.resize((resized_width, resized_height), image_module.Resampling.LANCZOS)
        canvas = image_module.new("RGB", (canvas_width, canvas_height), "white")
        paste_x = (canvas_width - resized_width) // 2
        paste_y = (canvas_height - resized_height) // 2
        canvas.paste(resized, (paste_x, paste_y))
        canvas.save(temp_pdf_path, "PDF", resolution=IMAGE_PAGE_DPI)


def merge_png_and_pdf(png_path: Path, answer_pdf_path: Path, output_pdf_path: Path) -> None:
    Image, PdfReader, PdfWriter = load_pdf_dependencies()

    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    temp_pdf_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".image.pdf",
            dir=output_pdf_path.parent,
            delete=False,
        ) as temp_file:
            temp_pdf_path = Path(temp_file.name)

        writer = PdfWriter()
        answer_pdf = PdfReader(str(answer_pdf_path))
        if not answer_pdf.pages:
            raise ValueError(f"Answer PDF has no pages: {answer_pdf_path}")

        answer_first_page = answer_pdf.pages[0]
        page_width = float(answer_first_page.mediabox.width)
        page_height = float(answer_first_page.mediabox.height)

        png_to_single_page_pdf(Image, png_path, temp_pdf_path, page_width, page_height)
        image_pdf = PdfReader(str(temp_pdf_path))
        image_pdf.pages[0].mediabox.upper_right = (page_width, page_height)

        writer.add_page(image_pdf.pages[0])
        for page in answer_pdf.pages:
            writer.add_page(page)

        with output_pdf_path.open("wb") as output_file:
            writer.write(output_file)
    finally:
        if temp_pdf_path and temp_pdf_path.exists():
            temp_pdf_path.unlink()


def process_category(
    category: str,
    frames_dir: Path,
    answers_dir: Path,
    output_dir: Path,
    dry_run: bool,
    overwrite: bool,
) -> tuple[int, int, list[Path], list[Path]]:
    frame_folder = frames_dir / category
    answer_folder = answers_dir / category
    output_folder = output_dir / category

    merged_count = 0
    skipped_count = 0
    missing_pdfs: list[Path] = []
    missing_pngs: list[Path] = []
    png_paths = iter_png_files(frame_folder)
    png_stems = {path.stem for path in png_paths}
    pdf_paths = sorted(
        [path for path in answer_folder.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"],
        key=lambda path: path.name.lower(),
    )
    pdf_stems = {path.stem for path in pdf_paths}

    for pdf_path in pdf_paths:
        if pdf_path.stem not in png_stems:
            print(f"[missing] {category}: no matching PNG for {pdf_path.name}")
            missing_pngs.append(pdf_path)

    for png_path in png_paths:
        answer_pdf_path = answer_folder / f"{png_path.stem}.pdf"
        output_pdf_path = output_folder / f"{png_path.stem}.pdf"

        if png_path.stem not in pdf_stems:
            print(f"[missing] {category}: no matching PDF for {png_path.name}")
            missing_pdfs.append(png_path)
            continue

        if output_pdf_path.exists() and not overwrite:
            print(f"[skip] {output_pdf_path} already exists")
            skipped_count += 1
            continue

        if dry_run:
            print(f"[match] {png_path} + {answer_pdf_path} -> {output_pdf_path}")
        else:
            merge_png_and_pdf(png_path, answer_pdf_path, output_pdf_path)
            print(f"[write] {output_pdf_path}")

        merged_count += 1

    return merged_count, skipped_count, missing_pdfs, missing_pngs


def main() -> int:
    args = parse_args()

    base_dir = args.base_dir
    frames_dir = base_dir / "frames"
    answers_dir = base_dir / "answers"
    output_dir = args.output_dir or base_dir / DEFAULT_OUTPUT_NAME

    if not frames_dir.is_dir():
        print(f"Frames folder not found: {frames_dir}", file=sys.stderr)
        return 1
    if not answers_dir.is_dir():
        print(f"Answers folder not found: {answers_dir}", file=sys.stderr)
        return 1

    categories = args.categories or discover_categories(frames_dir, answers_dir)
    if not categories:
        print("No category folders found under both frames/ and answers/.", file=sys.stderr)
        return 1

    total_merged = 0
    total_skipped = 0
    all_missing_pdfs: list[Path] = []
    all_missing_pngs: list[Path] = []

    print(f"Base: {base_dir}")
    print(f"Output: {output_dir}")
    print(f"Categories: {', '.join(categories)}")

    for category in categories:
        frame_folder = frames_dir / category
        answer_folder = answers_dir / category

        if not frame_folder.is_dir():
            print(f"[skip] frame category not found: {frame_folder}")
            continue
        if not answer_folder.is_dir():
            print(f"[skip] answer category not found: {answer_folder}")
            continue

        merged, skipped, missing_pdfs, missing_pngs = process_category(
            category=category,
            frames_dir=frames_dir,
            answers_dir=answers_dir,
            output_dir=output_dir,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
        total_merged += merged
        total_skipped += skipped
        all_missing_pdfs.extend(missing_pdfs)
        all_missing_pngs.extend(missing_pngs)

    action = "matched" if args.dry_run else "merged"
    print(
        f"Done: {total_merged} {action}, "
        f"{total_skipped} skipped, "
        f"{len(all_missing_pdfs)} PNGs missing PDFs, "
        f"{len(all_missing_pngs)} PDFs missing PNGs."
    )

    if all_missing_pdfs:
        print("\nPNGs without matching PDFs:")
        for path in all_missing_pdfs:
            print(f"  {path}")

    if all_missing_pngs:
        print("\nPDFs without matching PNGs:")
        for path in all_missing_pngs:
            print(f"  {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
