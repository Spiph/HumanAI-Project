"""
PDF Parser Module

This module contains the PDF parsing functionality from the original ollama_chatv35.py file.
It maintains the exact same functionality without any modifications.
"""

import os
import re
import pdfplumber
import numpy as np
from collections import defaultdict, OrderedDict
from sklearn.cluster import KMeans

SECTION_HEADER_MAP = {
    # Abstract
    "abstract": "abstract",
    "executive summary": "abstract",
    "summary": "abstract",

    # Introduction
    "introduction": "introduction",
    "related work": "introduction",
    "background": "introduction",
    "preliminaries": "introduction",
    "problem statement": "introduction",
    "motivation": "introduction",
    "paper organization": "introduction",
    "outline": "introduction",
    "contributions": "introduction",

    # Methods
    "methodology": "method",
    "methods": "method",
    "approach": "method",
    "materials and methods": "method",
    "method and materials": "method",
    "design and implementation": "method",
    "implementation": "method",
    "implementation details": "method",
    "model": "method",
    "model architecture": "method",
    "system architecture": "method",
    "algorithm": "method",
    "algorithms": "method",
    "nomenclature": "method",
    "notation": "method",

    # Data
    "data": "data",
    "datasets": "data",
    "dataset": "data",
    "data collection": "data",
    "data sources": "data",
    "data description": "data",
    "corpus": "data",
    "corpora": "data",

    # Experiments / Evaluation
    "experiment setup": "experiments",
    "experimental setup": "experiments",
    "experimental design": "experiments",
    "experiments": "experiments",
    "evaluation": "experiments",
    "evaluation methodology": "experiments",
    "evaluation metrics": "experiments",
    "benchmarks": "experiments",
    "benchmark": "experiments",
    "performance evaluation": "experiments",
    "experimental results": "experiments",
    "testbed": "experiments",
    "ablation study": "experiments",
    "ablation studies": "experiments",
    "case study": "experiments",
    "case studies": "experiments",
    "setup": "experiments",

    # Results & Analysis
    "results": "results",
    "findings": "results",
    "analysis": "results",
    "results and discussion": "results",
    "analysis and results": "results",
    "findings and discussion": "results",
    "outcomes": "results",
    "observations": "results",

    # Discussion & Conclusion
    "discussion": "conclusion",
    "concluding discussion": "conclusion",
    "conclusion": "conclusion",
    "conclusions and future work": "conclusion",
    "future work": "conclusion",
    "summary and future work": "conclusion",
    "summary and outlook": "conclusion",
    "concluding remarks": "conclusion",
    "takeaways": "conclusion",
    "limitations": "conclusion",

    # Back‐matter
    "references": "references",
    "bibliography": "references",
    "literature cited": "references",
    "citations": "references",

    # Acknowledgments & Declarations
    "acknowledgments": "acknowledgments",
    "acknowledgements": "acknowledgments",
    "funding": "acknowledgments",
    "author contributions": "acknowledgments",
    "conflicts of interest": "acknowledgments",
    "conflict of interest": "acknowledgments",
    "competing interests": "acknowledgments",

    # Declarations (often separate in some venues)
    "declarations": "declarations",
    "ethics statement": "declarations",
    "ethics approval": "declarations",
    "data availability": "declarations",
    "code availability": "declarations",

    # Supplementary / Appendix
    "appendix": "appendix",
    "appendices": "appendix",
    "supplementary material": "appendix",
    "supplementary materials": "appendix",
    "supplemental": "appendix",
    "supplemental materials": "appendix",

    # Keywords / Front Matter
    "keywords": "keywords",
    "index terms": "keywords",
}


# Header regex pattern
HEADER_REGEX = r"(?im)^\s*(?:\d+(?:\.\d+)*[.)]?\s*)?(?P<header>" + "|".join(re.escape(h) for h in SECTION_HEADER_MAP.keys()) + r")\s*$"

def detect_two_column_page_combined(page, threshold=0.95, white_ratio_threshold=0.65):
    """
    Detect if a page has a two-column layout using a combined approach.
    
    Args:
        page: The page to analyze
        threshold: Threshold for white pixel detection (default: 0.95)
        white_ratio_threshold: Threshold for white ratio in center (default: 0.65, lowered from 0.7)
        
    Returns:
        bool: True if the page has a two-column layout, False otherwise
    """
    im = page.to_image(resolution=150).original.convert("L")
    arr = np.array(im)
    height, width = arr.shape
    center_x = width // 2
    slice_width = width // 30
    center_band = arr[:, center_x - slice_width:center_x + slice_width]

    white_pixels = np.sum(center_band > 245, axis=1)
    white_fraction = white_pixels / center_band.shape[1]
    white_rows = np.sum(white_fraction > threshold)
    white_ratio = white_rows / height

    words = page.extract_words()
    x_vals = np.array([[w["x0"]] for w in words])
    if len(x_vals) < 10:
        return False

    kmeans = KMeans(n_clusters=2, n_init="auto").fit(x_vals)
    centers = sorted(kmeans.cluster_centers_.flatten())
    labels = kmeans.labels_
    cluster_distance = abs(centers[1] - centers[0])

    cluster_0 = np.sum(labels == 0)
    cluster_1 = np.sum(labels == 1)
    total = len(labels)
    balanced_clusters = (cluster_0 / total > 0.2 and cluster_1 / total > 0.2)

    return white_ratio > white_ratio_threshold and cluster_distance > 150 and balanced_clusters

def detect_document_layout(pdf):
    """
    Detect the layout of a document (single-column or two-column).
    
    Args:
        pdf: The PDF document to analyze
        
    Returns:
        str: "two-column" or "single-column"
    """
    page_limit = min(3, len(pdf.pages))
    two_column_votes = 0
    for i in range(page_limit):
        if detect_two_column_page_combined(pdf.pages[i]):
            two_column_votes += 1
    return "two-column" if two_column_votes >= 2 else "single-column"

def extract_text_from_pdf_with_sections(pdf_path):
    """
    Extract text from PDF with section detection.
    Returns both the full text and structured sections.
    """
    lines_with_positions = []
    with pdfplumber.open(pdf_path) as pdf:
        layout_mode = detect_document_layout(pdf)
        print(f"[INFO] Detected document layout: {layout_mode.upper()}")

        for page_num, page in enumerate(pdf.pages):
            words = page.extract_words(x_tolerance=1, y_tolerance=3, use_text_flow=True)
            if not words:
                continue

            is_two_column = layout_mode == "two-column"
            print(f"[INFO] Page {page_num + 1} processed as {'two-column' if is_two_column else 'single-column'}")

            if is_two_column:
                width = page.width
                left_col = page.within_bbox((0, 0, width / 2, page.height))
                right_col = page.within_bbox((width / 2, 0, width, page.height))
                columns = [(0, left_col), (1, right_col)]
            else:
                columns = [(0, page)]

            for col_idx, column in columns:
                col_words = column.extract_words(x_tolerance=1, y_tolerance=3, use_text_flow=True)
                if not col_words:
                    continue

                col_line_map = defaultdict(list)
                for word in col_words:
                    y = round(word['top'], 1)
                    col_line_map[y].append((word['x0'], word['text']))

                for y in sorted(col_line_map):
                    sorted_words = sorted(col_line_map[y], key=lambda x: x[0])
                    line_text = " ".join(w[1] for w in sorted_words)
                    lines_with_positions.append(((page_num, col_idx, y), line_text))

    lines_with_positions.sort(key=lambda x: x[0])
    full_text = "\n".join(line for _, line in lines_with_positions)
    
    # Extract title
    title = extract_title(pdf_path)
    
    # Split into sections
    sections = split_into_sections(full_text)
    
    return {
        "full_text": full_text,
        "title": title,
        "sections": sections
    }

def extract_title(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        first_page = pdf.pages[0]
        words = first_page.extract_words(use_text_flow=True, keep_blank_chars=False)
        if not words:
            return "Unknown Title"

        height_buckets = defaultdict(list)
        for word in words:
            height = round(word['height'], 1)
            height_buckets[height].append(word)

        largest_font_height = max(height_buckets.keys())
        title_words = height_buckets[largest_font_height]

        lines = defaultdict(list)
        for word in title_words:
            y0 = round(word['top'], 1)
            lines[y0].append(word['text'])

        ordered_lines = [" ".join(lines[y]) for y in sorted(lines)]
        title = " ".join(ordered_lines).strip()
        title = re.sub(r'\s+', ' ', title)

        if len(title.split()) < 3:
            return "Unknown Title"
        return title

def split_into_sections(text):
    raw_sections = OrderedDict()
    matches = list(re.finditer(HEADER_REGEX, text, re.IGNORECASE))

    if not matches:
        return {"content": text.strip()}

    for i, match in enumerate(matches):
        raw_header = match.group("header").strip().lower()
        if raw_header not in SECTION_HEADER_MAP:
            print(f"[INFO] Unmapped header found: {raw_header}")
            section_title = f"custom_{raw_header}"
        else:
            section_title = SECTION_HEADER_MAP[raw_header]
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        if section_title in raw_sections:
            raw_sections[section_title] += "\n" + content
        else:
            raw_sections[section_title] = content

    ordered_sections = OrderedDict()
    for key in SECTION_HEADER_MAP.values():
        if key in raw_sections:
            ordered_sections[key] = raw_sections[key]

    for key, val in raw_sections.items():
        if key not in ordered_sections:
            ordered_sections[key] = val

    return ordered_sections
