import pdfplumber
import re
from collections import defaultdict, OrderedDict
from sklearn.cluster import KMeans
import numpy as np

from config import SECTION_HEADER_MAP, HEADER_REGEX
from ollama_client import query_ollama_model

# Functions for PDF parsing
def detect_two_column_page_combined(page, threshold=0.95):
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

    return white_ratio > 0.7 and cluster_distance > 150 and balanced_clusters

def detect_document_layout(pdf):
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



# Extract section-specific information for the architectural diagram
def extract_section_information(parsed_data, model_name):
    # Define the sections we want to extract information from
    section_info = {
        "introduction": {
            "category": "Research Need",
            "prompt": "Extract 2-3 complete sentences that describe why this research is needed and what the authors want to do. Focus on the problem statement, research gap, and objectives."
        },
        "method": {
            "category": "Solution Approach",
            "prompt": "Extract 2-3 complete sentences that describe the solution or methodology proposed by the authors. Focus on the approach, techniques, and methods used."
        },
        "implementation": {
            "category": "Solution Approach",
            "prompt": "Extract 2-3 complete sentences that describe how the solution was implemented. Focus on the technical aspects, tools, and frameworks used."
        },
        "experiments": {
            "category": "Study Conduct",
            "prompt": "Extract 2-3 complete sentences that describe how the study was conducted. Focus on the experimental setup, datasets, and evaluation metrics."
        },
        "results": {
            "category": "Results",
            "prompt": "Extract 2-3 complete sentences that describe the results of the study. Focus on the main findings, performance metrics, and comparisons."
        },
        "conclusion": {
            "category": "Conclusion",
            "prompt": "Extract 2-3 complete sentences from the conclusion. Focus on the main takeaways, limitations, and future work."
        },
        "limitations": {
            "category": "Limitations",
            "prompt": "Extract 2-3 complete sentences that describe the limitations of the study. Focus on constraints, shortcomings, and areas for improvement."
        }
    }
    
    # Extract information from each available section
    extracted_info = {}
    
    for section_name, section_content in parsed_data["sections"].items():
        if section_name in section_info:
            info = section_info[section_name]
            category = info["category"]
            prompt = info["prompt"]
            
            # Truncate long sections to avoid context limits
            truncated_content = section_content[:2000] if len(section_content) > 2000 else section_content
            
            extraction_prompt = (
                f"From the following {section_name} section of a research paper, {prompt}\n\n"
                f"Return only the extracted sentences exactly as they appear in the text, no explanations or additional text.\n\n"
                f"Section content:\n{truncated_content}"
            )
            
            try:
                # Use streaming for all extractions to improve responsiveness
                full_response = ""
                for chunk in query_ollama_model(extraction_prompt, model_name, stream=True):
                    full_response = chunk["content"]
                
                # Store the extracted information
                if category not in extracted_info:
                    extracted_info[category] = []
                extracted_info[category].append({"section": section_name, "content": full_response})
                
            except Exception as e:
                print(f"Error extracting information from {section_name}: {str(e)}")
    
    return extracted_info

