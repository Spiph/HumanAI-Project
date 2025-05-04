import gradio as gr
import requests
import json
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
from io import BytesIO
import threading
import os
import datetime
import uuid
import re
import numpy as np
import random
from collections import defaultdict, OrderedDict
from sklearn.cluster import KMeans

# Define the Ollama API endpoint
OLLAMA_API_URL = "http://localhost:11434/api/chat"

# Directory to store user data
USER_DATA_DIR = "./user_data"
os.makedirs(USER_DATA_DIR, exist_ok=True)

# Maximum retries for MCQ generation
MAX_MCQ_GENERATION_ATTEMPTS = 5

# Section header mapping
SECTION_HEADER_MAP = {
    "abstract": "abstract",
    "introduction": "introduction",
    "related work": "introduction",
    "background": "introduction",
    "preliminaries": "introduction",
    "methodology": "method",
    "methods": "method",
    "approach": "method",
    "implementation": "method",
    "experiment setup": "experiments",
    "experimental setup": "experiments",
    "experimental design": "experiments",
    "experiments": "experiments",
    "evaluation": "experiments",
    "data": "data",
    "results": "results",
    "findings": "results",
    "analysis": "results",
    "discussion": "conclusion",
    "conclusion": "conclusion",
    "future work": "conclusion",
    "limitations": "conclusion",
    "references": "references",
    "acknowledgments": "acknowledgments",
    "acknowledgements": "acknowledgments"
}

# Header regex pattern
HEADER_REGEX = r"(?im)^\s*(?:\d+(?:\.\d+)*[.)]?\s*)?(?P<header>" + "|".join(re.escape(h) for h in SECTION_HEADER_MAP.keys()) + r")\s*$"

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

# Function to send a prompt to the Ollama model and retrieve the response
def query_ollama_model(prompt, model_name="gemma3:1b", context=None, stream=True):
    messages = [
        {"role": "user", "content": prompt}
    ]
    if context:
        messages.insert(0, {"role": "system", "content": context})  # Add context as system message
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": stream  # Enable or disable streaming
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, stream=stream)
        response.raise_for_status()  # Raise an error for bad responses
        if stream:
            full_response = ""
            for line in response.iter_lines():
                if line:
                    chunk = line.decode("utf-8")
                    data = json.loads(chunk)
                    if "message" in data and "content" in data["message"]:
                        full_response += data["message"]["content"]
                        yield {"role": "assistant", "content": full_response}  # Yield the cumulative response
        else:
            full_response = ""
            for line in response.iter_lines():
                if line:
                    chunk = line.decode("utf-8")
                    data = json.loads(chunk)
                    if "message" in data and "content" in data["message"]:
                        full_response += data["message"]["content"]
            yield {"role": "assistant", "content": full_response}  # Yield the complete response
    except requests.exceptions.RequestException as e:
        yield {"role": "assistant", "content": f"Error communicating with the Ollama API: {str(e)}"}

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

# Generate architectural diagram based on extracted section information
def generate_architectural_diagram(extracted_info, model_name, stream=True):
    # Create a prompt that uses the extracted section information directly
    sections_text = ""
    for category, items in extracted_info.items():
        sections_text += f"\n{category}:\n"
        for item in items:
            sections_text += f"- From {item['section']}: {item['content']}\n"
    
    # Updated prompt with specific instructions for centered rectangular blocks without arrows
    diagram_prompt = (
        "You are an educational visualization specialist. Based on the following extracted information from different "
        "sections of a research paper, create a comprehensive architectural diagram that visually represents the "
        "paper's framework using rectangular blocks.\n\n"
        f"Extracted information from paper sections:\n{sections_text}\n\n"
        "Follow these guidelines for creating the architectural diagram:\n"
        "1. Create a clear, hierarchical structure showing the flow of the research\n"
        "2. Use rectangular blocks with '+' for corners, '-' for horizontal borders, and '|' for vertical borders\n"
        "3. Make each block wide enough to accommodate the full content (not limited to a fixed width)\n"
        "4. Center the title of each block\n"
        "5. Include bullet points (•) for content inside each block\n"
        "6. Arrange blocks vertically with NO arrows or connecting symbols between them\n"
        "7. Center-align the entire diagram\n"
        "8. Ensure the diagram is readable in both light and dark modes\n\n"
        "Format your response as a text-based diagram using ASCII characters for blocks. "
        "Make sure the diagram is well-structured, easy to read, and captures the complete research framework."
    )
    
    try:
        if stream:
            return query_ollama_model(diagram_prompt, model_name, stream=True)
        else:
            full_response = ""
            for chunk in query_ollama_model(diagram_prompt, model_name, stream=False):
                full_response = chunk["content"]
            return full_response
    except Exception as e:
        print(f"Error generating architectural diagram: {str(e)}")
        return f"Error generating architectural diagram: {str(e)}"

# Function to handle PDF upload and summarization with streaming
def summarize_paper(file, model_name, summary_state):
    if not file:
        # Return properly formatted chatbot messages (list of tuples)
        yield [("", "Please upload a valid PDF file.")], summary_state
        return
    try:
        # Extract text and sections from the PDF using the parser
        parsed_data = extract_text_from_pdf_with_sections(file.name)
        
        # Check if text was successfully extracted
        if not parsed_data["full_text"].strip():
            yield [("", "No text could be extracted from the uploaded PDF.")], summary_state
            return
        
        # Step 1: Extract section-specific information
        # Use proper tuple format for chatbot messages
        chat_history = [("", "Analyzing the paper to extract key details from each section...")]
        yield chat_history, ""
        
        # Extract information from each section
        extracted_info = extract_section_information(parsed_data, model_name)
        
        # Store extracted information in summary_state for later use
        paper_details_state = extracted_info
        
        # Update chat history to indicate completion of extraction
        chat_history = [("", "Key details extracted from all sections. Generating architectural diagram...")]
        yield chat_history, paper_details_state
        
        # Step 2: Generate architectural diagram based on extracted section information
        for chunk in generate_architectural_diagram(extracted_info, model_name, stream=True):
            if "content" in chunk:
                # Replace the last assistant message with proper tuple format
                chat_history = [("", chunk["content"])]
                yield chat_history, paper_details_state
    except Exception as e:
        yield [("", f"An error occurred: {str(e)}")], summary_state

# Function to preprocess question text
def preprocess_question(question_text):
    """
    Remove "[Question]" prefix if present and return the cleaned question text.
    Returns None if the question becomes empty after preprocessing.
    """
    if not question_text:
        return None
        
    # Check if question starts with "[Question]"
    if question_text.startswith("[Question]"):
        # Remove the prefix
        cleaned_text = question_text[len("[Question]"):].strip()
        # Return None if empty, otherwise return cleaned text
        return cleaned_text if cleaned_text else None
    return question_text

# Function to create a default MCQ when generation fails
def create_default_mcq(index, paper_details):
    """
    Create a default MCQ when generation fails repeatedly.
    This ensures users always have questions to answer.
    """
    # Extract a key concept from paper details to use in the question
    concept = "the research"
    for category, items in paper_details.items():
        if items and items[0]["content"]:
            concept = items[0]["section"]
            break
    
    # Create different default questions based on the index
    if index == 0:
        return {
            "question": f"What is the main focus of {concept}?",
            "options": {
                "a": "Understanding theoretical concepts",
                "b": "Developing practical applications",
                "c": "Analyzing experimental results",
                "d": "Comparing with previous work"
            },
            "correct_answer": "b"
        }
    elif index == 1:
        return {
            "question": "Which approach best describes the methodology used?",
            "options": {
                "a": "Quantitative analysis of large datasets",
                "b": "Qualitative assessment through case studies",
                "c": "Mixed methods combining multiple approaches",
                "d": "Theoretical modeling without empirical validation"
            },
            "correct_answer": "c"
        }
    else:
        return {
            "question": "What is the most significant contribution of this research?",
            "options": {
                "a": "New theoretical framework",
                "b": "Improved methodology",
                "c": "Novel application of existing techniques",
                "d": "Comprehensive literature review"
            },
            "correct_answer": "c"
        }

# Enhanced function to generate a single MCQ with better error handling and formatting flexibility
def generate_single_mcq(paper_details, model_name, existing_mcqs=None, attempt=1):
    """
    Generate a single MCQ with enhanced error handling and formatting flexibility.
    Includes multiple attempts and fallback mechanisms.
    """
    if not paper_details:
        return None
    
    # Ensure existing_mcqs is a list, not None
    if existing_mcqs is None:
        existing_mcqs = []
    
    # Get all available sections from the extracted information
    sections_text = ""
    for category, items in paper_details.items():
        sections_text += f"\n{category}:\n"
        for item in items:
            sections_text += f"- From {item['section']}: {item['content']}\n"
    
    # Add diversity instructions
    diversity_instructions = (
        "IMPORTANT: Generate a UNIQUE question that is DIFFERENT from any previous questions. "
        "Focus on a DISTINCT concept or finding that hasn't been covered before."
    )
    
    # Add topic categories
    topic_categories = (
        "Choose ONE of these different categories for your question:\n"
        "- Theoretical concepts and frameworks\n"
        "- Methodological approaches and techniques\n"
        "- Empirical findings and results\n"
        "- Implications and applications\n"
    )
    
    # Enhanced exclusion mechanism
    exclusion_text = ""
    if existing_mcqs and len(existing_mcqs) > 0:
        exclusion_text = "IMPORTANT - Do NOT generate a question similar to these previous ones:\n"
        # Add the most recent questions first (more likely to be remembered by the LLM)
        recent_mcqs = existing_mcqs[-9:] if len(existing_mcqs) > 9 else existing_mcqs
        for mcq in recent_mcqs:
            if isinstance(mcq, dict) and "question" in mcq:
                exclusion_text += f"- {mcq['question']}\n"
    
    # Add random seed
    random_seed = random.randint(1, 1000)
    
    # Add formatting instructions based on attempt number
    # For later attempts, provide more explicit formatting instructions
    formatting_instructions = ""
    if attempt > 1:
        formatting_instructions = (
            "CRITICAL FORMATTING INSTRUCTIONS:\n"
            "1. Start with the question text directly (no numbering or 'Question:' prefix)\n"
            "2. Each option MUST start with the letter (a., b., c., or d.) followed by a space\n"
            "3. The correct answer MUST be specified as 'Correct Answer: [letter]'\n"
            "4. Follow this EXACT format - any deviation will cause rejection\n\n"
        )
    
    # Construct the enhanced prompt with clearer instructions
    mcq_prompt = (
        f"{formatting_instructions}"
        f"Based on the following extracted details from a research paper, generate exactly one multiple-choice question (MCQ) that tests theoretical understanding of the paper's key concepts and findings:\n"
        f"{sections_text}\n\n"
        f"{diversity_instructions}\n\n"
        f"{topic_categories}\n\n"
        f"{exclusion_text}\n"
        "The question should have four options labeled (a), (b), (c), and (d). Include the correct answer.\n"
        "Focus on testing understanding of the research problem, methodology, results, and implications rather than superficial details.\n"
        f"Use seed {random_seed} to ensure a unique question different from previous generations.\n"
        "Format the output as follows:\n"
        "[Your question text here without any prefix]\n"
        "a. [Option A]\n"
        "b. [Option B]\n"
        "c. [Option C]\n"
        "d. [Option D]\n"
        "Correct Answer: [Correct Option Letter]\n"
    )
    
    try:
        full_response = ""
        for chunk in query_ollama_model(mcq_prompt, model_name, stream=False):
            full_response = chunk["content"]
        
        # Debug output to help diagnose formatting issues
        if attempt > 1:
            print(f"Attempt {attempt} response:\n{full_response[:200]}...")
        
        # Enhanced parsing with more flexible pattern matching
        current_question = {}
        question_text = ""
        options = {}
        correct_answer = ""
        
        # Process the response line by line with more flexible parsing
        lines = full_response.split("\n")
        in_options = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for option lines with flexible pattern matching
            option_match = re.match(r'^([a-d])[.)]?\s+(.+)$', line)
            if option_match:
                in_options = True
                option_letter = option_match.group(1).lower()
                option_text = option_match.group(2).strip()
                options[option_letter] = option_text
                continue
                
            # Check for correct answer with flexible pattern matching
            if re.search(r'correct\s+answer\s*[:]?\s*([a-d])', line, re.IGNORECASE):
                correct_letter = re.search(r'correct\s+answer\s*[:]?\s*([a-d])', line, re.IGNORECASE).group(1).lower()
                correct_answer = correct_letter
                continue
                
            # If we haven't started processing options yet, this is part of the question text
            if not in_options and not line.lower().startswith("correct answer"):
                # Skip lines that look like formatting instructions or numbering
                if not re.match(r'^(\d+[.)]|question:|instructions:)', line.lower()):
                    question_text += line + " "
        
        # Clean up the question text
        question_text = question_text.strip()
        processed_question = preprocess_question(question_text)
        
        # Construct the MCQ if we have all required components
        if processed_question and options and len(options) >= 3 and correct_answer:
            # If we only have 3 options, add a fourth one
            if len(options) == 3:
                missing_letters = set(['a', 'b', 'c', 'd']) - set(options.keys())
                if missing_letters:
                    missing_letter = list(missing_letters)[0]
                    options[missing_letter] = f"None of the above"
            
            # Ensure we have exactly options a, b, c, d
            if set(options.keys()) == set(['a', 'b', 'c', 'd']):
                return {
                    "question": processed_question,
                    "options": options,
                    "correct_answer": correct_answer
                }
        
        # If we've reached the maximum number of attempts, create a default MCQ
        if attempt >= MAX_MCQ_GENERATION_ATTEMPTS:
            print(f"Failed to generate valid MCQ after {MAX_MCQ_GENERATION_ATTEMPTS} attempts, using default")
            # Determine which default MCQ to use based on existing MCQs
            default_index = len(existing_mcqs) % 3
            return create_default_mcq(default_index, paper_details)
            
        # Try again with a different approach if we haven't reached max attempts
        if attempt < MAX_MCQ_GENERATION_ATTEMPTS:
            print(f"Attempt {attempt} failed, trying again with more explicit instructions")
            return generate_single_mcq(paper_details, model_name, existing_mcqs, attempt + 1)
            
        print("Generated MCQ is invalid")
        return None
        
    except Exception as e:
        print(f"Error generating single MCQ: {str(e)}")
        
        # If we've reached the maximum number of attempts, create a default MCQ
        if attempt >= MAX_MCQ_GENERATION_ATTEMPTS:
            print(f"Exception in MCQ generation after {MAX_MCQ_GENERATION_ATTEMPTS} attempts, using default")
            # Determine which default MCQ to use based on existing MCQs
            default_index = len(existing_mcqs) % 3
            return create_default_mcq(default_index, paper_details)
            
        # Try again with a different approach if we haven't reached max attempts
        if attempt < MAX_MCQ_GENERATION_ATTEMPTS:
            print(f"Exception in attempt {attempt}, trying again")
            return generate_single_mcq(paper_details, model_name, existing_mcqs, attempt + 1)
            
        return None

# Enhanced function to generate practice MCQs with better error handling and fallback mechanisms
def generate_mcqs(paper_details, model_name, previous_mcqs=None):
    """
    Generate practice MCQs with enhanced error handling and fallback mechanisms.
    Ensures that valid MCQs are always returned.
    """
    if not paper_details:  # Ensure paper_details has content
        return []
    
    # Ensure previous_mcqs is a list, not None
    if previous_mcqs is None:
        previous_mcqs = []
    
    # Get all available sections from the extracted information
    sections_text = ""
    for category, items in paper_details.items():
        sections_text += f"\n{category}:\n"
        for item in items:
            sections_text += f"- From {item['section']}: {item['content']}\n"
    
    # Add diversity instructions
    diversity_instructions = (
        "IMPORTANT: Generate DIVERSE questions that cover DIFFERENT aspects of the paper. "
        "Each question must focus on a DISTINCT concept or finding. "
        "Specifically:\n"
        "- Question 1: Focus on the research problem or motivation\n"
        "- Question 2: Focus on the methodology or approach\n"
        "- Question 3: Focus on the results, findings, or implications\n"
    )
    
    # Add topic categories
    topic_categories = (
        "Ensure questions cover these different categories:\n"
        "- Theoretical concepts and frameworks\n"
        "- Methodological approaches and techniques\n"
        "- Empirical findings and results\n"
        "- Implications and applications\n"
    )
    
    # Enhanced exclusion mechanism
    exclusion_text = ""
    if previous_mcqs and len(previous_mcqs) > 0:
        exclusion_text = "IMPORTANT - Do NOT generate questions similar to these previous ones:\n"
        # Add the most recent questions first (more likely to be remembered by the LLM)
        recent_mcqs = previous_mcqs[-9:] if len(previous_mcqs) > 9 else previous_mcqs
        for mcq in recent_mcqs:
            if isinstance(mcq, dict) and "question" in mcq:
                exclusion_text += f"- {mcq['question']}\n"
    
    # Add random seed
    random_seed = random.randint(1, 1000)
    
    # Add clearer formatting instructions
    formatting_instructions = (
        "CRITICAL FORMATTING INSTRUCTIONS:\n"
        "1. Number each question as '1.', '2.', and '3.'\n"
        "2. Each option MUST start with the letter (a., b., c., or d.) followed by a space\n"
        "3. The correct answer MUST be specified as 'Correct Answer: [letter]'\n"
        "4. Follow this EXACT format - any deviation will cause rejection\n\n"
    )
    
    # Construct the enhanced prompt
    mcq_prompt = (
        f"{formatting_instructions}"
        f"Based on the following extracted details from a research paper, generate exactly three multiple-choice questions (MCQs) that test theoretical understanding of the paper's key concepts and findings:\n"
        f"{sections_text}\n\n"
        f"{diversity_instructions}\n\n"
        f"{topic_categories}\n\n"
        f"{exclusion_text}\n"
        "Each question should have four options labeled (a), (b), (c), and (d). Include the correct answer for each question.\n"
        "Focus on testing understanding of the research problem, methodology, results, and implications rather than superficial details.\n"
        f"Use seed {random_seed} to ensure unique questions different from previous generations.\n"
        "Format the output as follows:\n"
        "1. [Question text without any prefix]\n"
        "   a. [Option A]\n"
        "   b. [Option B]\n"
        "   c. [Option C]\n"
        "   d. [Option D]\n"
        "   Correct Answer: [Correct Option Letter]\n"
        "2. [Question text]...\n"
    )
    
    try:
        # First attempt: Try to generate all MCQs at once
        full_response = ""
        for chunk in query_ollama_model(mcq_prompt, model_name, stream=False):
            full_response = chunk["content"]  # Get the complete response
        
        # Parse the response into individual questions
        mcqs = []
        lines = full_response.split("\n")
        current_question = {}
        current_options = {}
        question_text = ""
        in_question = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for new question start
            if re.match(r'^\d+\.', line):
                # Save previous question if it exists
                if question_text and current_options:
                    processed_question = preprocess_question(question_text)
                    if processed_question and len(current_options) >= 3 and "correct_answer" in current_question:
                        # If we only have 3 options, add a fourth one
                        if len(current_options) == 3:
                            missing_letters = set(['a', 'b', 'c', 'd']) - set(current_options.keys())
                            if missing_letters:
                                missing_letter = list(missing_letters)[0]
                                current_options[missing_letter] = f"None of the above"
                        
                        # Ensure we have exactly options a, b, c, d
                        if set(current_options.keys()) == set(['a', 'b', 'c', 'd']):
                            mcqs.append({
                                "question": processed_question,
                                "options": current_options,
                                "correct_answer": current_question.get("correct_answer", "a")
                            })
                
                # Start new question
                question_text = line.split(".", 1)[1].strip()
                current_question = {}
                current_options = {}
                in_question = True
                continue
                
            # Check for option lines
            option_match = re.match(r'^([a-d])[.)]?\s+(.+)$', line)
            if option_match and in_question:
                option_letter = option_match.group(1).lower()
                option_text = option_match.group(2).strip()
                current_options[option_letter] = option_text
                continue
                
            # Check for correct answer
            if re.search(r'correct\s+answer\s*[:]?\s*([a-d])', line, re.IGNORECASE) and in_question:
                correct_letter = re.search(r'correct\s+answer\s*[:]?\s*([a-d])', line, re.IGNORECASE).group(1).lower()
                current_question["correct_answer"] = correct_letter
                
                # End of this question
                processed_question = preprocess_question(question_text)
                if processed_question and len(current_options) >= 3:
                    # If we only have 3 options, add a fourth one
                    if len(current_options) == 3:
                        missing_letters = set(['a', 'b', 'c', 'd']) - set(current_options.keys())
                        if missing_letters:
                            missing_letter = list(missing_letters)[0]
                            current_options[missing_letter] = f"None of the above"
                    
                    # Ensure we have exactly options a, b, c, d
                    if set(current_options.keys()) == set(['a', 'b', 'c', 'd']):
                        mcqs.append({
                            "question": processed_question,
                            "options": current_options,
                            "correct_answer": current_question.get("correct_answer", "a")
                        })
                
                # Reset for next question
                question_text = ""
                current_question = {}
                current_options = {}
                in_question = False
                continue
                
            # If none of the above, and we're in a question, append to question text
            if in_question and not option_match and not re.search(r'correct\s+answer', line, re.IGNORECASE):
                # Only append if it doesn't look like a formatting instruction
                if not re.match(r'^(question:|instructions:)', line.lower()):
                    question_text += " " + line
        
        # Add the last question if it exists and wasn't added yet
        if question_text and current_options and "correct_answer" in current_question:
            processed_question = preprocess_question(question_text)
            if processed_question and len(current_options) >= 3:
                # If we only have 3 options, add a fourth one
                if len(current_options) == 3:
                    missing_letters = set(['a', 'b', 'c', 'd']) - set(current_options.keys())
                    if missing_letters:
                        missing_letter = list(missing_letters)[0]
                        current_options[missing_letter] = f"None of the above"
                
                # Ensure we have exactly options a, b, c, d
                if set(current_options.keys()) == set(['a', 'b', 'c', 'd']):
                    mcqs.append({
                        "question": processed_question,
                        "options": current_options,
                        "correct_answer": current_question.get("correct_answer", "a")
                    })
        
        # Validate that we have valid MCQs
        valid_mcqs = []
        for mcq in mcqs:
            if "question" in mcq and "options" in mcq and "correct_answer" in mcq and len(mcq["options"]) == 4:
                valid_mcqs.append(mcq)
        
        # If we don't have 3 valid MCQs, generate them individually
        if len(valid_mcqs) < 3:
            print(f"Only have {len(valid_mcqs)} valid MCQs from batch generation, generating more individually...")
            
            # Generate individual MCQs to fill the gaps
            while len(valid_mcqs) < 3:
                # Generate a new MCQ with the enhanced single MCQ generator
                additional_mcq = generate_single_mcq(
                    paper_details, 
                    model_name, 
                    previous_mcqs + valid_mcqs
                )
                
                if additional_mcq:
                    valid_mcqs.append(additional_mcq)
                else:
                    # If we can't generate a valid MCQ, use a default one
                    default_index = len(valid_mcqs)
                    default_mcq = create_default_mcq(default_index, paper_details)
                    valid_mcqs.append(default_mcq)
        
        return valid_mcqs[:3]  # Return exactly 3 MCQs
        
    except Exception as e:
        print(f"Error in batch MCQ generation: {str(e)}")
        
        # Fallback: Generate MCQs individually
        print("Falling back to individual MCQ generation...")
        valid_mcqs = []
        
        # Try to generate 3 MCQs individually
        for i in range(3):
            try:
                # Generate a new MCQ with the enhanced single MCQ generator
                additional_mcq = generate_single_mcq(
                    paper_details, 
                    model_name, 
                    previous_mcqs + valid_mcqs
                )
                
                if additional_mcq:
                    valid_mcqs.append(additional_mcq)
                else:
                    # If we can't generate a valid MCQ, use a default one
                    default_mcq = create_default_mcq(i, paper_details)
                    valid_mcqs.append(default_mcq)
            except Exception as inner_e:
                print(f"Error generating individual MCQ {i+1}: {str(inner_e)}")
                # Use a default MCQ as fallback
                default_mcq = create_default_mcq(i, paper_details)
                valid_mcqs.append(default_mcq)
        
        return valid_mcqs  # Return whatever MCQs we managed to generate

# Modified: Extract explanation information for a single MCQ
def extract_explanation_information(paper_details, model_name, mcqs, mcq_index):
    """
    Extract relevant information from paper details that explains the correct answer
    for a single incorrect MCQ.
    """
    # Create a prompt that uses the paper details and the specific incorrect MCQ
    sections_text = ""
    for category, items in paper_details.items():
        sections_text += f"\n{category}:\n"
        for item in items:
            sections_text += f"- From {item['section']}: {item['content']}\n"
    
    # Build the explanation request with the specific question
    if mcq_index < len(mcqs):
        mcq = mcqs[mcq_index]
        question_text = f"\nQuestion: {mcq['question']}\n"
        question_text += f"Correct Answer: {mcq['correct_answer']}. {mcq['options'][mcq['correct_answer']]}\n"
    else:
        return {"questions": []}
    
    extraction_prompt = (
        "Based on the following extracted information from a research paper, identify and extract "
        "the specific sentences and concepts that explain why the correct answer is correct for "
        "the given question.\n\n"
        f"Extracted information from paper sections:\n{sections_text}\n\n"
        f"Question that needs explanation:\n{question_text}\n\n"
        "Extract 2-3 relevant sentences from the paper that directly support "
        "the correct answer. Focus on the most important concepts and evidence.\n\n"
        "Format your response as follows:\n"
        "QUESTION: [Question text]\n"
        "SUPPORTING EVIDENCE:\n"
        "- [Relevant sentence 1]\n"
        "- [Relevant sentence 2]\n"
        "KEY CONCEPT: [Brief description of the key concept that explains the answer]\n"
    )
    
    try:
        # Use non-streaming for extraction to get complete response
        full_response = ""
        for chunk in query_ollama_model(extraction_prompt, model_name, stream=False):
            full_response = chunk["content"]
        
        # Parse the extracted information into a structured format
        extracted_info = {
            "questions": []
        }
        
        # Process the response to extract structured information
        current_question = {}
        current_section = None
        
        for line in full_response.split('\n'):
            line = line.strip()
            if line.startswith('QUESTION'):
                if current_question and 'question_text' in current_question and 'evidence' in current_question:
                    extracted_info['questions'].append(current_question)
                
                question_text = line.split(':', 1)[1].strip() if ':' in line else line
                current_question = {
                    'question_text': question_text,
                    'evidence': [],
                    'key_concept': ''
                }
                current_section = 'question'
            
            elif line.startswith('SUPPORTING EVIDENCE:'):
                current_section = 'evidence'
            
            elif line.startswith('KEY CONCEPT:'):
                current_question['key_concept'] = line.split(':', 1)[1].strip() if ':' in line else ''
                current_section = 'concept'
            
            elif line.startswith('-') and current_section == 'evidence':
                evidence = line[1:].strip()
                if evidence:
                    current_question['evidence'].append(evidence)
        
        # Add the last question if it exists
        if current_question and 'question_text' in current_question and 'evidence' in current_question:
            extracted_info['questions'].append(current_question)
        
        # If no valid information was extracted, create a default explanation
        if not extracted_info['questions']:
            # Create a default explanation based on the MCQ
            if mcq_index < len(mcqs):
                mcq = mcqs[mcq_index]
                default_question = {
                    'question_text': mcq['question'],
                    'evidence': [
                        f"The correct answer is {mcq['correct_answer']}: {mcq['options'][mcq['correct_answer']]}.",
                        "This answer is supported by the paper's content and methodology."
                    ],
                    'key_concept': f"Understanding the paper's approach to {mcq['options'][mcq['correct_answer']].lower()}"
                }
                extracted_info['questions'].append(default_question)
        
        return extracted_info
    
    except Exception as e:
        print(f"Error extracting explanation information: {str(e)}")
        
        # Create a default explanation based on the MCQ
        if mcq_index < len(mcqs):
            mcq = mcqs[mcq_index]
            default_info = {
                "questions": [{
                    'question_text': mcq['question'],
                    'evidence': [
                        f"The correct answer is {mcq['correct_answer']}: {mcq['options'][mcq['correct_answer']]}.",
                        "This answer is supported by the paper's content and methodology."
                    ],
                    'key_concept': f"Understanding the paper's approach to {mcq['options'][mcq['correct_answer']].lower()}"
                }]
            }
            return default_info
        
        return {"questions": []}

# Modified: Generate explanation diagram for a single MCQ
def generate_explanation_diagram(extracted_explanation_info, model_name, mcq_index, stream=True):
    """
    Generate a visual diagram that explains the correct answer for a single MCQ
    based on pre-extracted explanation information.
    """
    # Check if we have valid extracted information
    if not extracted_explanation_info or 'questions' not in extracted_explanation_info or not extracted_explanation_info['questions']:
        return "Error: No explanation information available."
    
    # Create a prompt using the extracted explanation information for the single MCQ
    q_info = extracted_explanation_info['questions'][0]  # Get the first (and only) question info
    questions_info = f"\nQUESTION: {q_info['question_text']}\n"
    questions_info += "SUPPORTING EVIDENCE:\n"
    for evidence in q_info['evidence']:
        questions_info += f"- {evidence}\n"
    questions_info += f"KEY CONCEPT: {q_info['key_concept']}\n"
    
    # Updated prompt with specific instructions for centered rectangular blocks without arrows
    diagram_prompt = (
        "You are an educational visualization specialist. Based on the following extracted explanation "
        "information, create a comprehensive architectural diagram that visually explains why the correct "
        "answer is correct.\n\n"
        f"Extracted explanation information:\n{questions_info}\n\n"
        "Follow these guidelines for creating the explanation diagram:\n"
        "1. Create a clear, hierarchical structure showing the concepts that explain the correct answer\n"
        "2. Use rectangular blocks with '+' for corners, '-' for horizontal borders, and '|' for vertical borders\n"
        "3. Make each block wide enough to accommodate the full content (not limited to a fixed width)\n"
        "4. Center the title of each block\n"
        "5. Include bullet points (•) for content inside each block\n"
        "6. Arrange blocks vertically with NO arrows or connecting symbols between them\n"
        "7. Center-align the entire diagram\n"
        "8. Ensure the diagram is readable in both light and dark modes\n\n"
        "Format your response as a text-based diagram using ASCII characters for blocks. "
        "Make sure the diagram is well-structured, easy to read, and provides clear educational value."
    )
    
    try:
        if stream:
            return query_ollama_model(diagram_prompt, model_name, stream=True)
        else:
            full_response = ""
            for chunk in query_ollama_model(diagram_prompt, model_name, stream=False):
                full_response = chunk["content"]
            return full_response
    except Exception as e:
        print(f"Error generating explanation diagram: {str(e)}")
        
        # Create a simple fallback diagram if generation fails
        fallback_diagram = (
            "Error generating detailed diagram. Here's a simplified explanation:\n\n"
            "+---------------------+\n"
            "| Question            |\n"
            "+---------------------+\n"
            "| Supporting Evidence |\n"
            "+---------------------+\n"
            "| Correct Answer      |\n"
            "+---------------------+\n"
        )
        
        return fallback_diagram

# Modified: Generate explanation with sequential processing and display correct option
def generate_explanation_streaming(paper_details, model_name, mcqs, mcq_index, chatbot):
    """
    Generate explanation for a single incorrect MCQ.
    """
    if not paper_details or not mcqs or mcq_index >= len(mcqs):
        print("Missing data for explanation generation")
        # Use proper tuple format for chatbot messages
        chatbot.append(("", "Unable to generate explanation due to missing data."))
        yield chatbot
        return
    
    try:
        # Get the question number (1-based index for display)
        question_num = mcq_index + 1
        
        # Get the MCQ details
        mcq = mcqs[mcq_index]
        correct_letter = mcq["correct_answer"]
        correct_text = mcq["options"][correct_letter]
        
        # Display the correct option before starting the explanation
        correct_option_message = (
            f"For Question {question_num}, the correct answer is:\n\n"
            f"{correct_letter}. {correct_text}\n\n"
            f"Generating explanation..."
        )
        
        # Add the correct option message to the chatbot
        chatbot.append(("", correct_option_message))
        yield chatbot
        
        # Step 1: Extract explanation information for this specific MCQ
        # Use proper tuple format for chatbot messages
        chatbot.append(("", f"Analyzing incorrect answer for Question {question_num} and extracting relevant information..."))
        explanation_index = len(chatbot) - 1
        yield chatbot
        
        # Extract explanation information for this specific MCQ
        extracted_explanation_info = extract_explanation_information(paper_details, model_name, mcqs, mcq_index)
        
        # Update chat history to indicate completion of extraction
        # Use proper tuple format for chatbot messages
        chatbot[explanation_index] = ("", f"Information extracted for Question {question_num}. Generating visual explanation diagram...")
        yield chatbot
        
        # Step 2: Generate explanation diagram based on extracted information
        for chunk in generate_explanation_diagram(extracted_explanation_info, model_name, mcq_index, stream=True):
            if "content" in chunk:
                # Update the explanation message with the streaming content
                # Use proper tuple format for chatbot messages
                chatbot[explanation_index] = ("", f"Explanation Diagram for Question {question_num}:\n\n" + chunk["content"])
                yield chatbot
    except Exception as e:
        print(f"Error in explanation streaming: {str(e)}")
        # Use proper tuple format for chatbot messages
        chatbot.append(("", f"Error generating explanation diagram for Question {question_num}: {str(e)}"))
        yield chatbot

# Function to process the next explanation in the queue
def process_next_explanation(chatbot, paper_details, model_name, mcq_state):
    """
    Process the next explanation in the pending explanations queue.
    Returns a generator that yields updated chatbot messages.
    """
    # Check if there are any pending explanations
    if not mcq_state or "pending_explanations" not in mcq_state or not mcq_state["pending_explanations"]:
        # No more explanations to process, generate new MCQs
        return generate_new_mcqs(chatbot, paper_details, model_name, mcq_state)
    
    # Get the current MCQs
    mcqs = mcq_state["current_mcqs"]
    
    # Get the next MCQ index to explain
    mcq_index = mcq_state["pending_explanations"][0]
    
    # Remove this index from the pending list
    updated_pending = mcq_state["pending_explanations"][1:]
    mcq_state["pending_explanations"] = updated_pending
    
    # Generate explanation for this MCQ
    explanation_generator = generate_explanation_streaming(paper_details, model_name, mcqs, mcq_index, chatbot)
    
    # Process the streaming explanation
    for updated_chatbot in explanation_generator:
        yield updated_chatbot, mcq_state
    
    # Check if there are more explanations to process
    if mcq_state["pending_explanations"]:
        # There are more explanations to process
        next_explanation_generator = process_next_explanation(chatbot, paper_details, model_name, mcq_state)
        for updated_chatbot, updated_mcq_state in next_explanation_generator:
            yield updated_chatbot, updated_mcq_state
    else:
        # No more explanations to process, generate new MCQs
        new_mcqs_generator = generate_new_mcqs(chatbot, paper_details, model_name, mcq_state)
        for updated_chatbot, updated_mcq_state in new_mcqs_generator:
            yield updated_chatbot, updated_mcq_state

# Function to generate new MCQs after all explanations are processed
def generate_new_mcqs(chatbot, paper_details, model_name, mcq_state):
    """
    Generate new MCQs after all explanations have been processed.
    Returns a generator that yields updated chatbot messages and MCQ state.
    """
    # Add message with proper tuple format
    chatbot.append(("", "Here are your next questions:"))
    yield chatbot, mcq_state
    
    # Ensure mcq_state is a dictionary
    if mcq_state is None:
        mcq_state = {}
    
    # Get previous MCQs, ensuring it's a list
    previous_mcqs = mcq_state.get("previous_mcqs", [])
    if previous_mcqs is None:
        previous_mcqs = []
    
    # Generate new MCQs using paper details
    new_mcqs = generate_mcqs(paper_details, model_name, previous_mcqs)
    
    # We should always get MCQs now due to fallback mechanisms
    if not new_mcqs:
        print("Warning: generate_mcqs returned empty list despite fallbacks")
        # Create default MCQs as a last resort
        new_mcqs = [
            create_default_mcq(0, paper_details),
            create_default_mcq(1, paper_details),
            create_default_mcq(2, paper_details)
        ]
    
    # Ensure we have exactly 3 MCQs
    while len(new_mcqs) < 3:
        default_index = len(new_mcqs)
        new_mcqs.append(create_default_mcq(default_index, paper_details))
    
    # Update MCQ state
    new_mcq_state = {
        "current_mcqs": new_mcqs,
        "previous_mcqs": previous_mcqs + new_mcqs,  # Ensure both are lists before concatenating
        "attempts": mcq_state.get("attempts", []) if isinstance(mcq_state, dict) else [],
        "pending_explanations": []  # Reset pending explanations
    }
    
    # Display new MCQs
    mcq_message = "Answer the following questions based on the paper:\n\n"
    for i, mcq in enumerate(new_mcqs):
        mcq_message += f"{i + 1}. {mcq['question']}\n"
    
    # Add message with proper tuple format
    chatbot.append(("", mcq_message))
    
    yield chatbot, new_mcq_state

# Function to save user session data
def save_user_session(user_id, name, email, chat_history, mcq_data):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    user_dir = os.path.join(USER_DATA_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    
    # Save session info
    session_file = os.path.join(user_dir, f"session_{timestamp}.json")
    
    # Convert chat history to a serializable format if needed
    serializable_chat_history = []
    for msg in chat_history:
        if isinstance(msg, tuple) or isinstance(msg, list):
            # Convert tuple/list to a serializable format
            if len(msg) == 2:
                serializable_chat_history.append({"user": msg[0], "assistant": msg[1]})
            else:
                serializable_chat_history.append({"content": str(msg)})
        elif isinstance(msg, dict):
            serializable_chat_history.append(msg)
        else:
            serializable_chat_history.append({"content": str(msg)})
    
    session_data = {
        "user_id": user_id,
        "name": name,
        "email": email,
        "timestamp": timestamp,
        "chat_history": serializable_chat_history,
        "mcq_data": mcq_data
    }
    
    with open(session_file, "w") as f:
        json.dump(session_data, f, indent=2)
    
    return session_file

# Function to handle PDF upload, summarization, and automatic MCQ generation
def auto_summarize_with_mcqs(file, model_name, summary_state, mcq_state, user_id, name, email):
    if not file:
        yield (
            [("", "Please upload a valid PDF file.")],  # Chatbot output - proper tuple format
            summary_state,  # Summary state
            mcq_state,  # MCQ state
            gr.update(label="Question 1", choices=[]),  # Radio buttons for MCQ 1
            gr.update(label="Question 2", choices=[]),  # Radio buttons for MCQ 2
            gr.update(label="Question 3", choices=[]),  # Radio buttons for MCQ 3
            gr.update(interactive=False),  # Submit answers button
            user_id,  # User ID
            name,  # User name
            email  # User email
        )
        return
    
    # Generate the summary using the two-step process
    summary_generator = summarize_paper(file, model_name, summary_state)
    summary, paper_details = None, None
    
    for chunk, paper_details_state in summary_generator:
        summary = chunk
        paper_details = paper_details_state
        yield (
            summary,  # Chatbot output - already in proper tuple format from summarize_paper
            paper_details,  # Paper details state (replaces summary_state)
            mcq_state,  # MCQ state
            gr.update(label="Question 1", choices=[]),  # Radio buttons for MCQ 1
            gr.update(label="Question 2", choices=[]),  # Radio buttons for MCQ 2
            gr.update(label="Question 3", choices=[]),  # Radio buttons for MCQ 3
            gr.update(interactive=False),  # Submit answers button
            user_id,  # User ID
            name,  # User name
            email  # User email
        )
    
    # Automatically generate MCQs after summary is complete, using paper details instead of summary
    # Generate new MCQs
    mcqs = generate_mcqs(paper_details, model_name, [])  # Pass empty list instead of None
    
    # We should always get MCQs now due to fallback mechanisms
    if not mcqs:
        print("Warning: generate_mcqs returned empty list despite fallbacks")
        # Create default MCQs as a last resort
        mcqs = [
            create_default_mcq(0, paper_details),
            create_default_mcq(1, paper_details),
            create_default_mcq(2, paper_details)
        ]
    
    # Ensure we have exactly 3 MCQs
    while len(mcqs) < 3:
        default_index = len(mcqs)
        mcqs.append(create_default_mcq(default_index, paper_details))
    
    # Initialize previous_mcqs as empty list if mcq_state is None
    previous_mcqs = []
    if mcq_state is not None:
        # Get previous_mcqs from mcq_state, defaulting to empty list if not present or None
        previous_mcqs = mcq_state.get("previous_mcqs", [])
        if previous_mcqs is None:
            previous_mcqs = []
    
    # Update MCQ state
    new_mcq_state = {
        "current_mcqs": mcqs,
        "previous_mcqs": previous_mcqs + mcqs,  # Ensure both are lists before concatenating
        "attempts": [],
        "pending_explanations": []  # Initialize empty pending explanations list
    }
    
    # Display MCQs with radio buttons
    mcq_message = "Answer the following questions based on the paper:\n\n"
    for i, mcq in enumerate(mcqs):
        mcq_message += f"{i + 1}. {mcq['question']}\n"
    
    # Add message with proper tuple format
    summary.append(("", mcq_message))
    
    # Prepare radio button choices for each MCQ with full question text in label
    # Format options to display on separate lines
    radio_choices = []
    
    for i in range(3):
        if i < len(mcqs):
            # Create choices with option letter and text, each on a separate line
            choices = []
            for option, text in sorted(mcqs[i]["options"].items()):
                choices.append(f"{option}. {text}")
            radio_choices.append(choices)
        else:
            radio_choices.append([])
    
    # Save session data
    if user_id and name and email:
        save_user_session(user_id, name, email, summary, new_mcq_state)
    
    # Debug print to verify MCQs are being generated
    print(f"Generated MCQs: {len(mcqs)}")
    for i, mcq in enumerate(mcqs):
        print(f"MCQ {i+1}: {mcq['question']}")
        print(f"Options: {mcq['options']}")
        print(f"Correct Answer: {mcq['correct_answer']}")
    
    # Debug print for radio button updates
    print(f"Radio choices for MCQ 1: {radio_choices[0]}")
    print(f"Label for MCQ 1: Question 1: {mcqs[0]['question']}")
    
    yield (
        summary,  # Chatbot output - already in proper tuple format
        paper_details,  # Paper details state
        new_mcq_state,  # MCQ state
        gr.update(label=f"Question 1: {mcqs[0]['question']}", choices=radio_choices[0], visible=True),  # Radio buttons for MCQ 1
        gr.update(label=f"Question 2: {mcqs[1]['question']}", choices=radio_choices[1], visible=True),  # Radio buttons for MCQ 2
        gr.update(label=f"Question 3: {mcqs[2]['question']}", choices=radio_choices[2], visible=True),  # Radio buttons for MCQ 3
        gr.update(interactive=True, visible=True),  # Submit answers button
        user_id,  # User ID
        name,  # User name
        email  # User email
    )

# Modified: Function to handle MCQ answer submission with sequential explanations
def submit_mcq_answers(chatbot, paper_details, model_name, mcq_state, mcq1_answer, mcq2_answer, mcq3_answer, user_id, name, email):
    if not mcq_state or "current_mcqs" not in mcq_state or not mcq_state["current_mcqs"]:
        return (
            chatbot + [("", "Error: No active MCQs found.")],  # Chatbot output - proper tuple format
            paper_details,  # Paper details state
            mcq_state,  # MCQ state
            gr.update(),  # Radio buttons for MCQ 1
            gr.update(),  # Radio buttons for MCQ 2
            gr.update(),  # Radio buttons for MCQ 3
            gr.update(interactive=False),  # Submit answers button
            user_id,  # User ID
            name,  # User name
            email  # User email
        )
    
    # Get current MCQs
    mcqs = mcq_state["current_mcqs"]
    
    # Process user answers
    user_answers = [
        mcq1_answer.split(".")[0] if mcq1_answer else None,
        mcq2_answer.split(".")[0] if mcq2_answer else None,
        mcq3_answer.split(".")[0] if mcq3_answer else None
    ]
    
    # Check answers
    correct_answers = []
    incorrect_answers = []
    incorrect_indices = []
    
    for i, (mcq, user_answer) in enumerate(zip(mcqs, user_answers)):
        if user_answer and user_answer.lower() == mcq["correct_answer"].lower():
            correct_answers.append(i + 1)
        else:
            incorrect_answers.append(i + 1)
            incorrect_indices.append(i)
    
    # Record attempt
    attempt = {
        "mcqs": mcqs,
        "user_answers": user_answers,
        "correct_answers": correct_answers,
        "incorrect_answers": incorrect_answers,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # Update MCQ state with attempt
    mcq_state["attempts"] = mcq_state.get("attempts", []) + [attempt]
    
    # Display results
    result_message = f"Results: {len(correct_answers)}/{len(mcqs)} correct\n\n"
    
    if correct_answers:
        result_message += "Correct answers: " + ", ".join([f"Question {i}" for i in correct_answers]) + "\n"
    
    if incorrect_answers:
        result_message += "Incorrect answers: " + ", ".join([f"Question {i}" for i in incorrect_answers]) + "\n"
        if len(incorrect_answers) > 1:
            result_message += "\nGenerating visual explanations for each incorrect answer one by one..."
        else:
            result_message += "\nGenerating visual explanation for the incorrect answer..."
    
    # Add message with proper tuple format
    chatbot.append(("", result_message))
    
    # Save session data
    if user_id and name and email:
        save_user_session(user_id, name, email, chatbot, mcq_state)
    
    # If there are incorrect answers, store them in the MCQ state for sequential processing
    if incorrect_indices:
        # Store the incorrect indices in the MCQ state
        mcq_state["pending_explanations"] = incorrect_indices
        
        # Start processing the first explanation
        explanation_processor = process_next_explanation(chatbot, paper_details, model_name, mcq_state)
        
        # Process the explanations sequentially
        for updated_chatbot, updated_mcq_state in explanation_processor:
            # Update the MCQ state with the latest state
            mcq_state = updated_mcq_state
            
            # Prepare radio button choices for the new MCQs if they exist
            radio_choices = []
            if "current_mcqs" in mcq_state and mcq_state["current_mcqs"]:
                new_mcqs = mcq_state["current_mcqs"]
                for i in range(3):
                    if i < len(new_mcqs):
                        choices = []
                        for option, text in sorted(new_mcqs[i]["options"].items()):
                            choices.append(f"{option}. {text}")
                        radio_choices.append(choices)
                    else:
                        radio_choices.append([])
                
                # Yield the updated state with new MCQs
                yield (
                    updated_chatbot,  # Chatbot output with explanation - already in proper tuple format
                    paper_details,  # Paper details state
                    mcq_state,  # Updated MCQ state
                    gr.update(label=f"Question 1: {new_mcqs[0]['question']}", choices=radio_choices[0], value=None),  # Radio buttons for MCQ 1
                    gr.update(label=f"Question 2: {new_mcqs[1]['question']}", choices=radio_choices[1], value=None),  # Radio buttons for MCQ 2
                    gr.update(label=f"Question 3: {new_mcqs[2]['question']}", choices=radio_choices[2], value=None),  # Radio buttons for MCQ 3
                    gr.update(interactive=True),  # Submit answers button
                    user_id,  # User ID
                    name,  # User name
                    email  # User email
                )
            else:
                # Just yield the updated chatbot without new MCQs
                yield (
                    updated_chatbot,  # Chatbot output with explanation - already in proper tuple format
                    paper_details,  # Paper details state
                    mcq_state,  # Updated MCQ state
                    gr.update(value=None),  # Reset MCQ1 selection
                    gr.update(value=None),  # Reset MCQ2 selection
                    gr.update(value=None),  # Reset MCQ3 selection
                    gr.update(interactive=False),  # Submit answers button
                    user_id,  # User ID
                    name,  # User name
                    email  # User email
                )
    else:
        # No incorrect answers, generate new MCQs directly
        new_mcqs_generator = generate_new_mcqs(chatbot, paper_details, model_name, mcq_state)
        
        for updated_chatbot, updated_mcq_state in new_mcqs_generator:
            # Update the MCQ state with the latest state
            mcq_state = updated_mcq_state
            
            # Prepare radio button choices for the new MCQs
            radio_choices = []
            if "current_mcqs" in mcq_state and mcq_state["current_mcqs"]:
                new_mcqs = mcq_state["current_mcqs"]
                for i in range(3):
                    if i < len(new_mcqs):
                        choices = []
                        for option, text in sorted(new_mcqs[i]["options"].items()):
                            choices.append(f"{option}. {text}")
                        radio_choices.append(choices)
                    else:
                        radio_choices.append([])
                
                # Yield the updated state with new MCQs
                yield (
                    updated_chatbot,  # Chatbot output - already in proper tuple format
                    paper_details,  # Paper details state
                    mcq_state,  # Updated MCQ state
                    gr.update(label=f"Question 1: {new_mcqs[0]['question']}", choices=radio_choices[0], value=None),  # Radio buttons for MCQ 1
                    gr.update(label=f"Question 2: {new_mcqs[1]['question']}", choices=radio_choices[1], value=None),  # Radio buttons for MCQ 2
                    gr.update(label=f"Question 3: {new_mcqs[2]['question']}", choices=radio_choices[2], value=None),  # Radio buttons for MCQ 3
                    gr.update(interactive=True),  # Submit answers button
                    user_id,  # User ID
                    name,  # User name
                    email  # User email
                )

# Function to handle user registration and start the session
def register_user(name, email):
    if not name or not email:
        return gr.update(visible=True), gr.update(visible=False), "", "", "", "Please enter both name and email to continue."
    
    # Generate a unique user ID
    user_id = str(uuid.uuid4())
    
    # Return values to show the main interface and hide the registration form
    return gr.update(visible=False), gr.update(visible=True), user_id, name, email, ""

# Warm up the model when the app starts
def warm_up_model(model_name="gemma3:1b"):
    try:
        print("Warming up the model...")
        for _ in query_ollama_model("Hello, how are you?", model_name=model_name, stream=False):
            pass
        print("Model warmed up.")
    except Exception as e:
        print(f"Error warming up the model: {str(e)}")

# Custom CSS for improved visual appearance and dark mode compatibility
CUSTOM_CSS = """
/* Enhanced CSS for strict vertical layout of radio options */
.options-radio label {
    display: block !important;
    margin-bottom: 10px !important;
    width: 100% !important;
    clear: both !important;
    float: none !important;
}

/* Force each radio option to be on its own line */
.options-radio .gr-radio-row {
    display: block !important;
    margin-bottom: 8px !important;
}

/* Ensure radio buttons are properly aligned */
.options-radio input[type='radio'] {
    margin-right: 10px !important;
    vertical-align: middle !important;
}

/* Additional styling to prevent horizontal layout */
.options-radio .gr-form {
    display: block !important;
}

/* Prevent any flex or grid layout that might cause horizontal alignment */
.options-radio > div {
    display: block !important;
    flex-direction: column !important;
}

/* Home container styling */
.home-container {
    text-align: center;
    padding: 20px;
    max-width: 800px;
    margin: 0 auto;
}

/* Feature list styling */
.feature-list {
    text-align: left;
    margin: 20px auto;
    max-width: 600px;
}

/* Diagram container with dark mode compatibility */
.diagram-container {
    font-family: monospace;
    white-space: pre;
    overflow-x: auto;
    padding: 15px;
    border-radius: 5px;
    text-align: center;
    margin: 10px auto;
    max-width: 100%;
    display: inline-block;
}

/* Dark mode compatibility for diagrams */
.dark .diagram-container {
    background-color: #2a2a2a !important;
    color: #ffffff !important;
}

/* Light mode styling for diagrams */
.light .diagram-container {
    background-color: #f8f9fa !important;
    color: #000000 !important;
}

/* Center-align all pre elements (used for diagrams) */
pre {
    text-align: center !important;
    margin: 0 auto !important;
    display: inline-block !important;
    white-space: pre !important;
}

/* Ensure chatbot messages are visible in both light and dark modes */
.dark .message-bubble {
    color: #ffffff !important;
}

.light .message-bubble {
    color: #000000 !important;
}

/* Ensure diagrams are properly centered */
.message-bubble pre {
    display: block !important;
    margin: 0 auto !important;
    text-align: center !important;
}
"""

# Create the Gradio interface
with gr.Blocks(css=CUSTOM_CSS) as demo:
    # User session states
    user_id_state = gr.State("")
    user_name_state = gr.State("")
    user_email_state = gr.State("")
    
    # Research assistant states
    paper_details_state = gr.State("")  # Persistent state for the paper details (replaces summary_state)
    mcq_state = gr.State(None)  # Persistent state for MCQs
    
    # Home page with registration form
    with gr.Group(visible=True) as home_page:
        gr.Markdown("""
        # Research Paper Learning Assistant
        
        Welcome to the Research Paper Learning Assistant! This system helps you understand research papers through interactive learning.
        """, elem_classes=["home-container"])
        
        gr.Markdown("""
        ## Features:
        
        - Upload any research paper PDF
        - Get a visual architectural diagram of the paper's framework
        - Test your understanding with automatically generated MCQs
        - Receive visual explanations for incorrect answers
        - Track your learning progress
        """, elem_classes=["feature-list"])
        
        with gr.Row():
            with gr.Column(scale=1):
                pass
            with gr.Column(scale=2):
                name_input = gr.Textbox(label="Your Name", placeholder="Enter your full name")
                email_input = gr.Textbox(label="Your Email", placeholder="Enter your email address")
                register_button = gr.Button("Start Learning")
                registration_error = gr.Textbox(label="", visible=True)
            with gr.Column(scale=1):
                pass
    
    # Main interface (hidden initially)
    with gr.Group(visible=False) as main_interface:
        with gr.Row():
            with gr.Column(scale=2):
                # Model selection
                model_dropdown = gr.Dropdown(
                    choices=["gemma3:1b", "llama3:8b", "mistral:7b", "phi3:mini"],
                    value="gemma3:1b",
                    label="Select Model"
                )
                
                # File upload
                file_upload = gr.File(label="Upload Research Paper (PDF)", file_types=[".pdf"])
                
                # Chatbot for displaying summaries and explanations
                chatbot = gr.Chatbot(
                    label="Research Assistant",
                    height=500,
                    show_copy_button=True
                )
                
                # MCQ interface
                with gr.Group():
                    mcq1 = gr.Radio(label="Question 1", choices=[], visible=True, elem_classes=["options-radio"])
                    mcq2 = gr.Radio(label="Question 2", choices=[], visible=True, elem_classes=["options-radio"])
                    mcq3 = gr.Radio(label="Question 3", choices=[], visible=True, elem_classes=["options-radio"])
                    
                    with gr.Row():
                        submit_answers_button = gr.Button("Submit Answers", interactive=False)
    
    # Event handlers
    register_button.click(
        fn=register_user,
        inputs=[name_input, email_input],
        outputs=[home_page, main_interface, user_id_state, user_name_state, user_email_state, registration_error]
    )
    
    file_upload.change(
        fn=auto_summarize_with_mcqs,
        inputs=[file_upload, model_dropdown, paper_details_state, mcq_state, user_id_state, user_name_state, user_email_state],
        outputs=[chatbot, paper_details_state, mcq_state, mcq1, mcq2, mcq3, submit_answers_button, user_id_state, user_name_state, user_email_state]
    )
    
    submit_answers_button.click(
        fn=submit_mcq_answers,
        inputs=[chatbot, paper_details_state, model_dropdown, mcq_state, mcq1, mcq2, mcq3, user_id_state, user_name_state, user_email_state],
        outputs=[chatbot, paper_details_state, mcq_state, mcq1, mcq2, mcq3, submit_answers_button, user_id_state, user_name_state, user_email_state]
    )
    
    # Warm up the model when the app starts
    demo.load(fn=lambda: warm_up_model(model_name="gemma3:1b"))

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True)
