"""
Section Extractor Module

This module contains the section extraction functionality from the original ollama_chatv35.py file.
It maintains the exact same functionality without any modifications.
"""

from ollama_api import query_ollama_model

def extract_section_information(parsed_data, model_name):
    """
    Extract section-specific information for the architectural diagram.
    
    Args:
        parsed_data: Dictionary containing parsed paper data
        model_name: The name of the model to use
        
    Returns:
        Dictionary of extracted information organized by category
    """
    # Define the sections we want to extract information from
    section_info = {
        "introduction": {
            "category": "Introduction",
            "prompt": """
    You will be given the Introduction section of an academic paper.
    First, compose a concise narrative summary of the section.
    Next, distill that summary into a bulleted list highlighting:
    - Background & Motivation: the context, problem statement, and why this work matters
    - Related Work: the main prior approaches, their strengths, and remaining gaps
    - Preliminaries: essential definitions, notation, and key assumptions
    """
        },
        "method": {
            "category": "Method",
            "prompt": """
    You will be given the Method section.
    First, write a clear, cohesive summary of the authors’ methodology.
    Then, present a bulleted list of the core elements:
    - Overall approach or framework
    - Principal techniques or algorithms employed
    - Noteworthy implementation or architectural details
    """
        },
        "experiments": {
            "category": "Experiments",
            "prompt": """
    You will be given the Experiments section.
    Begin with a succinct summary of how the study was conducted.
    Then, provide a bulleted list covering:
    - Experimental setup and design
    - Datasets used and how they were partitioned
    - Evaluation metrics and procedures
    """
        },
        "results": {
            "category": "Results",
            "prompt": """
    You will be given the Results section.
    Start with a brief narrative summary of the main findings.
    Then, enumerate the key outcomes in bullet form:
    - Key performance metrics or quantitative outcomes
    - Comparisons to baselines or benchmarks
    - Any statistical significance or notable trends
    """
        },
        "conclusion": {
            "category": "Conclusion and Limitations",
            "prompt": """
    You will be given the Conclusion section.
    First, craft a concise summary of the authors’ final insights.
    Then, distill it into bullets that cover:
    - The primary conclusions drawn
    - Declared limitations of the study
    - Suggested directions for future work
    """
        },
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
