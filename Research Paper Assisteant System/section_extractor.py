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
