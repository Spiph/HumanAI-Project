"""
Diagram Generator Module with Modified Fallback Mechanism

This module contains the architectural diagram generation functionality from the original ollama_chatv35.py file,
with a modified fallback mechanism that reuses the same function for explanation diagrams.
"""

import re
from ollama_api import query_ollama_model

def generate_architectural_diagram(extracted_info, model_name, stream=True):
    """
    Generate architectural diagram based on extracted section information.
    
    Args:
        extracted_info: Dictionary of extracted information organized by category
        model_name: The name of the model to use
        stream: Whether to stream the response (default: True)
        
    Returns:
        Generator yielding diagram chunks if stream is True, or complete diagram if stream is False
    """
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

def is_empty_or_malformed_diagram(diagram_text):
    """
    Check if a diagram is empty or malformed (contains only lines without meaningful text).
    
    Args:
        diagram_text: The diagram text to check
        
    Returns:
        bool: True if the diagram is empty or malformed, False otherwise
    """
    # Check if the diagram is empty or very short
    if not diagram_text or len(diagram_text) < 50:
        return True
    
    # Check if the diagram contains only lines, +, -, |, and whitespace
    # First, count the number of lines with only these characters
    lines = diagram_text.split('\n')
    line_chars_pattern = r'^[+\-|\\/ \t]*$'
    empty_lines = sum(1 for line in lines if re.match(line_chars_pattern, line))
    
    # If more than 80% of the lines are just formatting characters, consider it malformed
    if empty_lines / max(1, len(lines)) > 0.8:
        return True
    
    # Check if there's any actual text content in the diagram (excluding formatting characters)
    content_pattern = r'[a-zA-Z0-9]'
    has_content = bool(re.search(content_pattern, diagram_text))
    
    return not has_content

# Generate explanation diagram for a single MCQ with modified fallback mechanism
def generate_explanation_diagram(extracted_explanation_info, model_name, mcq_index, stream=True, retry_attempt=False):
    """
    Generate a visual diagram that explains the correct answer for a single MCQ
    based on pre-extracted explanation information, with a modified fallback mechanism
    that reuses the same function instead of using a separate fallback function.
    
    Args:
        extracted_explanation_info: Dictionary containing explanation information
        model_name: The name of the model to use
        mcq_index: Index of the MCQ to explain
        stream: Whether to stream the response (default: True)
        retry_attempt: Whether this is a retry attempt after a failed generation (default: False)
        
    Returns:
        Generator yielding diagram chunks if stream is True, or complete diagram if stream is False
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
    
    # Use different prompts based on whether this is a retry attempt
    if not retry_attempt:
        # Standard prompt for first attempt - same as original
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
            "IMPORTANT: You must include actual text content in your diagram, not just formatting characters."
        )
    else:
        # Simplified prompt for retry attempt - focuses on creating a simpler diagram
        diagram_prompt = (
            "You are an educational visualization specialist. Based on the following extracted explanation "
            "information, create a SIMPLE diagram that explains why the correct answer is correct.\n\n"
            f"Extracted explanation information:\n{questions_info}\n\n"
            "IMPORTANT: Previous attempt failed to include actual content. Follow these guidelines STRICTLY:\n"
            "1. Create a SIMPLE structure with just 2-3 blocks maximum\n"
            "2. Use rectangular blocks with '+' for corners, '-' for horizontal borders, and '|' for vertical borders\n"
            "3. Make each block wide enough to accommodate the full content\n"
            "4. Include a clear title for each block\n"
            "5. MUST include actual text content explaining the concepts (not just formatting characters)\n"
            "6. Use bullet points (•) for listing key points inside each block\n"
            "7. Center-align the entire diagram\n\n"
            "Format your response as a text-based diagram using ASCII characters for blocks.\n"
            "CRITICAL: Your diagram MUST include actual text content explaining the concepts, not just empty boxes."
        )
    
    try:
        if stream:
            # For streaming, we need to collect the full response to check if it's malformed
            # before yielding it, so we'll use a different approach
            full_response = ""
            for chunk in query_ollama_model(diagram_prompt, model_name, stream=True):
                if "content" in chunk:
                    full_response = chunk["content"]
                    
                    # Check if the diagram is complete (has a reasonable amount of content)
                    if len(full_response) > 200:
                        # Check if the diagram is malformed
                        if is_empty_or_malformed_diagram(full_response) and not retry_attempt:
                            # If malformed and not already a retry, break out of the loop to retry
                            print("Detected malformed diagram, retrying with simplified prompt...")
                            break
                        
                    # Yield the current state of the diagram
                    yield chunk
            
            # If the final diagram is malformed and this is not already a retry attempt, retry with simplified prompt
            if is_empty_or_malformed_diagram(full_response) and not retry_attempt:
                # Retry with the same function but with retry_attempt=True
                print("Retrying explanation diagram generation with simplified prompt...")
                for chunk in generate_explanation_diagram(extracted_explanation_info, model_name, mcq_index, stream=True, retry_attempt=True):
                    yield chunk
            elif is_empty_or_malformed_diagram(full_response) and retry_attempt:
                # If still malformed after retry, create a very simple formatted explanation
                print("Diagram still malformed after retry, creating simple formatted explanation...")
                simple_explanation = create_simple_formatted_explanation(q_info)
                yield {"content": simple_explanation}
                
        else:
            # For non-streaming, collect the full response and check if it's malformed
            full_response = ""
            for chunk in query_ollama_model(diagram_prompt, model_name, stream=False):
                full_response = chunk["content"]
            
            # Check if the diagram is malformed and this is not already a retry attempt
            if is_empty_or_malformed_diagram(full_response) and not retry_attempt:
                # Retry with the same function but with retry_attempt=True
                print("Retrying explanation diagram generation with simplified prompt...")
                return generate_explanation_diagram(extracted_explanation_info, model_name, mcq_index, stream=False, retry_attempt=True)
            elif is_empty_or_malformed_diagram(full_response) and retry_attempt:
                # If still malformed after retry, create a very simple formatted explanation
                print("Diagram still malformed after retry, creating simple formatted explanation...")
                return create_simple_formatted_explanation(q_info)
            
            return full_response
            
    except Exception as e:
        print(f"Error generating explanation diagram: {str(e)}")
        if not retry_attempt:
            # If this is the first attempt and an error occurred, retry with simplified prompt
            print("Error in first attempt, retrying with simplified prompt...")
            try:
                return generate_explanation_diagram(extracted_explanation_info, model_name, mcq_index, stream, retry_attempt=True)
            except Exception as e2:
                print(f"Error in retry attempt: {str(e2)}")
                return create_simple_formatted_explanation(q_info)
        else:
            # If this is already a retry attempt and an error occurred, create a simple explanation
            return create_simple_formatted_explanation(q_info)

def create_simple_formatted_explanation(q_info):
    """
    Create a very simple formatted explanation when all diagram generation attempts fail.
    This is a last resort fallback.
    
    Args:
        q_info: Dictionary containing question information
        
    Returns:
        str: A simple formatted explanation
    """
    # Create a very simple formatted explanation
    simple_explanation = (
        "EXPLANATION (Diagram generation failed):\n\n"
        "+--------------------------------------------------------------+\n"
        "| EXPLANATION FOR CORRECT ANSWER                               |\n"
        "+--------------------------------------------------------------+\n"
        "|\n"
        f"| QUESTION: {q_info['question_text']}\n"
        "|\n"
        "| SUPPORTING EVIDENCE:\n"
    )
    
    for evidence in q_info['evidence']:
        simple_explanation += f"| - {evidence}\n"
    
    simple_explanation += "|\n"
    simple_explanation += f"| KEY CONCEPT: {q_info['key_concept']}\n"
    simple_explanation += "|\n"
    simple_explanation += "+--------------------------------------------------------------+\n"
    
    return simple_explanation
