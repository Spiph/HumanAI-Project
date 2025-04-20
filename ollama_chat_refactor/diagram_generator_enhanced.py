"""
Diagram Generator Module with Enhanced Fallback

This module contains the architectural diagram generation functionality from the original ollama_chatv35.py file,
with an added fallback mechanism for empty or malformed explanation diagrams that maintains the exact same format.
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

# Generate explanation diagram for a single MCQ with enhanced fallback
def generate_explanation_diagram(extracted_explanation_info, model_name, mcq_index, stream=True):
    """
    Generate a visual diagram that explains the correct answer for a single MCQ
    based on pre-extracted explanation information, with enhanced fallback for empty diagrams.
    
    Args:
        extracted_explanation_info: Dictionary containing explanation information
        model_name: The name of the model to use
        mcq_index: Index of the MCQ to explain
        stream: Whether to stream the response (default: True)
        
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
        "IMPORTANT: You must include actual text content in your diagram, not just formatting characters."
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
                        if is_empty_or_malformed_diagram(full_response):
                            # If malformed, break out of the loop to use the fallback
                            print("Detected malformed diagram, using fallback...")
                            break
                        
                    # Yield the current state of the diagram
                    yield chunk
            
            # If the final diagram is malformed, generate a fallback
            if is_empty_or_malformed_diagram(full_response):
                fallback_diagram = generate_fallback_explanation(q_info, model_name)
                yield {"content": fallback_diagram}
                
        else:
            # For non-streaming, collect the full response and check if it's malformed
            full_response = ""
            for chunk in query_ollama_model(diagram_prompt, model_name, stream=False):
                full_response = chunk["content"]
            
            # Check if the diagram is malformed
            if is_empty_or_malformed_diagram(full_response):
                # If malformed, use the fallback
                print("Detected malformed diagram, using fallback...")
                return generate_fallback_explanation(q_info, model_name)
            
            return full_response
            
    except Exception as e:
        print(f"Error generating explanation diagram: {str(e)}")
        return generate_fallback_explanation(q_info, model_name)

def generate_fallback_explanation(q_info, model_name):
    """
    Generate a fallback explanation when the diagram generation fails or produces malformed output.
    The fallback maintains the exact same format as the original explanation diagrams.
    
    Args:
        q_info: Dictionary containing question information
        model_name: The name of the model to use
        
    Returns:
        str: A fallback explanation in the same format as regular explanation diagrams
    """
    # Create a prompt for generating a diagram-formatted explanation
    fallback_prompt = (
        "You are an educational visualization specialist. Based on the following question and evidence, "
        "create a comprehensive architectural diagram that visually explains why the correct answer is correct.\n\n"
        f"QUESTION: {q_info['question_text']}\n\n"
        "SUPPORTING EVIDENCE:\n"
    )
    
    for evidence in q_info['evidence']:
        fallback_prompt += f"- {evidence}\n"
    
    fallback_prompt += f"\nKEY CONCEPT: {q_info['key_concept']}\n\n"
    fallback_prompt += (
        "Follow these guidelines for creating the explanation diagram:\n"
        "1. Create a clear, hierarchical structure with these EXACT blocks in this order:\n"
        "   - 'QUESTION ANALYSIS' block at the top\n"
        "   - 'KEY CONCEPTS' block in the middle\n" 
        "   - 'CORRECT ANSWER EXPLANATION' block at the bottom\n"
        "2. Use rectangular blocks with '+' for corners, '-' for horizontal borders, and '|' for vertical borders\n"
        "3. Make each block wide enough to accommodate the full content (at least 60 characters wide)\n"
        "4. Center the title of each block\n"
        "5. Include bullet points (•) for content inside each block\n"
        "6. Arrange blocks vertically with NO arrows or connecting symbols between them\n"
        "7. Center-align the entire diagram\n"
        "8. Ensure the diagram is readable in both light and dark modes\n\n"
        "IMPORTANT: You MUST follow the exact format specified above with the three named blocks. "
        "Do NOT include any text saying this is a fallback or alternative explanation."
    )
    
    try:
        # Generate a diagram-formatted explanation
        explanation = ""
        for chunk in query_ollama_model(fallback_prompt, model_name, stream=False):
            explanation = chunk["content"]
        
        # If the generated content still doesn't look like a diagram, create a standard one
        if is_empty_or_malformed_diagram(explanation):
            # Create a standard diagram format that matches the original style
            width = 70
            
            # Format the question for display (wrap long lines)
            question_lines = []
            words = q_info['question_text'].split()
            current_line = "• "
            for word in words:
                if len(current_line + word) > width - 4:  # -4 for margins
                    question_lines.append(current_line)
                    current_line = "  " + word
                else:
                    current_line += " " + word if current_line != "• " else word
            if current_line:
                question_lines.append(current_line)
            
            # Format the evidence points
            evidence_lines = []
            for evidence in q_info['evidence']:
                evidence_lines.append(f"• {evidence}")
            
            # Create the diagram
            diagram = []
            
            # Add the QUESTION ANALYSIS block
            title = "QUESTION ANALYSIS"
            diagram.append("+" + "-" * (width - 2) + "+")
            diagram.append("|" + title.center(width - 2) + "|")
            diagram.append("+" + "-" * (width - 2) + "+")
            for line in question_lines:
                diagram.append("| " + line.ljust(width - 4) + " |")
            diagram.append("+" + "-" * (width - 2) + "+")
            diagram.append("")  # Empty line between blocks
            
            # Add the KEY CONCEPTS block
            title = "KEY CONCEPTS"
            diagram.append("+" + "-" * (width - 2) + "+")
            diagram.append("|" + title.center(width - 2) + "|")
            diagram.append("+" + "-" * (width - 2) + "+")
            diagram.append("| • " + q_info['key_concept'].ljust(width - 5) + " |")
            diagram.append("+" + "-" * (width - 2) + "+")
            diagram.append("")  # Empty line between blocks
            
            # Add the CORRECT ANSWER EXPLANATION block
            title = "CORRECT ANSWER EXPLANATION"
            diagram.append("+" + "-" * (width - 2) + "+")
            diagram.append("|" + title.center(width - 2) + "|")
            diagram.append("+" + "-" * (width - 2) + "+")
            for line in evidence_lines:
                diagram.append("| " + line.ljust(width - 4) + " |")
            diagram.append("+" + "-" * (width - 2) + "+")
            
            return "\n".join(diagram)
        
        return explanation
        
    except Exception as e:
        print(f"Error generating fallback explanation: {str(e)}")
        
        # Create a standard diagram as a last resort
        width = 70
        diagram = []
        
        # Add the QUESTION ANALYSIS block
        title = "QUESTION ANALYSIS"
        diagram.append("+" + "-" * (width - 2) + "+")
        diagram.append("|" + title.center(width - 2) + "|")
        diagram.append("+" + "-" * (width - 2) + "+")
        diagram.append("| • " + q_info['question_text'].ljust(width - 5) + " |")
        diagram.append("+" + "-" * (width - 2) + "+")
        diagram.append("")  # Empty line between blocks
        
        # Add the KEY CONCEPTS block
        title = "KEY CONCEPTS"
        diagram.append("+" + "-" * (width - 2) + "+")
        diagram.append("|" + title.center(width - 2) + "|")
        diagram.append("+" + "-" * (width - 2) + "+")
        diagram.append("| • " + q_info['key_concept'].ljust(width - 5) + " |")
        diagram.append("+" + "-" * (width - 2) + "+")
        diagram.append("")  # Empty line between blocks
        
        # Add the CORRECT ANSWER EXPLANATION block
        title = "CORRECT ANSWER EXPLANATION"
        diagram.append("+" + "-" * (width - 2) + "+")
        diagram.append("|" + title.center(width - 2) + "|")
        diagram.append("+" + "-" * (width - 2) + "+")
        for evidence in q_info['evidence']:
            diagram.append("| • " + evidence.ljust(width - 5) + " |")
        diagram.append("+" + "-" * (width - 2) + "+")
        
        return "\n".join(diagram)
