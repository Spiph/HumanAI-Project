"""
Diagram Generator Module

This module contains the architectural diagram generation functionality from the original ollama_chatv35.py file.
It maintains the exact same functionality without any modifications.
"""

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

# Generate explanation diagram for a single MCQ
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
