from ollama_client import query_ollama_model

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
