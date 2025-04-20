from pdf_parser import query_ollama_model, extract_text_from_pdf_with_sections, extract_section_information

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