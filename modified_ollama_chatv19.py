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

# Define the Ollama API endpoint
OLLAMA_API_URL = "http://localhost:11434/api/chat"

# Directory to store user data
USER_DATA_DIR = "./user_data"
os.makedirs(USER_DATA_DIR, exist_ok=True)

# Function to extract text from a PDF file using pdfplumber
def extract_text_from_pdf(file):
    try:
        with pdfplumber.open(file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""  # Handle pages with no extractable text
        return text.strip()
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

# Function to extract text from a PDF file with OCR fallback
def extract_text_from_pdf_with_ocr(file):
    try:
        # Attempt regular extraction first
        text = extract_text_from_pdf(file)
        if text.strip():  # If text is successfully extracted, return it
            return text
        # If no text is extracted, fall back to OCR
        images = convert_from_path(file.name)  # Convert PDF pages to images
        ocr_text = ""
        for img in images:
            ocr_text += pytesseract.image_to_string(img)  # Perform OCR on each image
        return ocr_text.strip()
    except Exception as e:
        return f"Error extracting text from PDF using OCR: {str(e)}"

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

# Step 1: Extract important details about the paper framework
def extract_paper_details(extracted_text, model_name):
    # Prompt to extract important details about the paper
    extraction_prompt = (
        "You are an educational research assistant designed to help students understand research papers. "
        "Extract and organize the key components of the following research paper into these specific categories:\n\n"
        "1. Research Need/Problem: Why was this research conducted? What gap or problem does it address?\n"
        "2. Proposed Solution/Method: What approach or methodology does the paper propose?\n"
        "3. Experimental Setup: How were experiments designed and conducted?\n"
        "4. Results: What were the main findings or outcomes?\n"
        "5. Limitations: What constraints or shortcomings are mentioned?\n"
        "6. Future Work: What next steps or future research directions are suggested?\n\n"
        "For each category, provide 1-3 concise bullet points with the most important information. "
        "If a category is not explicitly addressed in the paper, indicate this with 'Not specified'.\n\n"
        f"Research paper text:\n{extracted_text}"
    )
    
    try:
        full_response = ""
        for chunk in query_ollama_model(extraction_prompt, model_name, stream=False):
            full_response = chunk["content"]  # Get the complete response
        return full_response
    except Exception as e:
        print(f"Error extracting paper details: {str(e)}")
        return f"Error extracting paper details: {str(e)}"

# Step 2: Generate architectural diagram based on extracted details
def generate_architectural_diagram(paper_details, model_name, stream=True):
    # Prompt to generate architectural diagram
    diagram_prompt = (
        "You are an educational visualization specialist. Based on the following extracted details about a research paper, "
        "create a comprehensive architectural diagram that visually represents the paper's framework using blocks and arrows.\n\n"
        f"Extracted paper details:\n{paper_details}\n\n"
        "Follow these guidelines for creating the architectural diagram:\n"
        "1. Create a clear, hierarchical structure showing the flow of the research\n"
        "2. Use blocks (represented as [ ]) for each major component\n"
        "3. Use arrows (-->, |, v) to show relationships and flow between components\n"
        "4. Include brief descriptions inside each block (1-2 sentences maximum)\n"
        "5. Organize in a logical flow (top-to-bottom or left-to-right)\n"
        "6. Include all relevant components: problem statement, methodology, experiments, results, limitations\n\n"
        "Format your response as a text-based diagram using ASCII characters for blocks and arrows. "
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

# Function to handle PDF upload and summarization with streaming (two-step process)
def summarize_paper(file, model_name, summary_state):
    if not file:
        yield [{"role": "assistant", "content": "Please upload a valid PDF file."}], summary_state
        return
    try:
        # Extract text from the PDF
        with open(file.name, "rb") as pdf_file:
            extracted_text = extract_text_from_pdf_with_ocr(pdf_file)  # Use improved extraction function
        # Truncate the extracted text to reduce context size
        extracted_text = extracted_text[:5000].strip()
        if not extracted_text.strip():
            yield [{"role": "assistant", "content": "No text could be extracted from the uploaded PDF."}], summary_state
            return
        
        # Step 1: Extract important details about the paper
        chat_history = [{"role": "assistant", "content": "Analyzing the paper to extract key details..."}]
        yield chat_history, ""
        
        paper_details = extract_paper_details(extracted_text, model_name)
        
        # Store paper details in summary_state for later use
        paper_details_state = paper_details
        
        chat_history = [{"role": "assistant", "content": "Key details extracted. Generating architectural diagram..."}]
        yield chat_history, paper_details_state
        
        # Step 2: Generate architectural diagram based on extracted details
        for chunk in generate_architectural_diagram(paper_details, model_name, stream=True):
            if chat_history and chat_history[-1]["role"] == "assistant":
                chat_history[-1] = chunk  # Replace the last assistant message
            else:
                chat_history.append(chunk)  # Add a new assistant message
            yield chat_history, paper_details_state
    except Exception as e:
        yield [{"role": "assistant", "content": f"An error occurred: {str(e)}"}], summary_state

# Function to generate practice MCQs based on the extracted paper details instead of summary
def generate_mcqs(paper_details, model_name, previous_mcqs=None):
    if not paper_details or len(paper_details.split()) < 50:  # Ensure paper details has sufficient content
        return []
    
    # If previous MCQs exist, ensure new ones are different
    exclusion_text = ""
    if previous_mcqs and len(previous_mcqs) > 0:
        exclusion_text = "Do NOT generate any of these previous questions again:\n"
        for mcq in previous_mcqs:
            if isinstance(mcq, dict) and "question" in mcq:
                exclusion_text += f"- {mcq['question']}\n"
    
    mcq_prompt = (
        f"Based on the following extracted details from a research paper, generate exactly three multiple-choice questions (MCQs) that test theoretical understanding of the paper's key concepts and findings:\n"
        f"{paper_details}\n"
        f"{exclusion_text}\n"
        "Each question should have four options labeled (a), (b), (c), and (d). Include the correct answer for each question.\n"
        "Focus on testing understanding of the research problem, methodology, results, and implications rather than superficial details.\n"
        "Format the output as follows:\n"
        "1. [Question]\n"
        "   a. [Option A]\n"
        "   b. [Option B]\n"
        "   c. [Option C]\n"
        "   d. [Option D]\n"
        "   Correct Answer: [Correct Option Letter]\n"
        "2. [Question]...\n"
    )
    try:
        full_response = ""
        for chunk in query_ollama_model(mcq_prompt, model_name, stream=False):
            full_response += chunk["content"]  # Accumulate the full response
        print(f"Raw MCQ response: {full_response}")  # Debugging: Log the raw response
        
        # Parse the response into individual questions
        mcqs = []
        lines = full_response.split("\n")
        current_question = {}
        for line in lines:
            line = line.strip()
            if line.startswith("Correct Answer:"):
                current_question["correct_answer"] = line.split(":")[1].strip().lower()
                mcqs.append(current_question)
                current_question = {}
            elif line.startswith(("1.", "2.", "3.")):
                if current_question:
                    mcqs.append(current_question)
                current_question = {"question": line.split(".")[1].strip(), "options": {}}
            elif line.startswith(("a.", "b.", "c.", "d.")):
                option_letter = line.split(".")[0].strip()
                option_text = line.split(".")[1].strip()
                current_question["options"][option_letter] = option_text
        
        # Add the last parsed question if it exists
        if current_question:
            mcqs.append(current_question)
        
        # Validate that we have at least 3 valid MCQs
        if len(mcqs) < 3:
            return []
        return mcqs
    except Exception as e:
        print(f"Error generating MCQs: {str(e)}")  # Debugging: Log the error
        return []

# Step 1: Generate detailed explanations for incorrect MCQ answers
def generate_detailed_explanation(paper_details, model_name, mcqs, incorrect_indices):
    if not paper_details or not mcqs or not incorrect_indices:
        return "Unable to generate explanation due to missing data."
    
    # Create prompt for detailed explanation
    explanation_prompt = (
        f"Based on the following extracted details from a research paper:\n{paper_details}\n\n"
        "Provide detailed explanations for why these answers are correct for the following questions:\n"
    )
    
    for idx in incorrect_indices:
        if idx < len(mcqs):
            mcq = mcqs[idx]
            explanation_prompt += f"\nQuestion: {mcq['question']}\n"
            explanation_prompt += f"Correct Answer: {mcq['correct_answer']}. {mcq['options'][mcq['correct_answer']]}\n"
    
    explanation_prompt += "\nFor each question, explain:\n"
    explanation_prompt += "1. Why the correct answer is right\n"
    explanation_prompt += "2. Key concepts from the paper that support this answer\n"
    explanation_prompt += "3. How this relates to the paper's overall framework\n"
    
    try:
        full_response = ""
        for chunk in query_ollama_model(explanation_prompt, model_name, stream=False):
            full_response = chunk["content"]
        return full_response
    except Exception as e:
        print(f"Error generating detailed explanation: {str(e)}")
        return f"Error generating detailed explanation: {str(e)}"

# Step 2: Generate architectural diagram for explanations
def generate_explanation_diagram(detailed_explanation, model_name, stream=True):
    # Prompt to generate architectural diagram for explanations
    diagram_prompt = (
        "You are an educational visualization specialist. Based on the following detailed explanations "
        "for incorrect MCQ answers, create an architectural diagram that visually represents the conceptual "
        "relationships between the questions, correct answers, and the main ideas in the research paper.\n\n"
        f"Detailed explanations:\n{detailed_explanation}\n\n"
        "Follow these guidelines for creating the architectural diagram:\n"
        "1. Create blocks for each question and its correct answer\n"
        "2. Connect these blocks to relevant concepts from the paper\n"
        "3. Use arrows to show how concepts relate to each other\n"
        "4. Include brief descriptions inside each block (1-2 sentences maximum)\n"
        "5. Organize in a logical flow that helps understand the relationships\n\n"
        "Format your response as a text-based diagram using ASCII characters for blocks and arrows. "
        "For example:\n\n"
        "[Question 1] --> [Correct Answer] --> [Related Concept from Paper]\n"
        "                                        |\n"
        "                                        v\n"
        "                                    [Supporting Evidence]\n\n"
        "Make sure the diagram is well-structured, easy to read, and effectively shows how the answers "
        "connect to the paper's framework."
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
        return f"Error generating explanation diagram: {str(e)}"

# Function to generate explanation for incorrect MCQ answers with streaming (two-step process)
def generate_explanation_streaming(paper_details, model_name, mcqs, incorrect_indices, chatbot):
    if not paper_details or not mcqs or not incorrect_indices:
        print("Missing data for explanation generation")
        chatbot.append({"role": "assistant", "content": "Unable to generate explanation due to missing data."})
        return chatbot
    
    try:
        # Step 1: Generate detailed explanations
        chatbot.append({"role": "assistant", "content": "Analyzing incorrect answers and generating detailed explanations..."})
        explanation_index = len(chatbot) - 1
        yield chatbot
        
        detailed_explanation = generate_detailed_explanation(paper_details, model_name, mcqs, incorrect_indices)
        
        chatbot[explanation_index] = {"role": "assistant", "content": "Explanations generated. Creating architectural diagram..."}
        yield chatbot
        
        # Step 2: Generate architectural diagram based on detailed explanations
        for chunk in generate_explanation_diagram(detailed_explanation, model_name, stream=True):
            # Update the explanation message with the latest content
            chatbot[explanation_index] = {"role": "assistant", "content": "Explanation:\n\n" + chunk["content"]}
            yield chatbot
            
        return chatbot
    except Exception as e:
        print(f"Error generating explanation: {str(e)}")
        chatbot.append({"role": "assistant", "content": f"Error: Unable to generate explanation. Technical details: {str(e)}"})
        return chatbot

# Function to save user session data
def save_user_session(user_id, name, email, chat_history, mcq_data):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    user_dir = os.path.join(USER_DATA_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    
    # Save session info
    session_file = os.path.join(user_dir, f"session_{timestamp}.json")
    session_data = {
        "user_id": user_id,
        "name": name,
        "email": email,
        "timestamp": timestamp,
        "chat_history": chat_history,
        "mcq_data": mcq_data
    }
    
    with open(session_file, "w") as f:
        json.dump(session_data, f, indent=2)
    
    return session_file

# Function to handle PDF upload, summarization, and automatic MCQ generation
def auto_summarize_with_mcqs(file, model_name, summary_state, mcq_state, user_id, name, email):
    if not file:
        yield (
            [(None, "Please upload a valid PDF file.")],  # Chatbot output
            summary_state,  # Summary state
            mcq_state,  # MCQ state
            gr.update(label="Question 1", choices=[]),  # Radio buttons for MCQ 1
            gr.update(label="Question 2", choices=[]),  # Radio buttons for MCQ 2
            gr.update(label="Question 3", choices=[]),  # Radio buttons for MCQ 3
            gr.update(interactive=False),  # Submit answers button
            gr.update(visible=False),  # Next MCQs button
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
            summary,  # Chatbot output
            paper_details,  # Paper details state (replaces summary_state)
            mcq_state,  # MCQ state
            gr.update(label="Question 1", choices=[]),  # Radio buttons for MCQ 1
            gr.update(label="Question 2", choices=[]),  # Radio buttons for MCQ 2
            gr.update(label="Question 3", choices=[]),  # Radio buttons for MCQ 3
            gr.update(interactive=False),  # Submit answers button
            gr.update(visible=False),  # Next MCQs button
            user_id,  # User ID
            name,  # User name
            email  # User email
        )
    
    # Automatically generate MCQs after summary is complete, using paper details instead of summary
    # Generate new MCQs
    mcqs = generate_mcqs(paper_details, model_name)
    if not mcqs or len(mcqs) < 3:
        yield (
            summary + [{"role": "assistant", "content": "Failed to generate MCQs. Please try uploading the PDF again."}],  # Chatbot output
            paper_details,  # Paper details state
            mcq_state,  # MCQ state
            gr.update(label="Question 1", choices=[]),  # Radio buttons for MCQ 1
            gr.update(label="Question 2", choices=[]),  # Radio buttons for MCQ 2
            gr.update(label="Question 3", choices=[]),  # Radio buttons for MCQ 3
            gr.update(interactive=False),  # Submit answers button
            gr.update(visible=False),  # Next MCQs button
            user_id,  # User ID
            name,  # User name
            email  # User email
        )
        return
    
    # Update MCQ state
    new_mcq_state = {
        "current_mcqs": mcqs,
        "previous_mcqs": (mcq_state.get("previous_mcqs", []) if mcq_state else []) + mcqs,
        "attempts": []
    }
    
    # Display MCQs with radio buttons
    mcq_message = "Answer the following questions based on the paper:\n\n"
    for i, mcq in enumerate(mcqs):
        mcq_message += f"{i + 1}. {mcq['question']}\n"
    
    summary.append({"role": "assistant", "content": mcq_message})
    
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
    
    yield (
        summary,  # Chatbot output
        paper_details,  # Paper details state
        new_mcq_state,  # MCQ state
        gr.update(label=f"Question 1: {mcqs[0]['question']}", choices=radio_choices[0]),  # Radio buttons for MCQ 1
        gr.update(label=f"Question 2: {mcqs[1]['question']}", choices=radio_choices[1]),  # Radio buttons for MCQ 2
        gr.update(label=f"Question 3: {mcqs[2]['question']}", choices=radio_choices[2]),  # Radio buttons for MCQ 3
        gr.update(interactive=True),  # Submit answers button
        gr.update(visible=False),  # Next MCQs button
        user_id,  # User ID
        name,  # User name
        email  # User email
    )

# Function to handle MCQ answer submission
def submit_mcq_answers(chatbot, paper_details, model_name, mcq_state, mcq1_answer, mcq2_answer, mcq3_answer, user_id, name, email):
    if not mcq_state or "current_mcqs" not in mcq_state or not mcq_state["current_mcqs"]:
        return (
            chatbot + [{"role": "assistant", "content": "Error: No active MCQs found."}],  # Chatbot output
            paper_details,  # Paper details state
            mcq_state,  # MCQ state
            gr.update(),  # Radio buttons for MCQ 1
            gr.update(),  # Radio buttons for MCQ 2
            gr.update(),  # Radio buttons for MCQ 3
            gr.update(interactive=False),  # Submit answers button
            gr.update(visible=False),  # Next MCQs button
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
        result_message += "\nGenerating explanations for incorrect answers..."
    
    chatbot.append({"role": "assistant", "content": result_message})
    
    # Save session data
    if user_id and name and email:
        save_user_session(user_id, name, email, chatbot, mcq_state)
    
    # If there are incorrect answers, generate explanations using paper details instead of summary
    if incorrect_indices:
        # Generate explanation using the two-step process
        explanation_generator = generate_explanation_streaming(paper_details, model_name, mcqs, incorrect_indices, chatbot)
        
        # Process the streaming explanation
        for updated_chatbot in explanation_generator:
            yield (
                updated_chatbot,  # Chatbot output with streaming explanation
                paper_details,  # Paper details state
                mcq_state,  # MCQ state
                gr.update(label=f"Question 1: {mcqs[0]['question']}", choices=[]),  # Radio buttons for MCQ 1
                gr.update(label=f"Question 2: {mcqs[1]['question']}", choices=[]),  # Radio buttons for MCQ 2
                gr.update(label=f"Question 3: {mcqs[2]['question']}", choices=[]),  # Radio buttons for MCQ 3
                gr.update(interactive=False),  # Submit answers button
                gr.update(visible=False),  # Next MCQs button
                user_id,  # User ID
                name,  # User name
                email  # User email
            )
        
        # Generate new MCQs
        chatbot.append({"role": "assistant", "content": "Here are your next questions:"})
        
        # Generate new MCQs using paper details instead of summary
        new_mcqs = generate_mcqs(paper_details, model_name, mcq_state.get("previous_mcqs", []))
        if not new_mcqs or len(new_mcqs) < 3:
            yield (
                chatbot + [{"role": "assistant", "content": "Failed to generate new MCQs. Please try again."}],  # Chatbot output
                paper_details,  # Paper details state
                mcq_state,  # MCQ state
                gr.update(label="Question 1", choices=[]),  # Radio buttons for MCQ 1
                gr.update(label="Question 2", choices=[]),  # Radio buttons for MCQ 2
                gr.update(label="Question 3", choices=[]),  # Radio buttons for MCQ 3
                gr.update(interactive=False),  # Submit answers button
                gr.update(visible=False),  # Next MCQs button
                user_id,  # User ID
                name,  # User name
                email  # User email
            )
            return
        
        # Update MCQ state
        new_mcq_state = {
            "current_mcqs": new_mcqs,
            "previous_mcqs": mcq_state.get("previous_mcqs", []) + new_mcqs,
            "attempts": mcq_state.get("attempts", [])
        }
        
        # Display new MCQs
        mcq_message = "Answer the following questions based on the paper:\n\n"
        for i, mcq in enumerate(new_mcqs):
            mcq_message += f"{i + 1}. {mcq['question']}\n"
        
        chatbot.append({"role": "assistant", "content": mcq_message})
        
        # Prepare radio button choices for each MCQ
        radio_choices = []
        
        for i in range(3):
            if i < len(new_mcqs):
                # Create choices with option letter and text, each on a separate line
                choices = []
                for option, text in sorted(new_mcqs[i]["options"].items()):
                    # Each option is a separate item in the list
                    choices.append(f"{option}. {text}")
                radio_choices.append(choices)
            else:
                radio_choices.append([])
        
        yield (
            chatbot,  # Chatbot output
            paper_details,  # Paper details state
            new_mcq_state,  # MCQ state
            gr.update(label=f"Question 1: {new_mcqs[0]['question']}", choices=radio_choices[0]),  # Radio buttons for MCQ 1
            gr.update(label=f"Question 2: {new_mcqs[1]['question']}", choices=radio_choices[1]),  # Radio buttons for MCQ 2
            gr.update(label=f"Question 3: {new_mcqs[2]['question']}", choices=radio_choices[2]),  # Radio buttons for MCQ 3
            gr.update(interactive=True),  # Submit answers button
            gr.update(visible=False),  # Next MCQs button
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

# Create the Gradio interface
with gr.Blocks(css="""
    .options-radio label {
        margin-bottom: 12px !important;
        display: block !important;
    }
    .options-radio input[type='radio'] {
        margin-right: 10px !important;
    }
    .home-container {
        text-align: center;
        padding: 20px;
        max-width: 800px;
        margin: 0 auto;
    }
    .feature-list {
        text-align: left;
        margin: 20px auto;
        max-width: 600px;
    }
""") as demo:
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
        
        - Upload any research paper in PDF format
        - Get an architectural diagram summary showing the paper's structure
        - Test your understanding with automatically generated multiple-choice questions
        - Receive explanations for incorrect answers
        - Track your learning progress
        """, elem_classes=["feature-list"])
        
        gr.Markdown("### Please register to begin:", elem_classes=["home-container"])
        
        with gr.Row():
            with gr.Column(scale=1):
                pass
            with gr.Column(scale=2):
                user_name = gr.Textbox(label="Name", placeholder="Enter your name")
                user_email = gr.Textbox(label="Email", placeholder="Enter your email")
                register_button = gr.Button("Start Learning")
                registration_message = gr.Markdown("")
            with gr.Column(scale=1):
                pass
    
    # Main application interface (initially hidden)
    with gr.Group(visible=False) as main_app:
        gr.Markdown("# Research Paper Learning Assistant")
        gr.Markdown("Upload a research paper, and the app will automatically generate an architectural diagram summary and interactive MCQs.")
        
        # Dropdown for model selection
        model_dropdown = gr.Dropdown(
            label="Select Model",
            choices=["gemma3:1b","qwen2.5:0.5b"],
            value="gemma3:1b"  # Default model
        )
        
        with gr.Tab("Research Paper Learning Assistant"):
            gr.Markdown("Upload a research paper in PDF format to begin the learning process.")
            
            # File upload
            pdf_upload = gr.File(label="Upload Research Paper (PDF Only)", file_types=[".pdf"])
            
            # Chatbot for displaying summary and MCQs
            chatbot = gr.Chatbot(label="Summary and MCQ Results", type="messages")
            
            # MCQ interface
            with gr.Group() as mcq_group:
                gr.Markdown("## Multiple Choice Questions")
                
                # Radio buttons for MCQs with question text in label and CSS class for styling
                mcq1_radio = gr.Radio(label="Question 1", choices=[], interactive=True, elem_classes=["options-radio"])
                mcq2_radio = gr.Radio(label="Question 2", choices=[], interactive=True, elem_classes=["options-radio"])
                mcq3_radio = gr.Radio(label="Question 3", choices=[], interactive=True, elem_classes=["options-radio"])
                
                # Submit answers button
                submit_answers_btn = gr.Button("Submit Answers", interactive=False)
                next_mcqs_btn = gr.Button("Next Questions", visible=False)
    
    # Connect registration button to show main app
    register_button.click(
        register_user,
        inputs=[user_name, user_email],
        outputs=[home_page, main_app, user_id_state, user_name_state, user_email_state, registration_message]
    )
    
    # Trigger summarization and automatic MCQ generation when a file is uploaded
    pdf_upload.change(
        auto_summarize_with_mcqs,
        inputs=[pdf_upload, model_dropdown, paper_details_state, mcq_state, user_id_state, user_name_state, user_email_state],
        outputs=[chatbot, paper_details_state, mcq_state, mcq1_radio, mcq2_radio, mcq3_radio, submit_answers_btn, next_mcqs_btn, 
                user_id_state, user_name_state, user_email_state]
    )
    
    # Handle MCQ answer submission
    submit_answers_btn.click(
        submit_mcq_answers,
        inputs=[chatbot, paper_details_state, model_dropdown, mcq_state, mcq1_radio, mcq2_radio, mcq3_radio, 
               user_id_state, user_name_state, user_email_state],
        outputs=[chatbot, paper_details_state, mcq_state, mcq1_radio, mcq2_radio, mcq3_radio, submit_answers_btn, next_mcqs_btn,
                user_id_state, user_name_state, user_email_state]
    )

# Launch the Gradio app
if __name__ == "__main__":
    warm_up_model()  # Warm up the model at startup
    demo.launch()
