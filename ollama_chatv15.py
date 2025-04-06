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

# Function to handle PDF upload and summarization with streaming
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
        
        # Modified prompt to generate architectural diagram summary
        predefined_prompt = (
            "You are an educational summarization assistant designed to help students understand research papers. "
            "When provided with academic text, create a summary in the form of an architectural diagram that shows "
            "the complete research paper framework/process using blocks and arrows. "
            "Follow these guidelines:\n"
            "1. Identify the main components of the research (problem statement, methodology, results, etc.)\n"
            "2. Represent each component as a block with a clear label\n"
            "3. Use arrows to show the flow and relationships between components\n"
            "4. Include brief descriptions for each block (1-2 sentences maximum)\n"
            "5. Organize the diagram in a logical flow (typically top-to-bottom or left-to-right)\n"
            "6. Start with the paper title and authors at the top\n\n"
            "Format your response as a text-based diagram using ASCII characters for blocks and arrows. "
            "Use [ ] for blocks, --> for arrows, and organize the layout clearly. "
            "For example:\n\n"
            "[Paper Title]\n"
            "      |\n"
            "      v\n"
            "[Problem Statement] --> [Methodology] --> [Results]\n"
            "                                            |\n"
            "                                            v\n"
            "                                       [Conclusion]\n\n"
            f"Create this architectural diagram for the following research paper:\n{extracted_text}"
        )
        
        print(extracted_text)  # Debugging: Log the extracted text
        chat_history = []
        for chunk in query_ollama_model(predefined_prompt, model_name, stream=True):
            if chat_history and chat_history[-1]["role"] == "assistant":
                chat_history[-1] = chunk  # Replace the last assistant message
            else:
                chat_history.append(chunk)  # Add a new assistant message
            yield chat_history, chunk["content"]
    except Exception as e:
        yield [{"role": "assistant", "content": f"An error occurred: {str(e)}"}], summary_state

# Function to generate practice MCQs based on the summary
def generate_mcqs(summary_state, model_name, previous_mcqs=None):
    if not summary_state or len(summary_state.split()) < 50:  # Ensure summary has sufficient content
        return []
    
    # If previous MCQs exist, ensure new ones are different
    exclusion_text = ""
    if previous_mcqs and len(previous_mcqs) > 0:
        exclusion_text = "Do NOT generate any of these previous questions again:\n"
        for mcq in previous_mcqs:
            if isinstance(mcq, dict) and "question" in mcq:
                exclusion_text += f"- {mcq['question']}\n"
    
    mcq_prompt = (
        f"Based on the following summary, generate exactly three multiple-choice questions (MCQs) that test theoretical understanding:\n"
        f"{summary_state}\n"
        f"{exclusion_text}\n"
        "Each question should have four options labeled (a), (b), (c), and (d). Include the correct answer for each question.\n"
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

# Function to generate explanation for incorrect MCQ answers with streaming
def generate_explanation_streaming(summary_state, model_name, mcqs, incorrect_indices, chatbot):
    if not summary_state or not mcqs or not incorrect_indices:
        print("Missing data for explanation generation")
        chatbot.append({"role": "assistant", "content": "Unable to generate explanation due to missing data."})
        return chatbot
    
    # Create prompt for explanation
    explanation_prompt = (
        f"Based on the following summary:\n{summary_state}\n\n"
        "Provide detailed explanations for why these answers are correct for the following questions:\n"
    )
    
    for idx in incorrect_indices:
        if idx < len(mcqs):
            mcq = mcqs[idx]
            explanation_prompt += f"\nQuestion: {mcq['question']}\n"
            explanation_prompt += f"Correct Answer: {mcq['correct_answer']}. {mcq['options'][mcq['correct_answer']]}\n"
    
    explanation_prompt += "\nFormat your response as an architectural diagram showing the conceptual relationships between the questions and the correct answers. Use blocks and arrows to illustrate how these concepts connect to the main ideas in the paper."
    
    try:
        print("Generating explanation with prompt:", explanation_prompt[:100] + "...")
        
        # Add initial explanation message to chatbot
        chatbot.append({"role": "assistant", "content": "Generating explanation..."})
        explanation_index = len(chatbot) - 1
        
        # Stream the explanation
        for chunk in query_ollama_model(explanation_prompt, model_name, stream=True):
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
    
    # Generate the summary
    summary_generator = summarize_paper(file, model_name, summary_state)
    summary, new_summary_state = None, None
    
    for chunk, new_summary_state in summary_generator:
        summary = chunk
        yield (
            summary,  # Chatbot output
            new_summary_state,  # Summary state
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
    
    # Automatically generate MCQs after summary is complete
    # Generate new MCQs
    mcqs = generate_mcqs(new_summary_state, model_name)
    if not mcqs or len(mcqs) < 3:
        yield (
            summary + [{"role": "assistant", "content": "Failed to generate MCQs. Please try uploading the PDF again."}],  # Chatbot output
            new_summary_state,  # Summary state
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
    mcq_message = "Answer the following questions:\n\n"
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
        new_summary_state,  # Summary state
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
def submit_mcq_answers(chatbot, summary_state, model_name, mcq_state, answer1, answer2, answer3, user_id, name, email):
    user_answers = [answer1, answer2, answer3]
    
    # Add user's answers to chat history
    answer_message = "My answers:\n"
    for i, answer in enumerate(user_answers):
        if answer:  # Only include non-empty answers
            answer_message += f"{i+1}. {answer}\n"
    
    chatbot.append({"role": "user", "content": answer_message})
    
    # Process user answers
    mcqs = mcq_state.get("current_mcqs", [])
    if not mcqs or len(mcqs) < 3 or not user_answers or len(user_answers) < 3:
        yield (
            chatbot,  # Chatbot output
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
    
    # Extract just the option letter from user answers (e.g., "a. Option text" -> "a")
    processed_answers = [ans.split(".")[0] if ans and "." in ans else "" for ans in user_answers]
    
    # Record this attempt in MCQ state
    if "attempts" not in mcq_state:
        mcq_state["attempts"] = []
    
    mcq_state["attempts"].append({
        "timestamp": datetime.datetime.now().isoformat(),
        "questions": mcqs,
        "user_answers": processed_answers
    })
    
    # Check if all answers are correct using stored correct answers
    all_correct = True
    incorrect_indices = []
    
    for i, mcq in enumerate(mcqs):
        if i < len(processed_answers):
            if processed_answers[i] != mcq["correct_answer"]:
                all_correct = False
                incorrect_indices.append(i)
    
    # Save session data with updated MCQ state
    if user_id and name and email:
        save_user_session(user_id, name, email, chatbot, mcq_state)
    
    if all_correct:
        # All answers correct - show success message and generate next set of MCQs
        chatbot.append({"role": "assistant", "content": "Congratulations! All answers are correct. Here are your next questions:"})
        
        # Generate new MCQs
        new_mcqs = generate_mcqs(summary_state, model_name, mcq_state.get("previous_mcqs", []))
        if not new_mcqs or len(new_mcqs) < 3:
            yield (
                chatbot + [{"role": "assistant", "content": "Failed to generate new MCQs. Please try again."}],  # Chatbot output
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
        
        # Update MCQ state
        new_mcq_state = {
            "current_mcqs": new_mcqs,
            "previous_mcqs": mcq_state.get("previous_mcqs", []) + new_mcqs,
            "attempts": mcq_state.get("attempts", [])
        }
        
        # Display new MCQs
        mcq_message = "Answer the following questions:\n\n"
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
            summary_state,  # Summary state
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
    else:
        # Some answers incorrect - generate explanation
        print(f"Generating explanation for incorrect answers at indices: {incorrect_indices}")
        
        # Show correct answers
        result_message = "Results:\n\n"
        for i, mcq in enumerate(mcqs):
            if i < len(processed_answers):
                is_correct = processed_answers[i] == mcq["correct_answer"]
                result_message += f"Question {i+1}: {'✓ Correct' if is_correct else '✗ Incorrect'}\n"
                if not is_correct:
                    result_message += f"Correct answer: {mcq['correct_answer']}. {mcq['options'][mcq['correct_answer']]}\n\n"
        
        chatbot.append({"role": "assistant", "content": result_message})
        
        # Stream the explanation
        explanation_generator = generate_explanation_streaming(summary_state, model_name, mcqs, incorrect_indices, chatbot)
        
        # Process the streaming explanation
        for updated_chatbot in explanation_generator:
            yield (
                updated_chatbot,  # Chatbot output with streaming explanation
                summary_state,  # Summary state
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
        
        # Generate new MCQs
        new_mcqs = generate_mcqs(summary_state, model_name, mcq_state.get("previous_mcqs", []))
        if not new_mcqs or len(new_mcqs) < 3:
            yield (
                chatbot + [{"role": "assistant", "content": "Failed to generate new MCQs. Please try again."}],  # Chatbot output
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
        
        # Update MCQ state
        new_mcq_state = {
            "current_mcqs": new_mcqs,
            "previous_mcqs": mcq_state.get("previous_mcqs", []) + new_mcqs,
            "attempts": mcq_state.get("attempts", [])
        }
        
        # Display new MCQs
        mcq_message = "Answer the following questions:\n\n"
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
            summary_state,  # Summary state
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
    summary_state = gr.State("")  # Persistent state for the summary
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
        inputs=[pdf_upload, model_dropdown, summary_state, mcq_state, user_id_state, user_name_state, user_email_state],
        outputs=[chatbot, summary_state, mcq_state, mcq1_radio, mcq2_radio, mcq3_radio, submit_answers_btn, next_mcqs_btn, 
                user_id_state, user_name_state, user_email_state]
    )
    
    # Handle MCQ answer submission
    submit_answers_btn.click(
        submit_mcq_answers,
        inputs=[chatbot, summary_state, model_dropdown, mcq_state, mcq1_radio, mcq2_radio, mcq3_radio, 
               user_id_state, user_name_state, user_email_state],
        outputs=[chatbot, summary_state, mcq_state, mcq1_radio, mcq2_radio, mcq3_radio, submit_answers_btn, next_mcqs_btn,
                user_id_state, user_name_state, user_email_state]
    )

# Launch the Gradio app
if __name__ == "__main__":
    warm_up_model()  # Warm up the model at startup
    demo.launch()
