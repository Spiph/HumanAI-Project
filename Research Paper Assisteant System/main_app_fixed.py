"""
Main Application Module with Exact UI Preservation

This module contains the main application functionality from the original ollama_chatv35.py file.
It maintains the exact same functionality and UI without any modifications.
"""

import gradio as gr
import os
import datetime
import uuid
import json
import threading
import random
from collections import defaultdict, OrderedDict
import base64, zlib


# Import from our modules
from pdf_parser_fix import extract_text_from_pdf_with_sections
from ollama_api import query_ollama_model
from section_extractor import extract_section_information
#from diagram_generator_enhanced import make_explanation_mermaid_diagram
from mcq_generator import (
    generate_mcqs, generate_single_mcq, generate_multiple_mcqs, 
    create_default_mcq, extract_explanation_information
)

import requests
import tempfile
import subprocess
import base64, zlib

# Directory to store user data
USER_DATA_DIR = "./user_data"
os.makedirs(USER_DATA_DIR, exist_ok=True)

def make_mermaid_diagram(extracted_info: dict[str, list[dict]], output_file=None) -> str:
    keys = list(extracted_info.keys())
    node_ids = [chr(65 + i) for i in range(len(keys))]
    lines = ["graph TD"]

    for idx, key in enumerate(keys):
        content = " ".join(str(item["content"]) for item in extracted_info[key])
        label = (key + ': ' + content.replace('\n', ' ').replace('"', "'")
                 .replace('[', '(').replace(']', ')'))
        lines.append(f'{node_ids[idx]}["<div style=\'width:1900px; font-size:40px;\'>{label}</div>"]')
        if idx:
            lines.append(f"{node_ids[idx - 1]} --> {node_ids[idx]}")

    mermaid_code = "\n".join(lines)
    #print(mermaid_code)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mmd", mode='w', encoding='utf-8') as tmp:
        tmp.write(mermaid_code)
        mmd_path = tmp.name

    if output_file is None:
        output_file = f"diagram_{os.getpid()}_{os.urandom(4).hex()}.png"

    try:
        mmdc_path = r"C:\\Users\\canno\\AppData\\Roaming\\npm\\mmdc.cmd"  # adjust path if needed
        subprocess.run([
            mmdc_path, "-i", mmd_path, "-o", output_file,
            "--configFile", "theme.json",
            "--scale", "4",  # 2x resolution
            "--width", "2600",  # Width in pixels
            
        ], check=True)

        print(f"✅ Saved {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Mermaid rendering failed: {e}")
    finally:
        os.remove(mmd_path)

    # Convert image to base64
    if os.path.exists(output_file):
        with open(output_file, "rb") as img:
            b64 = base64.b64encode(img.read()).decode("utf-8")
        return f"<div style='overflow-x:auto; text-align:center;'><img src='data:image/png;base64,{b64}' style='display:inline-block; width:1000px; height:1200; max-width:none; max-height:none;'></div>"

    return "<p><i>Diagram generation failed.</i></p>"

def make_explanation_mermaid_diagram(explanation_text, output_file) -> str:
    # diagram_data = {
    #     "Explanation": [{"content": explanation_text}]
    # }
    return make_mermaid_diagram(explanation_text, output_file=output_file)

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

    except Exception as e:
        yield [("", f"An error occurred: {str(e)}")], summary_state

def prepare_diagram_data(extracted_info):
    """
    Convert extracted_info into diagram_data format expected by make_mermaid_diagram.
    Ensures each section becomes a list of dicts with "content" as a string.
    """
    q_info = extracted_info

    diagram_data = {
        "Question": [{"section": "explanation", "content": q_info["question_text"]}],
        "Supporting Evidence": [{"section": "evidence", "content": e} for e in q_info["evidence"]],
        "Key Concept": [{"section": "concept", "content": q_info["key_concept"]}]
    }

    return diagram_data


# Function to generate explanation with sequential processing and display correct option
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

        # Instead of generating text-based ASCII diagram, use your mermaid image-based diagram
        output_file = f"diagram_explanation_q{question_num}.png"
        diagram_data = prepare_diagram_data(extracted_explanation_info["questions"][0])

        rendered_diagram_html = make_mermaid_diagram(diagram_data, output_file)
        #rendered_diagram_html = make_explanation_mermaid_diagram(extracted_explanation_info["questions"][0]["key_concept"], output_file)
        chatbot[explanation_index] = ("", f"Explanation Diagram for Question {question_num}:<br>{rendered_diagram_html}")
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
def auto_summarize_with_mcqs(file, model_name,
                             summary_state, mcq_state,
                             user_id, name, email):
    """Pipeline: PDF → summary → Mermaid PNG → MCQs (10 outputs)."""

    def make_out(chat, details_state, mcq_state_val,
                 mcq1_upd, mcq2_upd, mcq3_upd, submit_upd):
        return (
            chat, details_state, mcq_state_val,
            mcq1_upd, mcq2_upd, mcq3_upd,
            submit_upd,
            user_id, name, email
        )

    if not file:
        chat = [("", "Please upload a valid PDF file.")]
        yield make_out(chat, summary_state, mcq_state,
                       gr.update(label="Question 1", choices=[]),
                       gr.update(label="Question 2", choices=[]),
                       gr.update(label="Question 3", choices=[]),
                       gr.update(interactive=False))
        return

    for out in summarize_paper(file, model_name, summary_state):
        chat, paper_details_state = out[0], out[1]
        yield make_out(chat, paper_details_state, mcq_state,
                       gr.update(label="Question 1", choices=[]),
                       gr.update(label="Question 2", choices=[]),
                       gr.update(label="Question 3", choices=[]),
                       gr.update(interactive=False))
    paper_details = paper_details_state

    #print("Paper details:", paper_details)
    #print(type(paper_details))
    
    # Step 2: Mermaid diagram
    diagram_html = make_mermaid_diagram(paper_details)
    chat.append(("", diagram_html))
    yield make_out(chat, paper_details, mcq_state,
                   gr.update(label="Question 1", choices=[]),
                   gr.update(label="Question 2", choices=[]),
                   gr.update(label="Question 3", choices=[]),
                   gr.update(interactive=False))

    # Step 3: generate MCQs
    mcqs = generate_mcqs(paper_details, model_name, []) or [create_default_mcq(i, paper_details) for i in range(3)]
    while len(mcqs) < 3:
        mcqs.append(create_default_mcq(len(mcqs), paper_details))
    new_mcq_state = {
        "current_mcqs": mcqs,
        "previous_mcqs": mcqs,
        "attempts": [],
        "pending_explanations": []
    }
    intro = "Answer the following questions based on the paper:\n\n" + \
            "\n".join(f"{i+1}. {m['question']}" for i, m in enumerate(mcqs))
    chat.append(("", intro))
    radio_choices = [[f"{opt}. {txt}" for opt, txt in sorted(m["options"].items())] for m in mcqs]
    # Save session data
    if user_id and name and email:
        save_user_session(user_id, name, email, paper_details, new_mcq_state)

    yield make_out(chat, paper_details, new_mcq_state,
                   gr.update(label=f"Question 1: {mcqs[0]['question']}", choices=radio_choices[0], visible=True),
                   gr.update(label=f"Question 2: {mcqs[1]['question']}", choices=radio_choices[1], visible=True),
                   gr.update(label=f"Question 3: {mcqs[2]['question']}", choices=radio_choices[2], visible=True),
                   gr.update(interactive=True, visible=True))


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
def update_countdown(seconds_remaining):
    """
    Given how many seconds are left, format “M:SS” and tick down by one.
    """
    m, s = divmod(seconds_remaining, 60)
    return f"{m}:{s:02d}", seconds_remaining - 1

# show_quiz already defined:
# def show_quiz():
#     return gr.update(visible=False), gr.update(visible=True)

# ── right before def create_interface():
def show_quiz():
    # hide the learning/practice group, show the quiz group
    return gr.update(visible=False), gr.update(visible=True)

def create_interface():
    # Create the Gradio interface with the exact same UI as the original
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
        
        # Main interface (hidden initially) - EXACT MATCH to original UI
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
                        show_copy_button=True,
                        sanitize_html=False # Allow Mermaid markdown rendering
                    )
                    
                    
                    # MCQ interface
                    with gr.Group() as learning_group:
                        mcq1 = gr.Radio(label="Question 1", choices=[], visible=True, elem_classes=["options-radio"])
                        mcq2 = gr.Radio(label="Question 2", choices=[], visible=True, elem_classes=["options-radio"])
                        mcq3 = gr.Radio(label="Question 3", choices=[], visible=True, elem_classes=["options-radio"])
                        
                        with gr.Row():
                            submit_answers_button = gr.Button("Submit Answers", interactive=False)
           
                        # ── Proceed early button
                        proceed_button = gr.Button("Proceed to Quiz")

                        # ── Countdown display and internal state
                        timer_display = gr.Textbox(
                            label="Time until full quiz", 
                            value="10:00", 
                            interactive=False
                        )
                        time_state = gr.State(600)  # 600 seconds = 10 minutes

                        # ── Two timers:
                        #   * countdown_timer fires every second to update timer_display
                        #   * quiz_timer fires once at 600s to flip into the quiz
                        countdown_timer = gr.Timer(value=1.0, active=True, render=True)
                        quiz_timer      = gr.Timer(value=600.0, active=True, render=True)
                     # ── full 5‑question quiz screen (hidden by default)
                    with gr.Group(visible=False) as quiz_group:
                        gr.Markdown("## Take the Full Quiz")
                        q1 = gr.Radio(
                            choices=["A. …","B. …","C. …","D. …"], 
                            label="1. Placeholder Question", 
                            visible=True,
                            elem_classes=["options-radio"]
                        )
                        q2 = gr.Radio(choices=["A. …","B. …","C. …","D. …"], label="2. …", elem_classes=["options-radio"])
                        q3 = gr.Radio(choices=["A. …","B. …","C. …","D. …"], label="3. …", elem_classes=["options-radio"])
                        q4 = gr.Radio(choices=["A. …","B. …","C. …","D. …"], label="4. …", elem_classes=["options-radio"])
                        q5 = gr.Radio(choices=["A. …","B. …","C. …","D. …"], label="5. …", elem_classes=["options-radio"])
                        submit_quiz = gr.Button("Submit Quiz")
            
        
        # Event handlers - EXACT MATCH to original event handlers
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
        
        # file_upload.change(
        #     fn=summarize_paper,
        #     inputs=[file_upload, model_dropdown, paper_details_state],
        #     outputs=[chatbot, paper_details_state, diagram_display]   # ← add third slot
        # )


        submit_answers_button.click(
            fn=submit_mcq_answers,
            inputs=[chatbot, paper_details_state, model_dropdown, mcq_state, mcq1, mcq2, mcq3, user_id_state, user_name_state, user_email_state],
            outputs=[chatbot, paper_details_state, mcq_state, mcq1, mcq2, mcq3, submit_answers_button, user_id_state, user_name_state, user_email_state, proceed_button]
        )

        proceed_button.click(
            fn=show_quiz,
            inputs=[],
            outputs=[learning_group, quiz_group]
        )

        # every second, decrement the counter
        countdown_timer.tick(
            fn=update_countdown,
            inputs=[time_state],
            outputs=[timer_display, time_state]
        )

        # once after 10 minutes, auto‑swap screens
        quiz_timer.tick(
            fn=show_quiz,
            inputs=[],
            outputs=[learning_group, quiz_group]
        )

        
        # Warm up the model when the app starts
        demo.load(fn=lambda: warm_up_model(model_name="gemma3:1b"))
        
        return demo

# Main function to launch the app
def main():
    demo = create_interface()
    demo.launch(share=True)

if __name__ == "__main__":
    main()
