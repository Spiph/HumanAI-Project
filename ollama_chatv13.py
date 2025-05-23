import gradio as gr
import requests
import json
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
from io import BytesIO
import threading

# Define the Ollama API endpoint
OLLAMA_API_URL = "http://localhost:11434/api/chat"

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

# New function: retrieve_context
def retrieve_context(query, summary_state):
    """
    A simple retrieval augmented guidance (RAG) function.
    In a production system, this function could query a vector database
    or another knowledge source to fetch documents related to the query.
    Here, it scans the summary for sentences containing keywords from the query.
    """
    keywords = query.lower().split()
    context_sentences = []
    for sentence in summary_state.split('.'):
        for keyword in keywords:
            if keyword in sentence.lower():
                context_sentences.append(sentence.strip())
                break
    retrieved = " ".join(context_sentences)
    if not retrieved:
        retrieved = "No additional context available."
    return retrieved

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

# Function to generate follow-up question suggestions
def generate_follow_up_suggestions(summary_state, model_name):
    if not summary_state or len(summary_state.split()) < 50:  # Ensure summary has sufficient content
        return ["No valid suggestions available."]
    suggestion_prompt = (
        f"Based on the following summary, suggest exactly three potential actions or questions a user might want to take next:\n"
        f"{summary_state}\n"
        "Provide the suggestions as a numbered list in the following format:\n"
        "1. [Suggestion 1]\n"
        "2. [Suggestion 2]\n"
        "3. [Suggestion 3]\n"
        "Do not include any additional explanations, introductions, or text before or after the numbered list."
    )
    try:
        full_response = ""
        for chunk in query_ollama_model(suggestion_prompt, model_name, stream=False):
            full_response += chunk["content"]  # Accumulate the full response
        print(f"Raw model response: {full_response}")  # Debugging: Log the raw response
        # Split the response into individual questions
        questions = [q.strip() for q in full_response.split("\n") if q.strip().startswith(("1.", "2.", "3."))]
        # Validate the number of suggestions
        if len(questions) < 3:
            return ["No valid suggestions available."]  # Fallback if fewer than 3 suggestions are found
        return questions
    except Exception as e:
        print(f"Error generating suggestions: {str(e)}")  # Debugging: Log the error
        return ["Error: Unable to generate suggestions."]

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
        predefined_prompt = (
            "You are an educational summarization assistant designed to help students grasp key concepts and insights "
            "from academic materials, including research papers, textbook chapters, and presentation slides.\n"
            "When provided with academic text, follow these guidelines to produce the best summaries for student learning:\n"
            "- Identify Core Concepts: Clearly state the main ideas, theories, or arguments presented in the text.\n"
            "- Summarize Key Details: Include essential details such as methodologies, important findings, or critical examples that support understanding.\n"
            "- Simplify Language: Use clear, straightforward language suitable for undergraduate-level comprehension.\n"
            "- Highlight Connections: Mention how the information relates to broader themes, real-world applications, or previously known knowledge.\n"
            "- Structured Format: Present summaries in short, structured paragraphs or bullet points for clarity.\n"
            "- Length Constraint: Keep the summary concise but informative, ideally between 150-250 words unless otherwise instructed.\n"
            "Your goal is to create summaries that empower students by efficiently communicating complex information in an accessible, engaging, and easily reviewable format.\n"
            f"Summarize the following research paper and mention the paper title at the first:\n{extracted_text}"
        )
        chat_history = []
        for chunk in query_ollama_model(predefined_prompt, model_name, stream=True):
            if chat_history and chat_history[-1]["role"] == "assistant":
                chat_history[-1] = chunk  # Replace the last assistant message
            else:
                chat_history.append(chunk)  # Add a new assistant message
            yield chat_history, chunk["content"]
    except Exception as e:
        yield [{"role": "assistant", "content": f"An error occurred: {str(e)}"}], summary_state

# Function to handle PDF upload, summarization, and prompt suggestions
def auto_summarize(file, model_name, summary_state):
    if not file:
        yield (
            [(None, "Please upload a valid PDF file.")],  # Chatbot output
            gr.update(choices=[], interactive=False),  # Follow-up suggestions dropdown
            gr.update(value="", interactive=False),  # Custom question input box
            gr.update(interactive=False),  # Ask button (disabled)
            gr.update(interactive=False),  # MCQ button (disabled)
            summary_state  # Summary state
        )
        return
    # Generate the summary
    summary_generator = summarize_paper(file, model_name, summary_state)
    summary, new_summary_state = None, None
    suggestions = []
    summary_complete = False  # Flag to track if the summary is complete
    def generate_suggestions_after_summary():
        nonlocal suggestions, summary_complete
        if summary_complete:
            try:
                new_suggestions = generate_follow_up_suggestions(new_summary_state, model_name)
                suggestions.clear()
                suggestions.extend(new_suggestions)
            except Exception as e:
                print(f"Error in suggestion generation thread: {str(e)}")  # Debugging: Log the error
                suggestions.append("Error: Unable to generate suggestions.")
    for chunk, new_summary_state in summary_generator:
        summary = chunk
        yield (
            summary,  # Chatbot output
            gr.update(choices=[], interactive=False),  # Follow-up suggestions dropdown (disabled during summarization)
            gr.update(value="", interactive=False),  # Custom question input box (disabled during summarization)
            gr.update(interactive=False),  # Ask button (disabled during summarization)
            gr.update(interactive=False),  # MCQ button (disabled during summarization)
            new_summary_state  # Summary state
        )
        # Check if the summary is complete (at least 150 words)
        if len(new_summary_state.split()) > 150 and not summary_complete:
            summary_complete = True
            thread = threading.Thread(target=generate_suggestions_after_summary)
            thread.start()
            thread.join(timeout=1)
    # Finalize suggestions after the summary is complete
    if not suggestions:
        generate_suggestions_after_summary()
    yield (
        summary,  # Chatbot output
        gr.update(choices=suggestions, interactive=True),  # Follow-up suggestions dropdown (enabled after summarization)
        gr.update(value="", interactive=True),  # Custom question input box (enabled after summarization)
        gr.update(interactive=True),  # Ask button (enabled after summarization)
        gr.update(interactive=True),  # MCQ button (enabled after summarization)
        new_summary_state  # Summary state
    )

# Function to handle selection of a follow-up question suggestion
def handle_suggestion_selection(selected_question, model_name, chat_history, summary_state):
    if not selected_question or not summary_state.strip():
        yield (
            chat_history,  # Chatbot output
            gr.update(value=""),  # Clear the input field
            gr.update(choices=[], interactive=False),  # Disable follow-up suggestions dropdown
            gr.update(value="", interactive=False),  # Disable custom question input box
            gr.update(interactive=False),  # Ask button (disabled)
            summary_state  # Summary state
        )
        return
    # Add the selected question to the chat history
    chat_history.append({"role": "user", "content": selected_question})
    # Retrieve additional context using RAG and augment the context
    retrieved_context = retrieve_context(selected_question, summary_state)
    augmented_context = summary_state + "\nAdditional Context:\n" + retrieved_context
    # Query the model with the selected question and the augmented context
    full_response = ""
    for chunk in query_ollama_model(selected_question, model_name, context=augmented_context, stream=True):
        full_response = chunk["content"]
        # Update the last message in the chat history with the cumulative response
        if chat_history and chat_history[-1]["role"] == "assistant":
            chat_history[-1] = {"role": "assistant", "content": full_response}
        else:
            chat_history.append({"role": "assistant", "content": full_response})
        # Yield the updated chat history and keep inputs disabled
        yield (
            chat_history,  # Chatbot output
            gr.update(value=""),  # Clear the input field
            gr.update(choices=[], interactive=False),  # Keep follow-up suggestions dropdown disabled
            gr.update(value="", interactive=False),  # Keep custom question input box disabled
            gr.update(interactive=False),  # Ask button (disabled)
            summary_state  # Summary state
        )
    # Finalize by enabling the inputs and regenerating suggestions
    new_suggestions = generate_follow_up_suggestions(summary_state, model_name)
    yield (
        chat_history,  # Chatbot output
        gr.update(value=""),  # Clear the input field
        gr.update(choices=new_suggestions, interactive=True),  # Re-enable follow-up suggestions dropdown
        gr.update(value="", interactive=True),  # Re-enable custom question input box
        gr.update(interactive=True),  # Ask button (enabled)
        summary_state  # Summary state
    )

# Function to handle custom follow-up questions with streaming
def chat_wrapper(user_message, model_name, chat_history, summary_state):
    if not user_message.strip() or not summary_state.strip():
        yield (
            chat_history,  # Chatbot output
            gr.update(value=""),  # Clear the input field
            gr.update(choices=[], interactive=True),  # Keep follow-up suggestions dropdown enabled
            gr.update(value="", interactive=True),  # Keep custom question input box enabled
            gr.update(interactive=True),  # Ask button (enabled)
            summary_state  # Summary state
        )
        return
    # Add the user's question to the chat history
    chat_history.append({"role": "user", "content": user_message})
    # Retrieve additional context using RAG and augment the context
    retrieved_context = retrieve_context(user_message, summary_state)
    augmented_context = summary_state + "\nAdditional Context:\n" + retrieved_context
    # Query the model with the follow-up question and the augmented context
    full_response = ""
    for chunk in query_ollama_model(user_message, model_name, context=augmented_context, stream=True):
        full_response = chunk["content"]
        # Update the last message in the chat history with the cumulative response
        if chat_history and chat_history[-1]["role"] == "assistant":
            chat_history[-1] = {"role": "assistant", "content": full_response}
        else:
            chat_history.append({"role": "assistant", "content": full_response})
        # Yield the updated chat history and keep inputs disabled
        yield (
            chat_history,  # Chatbot output
            gr.update(value=""),  # Clear the input field
            gr.update(choices=[], interactive=False),  # Keep follow-up suggestions dropdown disabled
            gr.update(value="", interactive=False),  # Keep custom question input box disabled
            gr.update(interactive=False),  # Ask button (disabled)
            summary_state  # Summary state
        )
    # Finalize by enabling the inputs and regenerating suggestions
    new_suggestions = generate_follow_up_suggestions(summary_state, model_name)
    yield (
        chat_history,  # Chatbot output
        gr.update(value=""),  # Clear the input field
        gr.update(choices=new_suggestions, interactive=True),  # Re-enable follow-up suggestions dropdown
        gr.update(value="", interactive=True),  # Re-enable custom question input box
        gr.update(interactive=True),  # Ask button (enabled)
        summary_state  # Summary state
    )

# Function to generate practice MCQs based on the summary
def generate_mcqs(summary_state, model_name):
    if not summary_state or len(summary_state.split()) < 50:  # Ensure summary has sufficient content
        return []
    mcq_prompt = (
        f"Based on the following summary, generate exactly three multiple-choice questions (MCQs) that test theoretical understanding:\n"
        f"{summary_state}\n"
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
            return ["No valid MCQs could be generated."]
        return mcqs
    except Exception as e:
        print(f"Error generating MCQs: {str(e)}")  # Debugging: Log the error
        return ["Error: Unable to generate MCQs."]

# Function to handle MCQ generation and evaluation
def mcq_generator(chatbot, summary_state, model_name):
    if not summary_state.strip():
        yield (
            chatbot + [{"role": "assistant", "content": "Please upload a PDF file first to generate a summary."}],  # Chatbot output
            summary_state  # Summary state
        )
        return
    # Generate MCQs
    mcqs = generate_mcqs(summary_state, model_name)
    if not mcqs or len(mcqs) < 3:
        yield (
            chatbot + [{"role": "assistant", "content": "Failed to generate MCQs. Please try again."}],  # Chatbot output
            summary_state  # Summary state
        )
        return
    # Display MCQs without answers
    mcq_ui = []
    for i, mcq in enumerate(mcqs):
        question_text = f"{i + 1}. {mcq['question']}\n"
        for option, text in mcq["options"].items():
            question_text += f"   {option}. {text}\n"
        mcq_ui.append(question_text)
    chatbot.append({"role": "assistant", "content": "Answer the following questions:\n" + "\n".join(mcq_ui)})
    yield (
        chatbot,  # Chatbot output
        summary_state  # Summary state
    )

# Warm up the model when the app starts
def warm_up_model(model_name="gemma3:1b"):
    try:
        print("Warming up the model...")
        for _ in query_ollama_model("Hello, how are you?", model_name=model_name, stream=False):
            pass
        print("Model warmed up.")
    except Exception as e:
        print(f"Error warming up the model: {str(e)}")

def get_models() -> list:
    OLLAMA_URL = "http://localhost:11434"
    thelist = requests.get(OLLAMA_URL+"/api/tags")
    jsondata = thelist.json()
    result = list()

    for model in jsondata["models"]:
        result.append(model["model"])

    return result

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Research Paper Assistant")
    gr.Markdown("Upload a research paper, and the app will automatically generate its summary, follow-up questions, and practice MCQs.")
    # Dropdown for model selection
    model_dropdown = gr.Dropdown(
        label="Select Model",
        choices=get_models(),  # Use gemma3:1b as the only model #TODO add more models to select
        value="gemma3:1b"  # Default model
    )
    # State to store the summary
    summary_state = gr.State("")  # Persistent state for the summary
    # Single Tab for All Features
    with gr.Tab("Research Paper Assistant"):
        gr.Markdown("Upload a research paper in PDF format, and the system will automatically generate its summary, follow-up questions, and practice MCQs.")
        pdf_upload = gr.File(label="Upload Research Paper (PDF Only)", file_types=[".pdf"])
        chatbot = gr.Chatbot(label="Summary, Conversation, and MCQ Results", type="messages")  # Set type='messages'
        follow_up_suggestions = gr.Dropdown(label="Suggested Follow-Up Questions", choices=[], interactive=False)
        user_input = gr.Textbox(label="Your Question", placeholder="Type your follow-up question here...", interactive=False)
        submit_button = gr.Button("Ask", interactive=False)
        mcq_button = gr.Button("Generate Practice MCQs", interactive=False)
        # Trigger summarization and follow-up suggestions automatically when a file is uploaded
        pdf_upload.change(
            auto_summarize,
            inputs=[pdf_upload, model_dropdown, summary_state],
            outputs=[chatbot, follow_up_suggestions, user_input, submit_button, mcq_button, summary_state]
        )
        # Handle selection of a follow-up question suggestion
        follow_up_suggestions.change(
            handle_suggestion_selection,
            inputs=[follow_up_suggestions, model_dropdown, chatbot, summary_state],
            outputs=[chatbot, user_input, follow_up_suggestions, user_input, submit_button, summary_state]
        )
        # Handle custom follow-up questions with streaming
        user_input.submit(
            chat_wrapper,
            inputs=[user_input, model_dropdown, chatbot, summary_state],
            outputs=[chatbot, user_input, follow_up_suggestions, user_input, submit_button, summary_state]
        )
        # Handle Ask button click
        submit_button.click(
            chat_wrapper,
            inputs=[user_input, model_dropdown, chatbot, summary_state],
            outputs=[chatbot, user_input, follow_up_suggestions, user_input, submit_button, summary_state]
        )
        # Trigger MCQ generation
        mcq_button.click(
            mcq_generator,
            inputs=[chatbot, summary_state, model_dropdown],
            outputs=[chatbot, summary_state]
        )

# Launch the Gradio app
if __name__ == "__main__":
    warm_up_model()  # Warm up the model at startup
    demo.launch(server_name="localhost", server_port=7860, share=True)
