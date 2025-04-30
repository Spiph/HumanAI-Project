import gradio as gr
import requests
import json
import PyPDF2
from io import BytesIO
import threading

# Define the Ollama API endpoint
OLLAMA_API_URL = "http://localhost:11434/api/chat"

# Function to extract text from a PDF file
def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        return text.strip()
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

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

# Function to handle PDF upload, summarization, and prompt suggestions
def auto_summarize(file, model_name, summary_state):
    if not file:
        yield [(None, "Please upload a valid PDF file.")], gr.update(choices=["Generating suggestions..."], interactive=False), summary_state
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
        yield summary, gr.update(choices=suggestions, interactive=bool(suggestions)), new_summary_state  # Yield summary and suggestions
        
        # Check if the summary is complete (at least 150 words)
        if len(new_summary_state.split()) > 150 and not summary_complete:
            summary_complete = True
            thread = threading.Thread(target=generate_suggestions_after_summary)
            thread.start()
            thread.join(timeout=1)  # Allow a short timeout to avoid blocking the main thread
    
    # Finalize suggestions after the summary is complete
    if not suggestions:
        generate_suggestions_after_summary()
    yield summary, gr.update(choices=suggestions, interactive=True), new_summary_state

# Function to handle PDF upload and summarization with streaming
def summarize_paper(file, model_name, summary_state):
    if not file:
        yield [{"role": "assistant", "content": "Please upload a valid PDF file."}], summary_state
        return
    try:
        # Extract text from the PDF
        with open(file.name, "rb") as pdf_file:
            extracted_text = extract_text_from_pdf(pdf_file)
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
            f"Summarize the following research paper:\n{extracted_text}"
        )
        chat_history = []
        for chunk in query_ollama_model(predefined_prompt, model_name, stream=True):
            # Update the last message in the chat history with the cumulative response
            if chat_history and chat_history[-1]["role"] == "assistant":
                chat_history[-1] = chunk  # Replace the last assistant message
            else:
                chat_history.append(chunk)  # Add a new assistant message
            # Yield the updated chat history
            yield chat_history, chunk["content"]
    except Exception as e:
        yield [{"role": "assistant", "content": f"An error occurred: {str(e)}"}], summary_state

# Function to handle selection of a follow-up question suggestion
def handle_suggestion_selection(selected_question, model_name, chat_history, summary_state):
    if not selected_question:
        yield chat_history, ""  # Return unchanged history and clear input
        return
    # Add the selected question to the chat history
    chat_history.append({"role": "user", "content": selected_question})
    # Query the model with the selected question and stream the response
    full_response = ""
    for chunk in query_ollama_model(selected_question, model_name, context=summary_state, stream=True):
        full_response = chunk["content"]
        # Update the last message in the chat history with the cumulative response
        if chat_history and chat_history[-1]["role"] == "assistant":
            chat_history[-1] = {"role": "assistant", "content": full_response}
        else:
            chat_history.append({"role": "assistant", "content": full_response})
        yield chat_history, ""  # Clear the input field

# Function to handle custom follow-up questions with streaming
def chat_wrapper(user_message, model_name, chat_history, summary_state):
    if not user_message.strip():
        yield chat_history, ""  # Return unchanged history and clear input
        return
    # Add the user's question to the chat history
    chat_history.append({"role": "user", "content": user_message})
    # Query the model with the follow-up question and summary as context
    full_response = ""
    for chunk in query_ollama_model(user_message, model_name, context=summary_state, stream=True):
        full_response = chunk["content"]
        # Update the last message in the chat history with the cumulative response
        if chat_history and chat_history[-1]["role"] == "assistant":
            chat_history[-1] = {"role": "assistant", "content": full_response}
        else:
            chat_history.append({"role": "assistant", "content": full_response})
        yield chat_history, ""  # Clear the input field

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Chat with Ollama-Deployed LLM")
    gr.Markdown("Upload a research paper, and the app will automatically generate its summary.")
    
    # Dropdown for model selection
    model_dropdown = gr.Dropdown(
        label="Select Model",
        choices=["gemma3:1b"],  # Use gemma3:1b as the only model
        value="gemma3:1b"  # Default model
    )
    
    # State to store the summary
    summary_state = gr.State("")  # Persistent state for the summary
    
    # Single Tab for Upload, Summary, and Follow-Up Questions
    with gr.Tab("Research Paper Assistant"):
        gr.Markdown("Upload a research paper in PDF format, and the system will automatically generate its summary. You can also ask follow-up questions.")
        pdf_upload = gr.File(label="Upload Research Paper (PDF Only)", file_types=[".pdf"])
        chatbot = gr.Chatbot(label="Summary and Conversation", type="messages")  # Set type='messages'
        
        # Dropdown for follow-up question suggestions
        follow_up_suggestions = gr.Dropdown(label="Suggested Follow-Up Questions", choices=[], interactive=False)
        user_input = gr.Textbox(label="Your Question", placeholder="Type your follow-up question here...")
        submit_button = gr.Button("Ask")
        
        # Trigger summarization and follow-up suggestions automatically when a file is uploaded
        pdf_upload.change(
            auto_summarize,
            inputs=[pdf_upload, model_dropdown, summary_state],
            outputs=[chatbot, follow_up_suggestions, summary_state]
        )
        
        # Handle selection of a follow-up question suggestion
        follow_up_suggestions.change(
            handle_suggestion_selection,
            inputs=[follow_up_suggestions, model_dropdown, chatbot, summary_state],
            outputs=[chatbot, user_input]
        )
        
        # Handle custom follow-up questions with streaming
        user_input.submit(
            chat_wrapper,
            inputs=[user_input, model_dropdown, chatbot, summary_state],
            outputs=[chatbot, user_input]
        )
        
        # Handle Ask button click
        submit_button.click(
            chat_wrapper,
            inputs=[user_input, model_dropdown, chatbot, summary_state],
            outputs=[chatbot, user_input]
        )

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch(server_name="localhost", server_port=7860)