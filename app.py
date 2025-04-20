import gradio as gr
from quiz_flow import auto_summarize_with_mcqs, submit_mcq_answers
from user_session import register_user
from ollama_client import warm_up_model
import config
from pathlib import Path

CUSTOM_CSS = Path("static/styles.css").read_text()

# Create the Gradio interface
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
    
    # Main interface (hidden initially)
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
                    show_copy_button=True
                )
                
                # MCQ interface
                with gr.Group():
                    mcq1 = gr.Radio(label="Question 1", choices=[], visible=True, elem_classes=["options-radio"])
                    mcq2 = gr.Radio(label="Question 2", choices=[], visible=True, elem_classes=["options-radio"])
                    mcq3 = gr.Radio(label="Question 3", choices=[], visible=True, elem_classes=["options-radio"])
                    
                    with gr.Row():
                        submit_answers_button = gr.Button("Submit Answers", interactive=False)
    
    # Event handlers
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
    
    submit_answers_button.click(
        fn=submit_mcq_answers,
        inputs=[chatbot, paper_details_state, model_dropdown, mcq_state, mcq1, mcq2, mcq3, user_id_state, user_name_state, user_email_state],
        outputs=[chatbot, paper_details_state, mcq_state, mcq1, mcq2, mcq3, submit_answers_button, user_id_state, user_name_state, user_email_state]
    )
    
    # Warm up the model when the app starts
    demo.load(fn=lambda: warm_up_model(model_name="gemma3:1b"))

# Launch the app
if __name__ == "__main__":
    warm_up_model(config.DEFAULT_MODEL)
    demo.launch(share=True)