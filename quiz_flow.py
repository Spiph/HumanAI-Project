import gradio as gr
import datetime

from summary_generator import summarize_paper
from mcq_generator import generate_mcqs, create_default_mcq
from user_session import save_user_session
from explaination_generator import extract_explanation_information, generate_explanation_diagram

def auto_summarize_with_mcqs(file, model_name, summary_state, mcq_state, user_id, name, email):
    if not file:
        yield (
            [("", "Please upload a valid PDF file.")],  # Chatbot output - proper tuple format
            summary_state,  # Summary state
            mcq_state,  # MCQ state
            gr.update(label="Question 1", choices=[]),  # Radio buttons for MCQ 1
            gr.update(label="Question 2", choices=[]),  # Radio buttons for MCQ 2
            gr.update(label="Question 3", choices=[]),  # Radio buttons for MCQ 3
            gr.update(interactive=False),  # Submit answers button
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
            summary,  # Chatbot output - already in proper tuple format from summarize_paper
            paper_details,  # Paper details state (replaces summary_state)
            mcq_state,  # MCQ state
            gr.update(label="Question 1", choices=[]),  # Radio buttons for MCQ 1
            gr.update(label="Question 2", choices=[]),  # Radio buttons for MCQ 2
            gr.update(label="Question 3", choices=[]),  # Radio buttons for MCQ 3
            gr.update(interactive=False),  # Submit answers button
            user_id,  # User ID
            name,  # User name
            email  # User email
        )
    
    # Automatically generate MCQs after summary is complete, using paper details instead of summary
    # Generate new MCQs
    mcqs = generate_mcqs(paper_details, model_name, [])  # Pass empty list instead of None
    
    # We should always get MCQs now due to fallback mechanisms
    if not mcqs:
        print("Warning: generate_mcqs returned empty list despite fallbacks")
        # Create default MCQs as a last resort
        mcqs = [
            create_default_mcq(0, paper_details),
            create_default_mcq(1, paper_details),
            create_default_mcq(2, paper_details)
        ]
    
    # Ensure we have exactly 3 MCQs
    while len(mcqs) < 3:
        default_index = len(mcqs)
        mcqs.append(create_default_mcq(default_index, paper_details))
    
    # Initialize previous_mcqs as empty list if mcq_state is None
    previous_mcqs = []
    if mcq_state is not None:
        # Get previous_mcqs from mcq_state, defaulting to empty list if not present or None
        previous_mcqs = mcq_state.get("previous_mcqs", [])
        if previous_mcqs is None:
            previous_mcqs = []
    
    # Update MCQ state
    new_mcq_state = {
        "current_mcqs": mcqs,
        "previous_mcqs": previous_mcqs + mcqs,  # Ensure both are lists before concatenating
        "attempts": [],
        "pending_explanations": []  # Initialize empty pending explanations list
    }
    
    # Display MCQs with radio buttons
    mcq_message = "Answer the following questions based on the paper:\n\n"
    for i, mcq in enumerate(mcqs):
        mcq_message += f"{i + 1}. {mcq['question']}\n"
    
    # Add message with proper tuple format
    summary.append(("", mcq_message))
    
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
    
    # Debug print to verify MCQs are being generated
    print(f"Generated MCQs: {len(mcqs)}")
    for i, mcq in enumerate(mcqs):
        print(f"MCQ {i+1}: {mcq['question']}")
        print(f"Options: {mcq['options']}")
        print(f"Correct Answer: {mcq['correct_answer']}")
    
    # Debug print for radio button updates
    print(f"Radio choices for MCQ 1: {radio_choices[0]}")
    print(f"Label for MCQ 1: Question 1: {mcqs[0]['question']}")
    
    yield (
        summary,  # Chatbot output - already in proper tuple format
        paper_details,  # Paper details state
        new_mcq_state,  # MCQ state
        gr.update(label=f"Question 1: {mcqs[0]['question']}", choices=radio_choices[0], visible=True),  # Radio buttons for MCQ 1
        gr.update(label=f"Question 2: {mcqs[1]['question']}", choices=radio_choices[1], visible=True),  # Radio buttons for MCQ 2
        gr.update(label=f"Question 3: {mcqs[2]['question']}", choices=radio_choices[2], visible=True),  # Radio buttons for MCQ 3
        gr.update(interactive=True, visible=True),  # Submit answers button
        user_id,  # User ID
        name,  # User name
        email  # User email
    )

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

# Modified: Generate explanation with sequential processing and display correct option
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
        
        # Step 2: Generate explanation diagram based on extracted information
        for chunk in generate_explanation_diagram(extracted_explanation_info, model_name, mcq_index, stream=True):
            if "content" in chunk:
                # Update the explanation message with the streaming content
                # Use proper tuple format for chatbot messages
                chatbot[explanation_index] = ("", f"Explanation Diagram for Question {question_num}:\n\n" + chunk["content"])
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