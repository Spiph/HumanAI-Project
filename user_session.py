import os, json, datetime, uuid
from config import USER_DATA_DIR
import gradio as gr

def save_user_session(user_id, name, email, chat_history, mcq_data):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    user_dir = os.path.join(USER_DATA_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    
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

def register_user(name, email):
    # validation failed
    if not name or not email:
        return (
            gr.update(visible=True),    # keep home_page visible
            gr.update(visible=False),   # keep main_interface hidden
            "",                         # user_id_state
            "",                         # user_name_state
            "",                         # user_email_state
            "Please enter both name and email to continue."
        )
    # validation succeeded
    user_id = str(uuid.uuid4())
    return (
        gr.update(visible=False),   # hide home_page
        gr.update(visible=True),    # show main_interface
        user_id,                    # user_id_state
        name,                       # user_name_state
        email,                      # user_email_state
        ""                          # registration_error
    )
