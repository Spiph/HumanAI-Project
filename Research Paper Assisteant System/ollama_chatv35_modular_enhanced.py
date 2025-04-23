"""
Main Entry Point for Ollama Chat Application with Enhanced Fallback

This file imports and uses the modular components while maintaining
the exact same functionality and UI as the original ollama_chatv35.py,
with an added fallback mechanism for empty explanation diagrams.
"""

# Import from our modules
from pdf_parser_fix import *
from ollama_api import *
from section_extractor import *
#from diagram_generator_enhanced import *  # Using the enhanced version with fallback
from mcq_generator import *
from main_app_fixed import create_interface, main

# Run the application
if __name__ == "__main__":
    main()
