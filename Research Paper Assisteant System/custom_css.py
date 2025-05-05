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
.inline-radio .gr-radio-row {
  display: inline-block !important;
  margin-right: 12px !important;
}
"""