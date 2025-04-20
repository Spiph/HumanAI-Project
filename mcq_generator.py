from ollama_client import query_ollama_model
from config import MAX_MCQ_GENERATION_ATTEMPTS
import random, re

# Function to preprocess question text
def preprocess_question(question_text):
    """
    Remove "[Question]" prefix if present and return the cleaned question text.
    Returns None if the question becomes empty after preprocessing.
    """
    if not question_text:
        return None
        
    # Check if question starts with "[Question]"
    if question_text.startswith("[Question]"):
        # Remove the prefix
        cleaned_text = question_text[len("[Question]"):].strip()
        # Return None if empty, otherwise return cleaned text
        return cleaned_text if cleaned_text else None
    return question_text



# Function to create a default MCQ when generation fails
def create_default_mcq(index, paper_details):
    """
    Create a default MCQ when generation fails repeatedly.
    This ensures users always have questions to answer.
    """
    # Extract a key concept from paper details to use in the question
    concept = "the research"
    for category, items in paper_details.items():
        if items and items[0]["content"]:
            concept = items[0]["section"]
            break
    
    # Create different default questions based on the index
    if index == 0:
        return {
            "question": f"What is the main focus of {concept}?",
            "options": {
                "a": "Understanding theoretical concepts",
                "b": "Developing practical applications",
                "c": "Analyzing experimental results",
                "d": "Comparing with previous work"
            },
            "correct_answer": "b"
        }
    elif index == 1:
        return {
            "question": "Which approach best describes the methodology used?",
            "options": {
                "a": "Quantitative analysis of large datasets",
                "b": "Qualitative assessment through case studies",
                "c": "Mixed methods combining multiple approaches",
                "d": "Theoretical modeling without empirical validation"
            },
            "correct_answer": "c"
        }
    else:
        return {
            "question": "What is the most significant contribution of this research?",
            "options": {
                "a": "New theoretical framework",
                "b": "Improved methodology",
                "c": "Novel application of existing techniques",
                "d": "Comprehensive literature review"
            },
            "correct_answer": "c"
        }

# Enhanced function to generate a single MCQ with better error handling and formatting flexibility
def generate_single_mcq(paper_details, model_name, existing_mcqs=None, attempt=1):
    """
    Generate a single MCQ with enhanced error handling and formatting flexibility.
    Includes multiple attempts and fallback mechanisms.
    """
    if not paper_details:
        return None
    
    # Ensure existing_mcqs is a list, not None
    if existing_mcqs is None:
        existing_mcqs = []
    
    # Get all available sections from the extracted information
    sections_text = ""
    for category, items in paper_details.items():
        sections_text += f"\n{category}:\n"
        for item in items:
            sections_text += f"- From {item['section']}: {item['content']}\n"
    
    # Add diversity instructions
    diversity_instructions = (
        "IMPORTANT: Generate a UNIQUE question that is DIFFERENT from any previous questions. "
        "Focus on a DISTINCT concept or finding that hasn't been covered before."
    )
    
    # Add topic categories
    topic_categories = (
        "Choose ONE of these different categories for your question:\n"
        "- Theoretical concepts and frameworks\n"
        "- Methodological approaches and techniques\n"
        "- Empirical findings and results\n"
        "- Implications and applications\n"
    )
    
    # Enhanced exclusion mechanism
    exclusion_text = ""
    if existing_mcqs and len(existing_mcqs) > 0:
        exclusion_text = "IMPORTANT - Do NOT generate a question similar to these previous ones:\n"
        # Add the most recent questions first (more likely to be remembered by the LLM)
        recent_mcqs = existing_mcqs[-9:] if len(existing_mcqs) > 9 else existing_mcqs
        for mcq in recent_mcqs:
            if isinstance(mcq, dict) and "question" in mcq:
                exclusion_text += f"- {mcq['question']}\n"
    
    # Add random seed
    random_seed = random.randint(1, 1000)
    
    # Add formatting instructions based on attempt number
    # For later attempts, provide more explicit formatting instructions
    formatting_instructions = ""
    if attempt > 1:
        formatting_instructions = (
            "CRITICAL FORMATTING INSTRUCTIONS:\n"
            "1. Start with the question text directly (no numbering or 'Question:' prefix)\n"
            "2. Each option MUST start with the letter (a., b., c., or d.) followed by a space\n"
            "3. The correct answer MUST be specified as 'Correct Answer: [letter]'\n"
            "4. Follow this EXACT format - any deviation will cause rejection\n\n"
        )
    
    # Construct the enhanced prompt with clearer instructions
    mcq_prompt = (
        f"{formatting_instructions}"
        f"Based on the following extracted details from a research paper, generate exactly one multiple-choice question (MCQ) that tests theoretical understanding of the paper's key concepts and findings:\n"
        f"{sections_text}\n\n"
        f"{diversity_instructions}\n\n"
        f"{topic_categories}\n\n"
        f"{exclusion_text}\n"
        "The question should have four options labeled (a), (b), (c), and (d). Include the correct answer.\n"
        "Focus on testing understanding of the research problem, methodology, results, and implications rather than superficial details.\n"
        f"Use seed {random_seed} to ensure a unique question different from previous generations.\n"
        "Format the output as follows:\n"
        "[Your question text here without any prefix]\n"
        "a. [Option A]\n"
        "b. [Option B]\n"
        "c. [Option C]\n"
        "d. [Option D]\n"
        "Correct Answer: [Correct Option Letter]\n"
    )
    
    try:
        full_response = ""
        for chunk in query_ollama_model(mcq_prompt, model_name, stream=False):
            full_response = chunk["content"]
        
        # Debug output to help diagnose formatting issues
        if attempt > 1:
            print(f"Attempt {attempt} response:\n{full_response[:200]}...")
        
        # Enhanced parsing with more flexible pattern matching
        current_question = {}
        question_text = ""
        options = {}
        correct_answer = ""
        
        # Process the response line by line with more flexible parsing
        lines = full_response.split("\n")
        in_options = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for option lines with flexible pattern matching
            option_match = re.match(r'^([a-d])[.)]?\s+(.+)$', line)
            if option_match:
                in_options = True
                option_letter = option_match.group(1).lower()
                option_text = option_match.group(2).strip()
                options[option_letter] = option_text
                continue
                
            # Check for correct answer with flexible pattern matching
            if re.search(r'correct\s+answer\s*[:]?\s*([a-d])', line, re.IGNORECASE):
                correct_letter = re.search(r'correct\s+answer\s*[:]?\s*([a-d])', line, re.IGNORECASE).group(1).lower()
                correct_answer = correct_letter
                continue
                
            # If we haven't started processing options yet, this is part of the question text
            if not in_options and not line.lower().startswith("correct answer"):
                # Skip lines that look like formatting instructions or numbering
                if not re.match(r'^(\d+[.)]|question:|instructions:)', line.lower()):
                    question_text += line + " "
        
        # Clean up the question text
        question_text = question_text.strip()
        processed_question = preprocess_question(question_text)
        
        # Construct the MCQ if we have all required components
        if processed_question and options and len(options) >= 3 and correct_answer:
            # If we only have 3 options, add a fourth one
            if len(options) == 3:
                missing_letters = set(['a', 'b', 'c', 'd']) - set(options.keys())
                if missing_letters:
                    missing_letter = list(missing_letters)[0]
                    options[missing_letter] = f"None of the above"
            
            # Ensure we have exactly options a, b, c, d
            if set(options.keys()) == set(['a', 'b', 'c', 'd']):
                return {
                    "question": processed_question,
                    "options": options,
                    "correct_answer": correct_answer
                }
        
        # If we've reached the maximum number of attempts, create a default MCQ
        if attempt >= MAX_MCQ_GENERATION_ATTEMPTS:
            print(f"Failed to generate valid MCQ after {MAX_MCQ_GENERATION_ATTEMPTS} attempts, using default")
            # Determine which default MCQ to use based on existing MCQs
            default_index = len(existing_mcqs) % 3
            return create_default_mcq(default_index, paper_details)
            
        # Try again with a different approach if we haven't reached max attempts
        if attempt < MAX_MCQ_GENERATION_ATTEMPTS:
            print(f"Attempt {attempt} failed, trying again with more explicit instructions")
            return generate_single_mcq(paper_details, model_name, existing_mcqs, attempt + 1)
            
        print("Generated MCQ is invalid")
        return None
        
    except Exception as e:
        print(f"Error generating single MCQ: {str(e)}")
        
        # If we've reached the maximum number of attempts, create a default MCQ
        if attempt >= MAX_MCQ_GENERATION_ATTEMPTS:
            print(f"Exception in MCQ generation after {MAX_MCQ_GENERATION_ATTEMPTS} attempts, using default")
            # Determine which default MCQ to use based on existing MCQs
            default_index = len(existing_mcqs) % 3
            return create_default_mcq(default_index, paper_details)
            
        # Try again with a different approach if we haven't reached max attempts
        if attempt < MAX_MCQ_GENERATION_ATTEMPTS:
            print(f"Exception in attempt {attempt}, trying again")
            return generate_single_mcq(paper_details, model_name, existing_mcqs, attempt + 1)
            
        return None

# Enhanced function to generate practice MCQs with better error handling and fallback mechanisms
def generate_mcqs(paper_details, model_name, previous_mcqs=None):
    """
    Generate practice MCQs with enhanced error handling and fallback mechanisms.
    Ensures that valid MCQs are always returned.
    """
    if not paper_details:  # Ensure paper_details has content
        return []
    
    # Ensure previous_mcqs is a list, not None
    if previous_mcqs is None:
        previous_mcqs = []
    
    # Get all available sections from the extracted information
    sections_text = ""
    for category, items in paper_details.items():
        sections_text += f"\n{category}:\n"
        for item in items:
            sections_text += f"- From {item['section']}: {item['content']}\n"
    
    # Add diversity instructions
    diversity_instructions = (
        "IMPORTANT: Generate DIVERSE questions that cover DIFFERENT aspects of the paper. "
        "Each question must focus on a DISTINCT concept or finding. "
        "Specifically:\n"
        "- Question 1: Focus on the research problem or motivation\n"
        "- Question 2: Focus on the methodology or approach\n"
        "- Question 3: Focus on the results, findings, or implications\n"
    )
    
    # Add topic categories
    topic_categories = (
        "Ensure questions cover these different categories:\n"
        "- Theoretical concepts and frameworks\n"
        "- Methodological approaches and techniques\n"
        "- Empirical findings and results\n"
        "- Implications and applications\n"
    )
    
    # Enhanced exclusion mechanism
    exclusion_text = ""
    if previous_mcqs and len(previous_mcqs) > 0:
        exclusion_text = "IMPORTANT - Do NOT generate questions similar to these previous ones:\n"
        # Add the most recent questions first (more likely to be remembered by the LLM)
        recent_mcqs = previous_mcqs[-9:] if len(previous_mcqs) > 9 else previous_mcqs
        for mcq in recent_mcqs:
            if isinstance(mcq, dict) and "question" in mcq:
                exclusion_text += f"- {mcq['question']}\n"
    
    # Add random seed
    random_seed = random.randint(1, 1000)
    
    # Add clearer formatting instructions
    formatting_instructions = (
        "CRITICAL FORMATTING INSTRUCTIONS:\n"
        "1. Number each question as '1.', '2.', and '3.'\n"
        "2. Each option MUST start with the letter (a., b., c., or d.) followed by a space\n"
        "3. The correct answer MUST be specified as 'Correct Answer: [letter]'\n"
        "4. Follow this EXACT format - any deviation will cause rejection\n\n"
    )
    
    # Construct the enhanced prompt
    mcq_prompt = (
        f"{formatting_instructions}"
        f"Based on the following extracted details from a research paper, generate exactly three multiple-choice questions (MCQs) that test theoretical understanding of the paper's key concepts and findings:\n"
        f"{sections_text}\n\n"
        f"{diversity_instructions}\n\n"
        f"{topic_categories}\n\n"
        f"{exclusion_text}\n"
        "Each question should have four options labeled (a), (b), (c), and (d). Include the correct answer for each question.\n"
        "Focus on testing understanding of the research problem, methodology, results, and implications rather than superficial details.\n"
        f"Use seed {random_seed} to ensure unique questions different from previous generations.\n"
        "Format the output as follows:\n"
        "1. [Question text without any prefix]\n"
        "   a. [Option A]\n"
        "   b. [Option B]\n"
        "   c. [Option C]\n"
        "   d. [Option D]\n"
        "   Correct Answer: [Correct Option Letter]\n"
        "2. [Question text]...\n"
    )
    
    try:
        # First attempt: Try to generate all MCQs at once
        full_response = ""
        for chunk in query_ollama_model(mcq_prompt, model_name, stream=False):
            full_response = chunk["content"]  # Get the complete response
        
        # Parse the response into individual questions
        mcqs = []
        lines = full_response.split("\n")
        current_question = {}
        current_options = {}
        question_text = ""
        in_question = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for new question start
            if re.match(r'^\d+\.', line):
                # Save previous question if it exists
                if question_text and current_options:
                    processed_question = preprocess_question(question_text)
                    if processed_question and len(current_options) >= 3 and "correct_answer" in current_question:
                        # If we only have 3 options, add a fourth one
                        if len(current_options) == 3:
                            missing_letters = set(['a', 'b', 'c', 'd']) - set(current_options.keys())
                            if missing_letters:
                                missing_letter = list(missing_letters)[0]
                                current_options[missing_letter] = f"None of the above"
                        
                        # Ensure we have exactly options a, b, c, d
                        if set(current_options.keys()) == set(['a', 'b', 'c', 'd']):
                            mcqs.append({
                                "question": processed_question,
                                "options": current_options,
                                "correct_answer": current_question.get("correct_answer", "a")
                            })
                
                # Start new question
                question_text = line.split(".", 1)[1].strip()
                current_question = {}
                current_options = {}
                in_question = True
                continue
                
            # Check for option lines
            option_match = re.match(r'^([a-d])[.)]?\s+(.+)$', line)
            if option_match and in_question:
                option_letter = option_match.group(1).lower()
                option_text = option_match.group(2).strip()
                current_options[option_letter] = option_text
                continue
                
            # Check for correct answer
            if re.search(r'correct\s+answer\s*[:]?\s*([a-d])', line, re.IGNORECASE) and in_question:
                correct_letter = re.search(r'correct\s+answer\s*[:]?\s*([a-d])', line, re.IGNORECASE).group(1).lower()
                current_question["correct_answer"] = correct_letter
                
                # End of this question
                processed_question = preprocess_question(question_text)
                if processed_question and len(current_options) >= 3:
                    # If we only have 3 options, add a fourth one
                    if len(current_options) == 3:
                        missing_letters = set(['a', 'b', 'c', 'd']) - set(current_options.keys())
                        if missing_letters:
                            missing_letter = list(missing_letters)[0]
                            current_options[missing_letter] = f"None of the above"
                    
                    # Ensure we have exactly options a, b, c, d
                    if set(current_options.keys()) == set(['a', 'b', 'c', 'd']):
                        mcqs.append({
                            "question": processed_question,
                            "options": current_options,
                            "correct_answer": current_question.get("correct_answer", "a")
                        })
                
                # Reset for next question
                question_text = ""
                current_question = {}
                current_options = {}
                in_question = False
                continue
                
            # If none of the above, and we're in a question, append to question text
            if in_question and not option_match and not re.search(r'correct\s+answer', line, re.IGNORECASE):
                # Only append if it doesn't look like a formatting instruction
                if not re.match(r'^(question:|instructions:)', line.lower()):
                    question_text += " " + line
        
        # Add the last question if it exists and wasn't added yet
        if question_text and current_options and "correct_answer" in current_question:
            processed_question = preprocess_question(question_text)
            if processed_question and len(current_options) >= 3:
                # If we only have 3 options, add a fourth one
                if len(current_options) == 3:
                    missing_letters = set(['a', 'b', 'c', 'd']) - set(current_options.keys())
                    if missing_letters:
                        missing_letter = list(missing_letters)[0]
                        current_options[missing_letter] = f"None of the above"
                
                # Ensure we have exactly options a, b, c, d
                if set(current_options.keys()) == set(['a', 'b', 'c', 'd']):
                    mcqs.append({
                        "question": processed_question,
                        "options": current_options,
                        "correct_answer": current_question.get("correct_answer", "a")
                    })
        
        # Validate that we have valid MCQs
        valid_mcqs = []
        for mcq in mcqs:
            if "question" in mcq and "options" in mcq and "correct_answer" in mcq and len(mcq["options"]) == 4:
                valid_mcqs.append(mcq)
        
        # If we don't have 3 valid MCQs, generate them individually
        if len(valid_mcqs) < 3:
            print(f"Only have {len(valid_mcqs)} valid MCQs from batch generation, generating more individually...")
            
            # Generate individual MCQs to fill the gaps
            while len(valid_mcqs) < 3:
                # Generate a new MCQ with the enhanced single MCQ generator
                additional_mcq = generate_single_mcq(
                    paper_details, 
                    model_name, 
                    previous_mcqs + valid_mcqs
                )
                
                if additional_mcq:
                    valid_mcqs.append(additional_mcq)
                else:
                    # If we can't generate a valid MCQ, use a default one
                    default_index = len(valid_mcqs)
                    default_mcq = create_default_mcq(default_index, paper_details)
                    valid_mcqs.append(default_mcq)
        
        return valid_mcqs[:3]  # Return exactly 3 MCQs
        
    except Exception as e:
        print(f"Error in batch MCQ generation: {str(e)}")
        
        # Fallback: Generate MCQs individually
        print("Falling back to individual MCQ generation...")
        valid_mcqs = []
        
        # Try to generate 3 MCQs individually
        for i in range(3):
            try:
                # Generate a new MCQ with the enhanced single MCQ generator
                additional_mcq = generate_single_mcq(
                    paper_details, 
                    model_name, 
                    previous_mcqs + valid_mcqs
                )
                
                if additional_mcq:
                    valid_mcqs.append(additional_mcq)
                else:
                    # If we can't generate a valid MCQ, use a default one
                    default_mcq = create_default_mcq(i, paper_details)
                    valid_mcqs.append(default_mcq)
            except Exception as inner_e:
                print(f"Error generating individual MCQ {i+1}: {str(inner_e)}")
                # Use a default MCQ as fallback
                default_mcq = create_default_mcq(i, paper_details)
                valid_mcqs.append(default_mcq)
        
        return valid_mcqs  # Return whatever MCQs we managed to generate

