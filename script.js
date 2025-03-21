document.getElementById('pdf-upload').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
        const progressBar = document.getElementById('progress-bar');
        const progressBarFill = document.getElementById('progress-bar-fill');
        progressBar.classList.remove('hidden');
        progressBarFill.style.width = '0%';

        // Simulate PDF upload progress
        let progress = 0;
        const interval = setInterval(() => {
            progress += 10;
            progressBarFill.style.width = `${progress}%`;

            if (progress >= 100) {
                clearInterval(interval);
                progressBar.classList.add('hidden');
                document.getElementById('summary-section').classList.remove('hidden');
                document.getElementById('interaction-section').classList.remove('hidden');

                // Simulate loading summary
                document.getElementById('summary-content').textContent = 'Loading summary...';
                setTimeout(() => {
                    document.getElementById('summary-content').textContent = 'This is a summary of the research paper.';
                }, 1000);
            }
        }, 200);
    }
});

document.getElementById('send-btn').addEventListener('click', function() {
    const userInput = document.getElementById('user-input').value;
    if (userInput.trim() !== '') {
        const chatHistory = document.getElementById('chat-history');
        const userMessageDiv = document.createElement('div');
        userMessageDiv.className = 'user-message';
        userMessageDiv.textContent = userInput;
        chatHistory.appendChild(userMessageDiv);

        // Simulate LLM response
        setTimeout(() => {
            const botMessageDiv = document.createElement('div');
            botMessageDiv.className = 'bot-message';
            botMessageDiv.innerText = `LLM Response: ${userInput}`;
            chatHistory.appendChild(botMessageDiv);

            // Scroll to the bottom of the chat history
            chatHistory.scrollTop = chatHistory.scrollHeight;
        }, 1000); // Simulate a delay for response

        // Clear the input field
        document.getElementById('user-input').value = '';
    }
});