Required Softwares:
	1. Ollama
	2. NodeJS

Required Libraries:
	1.pdfplumber
	2.numpy
	3.scikit-learn
	4.requests
	5.gradio

Step 1: Install both the above softwares

Step 2: Open terminal and run the command: ollama run gemma3:1b

Step 3: Check id nodejs is installed properly: node -v

Step 4: Check if npm is working: npm -v

Step 5: By default you might get an error for the npm -v command as its set to Restricted: if yes continue to next step else skip to step 9

Step 6: Get-ExecutionPolicy 

Step 7: Set-ExecutionPolicy -Scope CurrentUser

Step 8: RemoteSigned

Step 9: Check again if npm -v command works

Step 10: npm install -g @mermaid-js/mermaid-cli

Step 11: Install all of the required libraries

Step 12: Go the main_app_fixed.py file and locate mmdc path

Step 13: Replace it with the path on which the code will be running.
		 Usually the mmdc path will be C:\\Users\\{user-name}\\AppData\\Roaming\\npm\\mmdc.cmd
		 
Note: AppData folder is by default hidden if you want to manually check the location

Step 14: Run the file ollama_chatv35_modular_enhanced.py

Note: If you get something like Could not create share link. Please check your internet connection or our status page: https://status.gradio.app.
	  Its mostly because the system antivirus is not letting the gradio library to craete a public sharable link.
	  You can manually allow this by letting the antivirus allow it.