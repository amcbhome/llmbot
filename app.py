
import PyPDF2
import re
import google.generativeai as genai
from google.colab import userdata
from IPython.display import display, HTML # Keep display for potential notebook use or adapt for web

# Configure the Gemini API
try:
    # It's best practice to get the key name from environment variables or a config file
    # For this example, we'll use the name you used in Colab secrets
    GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY') # Assuming you've named your secret 'GOOGLE_API_KEY'
    genai.configure(api_key=GOOGLE_API_KEY)
except userdata.SecretNotFoundError:
    print("API key not found. Please add your GOOGLE_API_KEY to Colab secrets or set it as an environment variable.")
    # In a script, you might want to exit or handle this differently
    raise

# Initialize the Generative Model
try:
    gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest') # Or your preferred model
except Exception as e:
    print(f"Error initializing Gemini model: {e}")
    print("Please check your API key and model name.")
    # In a script, you might want to exit or handle this differently
    raise

def load_and_process_pdf(pdf_path):
    """Loads a PDF, extracts text, and processes it into chunks."""
    try:
        pdf_file_object = open(pdf_path, 'rb')
        pdf_reader = PyPDF2.PdfReader(pdf_file_object)

        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        pdf_file_object.close()

        # Clean the extracted text
        text = text.strip()
        text = re.sub(r'\n+', '\n', text)
        text = text.replace('\f', '')

        # Split the text into chunks
        chunk_size = 1000  # Define a suitable chunk size
        text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

        return text_chunks
    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
        return None
    except Exception as e:
        print(f"An error occurred during PDF processing: {e}")
        return None

def answer_question_llm(question, text_chunks):
    """Answers a question based on the provided text chunks using a language model."""
    if not text_chunks:
        return "Error: No text content available from the PDF."

    # Combine the text chunks into a single context
    context = "\n".join(text_chunks)

    try:
        # Use the language model to generate an answer
        response = gemini_model.generate_content(f"Using the following text, answer the question: {question}\n\nText: {context}")
        return response.text
    except Exception as e:
        return f"An error occurred while generating the response: {e}"

if __name__ == "__main__":
    pdf_path = '/content/ISA_UK_200_Revised_June_2016_Updated_September_2025.pdf' # Specify your PDF path
    text_chunks = load_and_process_pdf(pdf_path)

    if text_chunks:
        print("Hello! I am an LLM bot that can answer questions about the ISA UK 200 document.")
        print("You can ask me questions related to the content of the document.")
        print("Type 'quit' to exit.")

        while True:
            user_input = input("Your question: ")
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break

            answer = answer_question_llm(user_input, text_chunks)
            print("\nAnswer:")
            # In a standalone script, you might just print the answer
            print(answer)
            print("-" * 50)
