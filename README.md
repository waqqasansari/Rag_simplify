# **LLM-PDF-Chatbot**

## **Overview**

This repository contains a Django REST Framework project that enables users to upload PDF files and interact with them through a chatbot. The chatbot utilizes the GPT-3.5 turbo model, OpenAI embeddings, and FAISS vector database to provide intelligent responses. Additionally, Langchain is used to create a chat QA chain that maintains conversation history.

### Installation

1. Clone the repository: `https://github.com/waqqasansari/Rag_simplify.git`
2. Navigate to the project directory: `cd Rag_simplify/`
3. Create a virtual environment: `python3 -m venv .venv`
4. Create a .env file where add `OPENAI_API_KEY`
5. Activate the virtual environment: `.venv\Scripts\activate`
6. Install dependencies: `pip install -r requirements.txt`
7. Navigate to the project directory: `cd task_simplyfy/`
8. Run migrations: `python manage.py makemigrations`
9. Apply database migrations: `python manage.py migrate`
10. Run the development server: `python manage.py runserver`

### Chat with PDF

- **Endpoint**: `http://127.0.0.1:8000/api/chat/`
- **Method**: `POST`
- **Description**: Interact with the uploaded PDF file through the
  chatbot.
- **Body**:

```json
{
  "user_input": "Your question or message here"
}
```

The API will respond with a JSON object containing the chatbot's response:

```json
{
  "ai_response": "The chatbot's response to your input"
}
```
