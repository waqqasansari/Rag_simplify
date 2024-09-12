from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from langchain import OpenAI, PromptTemplate
from langchain_openai import ChatOpenAI
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
import time
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
import time
import markdown

class ChatbotLogic:
    def __init__(self):
        load_dotenv()
        self.openai_api_key = os.environ.get('OPENAI_API_KEY')

        # Initialize OpenAI LLM
        # self.llm = OpenAI(api_key=self.openai_api_key, model_name="gpt-3.5-turbo-instruct")
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        print("Loading and processing PDF...")
        # Load the PDF and extract text
        pdf_file_path = "D:/task-simplyfyai/task_simplyfy/room-finder-company-info.pdf"
        pdf_reader = PdfReader(pdf_file_path)
        self.docs = ""
        for page in pdf_reader.pages:
            self.docs += page.extract_text()

        print("Initializing embeddings and text splitting...")
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.splits = self.text_splitter.split_text(self.docs)

        print("Setting up FAISS index...")
        self.vectorstore = FAISS.from_texts(texts=self.splits, embedding=self.embeddings)
        self.retriever = self.vectorstore.as_retriever()

        print("Vector store setup complete.")
        self.setup_chains()

        # Dictionary to store chat session histories
        self.store = {}

    def setup_chains(self):
        # System prompt to contextualize questions
        self.contextualize_q_system_prompt = """Given a chat history and the latest user question \
                                                which might reference context in the chat history, formulate a standalone question \
                                                which can be understood without the chat history. Do NOT answer the question, \
                                                just reformulate it if needed and otherwise return it as is."""

        self.contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # Combine chat history with retriever for question answering
        self.history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, self.contextualize_q_prompt
        )

        # System prompt for question answering with retrieved context
        self.qa_system_prompt =  """You are an assistant for question-answering tasks. \
                                    Use the following pieces of retrieved context to answer the question. \
                                    If you don't know the answer, just say that you don't know. \
                                    Use three sentences maximum and keep the answer concise.\

                                    {context}"""
        
        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        self.question_answer_chain = create_stuff_documents_chain(self.llm, self.qa_prompt)

        self.rag_chain = create_retrieval_chain(
            self.history_aware_retriever, self.question_answer_chain
        )

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Manage session-based chat history"""
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def process_message(self, session_id: str, user_input: str) -> str:
        """Process user input, retrieve relevant context, and maintain conversation memory."""
        conversational_rag_chain = RunnableWithMessageHistory(
            self.rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        return response