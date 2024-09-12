import os
import time
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain import OpenAI, PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
# from langchain_pinecone import PineconeVectorStore

# from pinecone import Pinecone, ServerlessSpec
# from google.colab import userdata
import time
import markdown

class ChatbotLogic:
    def __init__(self):
        load_dotenv()
        self.openai_api_key = userdata.get('OPENAI_API_KEY')
        self.pinecone_api_key = userdata.get('PINECONE_API_KEY')
        self.index_name = "testingpinecone"

        os.environ["OPENAI_API_KEY"] = self.openai_api_key
        os.environ["PINECONE_API_KEY"] = self.pinecone_api_key

        self.docsearch = None

    def initialize(self, pdf_docs):
        print("Initializing document search...")
        raw_text = self.get_pdf_text(pdf_docs)
        text_chunks = self.get_text_chunks(raw_text)
        self.docsearch = self.get_vector_store(text_chunks)
        print("Document search initialized successfully.")

    def get_pdf_text(self, pdf_docs):
        print("Extracting text from PDFs...")
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        print("Text extraction complete.")
        return text

    def get_text_chunks(self, text):
        print("Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
        text_chunks = text_splitter.split_text(text)
        print(f"Text split into {len(text_chunks)} chunks.")
        return text_chunks

    def get_vector_store(self, text_chunks):
        print("Setting up vector store...")
        embeddings = OpenAIEmbeddings()
        pc = Pinecone(api_key=self.pinecone_api_key)

        if self.index_name not in pc.list_indexes():
            print(f"Creating index '{self.index_name}'...")
            try:
                pc.create_index(
                    name=self.index_name,
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
                while not pc.describe_index(self.index_name).status["ready"]:
                    time.sleep(1)
                print("Index creation complete.")
            except:
                print("Index creation failed.")

        index = pc.Index(self.index_name)
        print(index.describe_index_stats())
        docsearch = PineconeVectorStore.from_texts(text_chunks, embeddings, index_name=self.index_name)
        print("Vector store setup complete.")
        print(docsearch)
        return docsearch

    def get_conversational_chain(self):
        print("Initializing conversational chain...")
        prompt_template = """
        Act as a chatbot (virtual assistance) and your name is Tom.
        Understand what the user wants and answer their questions based on the Context in a helpful way.
        Keep the conversation flowing and feel natural, like talking to a friend.
        If you're not sure, just say that you don't know the answer, don't try to make up an answer.\n\n

        Note if the Question is Greeting then Answer like this.
        Greeting: Hello
        Answer: Hello! How can I help you today?

        Greeting: Hi
        Answer: Hi! What would you like to chat about?

        Greeting: Hey there
        Answer: Hey there! What's on your mind?

        Greeting: Good morning/afternoon/evening
        Answer: Good [morning/afternoon/evening]! What can I do for you?

        Greeting: How are you?
        Answer: I'm doing well, thank you for asking! How about you?

        Greeting: What's up? (informal)
        Answer: Not much, just hanging out in the digital world. What can I help you with?

        If the Question is not a Greeting then you can avoid a greeting altogether and jump right into addressing the Question.

        Context:\n {context}?\n
        Question: \n{question}\n
        """

        model = OpenAI(api_key=self.openai_api_key, model_name="gpt-3.5-turbo-instruct")
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        print("Conversational chain initialized.")
        return chain

    def user_input(self, user_question):
        print("Handling user input...")
        if not self.docsearch:
            raise ValueError("Document search index not initialized. Call initialize() first.")

        docs = self.docsearch.similarity_search(user_question)
        print(f"Found {len(docs)} documents related to the user's question.")
        print("docs ", docs)

        chain = self.get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        markdown_text = response['output_text']

        # Render the markdown using a dedicated library like `markdown`
        html_content = markdown.markdown(markdown_text)
        print("User input handled successfully.")
        return html_content