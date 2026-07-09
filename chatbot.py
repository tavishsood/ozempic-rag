import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. Azure Configuration ---
# You will replace these with the actual details provided by your PM
os.environ["AZURE_OPENAI_API_KEY"] = "your_api_key_here"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-base-url.openai.azure.com/"
os.environ["AZURE_OPENAI_API_VERSION"] = "2024-02-15-preview" # Often this, check with your team

CHROMA_DB_DIR = "./chroma_db"

def start_chatbot():
    print("Initializing Chatbot...")

    # --- 2. Load the Database ---
    # We must use the exact same embedding model we used to create the DB
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding_model)
    
    # Create a retriever (fetches the top 3 most relevant chunks)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # --- 3. Connect to Azure GPT-4-mini ---
    llm = AzureChatOpenAI(
        azure_deployment="your-deployment-name", # Ask your PM for the deployment name
        temperature=0.2, # Low temperature keeps the bot factual and less "creative"
    )

    # --- 4. Define the Prompt Instructions ---
    system_prompt = (
        "You are a helpful and precise assistant. Use the following retrieved context "
        "to answer the user's question. If you don't know the answer based on the context, "
        "just say that you don't know. Keep the answer concise.\n\n"
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # --- 5. Build the RAG Chain ---
    # This automatically combines the retrieved chunks and feeds them into the prompt
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("\nSystem ready! Type 'exit' to quit.\n" + "-"*40)

    # --- 6. The Chat Loop ---
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
            
        # Run the chain
        response = rag_chain.invoke({"input": user_input})
        
        print(f"\nBot: {response['answer']}\n")
        print("-" * 40)

if __name__ == "__main__":
    start_chatbot()