import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
os.environ.setdefault("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

VECTOR_STORE_PATH = "vectorstore"  # folder to save/load from

try:
    # Embeddings
    print("Loading embeddings model...")
    embeddings = FastEmbedEmbeddings()

    # ── Load or Build Vector Store ────────────────────────────
    if os.path.exists(VECTOR_STORE_PATH):
        print("Loading vector store from disk (fast!)...")
        db = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        print("Vector store loaded from disk!")
    else:
        print("Building vector store for first time...")

        # Load
        loader = TextLoader("data/data.txt")
        documents = loader.load()
        print(f"Loaded {len(documents)} docs")

        # Split
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(documents)
        print(f"Split into {len(docs)} chunks")

        # Build + Save
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(VECTOR_STORE_PATH)
        print(f"Vector store saved to '{VECTOR_STORE_PATH}' folder!")

    retriever = db.as_retriever(search_kwargs={"k": 3})

    # Groq LLM
    print("Connecting to Groq...")
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    print("Groq ready")

    # Prompt with memory
    prompt = PromptTemplate.from_template("""
You are a helpful assistant. Use the context and chat history to answer the question.
If you don't know, say "I don't know(mujge pata nahi)".

Chat History:
{chat_history}

Context:
{context}

Question: {question}

Answer:""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def format_history(messages):
        if not messages:
            return "No previous conversation."
        history = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history += f"Human: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history += f"AI: {msg.content}\n"
        return history

    # Chat history
    chat_history = []

    # Chat loop
    print("\nRAG ready! Type 'exit' to quit or 'clear' to reset memory.\n")
    while True:
        query = input("Ask: ").strip()
        if query.lower() == "exit":
            break
        if query.lower() == "clear":
            chat_history = []
            print("Memory cleared!\n")
            continue
        if not query:
            continue

        chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
                "chat_history": lambda _: format_history(chat_history)
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        answer = chain.invoke(query)
        print(f"Answer: {answer}\n")

        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=answer))

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    input("Press Enter to exit...")
    