from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from dotenv import load_dotenv
import os
from pathlib import Path
os.chdir(Path(__file__).parent)


load_dotenv()

# Configure Azure OpenAI model
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment=os.getenv("DEPLOYMENT_NAME"),
    api_version=os.getenv("API_VERSION"),
)

# Configure embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment="text-embedding-3-small",
    api_version=os.getenv("API_VERSION"),
)

# Summarization chain
prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text in exactly 3 sentences:\n\n{text}"
)
parser = StrOutputParser()
summarization_chain = prompt | llm | parser

# Text splitter
splitter = CharacterTextSplitter(chunk_size=150, chunk_overlap=30)

# ── PART 1: Load PDF ──
print("=" * 60)
print("PART 1: Loading PDF with PyPDFLoader")
print("=" * 60)

pdf_loader = PyPDFLoader("ai_ethics.pdf")
pdf_documents = pdf_loader.load()
print(f"Loaded {len(pdf_documents)} pages from PDF")

pdf_chunks = splitter.split_documents(pdf_documents)
print(f"Split into {len(pdf_chunks)} chunks")

pdf_vector_store = InMemoryVectorStore(embedding=embeddings)
pdf_vector_store.add_documents(pdf_chunks)
pdf_retriever = pdf_vector_store.as_retriever(search_kwargs={"k": 3})

print("\nQuerying PDF with 'AI challenges'...")
pdf_docs = pdf_retriever.invoke("AI challenges")
pdf_retrieved_text = "\n".join([doc.page_content for doc in pdf_docs])
print(f"Retrieved {len(pdf_docs)} chunks from PDF")

pdf_summary = summarization_chain.invoke({"text": pdf_retrieved_text})
print("\nPDF Summary:")
print(pdf_summary)

# ── PART 2: Load Webpage ──
print("\n" + "=" * 60)
print("PART 2: Loading Webpage with WebBaseLoader")
print("=" * 60)

web_url = "https://www.ibm.com/think/topics/artificial-intelligence"
web_loader = WebBaseLoader(web_url)
web_documents = web_loader.load()
print(f"Loaded {len(web_documents)} document(s) from webpage")

web_chunks = splitter.split_documents(web_documents)
print(f"Split into {len(web_chunks)} chunks")

web_vector_store = InMemoryVectorStore(embedding=embeddings)
web_vector_store.add_documents(web_chunks)
web_retriever = web_vector_store.as_retriever(search_kwargs={"k": 3})

print("\nQuerying webpage with 'AI challenges'...")
web_docs = web_retriever.invoke("AI challenges")
web_retrieved_text = "\n".join([doc.page_content for doc in web_docs])
print(f"Retrieved {len(web_docs)} chunks from webpage")

web_summary = summarization_chain.invoke({"text": web_retrieved_text})
print("\nWebpage Summary:")
print(web_summary)

# ── COMPARISON ──
print("\n" + "=" * 60)
print("COMPARISON: PDF vs Webpage Summary Quality")
print("=" * 60)
print("\nPDF Summary (AI Ethics - European Parliament):")
print(pdf_summary)
print("\nWebpage Summary (IBM AI Trends):")
print(web_summary)