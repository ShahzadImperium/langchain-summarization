from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from dotenv import load_dotenv
import os

load_dotenv("../.env")

# Configure Azure OpenAI model
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment=os.getenv("DEPLOYMENT_NAME"),
    api_version=os.getenv("API_VERSION"),
)

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment="text-embedding-3-small",
    api_version=os.getenv("API_VERSION"),
)

# Step 1: Load the text file
print("=== Step 1: Loading Document ===")
loader = TextLoader("ai_intro.txt", encoding="utf-8")
documents = loader.load()
print(f"Loaded {len(documents)} document(s)")

# Step 2: Split into chunks
print("\n=== Step 2: Splitting into Chunks ===")
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents(documents)
print(f"Total chunks created: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk.page_content[:50]}...")

# Step 3: Set up in-memory vector store with embeddings
print("\n=== Step 3: Creating Vector Store ===")
vector_store = InMemoryVectorStore(embedding=embeddings)
vector_store.add_documents(chunks)
print("Vector store created and chunks indexed!")

# Step 4: Build retriever and query
print("\n=== Step 4: Querying Retriever ===")
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
retrieved_docs = retriever.invoke("AI milestones")
print(f"Retrieved {len(retrieved_docs)} chunks:")
for i, doc in enumerate(retrieved_docs):
    print(f"\nChunk {i+1}:\n{doc.page_content}")

# Step 5: Summarize retrieved text
print("\n=== Step 5: Summarizing Retrieved Text ===")
retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])

prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text in exactly 3 sentences:\n\n{text}"
)
parser = StrOutputParser()
chain = prompt | llm | parser
summary = chain.invoke({"text": retrieved_text})
print("\nSummary:")
print(summary)