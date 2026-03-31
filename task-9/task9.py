from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.retrievers.multi_query import MultiQueryRetriever
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

# Load and split ai_intro.txt
loader = TextLoader("../task-3/ai_intro.txt", encoding="utf-8")
documents = loader.load()
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents(documents)

# Build vector store
vector_store = InMemoryVectorStore(embedding=embeddings)
vector_store.add_documents(chunks)

# ── PART 1: Single Query Retriever (from Task 3) ──
print("=" * 60)
print("PART 1: Single Query Retriever")
print("=" * 60)

single_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
single_docs = single_retriever.invoke("AI advancements")
single_text = "\n".join([doc.page_content for doc in single_docs])

print(f"Query: 'AI advancements'")
print(f"Retrieved {len(single_docs)} chunks")
print("\nRetrieved chunks:")
for i, doc in enumerate(single_docs):
    print(f"\nChunk {i+1}: {doc.page_content[:100]}...")

single_summary = summarization_chain.invoke({"text": single_text})
print("\nSingle Query Summary:")
print(single_summary)

# ── PART 2: MultiQueryRetriever ──
print("\n" + "=" * 60)
print("PART 2: MultiQueryRetriever")
print("=" * 60)

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    llm=llm
)

import logging
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

print("\nQuery: 'AI advancements'")
multi_docs = multi_retriever.invoke("AI advancements")
print(f"\nRetrieved {len(multi_docs)} unique chunks (after deduplication)")
print("\nRetrieved chunks:")
for i, doc in enumerate(multi_docs):
    print(f"\nChunk {i+1}: {doc.page_content[:100]}...")

multi_text = "\n".join([doc.page_content for doc in multi_docs])
multi_summary = summarization_chain.invoke({"text": multi_text})
print("\nMulti Query Summary:")
print(multi_summary)

# ── COMPARISON ──
print("\n" + "=" * 60)
print("COMPARISON: Single vs Multi Query Retriever")
print("=" * 60)
print(f"\nSingle Query - Chunks retrieved: {len(single_docs)}")
print(f"Multi Query  - Chunks retrieved: {len(multi_docs)}")
print(f"\nSingle Query Summary:\n{single_summary}")
print(f"\nMulti Query Summary:\n{multi_summary}")


