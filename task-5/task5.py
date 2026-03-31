from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain.tools import Tool
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

# ── Task 3: Build retriever from ai_intro.txt ──
loader = TextLoader("../task-3/ai_intro.txt", encoding="utf-8")
documents = loader.load()
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents(documents)
vector_store = InMemoryVectorStore(embedding=embeddings)
vector_store.add_documents(chunks)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# ── Task 2: Summarization chain ──
prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text in exactly 3 sentences:\n\n{text}"
)
parser = StrOutputParser()
summarization_chain = prompt | llm | parser

# ── Tool 1: Retriever tool ──
def retrieve_text(query: str) -> str:
    docs = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in docs])

# ── Tool 2: Summarizer tool ──
def summarize_text(text: str) -> str:
    return summarization_chain.invoke({"text": text})

# ── Tool 3: Word counter tool ──
def count_words(text: str) -> str:
    count = len(text.split())
    return f"The text contains {count} words."

tools = [
    Tool(
        name="TextRetriever",
        func=retrieve_text,
        description="Useful for retrieving relevant text chunks from the AI document. Input should be a search query."
    ),
    Tool(
        name="TextSummarizer",
        func=summarize_text,
        description="Useful for summarizing any given text into 3 sentences. Input should be the text you want summarized."
    ),
    Tool(
        name="WordCounter",
        func=count_words,
        description="Useful for counting the number of words in a text. Input should be the text you want to count words for."
    )
]

# ── Build agent ──
react_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10
)

# ── Test 1: Find and summarize AI breakthroughs ──
print("=== Test 1: Find and Summarize AI Breakthroughs ===")
result1 = agent_executor.invoke({
    "input": "Find text about AI breakthroughs from the document, summarize it, then count the words in the summary."
})
print("\nFinal Answer:", result1["output"])

# ── Test 2: Just retrieval and summary ──
print("\n=== Test 2: Find and Summarize AI milestones ===")
result2 = agent_executor.invoke({
    "input": "Find text about AI milestones from the document and summarize it."
})
print("\nFinal Answer:", result2["output"])