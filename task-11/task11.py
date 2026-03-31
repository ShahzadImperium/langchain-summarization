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
from datetime import datetime
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

# ── Summarization chain ──
prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text in exactly 3 sentences:\n\n{text}"
)
parser = StrOutputParser()
summarization_chain = prompt | llm | parser

# ── Retriever from Task 3 ──
loader = TextLoader("../task-3/ai_intro.txt", encoding="utf-8")
documents = loader.load()
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents(documents)
vector_store = InMemoryVectorStore(embedding=embeddings)
vector_store.add_documents(chunks)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# ── Tool 1: Text Summarizer ──
def summarize_text(text: str) -> str:
    return summarization_chain.invoke({"text": text})

# ── Tool 2: Text Retriever ──
def retrieve_text(query: str) -> str:
    docs = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in docs])

# ── Tool 3: Word Counter ──
def count_words(text: str) -> str:
    count = len(text.split())
    return f"The text contains {count} words."

# ── Tool 4: Current Date (NEW!) ──
def get_current_date(input: str = "") -> str:
    now = datetime.now()
    return f"Today's date is {now.strftime('%B %d, %Y')} and the time is {now.strftime('%I:%M %p')}."

# ── Tool 5: Mock Web Search (NEW!) ──
def mock_web_search(query: str) -> str:
    return f"""Mock search results for '{query}':
    Recent AI trends show that large language models continue to advance rapidly 
    in 2025. Companies like OpenAI, Google, and Anthropic are releasing more 
    capable models. AI is being integrated into healthcare, education, and 
    business workflows at an unprecedented pace. Regulatory frameworks for AI 
    are being developed globally to ensure responsible deployment and usage."""

tools = [
    Tool(
        name="TextSummarizer",
        func=summarize_text,
        description="Useful for summarizing any given text into 3 sentences. Input should be the text you want summarized."
    ),
    Tool(
        name="TextRetriever",
        func=retrieve_text,
        description="Useful for retrieving relevant text chunks from the AI document. Input should be a search query."
    ),
    Tool(
        name="WordCounter",
        func=count_words,
        description="Useful for counting the number of words in a text. Input should be the text you want to count words for."
    ),
    Tool(
        name="CurrentDate",
        func=get_current_date,
        description="Useful for getting the current date and time. No input needed, just pass an empty string."
    ),
    Tool(
        name="WebSearch",
        func=mock_web_search,
        description="Useful for searching the web for recent updates and news. Input should be a search query string."
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

# ── Test 1: Summarize + Date ──
print("=" * 60)
print("Test 1: Summarize AI text and get today's date")
print("=" * 60)

ai_text = """
Artificial intelligence is transforming every industry at an unprecedented pace.
Machine learning algorithms now power recommendation systems, fraud detection,
and medical diagnostics. Natural language processing enables chatbots and virtual
assistants to communicate naturally with humans. Computer vision allows machines
to interpret and understand visual information from the world. Deep learning
models trained on massive datasets continue to push the boundaries of what
machines can achieve. The integration of AI into daily workflows is increasing
productivity and enabling new capabilities that were previously impossible.
Researchers continue to explore new frontiers in AI safety, interpretability,
and alignment to ensure these powerful systems benefit humanity.
"""

result1 = agent_executor.invoke({
    "input": f"Summarize this text about AI and tell me today's date: {ai_text}"
})
print("\nFinal Answer:", result1["output"])

# ── Test 2: Summarize + Mock Web Search ──
print("\n" + "=" * 60)
print("Test 2: Summarize AI trends and search for recent updates")
print("=" * 60)

result2 = agent_executor.invoke({
    "input": "Summarize AI trends from the document and search for recent updates on AI."
})
print("\nFinal Answer:", result2["output"])
