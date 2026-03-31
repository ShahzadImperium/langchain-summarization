from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

load_dotenv()

# Configure Azure OpenAI model
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment=os.getenv("DEPLOYMENT_NAME"),
    api_version=os.getenv("API_VERSION"),
)

# 100-word text about machine learning
ml_text = """
Machine learning is a subset of artificial intelligence that enables systems 
to learn and improve from experience without being explicitly programmed. 
It focuses on developing algorithms that can access data and use it to learn 
for themselves. The process begins with training data, which the algorithm 
uses to identify patterns and make decisions. Supervised learning, unsupervised 
learning, and reinforcement learning are the three main types. Applications 
include spam detection, recommendation systems, and fraud detection. Machine 
learning has transformed industries by enabling predictive analytics and 
automating complex decision-making processes that previously required human expertise.
"""

# 100-word text about deep learning
dl_text = """
Deep learning is a specialized subset of machine learning that uses neural 
networks with multiple layers to model complex patterns in data. Inspired by 
the human brain, these artificial neural networks consist of interconnected 
nodes that process information in hierarchical layers. Each layer learns 
increasingly abstract representations of the input data. Deep learning excels 
in tasks like image recognition, speech processing, and natural language 
understanding. It requires large amounts of data and significant computational 
power, typically using GPUs for training. Deep learning has driven major 
breakthroughs in computer vision, autonomous vehicles, and language models like GPT.
"""

# ── PART 1: ConversationBufferMemory ──
print("=" * 60)
print("PART 1: ConversationBufferMemory")
print("=" * 60)

buffer_memory = ConversationBufferMemory(
    memory_key="chat_history",
    k=3
)

buffer_prompt = PromptTemplate(
    input_variables=["chat_history", "text"],
    template="""You are a helpful assistant that summarizes text.
Previous conversation:
{chat_history}

Summarize the following text in exactly 3 sentences. 
If there is previous conversation, consider it for context:

{text}"""
)

buffer_chain = LLMChain(
    llm=llm,
    prompt=buffer_prompt,
    memory=buffer_memory,
    verbose=True
)

print("\n--- Summary 1: Machine Learning ---")
result1 = buffer_chain.invoke({"text": ml_text})
print("Summary:", result1["text"])

print("\n--- Summary 2: Deep Learning (with ML context) ---")
result2 = buffer_chain.invoke({"text": dl_text})
print("Summary:", result2["text"])

# ── PART 2: ConversationSummaryMemory ──
print("\n" + "=" * 60)
print("PART 2: ConversationSummaryMemory")
print("=" * 60)

summary_memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history"
)

summary_prompt = PromptTemplate(
    input_variables=["chat_history", "text"],
    template="""You are a helpful assistant that summarizes text.
Previous conversation summary:
{chat_history}

Summarize the following text in exactly 3 sentences.
If there is previous conversation, consider it for context:

{text}"""
)

summary_chain = LLMChain(
    llm=llm,
    prompt=summary_prompt,
    memory=summary_memory,
    verbose=True
)

print("\n--- Summary 1: Machine Learning ---")
result3 = summary_chain.invoke({"text": ml_text})
print("Summary:", result3["text"])

print("\n--- Summary 2: Deep Learning (with ML context) ---")
result4 = summary_chain.invoke({"text": dl_text})
print("Summary:", result4["text"])

# ── COMPARISON ──
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
print("\nBufferMemory - Deep Learning Summary:")
print(result2["text"])
print("\nSummaryMemory - Deep Learning Summary:")
print(result4["text"])