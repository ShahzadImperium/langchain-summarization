from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

# 200-word test paragraph about AI
ai_text = """
Artificial intelligence has transformed the way humans interact with technology 
over the past few decades. From its theoretical foundations in the 1950s, when 
Alan Turing proposed the concept of machine intelligence, AI has evolved into 
a powerful force shaping industries worldwide. Early AI systems were rule-based, 
relying on explicit programming to perform tasks. However, the advent of machine 
learning introduced a paradigm shift, allowing systems to learn from data rather 
than fixed rules. Deep learning further accelerated this progress, enabling 
breakthroughs in image recognition, natural language processing, and autonomous 
systems. Today, AI powers virtual assistants, recommendation engines, medical 
diagnostics, and self-driving cars. It has become a critical tool in healthcare, 
finance, education, and manufacturing. Despite its benefits, AI raises important 
ethical concerns, including bias in algorithms, job displacement, and privacy 
issues. Researchers and policymakers are actively working to address these 
challenges, developing frameworks for responsible AI use. As computing power 
continues to grow and datasets expand, AI is expected to become even more 
capable, potentially solving complex global problems like climate change and 
disease prevention in the coming decades.
"""

parser = StrOutputParser()

# --- 3-sentence summary ---
prompt_3 = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text in exactly 3 sentences:\n\n{text}"
)
chain_3 = prompt_3 | llm | parser
summary_3 = chain_3.invoke({"text": ai_text})
print("=== 3-Sentence Summary ===")
print(summary_3)

# --- 1-sentence summary ---
prompt_1 = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text in exactly 1 sentence:\n\n{text}"
)
chain_1 = prompt_1 | llm | parser
summary_1 = chain_1.invoke({"text": ai_text})
print("\n=== 1-Sentence Summary ===")
print(summary_1)