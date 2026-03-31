from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain.tools import Tool
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

# Summarization chain from Task 2
prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text in exactly 3 sentences:\n\n{text}"
)
parser = StrOutputParser()
summarization_chain = prompt | llm | parser

# Define the custom tool
def summarize_text(text: str) -> str:
    return summarization_chain.invoke({"text": text})

tools = [
    Tool(
        name="TextSummarizer",
        func=summarize_text,
        description="Useful for summarizing any given text into 3 sentences. Input should be the text you want summarized."
    )
]

# Get the react prompt from langchain hub
react_prompt = hub.pull("hwchase17/react")

# Initialize agent
agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# Test 1: Clear specific request
print("=== Test 1: Summarize AI impact on healthcare ===")
healthcare_text = """
Artificial intelligence is revolutionizing healthcare in numerous ways. 
AI-powered diagnostic tools can detect diseases like cancer and diabetes 
earlier and more accurately than traditional methods. Machine learning 
algorithms analyze medical images, identifying patterns that human doctors 
might miss. AI also accelerates drug discovery by predicting how molecules 
interact, reducing research time from decades to years. Virtual health 
assistants provide patients with 24/7 support, answering questions and 
monitoring symptoms. Predictive analytics help hospitals manage resources 
and reduce patient wait times. Despite these benefits, challenges remain 
around data privacy, algorithmic bias, and the need for regulatory frameworks 
to ensure safe AI deployment in clinical settings.
"""
result1 = agent_executor.invoke({"input": f"Summarize the impact of AI on healthcare: {healthcare_text}"})
print("\nFinal Answer:", result1["output"])

# Test 2: Vague request
print("\n=== Test 2: Vague request ===")
result2 = agent_executor.invoke({"input": "Summarize something interesting"})
print("\nFinal Answer:", result2["output"])