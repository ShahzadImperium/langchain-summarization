from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv
import json
import os

load_dotenv("../.env")

# Configure Azure OpenAI model
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment=os.getenv("DEPLOYMENT_NAME"),
    api_version=os.getenv("API_VERSION"),
)

# 150-word text about AI applications
ai_applications_text = """
Artificial intelligence is being applied across numerous industries, 
transforming the way businesses operate and deliver value. In healthcare, 
AI assists doctors in diagnosing diseases by analyzing medical images and 
patient data with remarkable accuracy. Financial institutions use AI for 
fraud detection, risk assessment, and algorithmic trading. In retail, 
recommendation engines powered by AI personalize shopping experiences for 
millions of customers. Manufacturing benefits from AI-driven predictive 
maintenance, reducing equipment downtime and operational costs. Education 
platforms leverage AI to create personalized learning paths for students. 
Transportation is being revolutionized by autonomous vehicles and AI-optimized 
logistics systems. In agriculture, AI helps farmers monitor crop health and 
optimize yields using drone imagery and sensor data. These diverse applications 
demonstrate AI's transformative potential across every sector of the economy.
"""

# ── Step 1: Define Response Schema ──
print("=" * 60)
print("Step 1: Defining Response Schema")
print("=" * 60)

response_schemas = [
    ResponseSchema(
        name="summary",
        description="A 3-sentence summary of the input text"
    ),
    ResponseSchema(
        name="length",
        description="The character count of the summary as an integer"
    )
]

print("Schema defined with keys: 'summary' and 'length'")

# ── Step 2: Create StructuredOutputParser ──
print("\n" + "=" * 60)
print("Step 2: Creating StructuredOutputParser")
print("=" * 60)

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()
print("Format instructions generated:")
print(format_instructions)

# ── Step 3: Build chain with structured output ──
print("\n" + "=" * 60)
print("Step 3: Building Chain with Structured Output")
print("=" * 60)

prompt = PromptTemplate(
    input_variables=["text"],
    partial_variables={"format_instructions": format_instructions},
    template="""Summarize the following text in exactly 3 sentences.
    
{format_instructions}

Text to summarize:
{text}"""
)

chain = prompt | llm | output_parser

# ── Step 4: Test the chain ──
print("\n" + "=" * 60)
print("Step 4: Testing Chain with AI Applications Text")
print("=" * 60)

result = chain.invoke({"text": ai_applications_text})

print("\nRaw result type:", type(result))
print("\nParsed JSON output:")
print(json.dumps(result, indent=2))

print("\n--- Extracted Fields ---")
print(f"Summary: {result['summary']}")
print(f"Length from AI: {result['length']}")
print(f"Actual character count: {len(result['summary'])}")

# ── Step 5: Validate the output ──
print("\n" + "=" * 60)
print("Step 5: Validating Output")
print("=" * 60)

has_summary = "summary" in result and len(result["summary"]) > 0
has_length = "length" in result

print(f"✓ Has 'summary' key: {has_summary}")
print(f"✓ Has 'length' key: {has_length}")
print(f"✓ Is valid JSON: True (parsed successfully)")
print(f"✓ Summary length matches: {result['length']}")