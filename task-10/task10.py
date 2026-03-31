from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
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

parser = StrOutputParser()

# ── Step 1: Load ai_intro.txt ──
print("=" * 60)
print("Step 1: Loading ai_intro.txt")
print("=" * 60)

loader = TextLoader("../task-3/ai_intro.txt", encoding="utf-8")
documents = loader.load()
full_text = documents[0].page_content
print(f"Loaded document with {len(full_text)} characters")

# ── Step 2: Summarize the document ──
print("\n" + "=" * 60)
print("Step 2: Summarizing the Document")
print("=" * 60)

summarization_prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text in exactly 3 sentences:\n\n{text}"
)
summarization_chain = summarization_prompt | llm | parser
summary = summarization_chain.invoke({"text": full_text})
print(f"Summary ({len(summary)} characters):")
print(summary)

# ── Step 3: Build QA chain ──
print("\n" + "=" * 60)
print("Step 3: Building Question Answering Chain")
print("=" * 60)

qa_prompt = PromptTemplate(
    input_variables=["text", "question"],
    template="""Answer the following question based only on the provided text.
Be concise and specific in your answer.

Text:
{text}

Question: {question}

Answer:"""
)
qa_chain = qa_prompt | llm | parser
print("QA chain created successfully!")

# ── Step 4: Ask question on summary ──
print("\n" + "=" * 60)
print("Step 4: Asking Question on SUMMARY")
print("=" * 60)

question = "What's the key event mentioned?"
print(f"Question: {question}")
print(f"Text used: Summary ({len(summary)} characters)")

summary_answer = qa_chain.invoke({
    "text": summary,
    "question": question
})
print(f"\nAnswer from Summary:\n{summary_answer}")

# ── Step 5: Ask same question on full document ──
print("\n" + "=" * 60)
print("Step 5: Asking Same Question on FULL DOCUMENT")
print("=" * 60)

print(f"Question: {question}")
print(f"Text used: Full document ({len(full_text)} characters)")

full_answer = qa_chain.invoke({
    "text": full_text,
    "question": question
})
print(f"\nAnswer from Full Document:\n{full_answer}")

# ── Step 6: Comparison ──
print("\n" + "=" * 60)
print("COMPARISON: Summary vs Full Document QA")
print("=" * 60)
print(f"\nQuestion: {question}")
print(f"\nAnswer from Summary ({len(summary)} chars):")
print(summary_answer)
print(f"\nAnswer from Full Document ({len(full_text)} chars):")
print(full_answer)
print(f"\nSummary answer length: {len(summary_answer)} characters")
print(f"Full doc answer length: {len(full_answer)} characters")