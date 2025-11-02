"""
Agent definitions for the multi-agent AI system.
Each agent has a specific role, goal, and backstory.
"""

import os
from crewai import Agent
from langchain_openai import ChatOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tools import search_tool
import torch

from local_llm import HuggingFaceLocalLLM

# Prepare tools list for Agent constructors. CrewAI expects tools to be
# either a dict or a crewai BaseTool instance. Our `scrape_tool` is a
# function (a lightweight wrapper), so we only include `search_tool` when
# it's an actual tool instance. This avoids pydantic validation errors.
tools_for_research = []
if search_tool is not None:
    tools_for_research.append(search_tool)


# Initialize LLM with fallback: OpenAI if API key available, otherwise use free Hugging Face model
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key and openai_key.strip() and not openai_key.startswith("your_"):
    # Use OpenAI GPT-3.5-turbo (faster, cheaper, higher rate limits)
    try:
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1500,  # Reduced to avoid context issues
            api_key=openai_key
        )
        print("✅ Using OpenAI GPT-3.5-turbo (API key found)")
    except Exception as e:
        print(f"❌ Error initializing OpenAI: {e}")
        raise
else:
    print("⚠️ No valid OpenAI API key found")
    print("💡 Using free local Hugging Face model (runs fully on your machine)")
    print("   First run will download weights (~2GB) and may take a minute")

    model_name = os.getenv("HF_LOCAL_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    max_new_tokens = int(os.getenv("HF_MAX_NEW_TOKENS", "512"))
    temperature = float(os.getenv("HF_TEMPERATURE", "0.7"))

    try:
        print(f"   Loading {model_name}...")
        model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device_map = "auto" if torch.cuda.is_available() else None

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=model_dtype,
            device_map=device_map
        )

        generation_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

        llm = HuggingFaceLocalLLM(
            model_name=model_name,
            generation_pipeline=generation_pipeline,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
        )
        print("✅ Free local model ready! (TinyLlama 1.1B chat)")
        if not torch.cuda.is_available():
            print("   Running on CPU – expect slower responses, but no API costs")

    except Exception as e:
        print(f"❌ Failed to load local model: {e}")
        print("   Please ensure you have enough disk space and RAM (>=6GB).")
        raise RuntimeError(
            "Unable to initialize free local model. Install a GPU-friendly model or "
            "set OPENAI_API_KEY for hosted inference."
        )


# Research Agent - Gathers information and conducts research
researcher = Agent(
    role="Senior Research Analyst",
    goal="Conduct thorough research on given topics and provide comprehensive, accurate information",
    backstory="""You are an experienced research analyst with a keen eye for detail.
    You excel at finding relevant information, analyzing data, and presenting findings
    in a clear and structured manner. You always verify your sources and provide
    evidence-based insights.""",
    verbose=True,
    allow_delegation=False,
    tools=tools_for_research,
    llm=llm
)


# Writer Agent - Creates content based on research
writer = Agent(
    role="Content Writer",
    goal="Create engaging, well-structured content based on research findings",
    backstory="""You are a skilled content writer with expertise in translating
    complex information into clear, engaging narratives. You have a talent for
    crafting compelling stories that resonate with readers while maintaining
    accuracy and professionalism.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)


# Reviewer Agent - Reviews and provides feedback
reviewer = Agent(
    role="Quality Assurance Reviewer",
    goal="Review content for accuracy, clarity, and quality, providing constructive feedback",
    backstory="""You are a meticulous reviewer with years of experience in quality
    assurance. You have a sharp eye for inconsistencies, errors, and areas for
    improvement. Your feedback is always constructive and aimed at elevating
    the quality of the final output.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)


# Analyst Agent - Analyzes data and draws insights
analyst = Agent(
    role="Data Analyst",
    goal="Analyze information, identify patterns, and provide actionable insights",
    backstory="""You are a data analyst with strong analytical skills and a talent
    for identifying trends and patterns. You excel at breaking down complex
    information into digestible insights and making data-driven recommendations.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)
