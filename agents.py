"""
Agent definitions for the multi-agent AI system.
Each agent has a specific role, goal, and backstory.
"""

import os
from crewai import Agent
from langchain_openai import ChatOpenAI
from tools import search_tool

# Prepare tools list for Agent constructors. CrewAI expects tools to be
# either a dict or a crewai BaseTool instance. Our `scrape_tool` is a
# function (a lightweight wrapper), so we only include `search_tool` when
# it's an actual tool instance. This avoids pydantic validation errors.
tools_for_research = []
if search_tool is not None:
    tools_for_research.append(search_tool)


# Initialize default LLM (for backwards compatibility)
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key and openai_key.strip() and not openai_key.startswith("your_"):
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=2000,
        api_key=openai_key
    )
    print("✅ Using OpenAI GPT-3.5-turbo (cheapest model)")
else:
    raise RuntimeError(
        "OPENAI_API_KEY is required. Please set it in your .env file."
    )


# Research Agent - Gathers information and conducts research
def make_agents_with_model(model_name="gpt-3.5-turbo"):
    """Create fresh Agent instances with specified model.
    
    Args:
        model_name: OpenAI model to use (e.g., "gpt-3.5-turbo", "gpt-4o-mini")
    
    Returns:
        Tuple of (researcher, writer, reviewer, analyst) agents
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if openai_key and openai_key.strip() and not openai_key.startswith("your_"):
        try:
            llm_instance = ChatOpenAI(
                model=model_name,
                temperature=0.7,
                max_tokens=2000,
                api_key=openai_key
            )
            print(f"✅ Using OpenAI {model_name}")
        except Exception as e:
            print(f"❌ Error initializing OpenAI: {e}")
            raise
    else:
        # Fallback to default llm
        llm_instance = llm

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
        llm=llm_instance
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
        llm=llm_instance
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
        llm=llm_instance
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
        llm=llm_instance
    )

    return researcher, writer, reviewer, analyst

def make_agents():
    """Create fresh Agent instances with default model (for backwards compatibility)."""
    return make_agents_with_model("gpt-3.5-turbo")

# Keep legacy names for backwards compatibility if other modules imported them
try:
    researcher, writer, reviewer, analyst  # type: ignore
except NameError:
    # Create one set at import time for compatibility, but prefer make_agents() per-run
    researcher, writer, reviewer, analyst = make_agents()


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
