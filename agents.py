"""
Agent definitions for the multi-agent AI system.
Each agent has a specific role, goal, and backstory.
"""

from crewai import Agent
from langchain_openai import ChatOpenAI
from tools import search_tool, scrape_tool

# Prepare tools list for Agent constructors. CrewAI expects tools to be
# either a dict or a crewai BaseTool instance. Our `scrape_tool` is a
# function (a lightweight wrapper), so we only include `search_tool` when
# it's an actual tool instance. This avoids pydantic validation errors.
tools_for_research = []
if search_tool is not None:
    tools_for_research.append(search_tool)


# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7
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
