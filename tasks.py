"""
Task definitions for the multi-agent system.
Tasks are assigned to agents and define what needs to be accomplished.
"""

from crewai import Task
from agents import researcher, writer, reviewer, analyst


def create_research_task(topic: str) -> Task:
    """Create a research task for the researcher agent."""
    return Task(
        description=f"""Conduct comprehensive research on the topic: '{topic}'.
        
        Your research should include:
        1. Key facts and background information
        2. Recent developments and trends
        3. Important statistics and data points
        4. Expert opinions and perspectives
        5. Credible sources and references
        
        Provide a well-organized summary of your findings.""",
        agent=researcher,
        expected_output="A detailed research report with key findings, statistics, and credible sources."
    )


def create_writing_task(topic: str) -> Task:
    """Create a writing task for the writer agent."""
    return Task(
        description=f"""Based on the research findings, write a comprehensive article about '{topic}'.
        
        Your article should:
        1. Have an engaging introduction that hooks the reader
        2. Present information in a logical, well-structured manner
        3. Include relevant data and examples from the research
        4. Be written in a clear, accessible style
        5. Have a strong conclusion that summarizes key points
        
        Target length: 800-1000 words.""",
        agent=writer,
        expected_output="A well-written, engaging article of 800-1000 words based on the research findings."
    )


def create_review_task() -> Task:
    """Create a review task for the reviewer agent."""
    return Task(
        description="""Review the written article for quality and accuracy.
        
        Your review should assess:
        1. Factual accuracy and consistency with research
        2. Clarity and readability
        3. Structure and flow
        4. Grammar, spelling, and punctuation
        5. Overall quality and impact
        
        Provide specific, constructive feedback and suggest improvements.
        If the article meets high standards, approve it for publication.""",
        agent=reviewer,
        expected_output="A detailed review with specific feedback and either approval or suggestions for improvement."
    )


def create_analysis_task(topic: str) -> Task:
    """Create an analysis task for the analyst agent."""
    return Task(
        description=f"""Analyze the information gathered about '{topic}' and provide actionable insights.
        
        Your analysis should include:
        1. Key patterns and trends identified
        2. Strengths and opportunities
        3. Challenges and risks
        4. Data-driven recommendations
        5. Future implications and predictions
        
        Present your findings in a clear, structured format.""",
        agent=analyst,
        expected_output="A comprehensive analysis with actionable insights, recommendations, and future predictions."
    )
