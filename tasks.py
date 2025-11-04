"""
Task definitions for the multi-agent system.
Tasks are assigned to agents and define what needs to be accomplished.
"""

from crewai import Task
from agents import researcher, writer, reviewer, analyst


def create_research_task(topic: str, agent: Task = None) -> Task:
    """Create a research task for the researcher agent.

    The `agent` parameter can be supplied to bind the task to a specific Agent
    instance (useful when creating fresh agents per run). If omitted, falls back
    to the module-level `researcher`.
    """
    assigned_agent = agent if agent is not None else researcher
    
    # Detect if this is a news query (looking for current/latest/today's news)
    topic_lower = topic.lower()
    is_news_query = any(keyword in topic_lower for keyword in 
                       ['news', 'latest', 'today', 'current events', 'breaking', 'headlines', 
                        'recent events', 'what happened', 'whats happening'])
    
    if is_news_query:
        description = f"""Use the 'Search the internet with Serper' tool to find REAL CURRENT NEWS about: '{topic}'.
        
        MANDATORY FIRST STEP: Call your tool 'Search the internet with Serper' with search_query='{topic}'
        
        After you receive the search results, format them EXACTLY like this:
        
        SEARCH RESULTS:
        ---
        1. Title: [copy exact title from search result]
           Snippet: [copy exact snippet from search result]
           URL: [copy exact URL from search result]
        
        2. Title: [copy exact title from search result]
           Snippet: [copy exact snippet from search result]
           URL: [copy exact URL from search result]
        
        (continue for all search results)
        ---
        
        RULES:
        - You HAVE a tool called 'Search the internet with Serper' - USE IT FIRST
        - Copy EXACTLY what the search tool returns - do not rewrite or embellish
        - Include ALL search results from the tool (at least 5-8 items)
        - Do NOT say you cannot access URLs - you have a search tool that CAN
        - Do NOT add commentary, analysis, or made-up information
        - Your job: 1) Use tool, 2) Copy results exactly"""
    else:
        description = f"""Conduct comprehensive research on the topic: '{topic}'.
        
        Your research MUST include SPECIFIC NUMBERS and DATA:
        1. Key facts and background information with specific statistics and percentages
        2. Recent developments and trends with exact figures, dates, and numerical comparisons
        3. Important statistics and data points - include as many concrete numbers as possible
        4. Year-over-year changes, growth rates, and quantitative metrics
        5. Expert opinions with cited numerical claims
        6. Credible sources and references (URLs when available)
        
        CRITICAL: Include specific numbers, percentages, rates, and metrics throughout.
        Avoid vague statements - use precise data points."""
    
    return Task(
        description=description,
        agent=assigned_agent,
        expected_output="Actual current news stories with headlines, summaries, dates, and source URLs" if is_news_query else "A detailed research report packed with specific numbers, statistics, percentages, and credible source citations."
    )


def create_writing_task(topic: str, agent: Task = None) -> Task:
    """Create a writing task for the writer agent; optional agent override."""
    assigned_agent = agent if agent is not None else writer
    
    # Detect if this is a news query
    topic_lower = topic.lower()
    is_news_query = any(keyword in topic_lower for keyword in 
                       ['news', 'latest', 'today', 'current events', 'breaking', 'headlines', 
                        'recent events', 'what happened', 'whats happening'])
    
    if is_news_query:
        description = f"""Format the researcher's search results into a clean news summary about '{topic}'.
        
        The researcher has provided EXACT search results. Your ONLY job is to format them nicely.
        
        Create this structure for EACH news item:
        
        ## [Number]. [Exact Title from search result]
        
        **Key Points:**
        - [Extract first key fact from snippet]
        - [Extract second key fact from snippet]
        - [Extract third key fact from snippet]
        - [Extract fourth key fact from snippet]
        - [Extract fifth key fact from snippet]
        
        **Source:** [Exact URL]
        
        INSTRUCTIONS:
        1. Break each snippet into 5 bullet points
        2. Each bullet point should be ONE distinct fact or detail from the snippet
        3. Keep bullets concise (one sentence, 10-20 words each)
        4. If snippet is short, break down the main information into smaller points
        5. Do NOT add information not in the snippet
        
        EXAMPLE FORMAT:
        
        ## 1. Candidates for governor of New Jersey spend final hours on the campaign trail
        
        **Key Points:**
        - Gubernatorial candidates are in their final hours of campaigning
        - The campaign trail activity is focused on New Jersey
        - Both major candidates are making last-minute voter outreach efforts
        - Election day is approaching for the New Jersey governor race
        - Campaigns are intensifying their efforts to reach undecided voters
        
        **Source:** https://newjersey.news12.com/
        
        CRITICAL RULES:
        - You MUST create exactly 5 bullet points per news item
        - Copy exact titles from researcher's results
        - Extract only what's in the snippets - no fabrication
        - If snippet has <5 distinct facts, rephrase or break down existing facts into separate points"""
    else:
        description = f"""Based on the research findings, write a comprehensive, data-rich article about '{topic}'.
        
        Your article MUST:
        1. Have an engaging introduction that hooks the reader with a compelling statistic
        2. Present information in a logical, well-structured manner with clear sections
        3. Include SPECIFIC NUMBERS, PERCENTAGES, and DATA POINTS throughout - cite exact figures
        4. Use tables or suggest charts/graphs where data comparisons would be helpful
        5. Include year-over-year trends, growth rates, and quantitative comparisons
        6. Be written in a clear, accessible style while maintaining numerical precision
        7. Have a strong conclusion that summarizes key numerical findings
        
        CRITICAL: Every major claim should be backed by specific numbers. Avoid vague statements.
        Suggest visualizations (e.g., "This data could be shown as a line graph comparing X vs Y").
        
        Target length: 800-1000 words."""
    
    return Task(
        description=description,
        agent=assigned_agent,
        expected_output="A news summary presenting actual current news stories with headlines, dates, details, and source links" if is_news_query else "A comprehensive, data-driven article with specific numbers, statistics, percentages throughout, and suggestions for data visualizations where appropriate."
    )


def create_review_task(agent: Task = None) -> Task:
    """Create a review task for the reviewer agent; optional agent override."""
    assigned_agent = agent if agent is not None else reviewer
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
        agent=assigned_agent,
        expected_output="A detailed review with specific feedback and either approval or suggestions for improvement."
    )


def create_analysis_task(topic: str, agent: Task = None) -> Task:
    """Create an analysis task for the analyst agent; optional agent override."""
    assigned_agent = agent if agent is not None else analyst
    
    # Detect if this is a news query
    topic_lower = topic.lower()
    is_news_query = any(keyword in topic_lower for keyword in 
                       ['news', 'latest', 'today', 'current events', 'breaking', 'headlines', 
                        'recent events', 'what happened', 'whats happening'])
    
    if is_news_query:
        description = f"""Extract and organize the ACTUAL NEWS STORIES found by the researcher about '{topic}'.
        
        CRITICAL RULES:
        - Use ONLY the news information provided by the researcher
        - Do NOT make up or fabricate any news stories, headlines, or statistics
        - Do NOT add analysis or generate hypothetical content
        - Simply extract and organize the actual news items found
        
        For each news story found, extract:
        1. The actual headline or title
        2. The publication date/time (if available)
        3. Key facts and details from the snippet
        4. The source name and URL
        5. Any numbers or statistics mentioned
        
        If the researcher didn't find specific article URLs, use the snippets and information provided.
        Format as a clean list of news items with all available details."""
    else:
        description = f"""Analyze the information gathered about '{topic}' and provide quantitative, data-driven insights.
        
        Your analysis MUST include SPECIFIC NUMBERS:
        1. Key patterns and trends with exact percentage changes, growth rates, and comparative metrics
        2. Strengths and opportunities quantified with specific data points
        3. Challenges and risks backed by numerical evidence
        4. Data-driven recommendations with projected impact (use percentages/ranges where possible)
        5. Future implications with numerical predictions and trend projections
        
        CRITICAL: Quantify everything possible. Use specific numbers, percentages, ratios, and metrics.
        Present findings in a clear, structured format with data comparisons."""
    
    return Task(
        description=description,
        agent=assigned_agent,
        expected_output="A quantitative analysis packed with specific numbers, percentage changes, growth rates, and data-driven projections."
    )
