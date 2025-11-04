"""
Streamlit Web Interface for Multi-Agent AI System
Provides an interactive UI for submitting topics and viewing agent outputs in real-time.
"""

import streamlit as st
import os
import sys
from pathlib import Path
from datetime import datetime
import json
import time
from typing import Optional
import threading

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()


# Page configuration
st.set_page_config(
    page_title="Multi-Agent AI Research System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 0.5rem 0;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
</style>
""", unsafe_allow_html=True)


def check_api_keys():
    """Check if required API keys are configured."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False, "OPENAI_API_KEY not found in environment"
    return True, "API keys configured"


def get_recent_runs(limit=10):
    """Get list of recent runs from the runs directory."""
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return []
    
    run_folders = [d for d in runs_dir.iterdir() if d.is_dir() and d.name != "tool_logs"]
    run_folders.sort(reverse=True)  # Most recent first
    
    runs = []
    for folder in run_folders[:limit]:
        summary_file = folder / "summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
                    runs.append({
                        "folder": folder,
                        "timestamp": summary.get("timestamp", folder.name),
                        "topic": summary.get("topic", "Unknown"),
                        "duration": summary.get("duration_seconds", 0)
                    })
            except Exception:
                pass
    
    return runs


def format_hms(seconds: float) -> str:
    """Format seconds as HH:MM:SS (hours may be > 99 if long runs).

    Rounds to nearest second for display.
    """
    try:
        total = int(round(float(seconds)))
    except Exception:
        total = 0
    hrs = total // 3600
    mins = (total % 3600) // 60
    secs = total % 60
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def display_task_output(task_file: Path):
    """Display a task output file."""
    if not task_file.exists():
        st.warning(f"Task file not found: {task_file.name}")
        return
    
    try:
        with open(task_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract task info from header comments
        lines = content.split('\n')
        task_name = task_file.stem.replace('task-', 'Task ').replace('-', ' ').title()
        
        with st.expander(f"📄 {task_name}", expanded=False):
            st.text_area(
                "Output",
                content,
                height=300,
                key=f"task_{task_file.stem}",
                label_visibility="collapsed"
            )
    except Exception as e:
        st.error(f"Error reading {task_file.name}: {e}")


def run_research_task(topic: str, progress_placeholder, status_placeholder, model_name="gpt-3.5-turbo"):
    """Execute the multi-agent research task."""
    try:
        # Import here to avoid issues if dependencies aren't installed
        from crewai import Crew, Process
        from tasks import create_research_task, create_writing_task, create_review_task, create_analysis_task
        # Create fresh agent instances per run with selected model
        from agents import make_agents_with_model
        researcher, writer, reviewer, analyst = make_agents_with_model(model_name)
        
        # Debug: Show which model is being used
        st.info(f"🤖 Using LLM: {model_name}")
        st.info(f"🔧 Researcher has {len(researcher.tools)} tool(s): {[t.name for t in researcher.tools] if researcher.tools else 'None'}")

        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("runs") / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        status_placeholder.info(f"📁 Output directory: `{run_dir}`")

        # Detect if this is a news query
        topic_lower = topic.lower()
        is_news_query = any(keyword in topic_lower for keyword in 
                           ['news', 'latest', 'today', 'current events', 'breaking', 'headlines', 
                            'recent events', 'what happened', 'whats happening'])

        # Create tasks bound to freshly-created agents
        progress_placeholder.progress(0.1, "Creating tasks...")
        research_task = create_research_task(topic, agent=researcher)
        writing_task = create_writing_task(topic, agent=writer)
        
        # For news queries, skip analyst and reviewer (2-agent workflow)
        # For standard research, use all 4 agents
        if is_news_query:
            status_placeholder.info("📰 News query detected - using streamlined 2-agent workflow")
            crew = Crew(
                agents=[researcher, writer],
                tasks=[research_task, writing_task],
                process=Process.sequential,
                verbose=True
            )
        else:
            analysis_task = create_analysis_task(topic, agent=analyst)
            review_task = create_review_task(agent=reviewer)
            crew = Crew(
                agents=[researcher, analyst, writer, reviewer],
                tasks=[research_task, analysis_task, writing_task, review_task],
                process=Process.sequential,
                verbose=True
            )

        # Create the crew
        progress_placeholder.progress(0.2, "Initializing crew...")

        # Execute the crew
        progress_placeholder.progress(0.3, "🔬 Research agent working...")
        start_time = datetime.now()

        # Run in separate thread to allow UI updates (simplified for demo)
        result = crew.kickoff()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        progress_placeholder.progress(1.0, "✅ Complete!")

        # Save outputs (similar to main.py logic)
        task_outputs = []
        tasks_info = [
            {"name": "research", "description": "Research findings", "agent": "researcher"},
            {"name": "analysis", "description": "Data analysis", "agent": "analyst"},
            {"name": "writing", "description": "Article draft", "agent": "writer"},
            {"name": "review", "description": "Quality review", "agent": "reviewer"}
        ]

        # Save individual task outputs and capture the article
        article_content = None
        if hasattr(crew, 'tasks'):
            for i, (task_info, task) in enumerate(zip(tasks_info, crew.tasks), 1):
                if hasattr(task, 'output') and task.output:
                    output_text = str(task.output)
                    task_file = run_dir / f"task-{i}-{task_info['name']}.txt"
                    with open(task_file, "w", encoding="utf-8") as f:
                        f.write(f"# Task {i}: {task_info['description']}\n")
                        f.write(f"# Agent: {task_info['agent']}\n")
                        f.write("=" * 60 + "\n\n")
                        f.write(output_text)

                    # Capture the writer's article (task 3) for final output
                    if task_info['name'] == 'writing':
                        article_content = output_text

                    task_outputs.append({
                        "task_number": i,
                        "task_name": task_info["name"],
                        "output_file": str(task_file)
                    })

        # Save final output - use the article from the writer, not the reviewer's feedback
        final_output_file = run_dir / "final_output.md"
        with open(final_output_file, "w", encoding="utf-8") as f:
            f.write(f"# {topic.title()}\n\n")
            f.write(f"**Generated:** {timestamp}\n")
            f.write(f"**Duration:** {format_hms(duration)}\n")
            f.write(f"**Model:** {model_name}\n\n")
            f.write("---\n\n")
            # Use the writer's article if available, otherwise fall back to final result
            content_to_save = article_content if article_content else str(result)
            f.write(content_to_save)

        # Save summary
        summary = {
            "topic": topic,
            "timestamp": timestamp,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "output_directory": str(run_dir),
            "final_output_file": str(final_output_file),
            "tasks": task_outputs
        }

        summary_file = run_dir / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return True, run_dir, duration, str(result)

    except Exception as e:
        progress_placeholder.empty()
        return False, None, 0, str(e)


# Main app
def main():
    # Header
    st.markdown('<div class="main-header">🤖 Multi-Agent AI Research System</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key status
        keys_ok, keys_msg = check_api_keys()
        if keys_ok:
            st.success(f"✅ Using OpenAI GPT-4")
        else:
            st.info(f"🆓 Using free Phi-2 model")
            st.caption("Add `OPENAI_API_KEY` to `.env` for GPT-4")
        
        st.divider()
        
        # System info
        st.header("📊 System Info")
        st.metric("Agents", "4")
        st.caption("Researcher • Analyst • Writer • Reviewer")
        
        runs = get_recent_runs(5)
        st.metric("Recent Runs", len(runs))
        
        st.divider()
        
        # Recent runs
        st.header("📂 Recent Runs")
        if runs:
            for run in runs:
                with st.expander(f"🕐 {run['timestamp']}", expanded=False):
                    st.write(f"**Topic:** {run['topic']}")
                    st.write(f"**Duration:** {format_hms(run['duration'])}")
                    if st.button("View", key=f"view_{run['timestamp']}"):
                        st.session_state['selected_run'] = run['folder']
        else:
            st.info("No runs yet")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["🚀 New Research", "📊 View Results", "ℹ️ About"])
    
    with tab1:
        st.header("Start New Research Task")
        
        if not keys_ok:
            st.warning("ℹ️ No OpenAI API key detected. Using free Microsoft Phi-2 model (slower but no cost).")
            st.info("💡 **Tip:** Add `OPENAI_API_KEY` to `.env` for faster GPT-4 performance.")
        
        # Model selection
        st.markdown("### 🤖 Select AI Model")
        model_choice = st.radio(
            "Choose your model:",
            options=["GPT-3.5-turbo (Cheapest)", "GPT-4o-mini (Better Tool Calling)"],
            index=0,
            help="GPT-3.5-turbo: Most economical | GPT-4o-mini: Better at using search tools reliably",
            horizontal=True
        )
        
        # Extract model name
        selected_model = "gpt-3.5-turbo" if "turbo" in model_choice else "gpt-4o-mini"
        
        # Show model info
        if selected_model == "gpt-3.5-turbo":
            st.info("💰 **GPT-3.5-turbo**: Most economical option. May have variable tool calling reliability.")
        else:
            st.info("⚡ **GPT-4o-mini**: Better tool calling reliability (~20-40s), excellent for news queries and research tasks requiring search.")
        
        st.divider()
        
        # Topic input
        topic = st.text_input(
            "Enter a research topic:",
            placeholder="e.g., The Impact of Artificial Intelligence on Modern Healthcare",
            help="The agents will research, analyze, write, and review content on this topic"
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            start_button = st.button("🚀 Start Research", type="primary", disabled=not topic)
        
        with col2:
            if st.button("📋 Use Example Topic"):
                topic = "The Future of Renewable Energy in Africa"
                st.rerun()
        
        if start_button and topic:
            st.divider()
            
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            result_placeholder = st.empty()
            
            with st.spinner("Initializing multi-agent system..."):
                success, run_dir, duration, result = run_research_task(
                    topic, progress_placeholder, status_placeholder, selected_model
                )
            
            if success:
                st.success(f"✅ Research completed in {format_hms(duration)}!")
                
                # Display results
                st.divider()
                st.subheader("📝 Final Output")
                
                final_file = run_dir / "final_output.md"
                if final_file.exists():
                    with open(final_file, 'r', encoding='utf-8') as f:
                        final_content = f.read()
                    st.markdown(final_content)
                
                # Display individual tasks
                st.divider()
                st.subheader("🔍 Individual Agent Outputs")
                
                task_files = sorted(run_dir.glob("task-*.txt"))
                for task_file in task_files:
                    display_task_output(task_file)
                
                # Download options
                st.divider()
                st.subheader("💾 Downloads")
                
                col1, col2 = st.columns(2)
                with col1:
                    if final_file.exists():
                        with open(final_file, 'r', encoding='utf-8') as f:
                            st.download_button(
                                "📥 Download Final Report (Markdown)",
                                f.read(),
                                file_name=f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                                mime="text/markdown"
                            )
                
                with col2:
                    summary_file = run_dir / "summary.json"
                    if summary_file.exists():
                        with open(summary_file, 'r', encoding='utf-8') as f:
                            st.download_button(
                                "📥 Download Summary (JSON)",
                                f.read(),
                                file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
            else:
                st.error(f"❌ Error: {result}")
    
    with tab2:
        st.header("View Previous Results")
        
        runs = get_recent_runs(20)
        
        if not runs:
            st.info("No previous runs found. Start a new research task to see results here!")
        else:
            # Select run
            selected_run = st.selectbox(
                "Select a run to view:",
                options=runs,
                format_func=lambda x: f"{x['timestamp']} - {x['topic'][:50]}... ({format_hms(x['duration'])})"
            )
            
            if selected_run:
                run_dir = selected_run['folder']
                
                st.divider()
                
                # Display summary
                summary_file = run_dir / "summary.json"
                if summary_file.exists():
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        summary = json.load(f)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Topic", summary.get('topic', 'Unknown')[:30] + "...")
                    col2.metric("Duration", format_hms(summary.get('duration_seconds', 0)))
                    col3.metric("Tasks", len(summary.get('tasks', [])))
                
                # Display final output
                final_file = run_dir / "final_output.md"
                if final_file.exists():
                    st.subheader("📝 Final Output")
                    with open(final_file, 'r', encoding='utf-8') as f:
                        st.markdown(f.read())
                
                # Display task outputs
                st.divider()
                task_files = sorted(run_dir.glob("task-*.txt"))
                for task_file in task_files:
                    display_task_output(task_file)
    
    with tab3:
        st.header("About This System")
        
        st.markdown("""
        ### 🤖 Multi-Agent AI Research System
        
        This system uses **CrewAI** and **LangChain** to orchestrate multiple AI agents that work together
        to research, analyze, write, and review content on any given topic.
        
        #### The Four Agents:
        
        1. **🔬 Research Agent**
           - Conducts thorough research on the topic
           - Uses web search and scraping tools
           - Gathers facts, statistics, and sources
        
        2. **📊 Analyst Agent**
           - Analyzes the research findings
           - Identifies patterns and trends
           - Provides data-driven insights
        
        3. **✍️ Writer Agent**
           - Creates engaging, well-structured content
           - Synthesizes research and analysis
           - Produces professional articles
        
        4. **✅ Reviewer Agent**
           - Reviews content for quality and accuracy
           - Provides constructive feedback
           - Ensures high standards
        
        #### Technology Stack:
        
        - **CrewAI**: Agent orchestration
        - **LangChain**: LLM integration
        - **OpenAI GPT-4** or **Microsoft Phi-2**: Language models (auto-fallback to free model)
        - **Streamlit**: Web interface
        - **BeautifulSoup**: Web scraping
        - **Python 3.11+**: Core language
        
        #### Features:
        
        - ✅ Sequential agent execution
        - ✅ Real-time progress tracking
        - ✅ Individual task outputs saved
        - ✅ Final report in Markdown
        - ✅ Comprehensive error handling
        - ✅ Cost-efficient token usage
        
        ---
        
        **Version:** 1.0.0  
        **License:** MIT  
        **Repository:** [GitHub](https://github.com/yourusername/multi-agents-ai)
        """)
        
        # Tool status
        st.divider()
        st.subheader("🔧 Tools Status")
        
        from tools import get_tools_info
        tools_info = get_tools_info()
        
        for tool_name, status in tools_info.items():
            if "available" in str(status).lower():
                if "not available" in str(status).lower():
                    st.warning(f"⚠️ **{tool_name}**: {status}")
                else:
                    st.success(f"✅ **{tool_name}**: {status}")
            else:
                st.info(f"ℹ️ **{tool_name}**: {status}")


if __name__ == "__main__":
    main()
