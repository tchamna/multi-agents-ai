"""Headless runner that saves outputs like app.py does.

Executes the multi-agent crew and saves results to runs/ folder.
"""
import sys
import json
from datetime import datetime
from pathlib import Path
from crewai import Crew, Process
from tasks import create_research_task, create_writing_task, create_review_task, create_analysis_task
from agents import make_agents


def format_hms(seconds):
    """Format seconds as HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def run_and_save(topic: str):
    """Run the multi-agent crew and save all outputs."""
    print(f"Starting headless multi-agent run with topic: {topic}")
    
    # Create fresh agent instances per run
    researcher, writer, reviewer, analyst = make_agents()
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {run_dir}")
    
    # Detect if this is a news query
    topic_lower = topic.lower()
    is_news_query = any(keyword in topic_lower for keyword in 
                       ['news', 'latest', 'today', 'current events', 'breaking', 'headlines', 
                        'recent events', 'what happened', 'whats happening'])
    
    # Create tasks
    research_task = create_research_task(topic, agent=researcher)
    writing_task = create_writing_task(topic, agent=writer)
    
    # For news queries, use 2-agent workflow; for standard research, use all 4
    if is_news_query:
        print("📰 News query detected - using streamlined 2-agent workflow")
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
    
    # Execute
    print("Kicking off crew...")
    start_time = datetime.now()
    result = crew.kickoff()
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n✅ Kickoff duration: {format_hms(duration)}")
    
    # Save outputs
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
                print(f"  ✓ Saved {task_file.name}")
    
    # Save final output - use the article from the writer, not the reviewer's feedback
    final_output_file = run_dir / "final_output.md"
    with open(final_output_file, "w", encoding="utf-8") as f:
        f.write(f"# {topic.title()}\n\n")
        f.write(f"**Generated:** {timestamp}\n")
        f.write(f"**Duration:** {format_hms(duration)}\n\n")
        f.write("---\n\n")
        # Use the writer's article if available, otherwise fall back to final result
        content_to_save = article_content if article_content else str(result)
        f.write(content_to_save)
    print(f"  ✓ Saved {final_output_file.name}")
    
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
    print(f"  ✓ Saved {summary_file.name}")
    
    print(f"\n🎉 Complete! Results saved to: {run_dir}")
    return run_dir, duration


if __name__ == '__main__':
    topic = sys.argv[1] if len(sys.argv) > 1 else "The environmental impact of electric vehicles"
    run_and_save(topic)
