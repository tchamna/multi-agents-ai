"""
Main script to run the multi-agent AI system using CrewAI and LangChain.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from crewai import Crew, Process
from tasks import create_research_task, create_writing_task, create_review_task, create_analysis_task
from agents import researcher, writer, reviewer, analyst


def main():
    """Main function to orchestrate the multi-agent system."""
    
    # Load environment variables
    load_dotenv()
    
    # Verify API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with your OpenAI API key.")
        return
    
    # Define the topic for the crew to work on
    topic = input("Enter a topic for the AI agents to research and analyze: ").strip()
    
    if not topic:
        topic = "The Impact of Artificial Intelligence on Modern Healthcare"
        print(f"\nUsing default topic: {topic}\n")
    
    print(f"\n{'='*60}")
    print(f"Starting Multi-Agent AI System")
    print(f"Topic: {topic}")
    print(f"{'='*60}\n")
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {run_dir}\n")
    
    # Create tasks
    research_task = create_research_task(topic)
    analysis_task = create_analysis_task(topic)
    writing_task = create_writing_task(topic)
    review_task = create_review_task()
    
    # Store task metadata
    tasks_info = [
        {"name": "research", "description": "Research findings", "agent": "researcher", "task": research_task},
        {"name": "analysis", "description": "Data analysis and insights", "agent": "analyst", "task": analysis_task},
        {"name": "writing", "description": "Written article", "agent": "writer", "task": writing_task},
        {"name": "review", "description": "Quality review and feedback", "agent": "reviewer", "task": review_task}
    ]
    
    # Create the crew
    crew = Crew(
        agents=[researcher, analyst, writer, reviewer],
        tasks=[research_task, analysis_task, writing_task, review_task],
        process=Process.sequential,  # Tasks will be executed in order
        verbose=True
    )
    
    # Execute the crew
    print("\nCrew is starting work...\n")
    start_time = datetime.now()
    result = crew.kickoff()
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Display results
    print("\n" + "="*60)
    print("FINAL RESULT")
    print("="*60 + "\n")
    print(result)
    
    # Save individual task outputs
    task_outputs = []
    
    # Try to extract individual task results from the crew's task_output attribute
    if hasattr(crew, 'tasks'):
        for i, (task_info, task) in enumerate(zip(tasks_info, crew.tasks), 1):
            task_output_data = {
                "task_number": i,
                "task_name": task_info["name"],
                "description": task_info["description"],
                "agent": task_info["agent"],
                "timestamp": datetime.now().isoformat()
            }
            
            # Try to get task output
            if hasattr(task, 'output') and task.output:
                output_text = str(task.output)
                task_output_data["output_file"] = f"task-{i}-{task_info['name']}.txt"
                task_output_data["output_length"] = len(output_text)
                
                # Save individual task output
                task_file = run_dir / f"task-{i}-{task_info['name']}.txt"
                with open(task_file, "w", encoding="utf-8") as f:
                    f.write(f"# Task {i}: {task_info['description']}\n")
                    f.write(f"# Agent: {task_info['agent']}\n")
                    f.write(f"# Timestamp: {task_output_data['timestamp']}\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(output_text)
                
                print(f"Saved: {task_file}")
            else:
                task_output_data["output_file"] = None
                task_output_data["output_length"] = 0
            
            task_outputs.append(task_output_data)
    
    # Save final result as markdown
    final_output_file = run_dir / "final_output.md"
    with open(final_output_file, "w", encoding="utf-8") as f:
        f.write(f"# Multi-Agent AI System Output\n\n")
        f.write(f"**Topic:** {topic}\n\n")
        f.write(f"**Generated:** {timestamp}\n\n")
        f.write(f"**Duration:** {duration:.2f} seconds\n\n")
        f.write("---\n\n")
        f.write(str(result))
    
    # Save metadata summary as JSON
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
    
    print(f"\n{'='*60}")
    print("OUTPUT FILES")
    print("="*60)
    print(f"Final output (MD):  {final_output_file}")
    print(f"Summary (JSON):     {summary_file}")
    print(f"Run directory:      {run_dir}")
    print(f"Duration:           {duration:.2f} seconds")
    print("="*60)


if __name__ == "__main__":
    main()
