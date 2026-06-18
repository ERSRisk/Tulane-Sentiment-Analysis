"""
Nightly pipeline orchestrator.
"""

from src.topics.bertopic_update import run_bertopic_update

from src.stories.build_stories import run_story_build

def main():
    print("Starting nightly risk intelligence pipeline...")
    print("Step 1: Updating BERTopic risk intelligence outputs...")
    run_bertopic_update()
    print("Step 2: Building story clusters and dashboard outputs...")
    run_story_build()

    print("Nightly pipeline completed.")

if __name__ == "__main__":
    main()