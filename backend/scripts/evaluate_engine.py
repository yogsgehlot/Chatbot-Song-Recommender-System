"""
Evaluate the recommendation engine offline using academic metrics.

Usage:
    cd backend
    .venv\\Scripts\\python scripts\\evaluate_engine.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.recommender.engine import RecommendationEngine
from app.recommender.evaluator import RecommendationEvaluator


def main() -> None:
    dataset_path = "./data/songs.csv"
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}. Run generate_sample_data.py first.")
        sys.exit(1)

    print("Loading recommendation engine...")
    engine = RecommendationEngine.load(dataset_path)

    print("Initializing evaluator...")
    evaluator = RecommendationEvaluator(engine)

    print(f"\nRunning evaluation across {len(evaluator.engine._df)} songs in catalog...")
    metrics = evaluator.run_evaluation(n=5)

    print("\n" + "="*50)
    print("MusicMind Recommendation Engine Evaluation Report")
    print("="*50)
    print(f"Catalog Size:      {metrics['catalog_size']} songs")
    print(f"Test Profiles:     {metrics['test_profiles_count']} unique personas")
    print("-"*50)
    print(f"Catalog Coverage:  {metrics['catalog_coverage'] * 100:.2f}%")
    print(f"Mean Diversity:    {metrics['mean_diversity']:.4f}")
    print(f"Personalization:   {metrics['personalization'] * 100:.2f}%")
    print(f"Avg Popularity:    {metrics['average_popularity']:.2f} / 100")
    print("="*50)
    print("\nMetrics Definitions:")
    print("- Catalog Coverage: % of total catalog recommended at least once.")
    print("- Mean Diversity: Pairwise distance of song features (higher = more varied styles).")
    print("- Personalization: Uniqueness of recommendations across different users.")
    print("- Avg Popularity: Average popularity score of recommended tracks.")
    print("="*50)


if __name__ == "__main__":
    main()
