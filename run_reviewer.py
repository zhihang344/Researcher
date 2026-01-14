# Import necessary libraries
from ai_researcher import CycleReviewer

# Initialize CycleReviewer with the default 8B model
reviewer = CycleReviewer(model_size="8B")

# Review a paper (assuming paper_text contains the paper content)
review_results = reviewer.evaluate(paper_text)

# Print review results
print(f"Average score: {review_results[0]['avg_rating']}")
print(f"Decision: {review_results[0]['paper_decision']}")