# Import necessary libraries
from ai_researcher import CycleResearcher
from ai_researcher.utils import print_paper_summary

# Initialize CycleResearcher with the default 12B model
# Use 4-bit quantization to fit in 12GB VRAM
researcher = CycleResearcher(
    model_size="12B", 
    gpu_memory_utilization=0.9, 
    quantization="bitsandbytes", 
    load_format="bitsandbytes"
)

# Load references from BibTeX file
with open('Tutorial/cycleresearcher_references.bib', 'r') as f:
    references_content = f.read()

# Generate a paper with specific references
generated_papers = researcher.generate_paper(
    topic = "AI Researcher",
    references = references_content,
    n = 1  # Generate a single paper
)

# Print summary of generated paper
print_paper_summary(generated_papers[0])