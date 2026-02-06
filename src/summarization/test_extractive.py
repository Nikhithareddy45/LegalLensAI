from src.summarization.extractive import ExtractiveSummarizer

TEXT = """
This Agreement may be terminated by either party upon written notice.
The Client shall pay all outstanding dues before termination.
Confidential information must not be disclosed.
Liability is limited to the amount paid under this Agreement.
Governing law shall be the laws of India.
"""

summarizer = ExtractiveSummarizer(top_k=3)
summary = summarizer.summarize(TEXT)

print("\n--- SUMMARY ---")
for s in summary:
    print("-", s)
