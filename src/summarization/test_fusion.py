from src.summarization.fusion import FusionSummarizer

TEXT = """
This Agreement may be terminated by either party upon written notice.
The Client shall pay all outstanding dues before termination.
Confidential information must not be disclosed.
Liability is limited to the amount paid under this Agreement.
Governing law shall be the laws of India.
"""

summarizer = FusionSummarizer()
summary = summarizer.summarize(TEXT)

print("\n--- FUSION SUMMARY ---")
print(summary)
