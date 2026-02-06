"""
Builds model-ready sequences
"""

from src.data_preprocessing.text_cleaner import TextCleaner


class SequenceBuilder:
    @staticmethod
    def build(context, question):
        context = TextCleaner.clean(context)
        question = TextCleaner.clean(question)

        return f"question: {question} context: {context}"
