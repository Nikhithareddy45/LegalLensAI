"""
Encodes CUAD QA pairs into binary clause labels
"""


class LabelEncoder:
    @staticmethod
    def encode_answer(answer_list):
        """
        CUAD answer format:
        - empty list => clause not present (0)
        - non-empty list => clause present (1)
        """
        if not answer_list:
            return 0
        return 1
