class TokenCounter:
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0

    def add(self, input_tokens, output_tokens):
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def totals(self):
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }

    def show(self):
        print(f"Input tokens: {self.input_tokens:,}")
        print(f"Output tokens: {self.output_tokens:,}")