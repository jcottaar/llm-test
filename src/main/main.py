import sys
import os

from lib.lib import load_model, prompt_model


def main():
    tokenizer, model = load_model()
    
    question = "What is 3+5?"
    print(f"\nQuestion: {question}")
    answer = prompt_model(tokenizer, model, question)
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()