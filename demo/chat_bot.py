import argparse
import json

from reactagenthlt.agent import Agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="filepath to model config")
    parser.add_argument("--think", action="store_true", help="enable agent thinking")
    args = parser.parse_args()

    config_file = open(args.config, "r")
    config = json.load(config_file)
    config_file.close()

    agent = Agent(config, thinking=args.think)
    while True:
        prompt = input("> ")
        response = agent.ask(prompt)

        print(f"Thinking: {response.thinking}")
        print(f"Content: {response.content}")
        print(response.stats)
