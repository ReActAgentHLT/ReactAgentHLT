import json

from reactagenthlt import tools

if __name__ == "__main__":
    se = tools.SearchEngine(k=2)
    results = se.search("something")

    print(json.dumps(results, indent=2))
