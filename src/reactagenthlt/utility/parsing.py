import re


def parse_thinking(content: str) -> list[str]:
    match = re.search(r"<think>(.*?)</think>(.*)", content, flags=re.DOTALL)
    if match:
        return [match.group(1).strip(), match.group(2).strip()]

    return ["", content.strip()]
