from abc import ABC, abstractmethod

from reactagenthlt.agents.language_model import Response


class Agent(ABC):
    @abstractmethod
    def add_sys_prompt(self, content: str) -> None: ...

    @abstractmethod
    def ask(self, content: str) -> Response: ...

    @abstractmethod
    def total_usage(self) -> int: ...

    @abstractmethod
    def clear_context(self) -> None: ...
