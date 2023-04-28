"""Tool for searching the web."""

from langchain.agents import Tool
from steamship import Steamship
from steamship_langchain.tools import SteamshipSERP

NAME = "Search"

DESCRIPTION = """
Useful for when you need to answer questions about current events
"""


class SearchTool(Tool):
    """Tool used to schedule reminders via the Steamship Task system."""

    search: SteamshipSERP

    def __init__(self, client: Steamship):
        super().__init__(
            name=NAME,
            func=self.run,
            description=DESCRIPTION,
            search=SteamshipSERP(client=client),
        )

    def run(self, prompt: str, **kwargs) -> str:
        """Respond to LLM prompts."""
        return self.search.search(prompt)


if __name__ == "__main__":
    with Steamship.temporary_workspace() as client:
        my_tool = SearchTool(client)
        result = my_tool.run("What's the weather today?")
        print(result)