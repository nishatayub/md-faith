from .base import Condition, Explanation, Task
from .image_only import ImageOnly
from .table_only import TableOnly
from .tool_agent import ToolAgent

CONDITIONS = {"image_only": ImageOnly, "table_only": TableOnly, "tool_agent": ToolAgent}

__all__ = ["Condition", "Explanation", "Task", "ImageOnly", "TableOnly", "ToolAgent", "CONDITIONS"]
