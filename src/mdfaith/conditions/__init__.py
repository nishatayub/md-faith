from .base import Condition, Explanation, Task
from .image_only import ImageOnly
from .qc_gated import QCGated
from .table_only import TableOnly
from .tool_agent import ToolAgent

CONDITIONS = {"image_only": ImageOnly, "table_only": TableOnly, "tool_agent": ToolAgent, "qc_gated": QCGated}

__all__ = ["CONDITIONS", "Condition", "Explanation", "ImageOnly", "TableOnly", "Task", "ToolAgent"]
