"""The README's code samples import names that exist."""

import ast
import re
import unittest
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"


def python_blocks() -> list[str]:
    return re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), flags=re.S)


class TestReadmeSamples(unittest.TestCase):
    def test_there_are_samples(self) -> None:
        self.assertGreaterEqual(len(python_blocks()), 2)

    def test_every_imported_name_exists(self) -> None:
        checked = 0
        for block in python_blocks():
            for node in ast.parse(block).body:
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("ml_meta_perf"):
                    module = __import__(node.module, fromlist=["_"])
                    for alias in node.names:
                        self.assertTrue(hasattr(module, alias.name), f"{node.module}.{alias.name} is in the README")
                        checked += 1
        self.assertGreater(checked, 5)


if __name__ == "__main__":
    unittest.main()
