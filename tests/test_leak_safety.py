"""Structural firewall (project rule 1): the generation path can NEVER reach
the gold answers.

We assert this two ways:
  1. Import-level (AST): no generation-path module imports the evaluation
     loader. Docstrings may *mention* it for cross-reference; only real
     ``import`` statements count.
  2. Object-level: a GenerationLoader built from config exposes no path that
     points inside notes/ or human_eval/.

If a future edit tries to feed answers into the generator, one of these fails.
"""

import ast
import inspect
from pathlib import Path

from s2n.data import generation_data, textgrid
from s2n.data.generation_data import GenerationLoader
from s2n.generation import generator, prompts

# Every module the generator is allowed to depend on.
GENERATION_PATH_MODULES = [generation_data, textgrid, generator, prompts]

# Modules that hold the gold answers — off-limits to the generation path.
FORBIDDEN_IMPORTS = {"evaluation_data", "s2n.data.evaluation_data"}


def _imported_names(module) -> set[str]:
    tree = ast.parse(inspect.getsource(module))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
                names.update(f"{node.module}.{a.name}" for a in node.names)
    return names


def test_generation_never_imports_the_answer_loader():
    for module in GENERATION_PATH_MODULES:
        imported = _imported_names(module)
        leaks = {name for name in imported if name.split(".")[-1] == "evaluation_data"}
        leaks |= imported & FORBIDDEN_IMPORTS
        assert not leaks, f"{module.__name__} imports answer module(s): {leaks}"


def test_generation_loader_paths_stay_inside_transcripts():
    loader = GenerationLoader.from_config()
    for value in vars(loader).values():
        if isinstance(value, Path):
            parts = set(value.parts)
            assert "notes" not in parts
            assert "human_eval_data" not in parts
