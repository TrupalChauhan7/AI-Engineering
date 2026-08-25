"""Load and render versioned prompt templates from prompts/.

WHY: prompts are CODE. Versioning them (v1.0, v1.1, ...) lets us prove which
prompt produced which result — essential for the viva and A2/A3. Templates use
``## System`` / ``## User`` markdown sections and ``{{placeholders}}``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from s2n.config import ROOT

# Split a prompt .md into its "## System" / "## User" sections.
_SECTION_RE = re.compile(r"^##\s+(System|User)\s*$", re.IGNORECASE | re.MULTILINE)


@dataclass
class Prompt:
    system: str
    user: str

    def render(self, **fields: str) -> "Prompt":
        """Substitute {{placeholders}} in both sections."""
        return Prompt(_fill(self.system, fields), _fill(self.user, fields))


def _fill(text: str, fields: dict[str, str]) -> str:
    for key, value in fields.items():
        text = text.replace("{{" + key + "}}", str(value))
    return text


def load_prompt(kind: str, version: str) -> Prompt:
    """kind = 'note_generation' or 'judge'; version = 'v1.0'.

    Parses the ``## System`` and ``## User`` sections; any leading prose above
    the first section header (title, versioning note) is ignored.
    """
    path = ROOT / "prompts" / kind / f"{version}.md"
    text = path.read_text()

    sections: dict[str, str] = {}
    parts = _SECTION_RE.split(text)
    # parts = [preamble, name1, body1, name2, body2, ...]
    for name, body in zip(parts[1::2], parts[2::2]):
        sections[name.lower()] = body.strip()

    if "user" not in sections:
        raise ValueError(f"Prompt {kind}/{version} has no '## User' section.")
    return Prompt(system=sections.get("system", ""), user=sections["user"])
