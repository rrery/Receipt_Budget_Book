"""Static syntax and DOM contract checks, distinct from browser visual QA."""
from html.parser import HTMLParser
from pathlib import Path
import re

from tree_sitter import Language, Parser
import tree_sitter_javascript

ROOT = Path(__file__).resolve().parents[1]


class IDs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []

    def handle_starttag(self, tag, attrs):
        self.ids.extend(value for key, value in attrs if key == "id")


source = (ROOT / "frontend/app.js").read_bytes()
parser = Parser(Language(tree_sitter_javascript.language()))
for script in (ROOT / "frontend").glob("*.js"):
    assert not parser.parse(script.read_bytes()).root_node.has_error, f"JavaScript syntax error: {script.name}"
html = IDs()
html.feed((ROOT / "frontend/index.html").read_text(encoding="utf-8"))
assert len(html.ids) == len(set(html.ids)), "Duplicate DOM IDs"
referenced = set(re.findall(r"\$\('([^']+)'\)", source.decode()))
assert not referenced - set(html.ids), f"Missing DOM IDs: {referenced - set(html.ids)}"
print(f"JavaScript syntax OK; {len(referenced)} referenced DOM IDs exist")
