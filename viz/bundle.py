"""Bundle renderer + script.json into a single self-contained HTML file."""

from __future__ import annotations

import sys
from pathlib import Path

RENDERER_DIR = Path(__file__).parent / "renderer"
JS_FILES = ["room.js", "character.js", "sound.js", "ui.js", "script.js", "engine.js"]


def _script_tag_safe_json(raw_json: str) -> str:
    """Escape HTML-significant characters inside embedded JSON script tags."""
    return (
        raw_json.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    )


def bundle(script_json_path: str, output_path: str | None = None) -> None:
    script_path = Path(script_json_path)
    if not script_path.exists():
        print(f"Error: {script_path} not found")
        sys.exit(1)

    script_data = script_path.read_text()
    safe_script_data = _script_tag_safe_json(script_data)

    # Read HTML template
    html_path = RENDERER_DIR / "index.html"
    html = html_path.read_text()

    # Collect all JS
    all_js = []
    for js_file in JS_FILES:
        js_path = RENDERER_DIR / js_file
        if not js_path.exists():
            print(f"Warning: {js_path} not found, skipping")
            continue
        all_js.append(f"// === {js_file} ===\n" + js_path.read_text())
    combined_js = "\n\n".join(all_js)

    # Replace script-data placeholder with actual data
    html = html.replace(
        '<script id="script-data" type="application/json">null</script>',
        f'<script id="script-data" type="application/json">{safe_script_data}</script>',
    )

    # Replace separate script tags with single inline block
    for js_file in JS_FILES:
        html = html.replace(f'<script src="{js_file}"></script>', "")
    html = html.replace(
        "</body>",
        f"<script>\n{combined_js}\n</script>\n</body>",
    )

    # Output
    if output_path is None:
        output_path = str(script_path.parent / "replay.html")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    size_kb = out.stat().st_size / 1024
    print(f"Bundled → {out} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python viz/bundle.py viz/output/<id>/script.json [output.html]")
        sys.exit(1)
    out = sys.argv[2] if len(sys.argv) > 2 else None
    bundle(sys.argv[1], out)
