"""Run a small, reproducible qualitative showcase of Calin-GPT."""

from datetime import datetime
from pathlib import Path
import subprocess
import sys

TESTS = [
    ("Story opening", "Once upon a time", 0.5),
    ("Story opening, more randomness", "Once upon a time", 0.9),
    ("Historical knowledge", "The history of the Roman Empire", 0.5),
    ("Science explanation", "The reason the sky appears blue is", 0.5),
    ("Technology explanation", "A computer works by", 0.6),
    ("Geography", "France is a country in Europe. Its capital city is", 0.5),
    ("Instructions", "How to write a letter:\n1.", 0.6),
    ("Dialogue", '"Where are you going?" she asked.\n"', 0.6),
    ("Python code", "def fibonacci(n):\n", 0.5),
    ("Long-form coherence", "In a small village near the mountains, a young scientist discovered", 0.7),
]

OUTPUT = Path("results/model_showcase.txt")
MAX_NEW_TOKENS = 120


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Calin-GPT — Qualitative Generation Showcase",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Python: {sys.version.split()[0]}",
        f"Max new tokens per test: {MAX_NEW_TOKENS}",
        "Note: Samples are stochastic; results are not a quantitative benchmark.",
        "",
    ]

    for index, (name, prompt, temperature) in enumerate(TESTS, 1):
        command = [
            sys.executable, "-m", "src.generate", "--gpu",
            "--prompt", prompt,
            "--temperature", str(temperature),
            "--max-new-tokens", str(MAX_NEW_TOKENS),
        ]
        print(f"[{index}/{len(TESTS)}] {name}", flush=True)
        result = subprocess.run(command, capture_output=True, text=True, errors="replace")
        lines.extend([
            "=" * 72,
            f"TEST {index:02d}: {name}",
            f"Prompt: {prompt!r}",
            f"Temperature: {temperature}",
            f"Exit code: {result.returncode}",
            "-" * 72,
            result.stdout.rstrip() or "(no output)",
        ])
        if result.stderr.strip():
            lines.extend(["STDERR:", result.stderr.rstrip()])
        lines.append("")
        OUTPUT.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nSaved all results to {OUTPUT}")


if __name__ == "__main__":
    main()
