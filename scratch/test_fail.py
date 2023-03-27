from pathlib import Path

try:
    exec(Path("fail.py").read_text())
except SyntaxError as e:
    print(f"Caught exception on line {e.lineno}")
except:
    print("Code failure")
