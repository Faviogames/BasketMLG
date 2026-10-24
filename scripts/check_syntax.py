import py_compile
try:
    py_compile.compile('core/live_api.py', doraise=True)
    print("Syntax OK")
except Exception as e:
    print(f"Syntax Error: {e}")
