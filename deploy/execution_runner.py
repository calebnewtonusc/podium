"""
Execution runner — runs inside the Docker sandbox for CV-RL reward computation.
Receives code via stdin, executes it, returns CV score as JSON.
"""

import json
import sys
import traceback


def run_user_code(code: str) -> dict:
    """Execute user-generated code and capture cv_score variable."""
    namespace = {}
    try:
        exec(compile(code, "<generated>", "exec"), namespace)
        cv_score = namespace.get("cv_score")
        if cv_score is None:
            return {"success": False, "error": "cv_score variable not set"}
        return {"success": True, "cv_score": float(cv_score)}
    except SystemExit as e:
        # sys.exit() inside generated code should be treated as a failure, not propagate
        return {"success": False, "error": f"Generated code called sys.exit({e.code})"}
    except Exception as e:
        tb = traceback.format_exc()
        return {"success": False, "error": str(e), "traceback": tb[-500:]}


if __name__ == "__main__":
    code = sys.stdin.read()
    result = run_user_code(code)
    print(json.dumps(result))
