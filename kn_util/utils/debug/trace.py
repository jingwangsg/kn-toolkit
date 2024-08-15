import sys
import os


def trace_lines(filename):
    def _trace_lines(frame, event, arg):
        if event != "line":
            return
        co = frame.f_code
        func_name = co.co_name
        line_no = frame.f_lineno
        line_filename = co.co_filename
        with open(filename, "a") as f:
            print(f"{line_filename}:{line_no} - {func_name}")
            f.write(f"{line_filename}:{line_no} - {func_name}\n")

    return _trace_lines


def set_trace_lines(filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if os.getenv("TRACE_LINES", False):
        print(f"Tracing lines to {filename}")
        sys.settrace(trace_lines(filename))
