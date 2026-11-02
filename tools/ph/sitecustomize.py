# rename every python process started with PROC_TITLE set (and its children, which inherit the env)
import os
if os.environ.get("PROC_TITLE"):
    try:
        import setproctitle
        setproctitle.setproctitle(os.environ["PROC_TITLE"])
    except Exception:
        pass
