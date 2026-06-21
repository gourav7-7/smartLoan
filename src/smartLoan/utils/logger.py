# ─────────────────────────────────────────────────────────────────────────────
# utils/logger.py
#
# Reads log level, format, and file path from settings (.env) instead of
# hardcoding them. Falls back to safe defaults if settings can't be loaded.
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import logging

try:
    from smartLoan.config.settings import settings
    _log_level  = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    _log_format = settings.LOG_FORMAT
    _log_file   = settings.LOG_FILE
except Exception:
    # Fallback — keeps logger usable even before settings are fully wired
    _log_level  = logging.INFO
    _log_format = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"
    _log_file   = "logs/running_logs.log"

# Ensure log directory exists
os.makedirs(os.path.dirname(_log_file), exist_ok=True)

logging.basicConfig(
    level=_log_level,
    format=_log_format,
    handlers=[
        logging.FileHandler(_log_file),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("smartLoanLogger")
