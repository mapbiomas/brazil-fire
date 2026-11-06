"""Small helpers for approximate, rate-based progress reporting."""
import math
from M_lang import L


def format_duration(seconds):
    seconds = max(0, int(math.ceil(seconds)))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def progress_message(phase, done, total, elapsed, new_done, new_total, failures=0):
    """Report completed items; ETA uses only successes from this session.

    At least three completed items are required. An export with failures does
    not get a misleading ETA because retries will require a separate run.
    """
    eta = L.ETA_PENDING
    if not failures and new_done >= 3 and elapsed > 0:
        eta = format_duration(elapsed * max(0, new_total - new_done) / new_done)
    return L.ETA_PROGRESS.format(phase=phase, done=done, total=total,
                                 elapsed=format_duration(elapsed), eta=eta)
