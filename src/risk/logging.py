"""Structured risk logging helper"""
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, UTC

LOGGER = logging.getLogger(__name__)


# Launch Punch List – Quick Win #1: structured risk logging
def log_risk_event(
    event_type: str,
    payload: Dict[str, Any],
    logger: Optional[List[Dict]] = None,
    timestamp: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Log a structured risk event.
    
    Args:
        event_type: Type of event (e.g., "trim_fail", "es_block", "daily_kill", "beta_block", "state_change")
        payload: Event-specific data (symbol, regime, numeric details, etc.)
        logger: Optional list to append to (e.g., forensic_log)
        timestamp: Optional timestamp (defaults to now)
    
    Returns:
        Structured log entry dict
    """
    if timestamp is None:
        timestamp = datetime.now(UTC)
    
    log_entry = {
        'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
        'event_type': event_type,
        **payload
    }
    
    # Write to logger if provided
    if logger is not None:
        logger.append(log_entry)
    
    # Also emit structured JSON for immediate visibility
    LOGGER.info("%s: %s", event_type, json.dumps(payload, default=str))
    
    return log_entry

