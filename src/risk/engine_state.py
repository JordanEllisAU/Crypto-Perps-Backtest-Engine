"""Global trading state machine for risk and technical halts"""
import json
import logging
from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
from engine_core.src.risk.logging import log_risk_event

LOGGER = logging.getLogger(__name__)


class TradingState(Enum):
    """Trading state enumeration"""
    RUNNING = "RUNNING"
    RISK_HALT = "RISK_HALT"
    TECH_HALT = "TECH_HALT"
    NEUTRAL_ONLY = "NEUTRAL_ONLY"


class EngineStateManager:
    """Manages global trading state machine"""
    
    def __init__(self, initial_state: TradingState = TradingState.RUNNING):
        self.current_state = initial_state
        self.state_history: List[Dict[str, Any]] = []
        self._last_reason = None
        self._last_transition_ts = None
    
    def get_state(self) -> TradingState:
        """Get current trading state"""
        return self.current_state
    
    def set_state(self, new_state: TradingState, reason: str, timestamp: Optional[datetime] = None):
        """
        Transition to new state with logging
        
        Args:
            new_state: Target state
            reason: Reason for transition (e.g., "risk:daily_loss", "tech:api_errors")
            timestamp: Optional timestamp (defaults to now)
        """
        if timestamp is None:
            from datetime import datetime, UTC
            timestamp = datetime.now(UTC)
        
        previous_state = self.current_state
        self.current_state = new_state
        self._last_reason = reason
        self._last_transition_ts = timestamp
        
        # Record transition
        transition_record = {
            'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
            'previous_state': previous_state.value,
            'new_state': new_state.value,
            'reason': reason
        }
        self.state_history.append(transition_record)
        
        # Launch Punch List – Blocker #5: global trading state machine
        # Launch Punch List – Quick Win #1: structured risk logging
        log_risk_event(
            'state_change',
            {
                'previous_state': previous_state.value,
                'new_state': new_state.value,
                'reason': reason
            },
            timestamp=timestamp if hasattr(timestamp, 'isoformat') else None
        )
    
    def can_trade(self, module: Optional[str] = None) -> bool:
        """
        Check if trading is allowed
        
        Args:
            module: Optional module name (engine-agnostic: not used for filtering)
        
        Returns:
            True if trading is allowed, False otherwise
        """
        if self.current_state == TradingState.RUNNING:
            return True
        elif self.current_state == TradingState.NEUTRAL_ONLY:
            # Engine-agnostic: allow all trading in NEUTRAL_ONLY state
            # Strategy-specific module filtering removed
                return True
        else:
            # RISK_HALT or TECH_HALT - no trading allowed
            return False
    
    def save_state(self, filepath: str):
        """Save state to JSON file"""
        state_data = {
            'current_state': self.current_state.value,
            'last_reason': self._last_reason,
            'last_transition_ts': self._last_transition_ts.isoformat() if self._last_transition_ts else None,
            'state_history': self.state_history[-10:]  # Keep last 10 transitions
        }
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(state_data, f, indent=2)
    
    def load_state(self, filepath: str) -> bool:
        """
        Load state from JSON file
        
        Returns:
            True if state was loaded, False if file doesn't exist
        """
        path = Path(filepath)
        if not path.exists():
            return False
        
        try:
            with open(path, 'r') as f:
                state_data = json.load(f)
            
            # Restore state
            self.current_state = TradingState(state_data.get('current_state', 'RUNNING'))
            self._last_reason = state_data.get('last_reason')
            if state_data.get('last_transition_ts'):
                from datetime import datetime
                self._last_transition_ts = datetime.fromisoformat(state_data['last_transition_ts'])
            
            # Restore recent history
            self.state_history = state_data.get('state_history', [])
            
            return True
        except Exception as e:
            LOGGER.warning("Failed to load state from %s: %s", filepath, e)
            return False
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information"""
        return {
            'current_state': self.current_state.value,
            'last_reason': self._last_reason,
            'last_transition_ts': self._last_transition_ts.isoformat() if self._last_transition_ts else None,
            'can_trade': self.can_trade()
        }

