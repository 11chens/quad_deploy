import time
from typing import List, Dict, Callable, Any, Tuple

from ros_base.agents.base_agent import BaseAgent

# ---------------------------------------------------------
# Default Configurations (Macro Definitions)
# ---------------------------------------------------------
DEFAULT_AUTO_RELEASE_THRESHOLD = 0.25
DEFAULT_AUTO_RELEASE_DURATION = 1.0
DEFAULT_AUTO_RELEASE_COOLDOWN = 5.0
DEFAULT_AUTO_RELEASE_FILTER_ALPHA = 0.1  # EMA filter coefficient (1.0 = raw, 0.1 = strong smoothing)
# ---------------------------------------------------------

class BaseCondition:
    """
    Base condition class. All custom conditions should inherit from this and implement the evaluate method.
    """
    def __init__(self, name: str):
        self.name = name

    def evaluate(self, **state_kwargs) -> Tuple[bool, str]:
        """
        Evaluate the condition based on the provided state dictionary.
        Returns:
            tuple: (bool: True if condition is met, str: debug/log message)
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset any internal timers or states.
        """
        pass

class ValueDurationCondition(BaseCondition):
    """
    Condition that triggers when a value stays below/above/equal to a threshold for a duration.
    """
    def __init__(self, name: str, state_key: str, threshold: float, duration: float, mode: str = "less_than", filter_alpha: float = 1.0):
        super().__init__(name)
        self.state_key = state_key          # State variable name (e.g., 'lin_vel_norm')
        self.threshold = threshold          # Threshold value
        self.duration = duration            # Duration in seconds
        self.mode = mode                    # 'less_than', 'greater_than', 'equal'
        self.filter_alpha = filter_alpha    # EMA Filter coefficient
        
        self.filtered_val = None
        self.condition_met_start_time = None

    def evaluate(self, **state_kwargs) -> Tuple[bool, str]:
        raw_val = state_kwargs.get(self.state_key, None)
        
        # If the expected state is missing, interrupt accumulated condition
        if raw_val is None:
            self.condition_met_start_time = None
            self.filtered_val = None
            return False, f"{self.state_key} not found"
            
        # Apply Exponential Moving Average (EMA) Filter
        if self.filtered_val is None:
            self.filtered_val = raw_val
        else:
            self.filtered_val = (1.0 - self.filter_alpha) * self.filtered_val + self.filter_alpha * raw_val
            
        val = self.filtered_val

        # Determine if the current instantaneous state satisfies the logic
        if self.mode == "less_than":
            met = val < self.threshold
        elif self.mode == "greater_than":
            met = val > self.threshold
        elif self.mode == "equal":
            met = val == self.threshold
        else:
            met = False

        # Time accumulation logic
        if met:
            current_time = time.time()
            if self.condition_met_start_time is None:
                self.condition_met_start_time = current_time
            
            # If the condition is met for longer than the required duration, return True
            elapsed = current_time - self.condition_met_start_time
            if elapsed >= self.duration:
                return True, f"DONE"
            
            return False, f"[ OK ] {elapsed:.1f}/{self.duration}s (v={val:.3f})"
        else:
            self.reset()
            return False, f"[ XX ] (v={val:.3f})"

    def reset(self):
        self.condition_met_start_time = None

class AutoTriggerAgent(BaseAgent):
    """
    Agent to automatically trigger events based on external states. 
    Can be flexibly configured with multiple rules.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Structure: { "event_name" : [condition, condition2, ...] }
        # AND logic is applied by default (all conditions must be true to trigger)
        self.triggers: Dict[str, List[BaseCondition]] = {}
        self.triggers_cooldown: Dict[str, float] = {}
        self.last_trigger_times: Dict[str, float] = {}
        
        # Load default triggers from macros
        self._load_default_triggers()

    def _load_default_triggers(self):
        """
        Register default triggers based on macro definitions.
        """
        release_conds = [
            ValueDurationCondition(
                name="vel_stop", 
                state_key="lin_vel_norm", 
                threshold=DEFAULT_AUTO_RELEASE_THRESHOLD, 
                duration=DEFAULT_AUTO_RELEASE_DURATION, 
                mode="less_than",
                filter_alpha=DEFAULT_AUTO_RELEASE_FILTER_ALPHA
            )
        ]
        self.register_trigger("auto_release", release_conds, cooldown=DEFAULT_AUTO_RELEASE_COOLDOWN)

    def register_trigger(self, trigger_name: str, conditions: List[BaseCondition], cooldown: float = 0.0):
        """
        Register a set of conditions for a specific action (e.g., 'auto_release').
        """
        self.triggers[trigger_name] = conditions
        self.triggers_cooldown[trigger_name] = cooldown
        self.last_trigger_times[trigger_name] = 0.0

    def evaluate_triggers(self, **state_kwargs) -> Dict[str, bool]:
        """
        To be called in each step/cycle. Takes external states and returns triggered events.
        Example: {'auto_release': True, 'auto_grasp': False}
        """
        results = {}
        for trigger_name, conditions in self.triggers.items():
            results[trigger_name] = False
            
            # Check cooldown duration
            current_time = time.time()
            if (current_time - self.last_trigger_times.get(trigger_name, 0.0)) < self.triggers_cooldown.get(trigger_name, 0.0):
                continue
            
            # AND logic: all conditions must evaluate to True
            all_met = True
            debug_msgs = []
            for cond in conditions:
                met, msg = cond.evaluate(**state_kwargs)
                debug_msgs.append(f"[{cond.name}] {msg}")
                if not met:
                    all_met = False
            
            # Log debug info periodically to avoid terminal flooding
            msg_str = " | ".join(debug_msgs)
            if self.logger:
                if hasattr(self.logger, "log_throttle"):
                    self.logger.log_throttle(f"[Auto] {trigger_name}: {msg_str}", seconds=1.0)
                else:
                    self.logger.info(f"[Auto] {trigger_name}: {msg_str}")
            
            if all_met and len(conditions) > 0:
                results[trigger_name] = True
                self.last_trigger_times[trigger_name] = current_time
                self.reset_conditions(trigger_name)
                
                trigger_msg = f"AutoTrigger Event '{trigger_name}' TRIGGERED! ({msg_str})"
                if self.logger and hasattr(self.logger, "important"):
                    self.logger.important(trigger_msg)
                elif self.logger:
                    self.logger.info(trigger_msg)
                else:
                    print(trigger_msg)

        return results

    def reset_conditions(self, trigger_name: str = None):
        """
        Reset timers and states of conditions.
        """
        if trigger_name and trigger_name in self.triggers:
            for cond in self.triggers[trigger_name]:
                cond.reset()
        else:
            for conds in self.triggers.values():
                for cond in conds:
                    cond.reset()

    def reset(self):
        # Call BaseAgent's reset if it exists
        super().reset()
        self.reset_conditions()
