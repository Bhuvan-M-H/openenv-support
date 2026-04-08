"""
grader.py — Per-task scoring functions for the Support Ticket RL Environment.

All functions return a float in (0.0, 1.0).
Called by both run_task.py (local validation) and inference.py (submission).
"""

from typing import List


def grade(task: str, history: List[str], tickets: list) -> float:
    """
    Enhanced grading with multi-objective considerations.

    Args:
        task:     One of "easy", "medium", "hard"
        history:  List of step log strings from SupportEnv
        tickets:  List of Ticket objects at end of episode

    Returns:
        Float score in (0.0, 1.0)
    """
    total = len(tickets)
    if total == 0:
        return 0.001

    resolved = [t for t in tickets if t.resolved]
    n_resolved = len(resolved)

    # Extract impact metrics from history
    customer_satisfaction_impacts = []
    team_efficiency_impacts = []
    long_term_value_impacts = []

    for line in history:
        if "satisfaction=" in line and "efficiency=" in line:
            try:
                parts = line.split()
                for part in parts:
                    if part.startswith("satisfaction="):
                        customer_satisfaction_impacts.append(float(part.split("=")[1]))
                    elif part.startswith("efficiency="):
                        team_efficiency_impacts.append(float(part.split("=")[1]))
                    elif part.startswith("value="):
                        long_term_value_impacts.append(float(part.split("=")[1]))
            except (ValueError, IndexError):
                continue

    avg_customer_satisfaction = sum(customer_satisfaction_impacts) / len(customer_satisfaction_impacts) if customer_satisfaction_impacts else 0.001
    avg_team_efficiency = sum(team_efficiency_impacts) / len(team_efficiency_impacts) if team_efficiency_impacts else 0.001
    avg_long_term_value = sum(long_term_value_impacts) / len(long_term_value_impacts) if long_term_value_impacts else 0.001

    if task == "easy":
        # Focus on basic resolution with some customer satisfaction
        resolution_score = max(0.001, min(0.798, n_resolved / total))  # Ensure > 0 and < 0.8
        satisfaction_bonus = max(0.001, min(0.198, avg_customer_satisfaction * 0.5))  # Ensure > 0 and < 0.2
        score = resolution_score + satisfaction_bonus

    elif task == "medium":
        # Balance resolution, SLA compliance, and team efficiency
        resolution_score = max(0.001, 0.4 * (n_resolved / total))

        sla_breached_open = sum(1 for t in tickets if (not t.resolved) and t.sla_deadline <= 0)
        sla_score = max(0.001, 0.3 * max(0.001, 1.0 - (sla_breached_open / total)))

        efficiency_score = max(0.001, 0.3 * min(0.998, avg_team_efficiency + 0.5))  # Ensure < 1.0

        score = resolution_score + sla_score + efficiency_score

    elif task == "hard":
        # Multi-objective optimization with complexity weighting
        steps_taken = max(len(history), 1)

        # Resolution weighted by complexity
        complexity_weights = {"simple": 1.0, "moderate": 1.2, "complex": 1.5}
        weighted_resolutions = sum(
            complexity_weights.get(t.resolution_complexity or "simple", 1.0)
            for t in resolved
        )
        total_weighted_complexity = sum(
            complexity_weights.get(t.resolution_complexity or "simple", 1.0)
            for t in tickets
        )
        resolution_score = max(0.001, 0.3 * (weighted_resolutions / total_weighted_complexity if total_weighted_complexity > 0 else 0.001))

        # Efficiency considering complexity
        efficiency_score = max(0.001, 0.2 * min(0.998, (n_resolved / steps_taken) * 2.0))

        # Customer satisfaction and long-term value
        satisfaction_score = max(0.001, 0.25 * min(0.998, avg_customer_satisfaction + 0.5))
        value_score = max(0.001, 0.25 * min(0.998, avg_long_term_value + 0.5))

        score = resolution_score + efficiency_score + satisfaction_score + value_score

    else:
        raise ValueError(f"Unknown task: '{task}'. Must be one of: easy, medium, hard")

    # Ensure score is strictly between 0 and 1
    score = round(score, 4)
    # Additional safety check after rounding - be more conservative
    if score <= 0.001:  # Any value at or below 0.001
        score = 0.001
    elif score >= 0.998:  # Any value at or above 0.998
        score = 0.998

    return score