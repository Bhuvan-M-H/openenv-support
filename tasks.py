"""
Task definitions for the Support Ticket RL Environment.
Each task represents an increasing level of difficulty for the agent.
"""

TASKS = {
    "easy": (
        "Resolve as many support tickets as possible within 25 steps. "
        "Score = resolved_tickets / total_tickets."
    ),
    "medium": (
        "Resolve tickets without breaching SLA deadlines. "
        "Score = 0.8*(resolved/total_tickets) + 0.2*(1 - open_sla_breached/total_tickets)."
    ),
    "hard": (
        "Maximize completion and step-efficiency. "
        "Score = min(0.999, 0.6*completion + 0.4*min(0.999, 4*resolved/steps_taken))."
    ),
}