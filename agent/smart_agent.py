# -*- coding: utf-8 -*-
"""
smart_agent.py -- Priority-aware rule-based agent for the Support Ticket environment.

Decision logic (per ticket, in order):
  1. classify  -- if ticket has no category yet
  2. respond   -- if ticket has no response yet
  3. escalate  -- ONCE, only for truly urgent tickets (urgency>=8 AND high priority AND sla<=1)
  4. resolve   -- all other cases
"""

from support_env.models import Action

# Track which tickets this agent has already escalated this episode
_escalated: set = set()


def reset_agent() -> None:
    """Call this at the start of a new episode to clear escalation memory."""
    _escalated.clear()


def decide_action(obs) -> Action:
    """
    Pick the best next action for the current observation.

    Args:
        obs: Observation with .tickets, .current_ticket, .step_count, .history

    Returns:
        Action to take
    """
    # Sort unresolved tickets by urgency desc, then sla_deadline asc
    unresolved = [t for t in obs.tickets if not t.resolved]
    if not unresolved:
        # All done — return a no-op (won't happen in normal flow)
        return Action(action_type="resolve", ticket_id=obs.tickets[0].id)

    tickets_sorted = sorted(unresolved, key=lambda t: (-t.urgency, t.sla_deadline))
    t = tickets_sorted[0]
    text = t.text.lower()

    # --- Step 1: classify ---
    if not t.category:
        if any(k in text for k in ("refund", "invoice", "charged", "payment", "subscription")):
            category = "billing"
        elif any(k in text for k in ("cancel", "card", "details", "access", "update")):
            category = "account"
        else:
            category = "technical"
        return Action(action_type="classify", ticket_id=t.id, category=category)

    # --- Step 2: respond ---
    if not t.response:
        if t.sentiment == "angry":
            tone = (
                "We sincerely apologize for the frustration this has caused. "
                "We are prioritizing your issue and will resolve it immediately."
            )
        elif t.urgency > 7:
            tone = (
                "We understand the urgency and are immediately prioritizing your ticket "
                "to ensure the fastest possible resolution."
            )
        else:
            tone = (
                "Thank you for reaching out. We have received your request and are "
                "working to resolve it as quickly as possible."
            )
        return Action(action_type="respond", ticket_id=t.id, response=tone)

    # --- Step 3: escalate ONCE for critically urgent tickets ---
    is_critical = (t.priority == "high" or t.urgency >= 8) and t.sla_deadline <= 1
    if is_critical and t.id not in _escalated:
        _escalated.add(t.id)
        return Action(action_type="escalate", ticket_id=t.id)

    # --- Step 4: resolve ---
    return Action(action_type="resolve", ticket_id=t.id)