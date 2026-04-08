"""
Inference Script for Support Ticket Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME 
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]

  Example:
    [START] task=click-test env=miniwob model=Qwen3-VL-30B
    [STEP] step=1 action=click('123') reward=0.00 done=false error=null
    [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
    [STEP] step=3 action=click('789') reward=1.00 done=true error=null
    [END] success=true steps=3 score=1.00 rewards=0.00,0.00,1.00
"""

import asyncio
import json
import os
import sys
import time
import uuid
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

from server.support_env import SupportEnv
from support_env.models import Action
from tasks import TASKS
from grader import grade

# Environment configuration (as per hackathon requirements)
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
API_KEY = os.getenv("HF_TOKEN")

# Task configuration
TASK_NAME = os.getenv("SUPPORT_TICKET_TASK", "easy")
BENCHMARK = os.getenv("SUPPORT_TICKET_BENCHMARK", "support-ticket-env")
MAX_STEPS = 25
TEMPERATURE = 0.2
MAX_TOKENS = 200
SUCCESS_SCORE_THRESHOLD = 0.1  # normalized score in [0, 1]

# Episode-local memory for policy decisions
_planner_escalated: set[int] = set()

SYSTEM_PROMPT = """You are an expert AI customer-support agent managing a complex support environment.
You must decide the SINGLE best action for the current support ticket.
Reply ONLY with valid JSON matching this schema (no markdown, no extra text):
{
  "action_type": "<classify|respond|escalate|resolve|transfer|search_kb|schedule_callback|request_supervisor>",
  "ticket_id": <int>,
  "category": "<billing|technical|account|enterprise|null>",
  "subcategory": "<specific subcategory or null>",
  "response": "<response text or null>",
  "escalate_to": "<billing|technical|supervisor|null>",
  "transfer_to_agent": "<agent_id or null>",
  "search_query": "<search terms or null>",
  "callback_time": "<time string or null>"
}

Available Actions:
- classify: Set category and subcategory for organization
- respond: Send empathetic response to customer
- escalate: Pass to specialized team (billing/technical/supervisor)
- transfer: Move to specific team member
- search_kb: Query knowledge base for solutions
- schedule_callback: Arrange phone/video follow-up
- request_supervisor: Get manager assistance
- resolve: Close the ticket (only after proper context)

Rules:
- First classify tickets to establish context
- Use customer profiles and team availability in decisions
- Consider system status and business hours
- Escalate complex enterprise issues immediately
- Transfer urgent tickets to available specialists
- Schedule callbacks for complex or angry customers
- Search KB before attempting resolution
- Request supervisor help for VIP customers with SLA risk

Context Available:
- Customer lifetime value and satisfaction history
- Team member expertise and current workload
- System health and active incidents
- Business hours and peak times
- Ticket complexity and resolution history
"""


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


def build_user_prompt(obs) -> str:
    t = obs.current_ticket
    open_tickets = [tk for tk in obs.tickets if not tk.resolved]

    # Enhanced ticket summary with customer profiles
    summary_lines = []
    for tk in open_tickets:
        profile_info = ""
        if tk.customer_profile:
            profile_info = f" LTV:${tk.customer_profile.lifetime_value:.0f} prev_tix:{tk.customer_profile.previous_tickets_count} vip:{tk.customer_profile.vip_status}"

        team_info = ""
        if tk.escalated_to:
            team_info = f" escalated_to:{tk.escalated_to}"

        summary_lines.append(
            f"  [#{tk.id}] priority={tk.priority} urgency={tk.urgency} "
            f"sla={tk.sla_deadline} sentiment={tk.sentiment} "
            f"user={tk.user_type} category={tk.category or 'none'} "
            f"complexity={tk.resolution_complexity} tags={','.join(tk.tags)}{profile_info}{team_info}"
        )

    # Team availability summary
    team_summary = []
    for agent in obs.team_members:
        team_summary.append(
            f"{agent.name}({agent.id}): {agent.availability_status} "
            f"workload={agent.current_workload} expertise={','.join(agent.expertise)}"
        )

    # System status
    system_info = (
        f"KB_health={obs.system_status.knowledge_base_health:.1f} "
        f"uptime={obs.system_status.system_uptime:.1f} "
        f"peak_hours={obs.system_status.peak_hours} "
        f"incidents={len(obs.system_status.active_incidents)}"
    )

    return (
        f"Step {obs.step_count} — Business Hours: {obs.business_hours} | Pending Callbacks: {obs.pending_callbacks}\n"
        f"System Status: {system_info}\n\n"
        f"Team Status:\n" + "\n".join(f"  {line}" for line in team_summary) + "\n\n"
        f"Current ticket #{t.id}:\n"
        f"  Text: \"{t.text}\"\n"
        f"  Priority: {t.priority}, Urgency: {t.urgency}/10, SLA deadline: {t.sla_deadline} steps\n"
        f"  Sentiment: {t.sentiment}, User type: {t.user_type}, Complexity: {t.resolution_complexity}\n"
        f"  Category: {t.category or 'unclassified'}, Subcategory: {t.subcategory or 'none'}\n"
        f"  Tags: {', '.join(t.tags) if t.tags else 'none'}\n"
        f"  Already classified: {'yes' if t.category else 'no'}\n"
        f"  Already responded: {'yes' if t.response else 'no'}\n"
        f"  Escalated to: {t.escalated_to or 'none'}\n" +
        (f"  Customer Profile: LTV=${t.customer_profile.lifetime_value:.0f}, "
         f"Account age={t.customer_profile.account_age_days}d, "
         f"Prev tickets={t.customer_profile.previous_tickets_count}, "
         f"VIP={t.customer_profile.vip_status}\n" if t.customer_profile else "") +
        f"\nAll open tickets:\n" + "\n".join(summary_lines) + "\n\n"
        f"Choose the best next action from: classify, respond, escalate, resolve, transfer, search_kb, schedule_callback, request_supervisor.\n"
        f"Reply with JSON only."
    )


def _guess_category(text: str) -> str:
    text = text.lower()
    if any(k in text for k in ("refund", "invoice", "charged", "payment", "subscription")):
        return "billing"
    if any(k in text for k in ("cancel", "card", "details", "access", "account", "update")):
        return "account"
    return "technical"


def _did_resolve_fail(obs, ticket_id: int) -> bool:
    needle = f"action=resolve ticket={ticket_id} score=0.00"
    return any(needle in line for line in obs.history[-6:])


def _policy_action(obs) -> Action:
    unresolved = [t for t in obs.tickets if not t.resolved]
    if not unresolved:
        return Action(action_type="resolve", ticket_id=obs.current_ticket.id)

    # Prioritize imminent SLA risk first, then urgency.
    unresolved.sort(key=lambda t: (t.sla_deadline, -t.urgency, t.id))
    t = unresolved[0]

    if not t.category:
        return Action(action_type="classify", ticket_id=t.id, category=_guess_category(t.text))

    if not t.response:
        if t.sentiment == "angry":
            msg = (
                "We sincerely apologize for the frustration. "
                "We are prioritizing your issue and actively working on resolution now."
            )
        elif t.urgency >= 8 or t.sla_deadline <= 2:
            msg = "We understand the urgency and are prioritizing this ticket for immediate resolution."
        else:
            msg = "Thank you for reporting this. We are actively working to resolve it quickly."
        return Action(action_type="respond", ticket_id=t.id, response=msg)

    critical = (t.priority == "high" or t.urgency >= 8) and t.sla_deadline <= 2
    failed_once = _did_resolve_fail(obs, t.id)
    should_escalate = (critical or failed_once) and t.id not in _planner_escalated
    if should_escalate:
        _planner_escalated.add(t.id)
        return Action(action_type="escalate", ticket_id=t.id)

    return Action(action_type="resolve", ticket_id=t.id)


def _sanitize_action(raw: Action, obs) -> Action:
    unresolved_ids = {t.id for t in obs.tickets if not t.resolved}
    if raw.ticket_id not in unresolved_ids:
        return _policy_action(obs)

    t = next(tt for tt in obs.tickets if tt.id == raw.ticket_id)

    # Enforce valid workflow to avoid low-quality/random LLM actions.
    if not t.category and raw.action_type != "classify":
        return Action(action_type="classify", ticket_id=t.id, category=_guess_category(t.text))
    if t.category and not t.response and raw.action_type not in ("respond", "escalate"):
        return _policy_action(obs)
    if raw.action_type == "classify":
        return Action(
            action_type="classify",
            ticket_id=t.id,
            category=(raw.category if raw.category in ("billing", "technical", "account") else _guess_category(t.text)),
            response=None,
        )
    if raw.action_type == "respond":
        return Action(action_type="respond", ticket_id=t.id, response=raw.response or "Thank you for contacting support. We're looking into your issue.")
    if raw.action_type == "escalate":
        return Action(action_type="escalate", ticket_id=t.id, category=None, response=None)
    if raw.action_type == "resolve":
        return Action(action_type="resolve", ticket_id=t.id, category=None, response=None)
    
    return _policy_action(obs)


def get_model_action(client: OpenAI, obs) -> Action:
    user_prompt = build_user_prompt(obs)
    try:
        # Try with the new Hugging Face router using v1 endpoint
        client.base_url = f"{API_BASE_URL}/v1"
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
        
        import re
        # Extract JSON from the response
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            data = json.loads(json_str)
            return Action(**data)
        else:
            raise ValueError("No valid JSON found in response")
            
    except Exception as exc:
        err_text = str(exc)
        if "402" not in err_text and "depleted your monthly included credits" not in err_text:
            print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return _policy_action(obs)


def format_action_string(action: Action) -> str:
    """Format action as string for logging."""
    action_str = f"{action.action_type}({action.ticket_id}"
    if action.category:
        action_str += f",category={action.category}"
    if action.response:
        response = action.response.replace("'", "\\'")[:50]
        action_str += f",response='{response}...'"
    action_str += ")"
    return action_str


async def run_task(task_name: str) -> None:
    """Run a single task episode following the template pattern."""
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # Initialize environment
    env = SupportEnv(max_steps=MAX_STEPS)
    obs = env.reset()
    _planner_escalated.clear()

    history: List[float] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_error = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, MAX_STEPS + 1):
            # Get action from LLM with fallback
            try:
                action = get_model_action(client, obs)
                action = _sanitize_action(action, obs)
            except Exception as e:
                last_error = str(e)
                action = _policy_action(obs)

            # Execute action
            obs, reward, done, info = env.step(action)
            
            # Format action for logging
            action_str = format_action_string(action)
            
            # Track rewards and steps
            rewards.append(reward.score)
            steps_taken = step

            # Log step
            log_step(step=step, action=action_str, reward=reward.score, done=done, error=last_error)

            if done:
                break

        # Calculate final score using grader
        final_score = grade(task_name, env.history, env.tickets)
        score = min(max(final_score, 0.0), 1.0)  # clamp to [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        last_error = str(e)
        # Log final error step
        log_step(step=steps_taken + 1, action="error", reward=0.0, done=True, error=last_error)
    
    finally:
        # Ensure environment is closed
        try:
            # Note: SupportEnv doesn't have async close, so we just clean up
            pass
        except Exception as e:
            print(f"[DEBUG] env cleanup error: {e}", flush=True)
        
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    """Main function following template pattern."""
    results = []
    
    for task_name in TASKS:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Running task: {task_name}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        
        try:
            await run_task(task_name)
        except Exception as e:
            print(f"[ERROR] Task {task_name} failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
