# -*- coding: utf-8 -*-
"""
run_task.py -- Runs all 3 environment tasks using the built-in smart agent
and reports per-task graded scores. Useful for quick local validation
before running the full LLM-powered inference.py.

Usage:
    python run_task.py
    python run_task.py --task easy
    python run_task.py --task all
"""

import argparse
import sys

from server.support_env import SupportEnv
from agent.smart_agent import decide_action, reset_agent
from tasks import TASKS
from grader import grade


def run_task(task_name: str, max_steps: int = 25, verbose: bool = True) -> float:
    """Run one episode for the given task. Returns the graded score [0, 1]."""
    reset_agent()   # clear escalation memory from previous episode
    env = SupportEnv(max_steps=max_steps)
    obs = env.reset()
    total_step_reward = 0.001

    if verbose:
        print(f"\n{'-'*55}")
        print(f"  Task      : {task_name}")
        print(f"  Goal      : {TASKS[task_name]}")
        print(f"  Max steps : {max_steps}")
        print(f"{'-'*55}")

    for step_num in range(max_steps):
        action = decide_action(obs)
        obs, reward, done, info = env.step(action)
        total_step_reward += reward.score

        if verbose:
            print(
                f"  step={step_num+1:02d}  {action.action_type:10s}  "
                f"ticket=#{action.ticket_id}  "
                f"reward={reward.score:.3f}  "
                f"resolved={info['resolved_count']}/{env.ticket_count}"
            )

        if done:
            break

    final_score = grade(task_name, env.history, env.tickets)
    final_score = max(0.001, min(0.999, final_score))  # strict clamp for safety

    if verbose:
        resolved = sum(1 for t in env.tickets if t.resolved)
        print(f"\n  Final score  : {final_score:.4f}")
        print(f"  Resolved     : {resolved}/{env.ticket_count}")
        print(f"  Total steps  : {env.step_count}")

    return final_score


def main():
    parser = argparse.ArgumentParser(description="Run support-ticket RL tasks.")
    parser.add_argument(
        "--task",
        choices=list(TASKS.keys()) + ["all"],
        default="all",
        help="Which task to run (default: all)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-step output")
    args = parser.parse_args()

    tasks_to_run = list(TASKS.keys()) if args.task == "all" else [args.task]
    scores = {}

    for task_name in tasks_to_run:
        score = run_task(task_name, verbose=not args.quiet)
        scores[task_name] = score

    print(f"\n{'='*55}")
    print("  SUMMARY")
    print(f"{'='*55}")
    for task_name, score in scores.items():
        score = max(0.001, min(0.999, score))  # strict clamp for safety
        bar_len = int(score * 30)
        bar = "#" * bar_len + "." * (30 - bar_len)
        print(f"  {task_name:8s}  [{bar}]  {score:.4f}")

    if len(scores) > 1:
        overall = sum(scores.values()) / len(scores)
        overall = max(0.001, min(0.999, overall))
        print(f"\n  OVERALL   {overall:.4f}")

    print()


if __name__ == "__main__":
    main()

