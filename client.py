import random
from server.support_env import SupportEnv
from support_env.models import Action

env = SupportEnv()
obs = env.reset()

history = []

for _ in range(25):
    ticket = obs.current_ticket

    action = Action(
        action_type=random.choice(["classify", "respond", "resolve"]),
        ticket_id=ticket.id,
        category="billing",
        response="We are checking your issue."
    )

    obs, reward, done, _ = env.step(action)

    history.append({"reward": reward.score})

    if done:
        break

print("Baseline Score:", sum(h["reward"] for h in history))
