from server.support_env import SupportEnv
from agent.smart_agent import decide_action

env = SupportEnv()
obs = env.reset()

total = 0

for i in range(25):
    action = decide_action(obs)
    obs, reward, done, _ = env.step(action)
    print(i, action.action_type, reward.score)
    total += reward.score
    if done:
        break

print("Final Score:", total)