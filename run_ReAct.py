from openai import OpenAI
import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic
import os

OPENAI_API_KEY = "xxxxxx"  # Replace with your OpenAI API key

client = OpenAI(
    api_key=OPENAI_API_KEY,  # This is the default and can be omitted
)
def llm(prompt):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4o-mini",
    )
    return chat_completion.choices[0].message.content

class ReActAlfworldAgent:
    def __init__(self, env):
        self.env = env
        self.history = []
    
    def run(self, task):
        """Reasoning + Acting loop"""
        self.history.append(f"Task: {task}")
        
        obs, info = self.env.reset()  # Reset environment for a new episode
        self.history.append(f"Observation: {obs}")
        print(f"\nInitial Observation: {obs}")

        for _ in range(10):  # Max 10 reasoning steps
            # Format the LLM prompt
            prompt = "\n".join(self.history) + "\nWhat should I do next?"
            response = llm(prompt)
            self.history.append(f"LLM: {response}")
            print(f"\nAI: {response}")

            # Extract action from LLM response
            if "ACTION(" in response:
                action = response.split("ACTION(")[-1].split(")")[0]
                print(f"\nExecuting action: {action}")

                obs, reward, done, _ = self.env.step([action])  # Execute in ALFWorld
                self.history.append(f"Result: {obs}")

                if done:
                    print("\n✅ Task Completed!")
                    return obs  # Return final state

            elif "ANSWER:" in response:
                print("\nFinal Answer:", response.split("ANSWER:")[-1])
                return response.split("ANSWER:")[-1]

        print("\n❌ Task Failed: Max steps reached")
        return "Task failed."

# Load ALFWorld environment
#env = environment.AlfredTWEnv()
#env.load(["alfworld/data/json_2.1.0"])
import alfworld.agents.modules.generic as generic 
config = generic.load_config()
alfred_env = getattr(environment, config["general"]["evaluate"]["env"]["type"])(config, train_eval="eval_out_of_distribution")
env = alfred_env.init_env(batch_size=1)

# Create and run ReAct agent
agent = ReActAlfworldAgent(env)
agent.run("Put a warm apple on the table.")
        
