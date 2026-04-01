import os
from google import genai
from google.genai import types

os.environ["GEMINI_API_KEY"] = "your_api_key"


class BaseAgent():
    def __init__(self, name, sys_prompt, model="gemini-2.5-flash-lite"):
        self.name = name
        self.sys_prompt = sys_prompt
        self.memory = []
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.model = model

    def think(self, input_text):

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=input_text)]
            )
        ]

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            system_instruction=[types.Part.from_text(self.sys_prompt)]
        )

        response = ""

        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config
        ):
            if chunk.text:
                print(chunk.text, end="")
                response += chunk.text

        print()
        self.memory.append((input_text, response))
        return response


class Human(BaseAgent):
    def __init__(self, name):
        super().__init__(
            name,
            f"You are {name}, you are a human. Talk naturally."
        )


class Dog(BaseAgent):
    def __init__(self, name):
        super().__init__(
            name,
            f"You are {name}, you are a dog. Bark, be playful."
        )


def orchestrate(human, dog, turns=5):
    message = "Hi Dog!"

    for _ in range(turns):
        print(f"\n{human.name}: ", end="")
        human_out = human.think(message)

        print(f"\n{dog.name}: ", end="")
        dog_out = dog.think(human_out)

        message = dog_out


if __name__ == "__main__":
    human = Human("Shashank")
    dog = Dog("Bruno")
    orchestrate(human, dog)
