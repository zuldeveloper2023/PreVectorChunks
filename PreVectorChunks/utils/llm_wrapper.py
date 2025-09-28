import openai

class LLMClientWrapper:
    def __init__(self, client, model="gpt-4o-mini", temperature=0, system_prompt=None):
        """
        client: OpenAI client instance
        model: LLM model name
        temperature: randomness/creativity
        system_prompt: default system instructions for the LLM
        """
        self.client = client
        self.model = model
        self.temperature = temperature
        # Use default system prompt if not provided
        self.system_prompt = system_prompt or ""

    def chat(self, context, user_query):
        """
        context: content retrieved from RAG or other sources
        user_query: actual question from user
        """
        full_query = f"{context}\nQuestion: {user_query}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": full_query}
            ],
            temperature=self.temperature
        )

        return response.choices[0].message.content

