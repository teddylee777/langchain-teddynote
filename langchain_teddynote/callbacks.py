from langchain_core.callbacks.base import BaseCallbackHandler


class StreamingCallback(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"{token}", end="", flush=True)