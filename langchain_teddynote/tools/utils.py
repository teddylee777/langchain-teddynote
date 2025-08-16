import json

def update_tool_call(ai_message, tool_name: str, tool_args: dict):
    """
    AIMessage의 도구 호출을 업데이트하는 함수

    Args:
        ai_message: AIMessage
        tool_name: 도구 이름 (예: 'tavily_search')
        tool_args: 도구 인자 딕셔너리 (예: {'query': '...', 'search_depth': '...'})

    Returns:
        업데이트된 AIMessage
    """

    # AIMessage 복사
    new_message = ai_message.model_copy()

    # additional_kwargs의 tool_calls 업데이트
    if new_message.additional_kwargs.get("tool_calls"):
        # tool_calls 리스트 복사
        updated_tool_calls = new_message.additional_kwargs["tool_calls"].copy()

        # 첫 번째 tool_call 업데이트
        updated_tool_calls[0] = updated_tool_calls[0].copy()
        updated_tool_calls[0]["function"]["name"] = tool_name
        updated_tool_calls[0]["function"]["arguments"] = json.dumps(
            tool_args, ensure_ascii=False
        )

        # additional_kwargs 업데이트
        new_message.additional_kwargs["tool_calls"] = updated_tool_calls

    # tool_calls 속성 업데이트
    if new_message.tool_calls:
        # tool_calls 리스트 복사
        updated_tool_calls = new_message.tool_calls.copy()

        # 첫 번째 tool_call 업데이트
        updated_tool_calls[0] = updated_tool_calls[0].copy()
        updated_tool_calls[0]["name"] = tool_name
        updated_tool_calls[0]["args"] = tool_args

        # tool_calls 업데이트
        new_message.tool_calls = updated_tool_calls

    return new_message