from langchain_core.messages import AIMessageChunk
from typing import Any, Dict, List, Callable
from dataclasses import dataclass
from langchain_core.agents import AgentAction, AgentFinish, AgentStep
from langchain.agents.output_parsers.tools import ToolAgentAction
from langchain_core.messages import BaseMessage


def stream_response(response, return_output=False):
    """
    AI 모델로부터의 응답을 스트리밍하여 각 청크를 처리하면서 출력합니다.

    이 함수는 `response` 이터러블의 각 항목을 반복 처리합니다. 항목이 `AIMessageChunk`의 인스턴스인 경우,
    청크의 내용을 추출하여 출력합니다. 항목이 문자열인 경우, 문자열을 직접 출력합니다. 선택적으로, 함수는
    모든 응답 청크의 연결된 문자열을 반환할 수 있습니다.

    매개변수:
    - response (iterable): `AIMessageChunk` 객체 또는 문자열일 수 있는 응답 청크의 이터러블입니다.
    - return_output (bool, optional): True인 경우, 함수는 연결된 응답 문자열을 문자열로 반환합니다. 기본값은 False입니다.

    반환값:
    - str: `return_output`이 True인 경우, 연결된 응답 문자열입니다. 그렇지 않으면, 아무것도 반환되지 않습니다.
    """
    answer = ""
    for token in response:
        if isinstance(token, AIMessageChunk):
            answer += token.content
            print(token.content, end="", flush=True)
        elif isinstance(token, str):
            answer += token
            print(token, end="", flush=True)
    if return_output:
        return answer


# 도구 호출 시 실행되는 콜백 함수입니다.
def tool_callback(tool) -> None:
    print("[도구 호출]")
    print(f"Tool: {tool.get('tool')}")  # 사용된 도구의 이름을 출력합니다.
    if tool_input := tool.get("tool_input"):  # 도구에 입력된 값이 있다면
        for k, v in tool_input.items():
            print(f"{k}: {v}")  # 입력값의 키와 값을 출력합니다.
    print(f"Log: {tool.get('log')}")  # 도구 실행 로그를 출력합니다.


# 관찰 결과를 출력하는 콜백 함수입니다.
def observation_callback(observation) -> None:
    print("[관찰 내용]")
    print(f"Observation: {observation.get('observation')}")  # 관찰 내용을 출력합니다.


# 최종 결과를 출력하는 콜백 함수입니다.
def result_callback(result: str) -> None:
    print("[최종 답변]")
    print(result)  # 최종 답변을 출력합니다.


@dataclass
class AgentCallbacks:
    """
    에이전트 콜백 함수들을 포함하는 데이터 클래스입니다.

    Attributes:
        tool_callback (Callable[[Dict[str, Any]], None]): 도구 사용 시 호출되는 콜백 함수
        observation_callback (Callable[[Dict[str, Any]], None]): 관찰 결과 처리 시 호출되는 콜백 함수
        result_callback (Callable[[str], None]): 최종 결과 처리 시 호출되는 콜백 함수
    """

    tool_callback: Callable[[Dict[str, Any]], None] = tool_callback
    observation_callback: Callable[[Dict[str, Any]], None] = observation_callback
    result_callback: Callable[[str], None] = result_callback


class AgentStreamParser:
    """
    에이전트의 스트림 출력을 파싱하고 처리하는 클래스입니다.
    """

    def __init__(self, callbacks: AgentCallbacks = AgentCallbacks()):
        """
        AgentStreamParser 객체를 초기화합니다.

        Args:
            callbacks (AgentCallbacks, optional): 파싱 과정에서 사용할 콜백 함수들. 기본값은 AgentCallbacks()입니다.
        """
        self.callbacks = callbacks
        self.output = None

    def process_agent_steps(self, step: Dict[str, Any]) -> None:
        """
        에이전트의 단계를 처리합니다.

        Args:
            step (Dict[str, Any]): 처리할 에이전트 단계 정보
        """
        if "actions" in step:
            self._process_actions(step["actions"])
        elif "steps" in step:
            self._process_observations(step["steps"])
        elif "output" in step:
            self._process_result(step["output"])

    def _process_actions(self, actions: List[Any]) -> None:
        """
        에이전트의 액션들을 처리합니다.

        Args:
            actions (List[Any]): 처리할 액션 리스트
        """
        for action in actions:
            if isinstance(action, (AgentAction, ToolAgentAction)) and hasattr(
                action, "tool"
            ):
                self._process_tool_call(action)

    def _process_tool_call(self, action: Any) -> None:
        """
        도구 호출을 처리합니다.

        Args:
            action (Any): 처리할 도구 호출 액션
        """
        tool_action = {
            "tool": getattr(action, "tool", None),
            "tool_input": getattr(action, "tool_input", None),
            "log": getattr(action, "log", None),
        }
        self.callbacks.tool_callback(tool_action)

    def _process_observations(self, observations: List[Any]) -> None:
        """
        관찰 결과들을 처리합니다.

        Args:
            observations (List[Any]): 처리할 관찰 결과 리스트
        """
        for observation in observations:
            observation_dict = {}
            if isinstance(observation, AgentStep):
                observation_dict["observation"] = getattr(
                    observation, "observation", None
                )
            self.callbacks.observation_callback(observation_dict)

    def _process_result(self, result: str) -> None:
        """
        최종 결과를 처리합니다.

        Args:
            result (str): 처리할 최종 결과
        """
        self.callbacks.result_callback(result)
        self.output = result


def pretty_print_messages(messages: list[BaseMessage]):
    for message in messages:
        message.pretty_print()
