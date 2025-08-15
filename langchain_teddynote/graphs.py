import random
from IPython.display import Image, display
from langgraph.graph.state import CompiledStateGraph
from dataclasses import dataclass


@dataclass
class NodeStyles:
    default: str = (
        "fill:#45C4B0, fill-opacity:0.3, color:#23260F, stroke:#45C4B0, stroke-width:1px, font-weight:bold, line-height:1.2"  # 기본 색상
    )
    first: str = (
        "fill:#45C4B0, fill-opacity:0.1, color:#23260F, stroke:#45C4B0, stroke-width:1px, font-weight:normal, font-style:italic, stroke-dasharray:2,2"  # 점선 테두리
    )
    last: str = (
        "fill:#45C4B0, fill-opacity:1, color:#000000, stroke:#45C4B0, stroke-width:1px, font-weight:normal, font-style:italic, stroke-dasharray:2,2"  # 점선 테두리
    )


def visualize_graph(graph, xray=False, ascii=False):
    """
    CompiledStateGraph 객체를 시각화하여 표시합니다.

    이 함수는 주어진 그래프 객체가 CompiledStateGraph 인스턴스인 경우
    해당 그래프를 Mermaid 형식의 PNG 이미지로 변환하여 표시합니다.

    Args:
        graph: 시각화할 그래프 객체. CompiledStateGraph 인스턴스여야 합니다.
        xray: 그래프 내부 상태를 표시할지 여부.
        ascii: ASCII 형식으로 그래프를 표시할지 여부.
    """

    if not ascii:
        try:
            # 그래프 시각화
            if isinstance(graph, CompiledStateGraph):
                display(
                    Image(
                        graph.get_graph(xray=xray).draw_mermaid_png(
                            background_color="white",
                            node_colors=NodeStyles(),
                        )
                    )
                )
        except Exception as e:
            print(f"그래프 시각화 실패 (추가 종속성 필요): {e}")
            print("ASCII로 그래프 표시:")
            try:
                print(graph.get_graph(xray=xray).draw_ascii())
            except Exception as ascii_error:
                print(f"ASCII 표시도 실패: {ascii_error}")
    else:
        print(graph.get_graph(xray=xray).draw_ascii())


def generate_random_hash():
    return f"{random.randint(0, 0xffffff):06x}"
