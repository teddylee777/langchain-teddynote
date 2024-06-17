import os


def langsmith(project_name=None, set_enable=True):
    if set_enable:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"  # true: 활성화
        os.environ["LANGCHAIN_PROJECT"] = project_name  # 프로젝트명
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"  # false: 비활성화
