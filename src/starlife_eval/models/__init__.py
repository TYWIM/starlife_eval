"""模型适配器层。

- :mod:`starlife_eval.models.heads`：Head 结构与 checkpoint 加载（torch）
- :mod:`starlife_eval.models.local_heads`：本地五头推理适配器（torch）
- :mod:`starlife_eval.models.replay_cloud`：云端 raw jsonl 离线回放（numpy）
- :mod:`starlife_eval.models.openai_compat`：OpenAI 兼容 live 适配器（stdlib http）
- :mod:`starlife_eval.models.registry`：按 ``ModelSpec.kind`` 造适配器 / unavailable

注意：本 ``__init__`` 不做 eager import，避免只跑离线回放时被强制依赖 torch；
请直接从子模块导入（例如 ``from starlife_eval.models import registry``）。
"""
