"""StarLife observation-model evaluation harness.

唯一真后端：读真实产物（特征 npz + manifest + checkpoint + 云端缓存），
算头级/事件级/世界状态指标，产出 report.json / report.md / 前端排行榜数据。

口径与 starlife/starlife_scripts 下训练/评测脚本严格对齐，见 ARCHITECTURE.md。
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
