"""pytest 共享配置：注入插件目录到 sys.path，使纯逻辑模块 utils 可被导入。

运行方式（插件目录下）：
    python3 -m pytest tests/ -v

说明：utils.py 刻意零框架耦合，测试不依赖 astrbot 运行时，本地即可执行。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 将插件根目录（utils.py 所在目录）注入 sys.path 前端，
# 使 `from utils import ...` 与运行时相对导入指向同一份代码。
_PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))