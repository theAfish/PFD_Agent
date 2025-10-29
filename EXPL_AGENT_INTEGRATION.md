# Exploration Agent 集成文档

## 概述

已成功将 `/personal/Hub/PFD_Agent/src/pfd_agent_tool/modules/expl` 模块集成到 `/personal/PFD_Agent` 项目中。

## 文件结构

### 1. 工具模块 (`/personal/PFD_Agent/src/pfd_agent/tools/expl/`)

```
tools/expl/
├── __init__.py           # 导出所有工具函数
├── ase_tools.py          # ASE 相关工具（结构优化、MD）
├── atoms_tools.py        # 原子结构工具（熵选择）
├── calculator.py         # 计算器包装器
└── filter.py             # 熵过滤算法
```

**关键工具函数**:
- `list_calculators()` - 列出可用的计算器
- `optimize_structure()` - 结构优化
- `run_molecular_dynamics()` - 分子动力学模拟
- `get_base_model_path()` - 模型路径解析
- `filter_by_entropy()` - 基于熵的数据集筛选

### 2. Exploration Agent (`/personal/PFD_Agent/src/pfd_agent/expl_agent/`)

```
expl_agent/
├── __init__.py           # 模块导出
├── agent.py              # Agent 初始化
└── prompt.py             # Agent 指令和描述
```

## 功能特性

### 1. 数据集筛选 (Dataset Curation)
- **工具**: `filter_by_entropy`
- **用途**: 从大量候选结构中选择最有代表性的子集
- **应用**: 主动学习、训练数据生成

### 2. 结构优化 (Structure Optimization)
- **工具**: `optimize_structure`
- **支持的计算器**: DPA, DeepMD, MatterSim, MACE, EMT, LJ
- **用途**: 弛豫原子结构到稳定构型

### 3. 分子动力学 (Molecular Dynamics)
- **工具**: `run_molecular_dynamics`
- **支持的系综**: NVT, NPT, NVE (多种变体)
- **用途**: 探索势能面、生成训练数据

## 集成到主 Agent

已在 `/personal/PFD_Agent/src/pfd_agent/agent.py` 中集成:

```python
from .expl_agent.agent import init_expl_agent

class PFDAgent(Agent):
    def __init__(self, llm_config):
        ft_agent = init_ft_agent(llm_config)
        expl_agent = init_expl_agent(llm_config)  # ✓ 新增
        
        super().__init__(
            name="pfd_agent",
            sub_agents=[
                ft_agent,
                expl_agent,  # ✓ 新增
            ],
            ...
        )
```

## 路由规则

更新了 `/personal/PFD_Agent/src/pfd_agent/prompt.py`:

```
- 探索任务 → transfer to "expl_agent"
  * 结构优化
  * MD 模拟
  * 熵选择

- 训练任务 → transfer to "ft_agent"
  * 模型训练
  * 配置生成
```

## 典型工作流

### 工作流 1: 主动学习
```
User → Root Agent → expl_agent (MD + 熵选择) → ft_agent (训练)
```

### 工作流 2: 结构优化 + 训练
```
User → Root Agent → expl_agent (优化结构) → ft_agent (以优化结构为初始)
```

### 工作流 3: 数据策划
```
User → Root Agent → expl_agent (熵过滤) → ft_agent (在筛选数据上训练)
```

## 修改说明

### 1. 移除 MCP 装饰器
原始代码使用 `@mcp.tool()` 装饰器用于 MCP 服务器，已移除并转换为普通函数。

### 2. 路径更新
```python
# 原始:
from pfd_agent_tool.modules.expl import ...
from pfd_agent_tool.init_mcp import mcp

# 更新后:
from pfd_agent.tools.expl import ...
# (移除 mcp 导入)
```

### 3. 文件重命名
- `ase.py` → `ase_tools.py` (避免与 ASE 包冲突)
- `atoms.py` → `atoms_tools.py`

## 依赖项

确保安装以下依赖:
- `ase` - 原子模拟环境
- `numpy` - 数值计算
- `deepmd-kit` - DeepMD 计算器（可选）
- `mattersim` - MatterSim 计算器（可选）
- `quests` - 熵计算（用于 filter_by_entropy）

## 使用示例

### 示例 1: 熵选择
```
User: "从我的轨迹中选择50个最有代表性的结构"
→ Root Agent 路由到 expl_agent
→ expl_agent 调用 filter_by_entropy(iter_confs="traj.xyz", max_sel=50)
→ 返回 selected.extxyz
```

### 示例 2: 结构优化 + MD
```
User: "优化这个POSCAR并运行10ps NPT MD"
→ Root Agent 路由到 expl_agent
→ expl_agent:
  1. optimize_structure(input="POSCAR")
  2. run_molecular_dynamics(stages=[NPT 10ps])
→ 返回优化结构和MD轨迹
```

## 测试建议

1. 测试熵选择功能
2. 测试不同计算器的结构优化
3. 测试多阶段 MD 模拟
4. 测试 Root Agent 的路由逻辑

## 已知问题

1. 需要确保 `pfd_agent.tools.util.common` 模块存在（filter.py 依赖）
2. 某些计算器需要额外安装（DeepMD, MatterSim）

## 下一步

- [ ] 测试 expl_agent 功能
- [ ] 添加更多计算器支持
- [ ] 优化熵选择算法性能
- [ ] 编写单元测试
