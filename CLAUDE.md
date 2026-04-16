# CLAUDE.md

## Network Proxy

**所有涉及外部网络的操作都必须使用以下代理**，包括但不限于：

- Git 操作（clone、fetch、pull、push 等）
- pip / npm / curl / wget 等包管理或下载命令
- GitHub API 调用（gh CLI 等）
- 任何需要访问外部服务的命令

在执行上述操作前，先设置代理环境变量：

```bash
export http_proxy=http://221.194.188.92:3128
export https_proxy=http://221.194.188.92:3128
```

## Git Config

```
user.name = zhennan0521
user.email = 1641225799@qq.com
```

## GitHub Token

存放在 `.env` 文件中（已被 `.gitignore` 忽略）。

## Git 工作流

任何操作（新 feature、删除、debug、文档）都必须以 commit 形式固定，且**一个改动一个 commit**。

### 基本流程

1. 从 `main` 或 `dev` 新建分支
2. 写代码并 commit（规范 commit message）
3. 检查验证功能无误后，merge 到 `main`/`dev`

### Commit Message 规范

- `[feat]` 新功能
- `[fix]` 修 bug
- `[refactor]` 代码重构
- `[docs]` 文档
- `[chore]` 杂项（配置、脚本等）

示例：`[feat] add math/deepscaler accuracy logging for OPD`
