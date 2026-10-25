# Git 使用指南

## 仓库信息

| 项目 | 值 |
|------|------|
| 远程仓库 | `https://github.com/xinlee0606/football.git` |
| 默认分支 | `main` |
| 本地路径 | `D:\Coding\QiuSport\live` |

---

## 仓库结构

```
live/
├── .gitignore              # Git 忽略规则
├── config.py               # 配置文件（含数据库密码/邮箱授权码 ⚠️）
├── requirements.txt        # Python 依赖
── scraper.py              # 主爬虫
├── sp_scraper.py           # SP值爬虫
├── history_scraper.py      # 历史回填爬虫
├── scrape_euro_odds.py     # 欧赔爬虫
├── features_*.py           # 特征工程（over/under/upside）
── train_*.py              # 模型训练
├── predict_*.py            # 模型预测
├── backtest_*.py           # 回测
├── backfill_*.py           # 数据回填
├── filter_matches*.py      # 比赛筛选
── *.sh                    # Linux 部署脚本
├── crontab_linux.txt       # 定时任务配置
├── WIKI*.md                # 项目文档
├── models/                 # 模型文件目录（.gitignore 排除二进制产物）
└── .venv/                  # 虚拟环境（.gitignore 排除）
```

---

## .gitignore 规则说明

### 已排除的内容

| 规则 | 排除内容 | 原因 |
|------|----------|------|
| `.venv/`, `venv/`, `env/` | Python 虚拟环境 | 体积大、可通过 `requirements.txt` 重建 |
| `__pycache__/`, `*.pyc` | Python 字节码缓存 | 自动生成，无需版本控制 |
| `.idea/`, `.vscode/` | IDE 配置 | 个人开发环境配置 |
| `models/*.joblib`, `models/*.txt`, `models/*.json` | 模型二进制文件 | 体积大、训练可随时重建 |
| `*_cache.csv` | 特征缓存文件 | 运行时自动生成，体积大 |
| `backtest_result.json`, `backtest_filter_result.json`, `backtest_filter_detail.csv` | 回测结果 | 运行时产物 |
| `filter_results.jsonl`, `filter_bigqiu_results.jsonl` | 筛选结果 | 运行时产物 |
| `_*.py`, `_*.sh`, `_*.ps1`, `_*.bat` | 临时/调试脚本 | 一次性工具，不纳入版本管理 |
| `*.bat` | Windows 批处理文件 | 部署用脚本，非核心代码 |
| `.DS_Store`, `Thumbs.db` | 系统文件 | OS 自动生成 |
| `*.log` | 日志文件 | 运行时产物 |

### ⚠️ 敏感文件注意

`config.py` 包含数据库密码和邮箱授权码，当前已提交到仓库。**生产环境建议：**
1. 创建 `config.example.py` 作为模板（不含真实密码）
2. 将 `config.py` 加入 `.gitignore`
3. 部署时手动复制并填写真实凭据

---

## 常用 Git 操作

### 查看状态

```bash
cd D:\Coding\QiuSport\live
git status                    # 查看工作区状态
git log --oneline -10         # 查看最近10条提交
git diff                      # 查看未暂存的修改
git diff --cached             # 查看已暂存的修改
```

### 提交代码

```bash
git add .                     # 添加所有修改
git add <file>                # 添加指定文件
git commit -m "描述修改内容"    # 提交
git push                      # 推送到远程
```

### 拉取最新代码

```bash
git pull origin main          # 从远程拉取并合并
```

### 分支管理

```bash
git branch                    # 查看本地分支
git branch -a                 # 查看所有分支（含远程）
git checkout -b feature-xxx   # 创建并切换到新分支
git merge feature-xxx         # 合并分支到当前分支
```

---

## 同步到服务器

服务器路径：`/coding/live/`（ECS: 8.130.188.90）

### 方式一：Git Pull（推荐）

```bash
# SSH 登录服务器后
cd /coding/live
git pull origin main
```

### 方式二：SCP 上传

```powershell
# 本地 PowerShell
scp -r D:\Coding\QiuSport\live\*.py root@8.130.188.90:/coding/live/
```

### 方式三：Python 脚本自动上传

项目内置 `_ssh_put.py` 和 `_ssh_upload_*.py` 脚本，通过 paramiko 自动上传指定文件到服务器。

---

## 服务器 Git 配置

首次使用需在服务器上配置 Git：

```bash
# 安装 Git（CentOS）
sudo yum install -y git

# 配置用户信息
git config --global user.name "xinlee0606"
git config --global user.email "xinlee0606@github.com"

# 克隆仓库（首次）
cd /coding
git clone https://github.com/xinlee0606/football.git live

# 配置认证（二选一）
# 1. 使用 Personal Access Token
git remote set-url origin https://<TOKEN>@github.com/xinlee0606/football.git

# 2. 使用 SSH Key
ssh-keygen -t ed25519 -C "xinlee0606@github.com"
# 将 ~/.ssh/id_ed25519.pub 添加到 GitHub Settings → SSH Keys
git remote set-url origin git@github.com:xinlee0606/football.git
```

---

## 认证方式配置

### 方式一：Personal Access Token（推荐）

1. GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens
2. 创建 token，勾选 `Contents: Read and write` 权限
3. 推送时使用 token 作为密码：
   ```bash
   git push
   # Username: xinlee0606
   # Password: <粘贴token>
   ```
4. 或嵌入 URL（不推荐，token 会明文存储）：
   ```bash
   git remote set-url origin https://<TOKEN>@github.com/xinlee0606/football.git
   ```

### 方式二：GitHub CLI

```powershell
# 安装
winget install GitHub.cli

# 登录
gh auth login

# 登录后 git push 自动使用认证
git push
```

### 方式三：Git Credential Manager

```bash
git config --global credential.helper manager
git push   # 首次弹出登录窗口，后续自动记住
```

---

## 工作流建议

### 日常开发流程

```
1. 本地修改代码
2. git add + git commit 提交
3. git push 推送到 GitHub
4. 服务器 git pull 拉取最新代码
5. 在服务器上运行/测试
```

### 模型训练流程

```
1. 修改特征/训练脚本 → git commit + push
2. 服务器 git pull 同步
3. 服务器执行训练（后台运行）：
   setsid bash train_over_only.sh > logs/train_over.log 2>&1 &
4. 训练完成后模型产物保存在 models/ 目录（不纳入 git）
```

### 回滚操作

```bash
git log --oneline              # 查看提交历史
git revert <commit-hash>       # 安全回滚（生成新提交）
git checkout <commit-hash> -- <file>  # 恢复单个文件到指定版本
```

---

## 注意事项

1. **不要提交敏感信息**：数据库密码、邮箱授权码、API Key 等应通过环境变量或配置文件管理
2. **不要提交大文件**：模型文件（.joblib）、缓存文件（.csv）已通过 .gitignore 排除
3. **服务器同步**：修改代码后务必同步到服务器，避免本地和服务器代码不一致导致的问题
4. **分支策略**：目前使用单分支 `main`，如需并行开发可创建 feature 分支
5. **提交信息规范**：使用 `feat:` / `fix:` / `docs:` / `refactor:` 等前缀描述修改类型
