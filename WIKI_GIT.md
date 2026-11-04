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

## 完整推送流程（Gitee）

### 步骤 1：查看远程仓库配置

```bash
cd D:\Coding\QiuSport\live
git remote -v
```

**预期输出**：
```
gitee   https://xinlee0606:***@gitee.com/xinlee0606/football.git (fetch)
gitee   https://xinlee0606:***@gitee.com/xinlee0606/football.git (push)
origin  https://github.com/xinlee0606/football.git (fetch)
origin  https://github.com/xinlee0606/football.git (push)
```

**说明**：
- `gitee` = Gitee 远程仓库（主仓库，国内速度快）
- `origin` = GitHub 远程仓库（备用，用于备份）

---

### 步骤 2：查看当前分支状态

```bash
git branch -vv
```

**预期输出**：
```
* main afe8712 [gitee/main] WIKI_GIT.md: 更新为 Gitee 优先的 Git 使用指南
```

**说明**：
- `*` 表示当前所在分支
- `[gitee/main]` 表示本地 `main` 分支跟踪的是 `gitee` 远程的 `main` 分支

---

### 步骤 3：查看工作区状态

```bash
git status
```

**预期输出示例**：
```
On branch main
Your branch is up to date with 'gitee/main'.

Changes not staged for commit:
  modified:   features_under.py
  modified:   train_under.py

no changes added to commit
```

**说明**：
- 如果有修改的文件，会显示在 "Changes not staged for commit" 下
- 如果有未跟踪的新文件，会显示在 "Untracked files" 下

---

### 步骤 4：查看具体变更内容（可选）

```bash
git diff                      # 查看所有未暂存的修改
git diff features_under.py    # 查看指定文件的修改
```

---

### 步骤 5：添加文件到暂存区

```bash
# 添加所有修改的文件
git add .

# 或添加指定文件
git add features_under.py train_under.py

# 查看已暂存的文件
git status
```

**预期输出**：
```
Changes to be committed:
  modified:   features_under.py
  modified:   train_under.py
```

---

### 步骤 6：提交代码

```bash
git commit -m "feat: 优化 UNDER 模型大小球上限配置

- MAX_OU_LINE_UNDER 从 3.25 降至 2.75
- 优化覆盖率权重特征添加逻辑，避免 DataFrame 碎片化
- 更新 WIKI_UNDER.md 文档"
```

**提交信息规范**：
- `feat:` 新功能
- `fix:` 修复 bug
- `docs:` 文档更新
- `refactor:` 代码重构
- `perf:` 性能优化
- `test:` 测试相关

---

### 步骤 7：推送到 Gitee

```bash
# 推送到 Gitee 的 main 分支
git push gitee main
```

**预期输出**：
```
Enumerating objects: 15, done.
Counting objects: 100% (15/15), done.
Delta compression using up to 8 threads
Compressing objects: 100% (8/8), done.
Writing objects: 100% (8/8), 1.2 KiB | 1.2 MiB/s, done.
Total 8 (delta 6), reused 0 (delta 0), pack-reused 0
remote: Powered by GITEE.COM [1.1.23]
remote: Set trace flag a1b2c3d4
To https://gitee.com/xinlee0606/football.git
   afe8712..b5c6d7e  main -> main
```

**说明**：
- 最后一行显示推送成功：`afe8712..b5c6d7e  main -> main`
- 如果显示 `Everything up-to-date`，表示远程已经是最新状态，无需推送

---

### 步骤 8：验证推送结果

```bash
# 查看远程分支列表
git branch -r

# 查看最近提交记录
git log --oneline -5

# 查看远程仓库最新提交
git log gitee/main --oneline -5
```

**预期输出**：
```
gitee/main
gitee/feature-tech
origin/main

b5c6d7e (HEAD -> main, gitee/main) feat: 优化 UNDER 模型大小球上限配置
afe8712 WIKI_GIT.md: 更新为 Gitee 优先的 Git 使用指南
a449d54 从服务器同步最新代码（2026-11-03）
```

---

### 步骤 9：推送到 GitHub（可选，用于备份）

```bash
git push origin main
```

**说明**：
- GitHub 在国内访问可能不稳定，推送失败可忽略
- 等网络恢复后重新推送即可

---

## 推送流程完整示例

```bash
# 1. 进入项目目录
cd D:\Coding\QiuSport\live

# 2. 查看当前状态
git status

# 3. 添加所有修改
git add .

# 4. 提交代码
git commit -m "feat: 添加新功能描述"

# 5. 推送到 Gitee
git push gitee main

# 6. 推送到 GitHub（可选）
git push origin main
```

---

## 常见问题处理

### 问题 1：推送失败 - 远程分支不存在

**错误信息**：
```
fatal: 'gitee' does not appear to be a git repository
```

**原因**：没有配置名为 `gitee` 的远程仓库

**解决方案**：
```bash
# 添加 gitee 远程仓库
git remote add gitee https://gitee.com/xinlee0606/football.git

# 验证
git remote -v
```

---

### 问题 2：推送失败 - 本地落后于远程

**错误信息**：
```
! [rejected] main -> main (fetch first)
error: failed to push some refs to 'https://gitee.com/...'
```

**原因**：远程仓库有新的提交，本地落后于远程

**解决方案**：
```bash
# 先拉取远程最新代码
git pull gitee main

# 解决冲突（如果有）
# 然后重新提交
git add .
git commit -m "merge: 合并远程更新"

# 再次推送
git push gitee main
```

---

### 问题 3：推送失败 - 认证错误

**错误信息**：
```
remote: Incorrect username or password (access token)
fatal: Authentication failed
```

**原因**：Gitee 密码或 Token 不正确

**解决方案**：
```bash
# 更新远程仓库 URL（包含新的 Token）
git remote set-url gitee https://xinlee0606:<NEW_TOKEN>@gitee.com/xinlee0606/football.git

# 或使用 Git Credential Manager 重新登录
git credential-manager erase
# 然后重新推送，会弹出登录窗口
git push gitee main
```

---

### 问题 4：PowerShell 命令语法错误

**错误信息**：
```
标记"&&"不是此版本中的有效语句分隔符
```

**原因**：PowerShell 不支持 `&&` 语法

**解决方案**：使用分号 `;` 替代
```powershell
# 错误写法
cd D:\Coding\QiuSport\live && git push gitee main

# 正确写法
cd D:\Coding\QiuSport\live; git push gitee main
```

---

### 问题 5：推送后显示 Everything up-to-date

**说明**：这不是错误！表示本地和远程已经完全同步，没有新的提交需要推送。

**验证**：
```bash
git log gitee/main --oneline -1
# 应该显示最新的 commit
```

---

## 服务器推送流程

服务器上的 Git 配置与本地不同，通常只有 `origin` 一个远程（指向 Gitee）。

### 服务器推送步骤

```bash
# 1. 查看远程配置
git remote -v
# 输出：origin  https://gitee.com/xinlee0606/football.git

# 2. 查看当前分支
git branch
# 输出：* main

# 3. 推送到 Gitee（使用 origin）
git push origin main

# 或推送其他分支
git push origin feature-tech
```

**注意**：
- 服务器上使用 `origin` 而不是 `gitee`
- 因为服务器的 `origin` 已经指向 Gitee

---

## 本地 vs 服务器配置对比

| 环境 | 远程名称 | 仓库地址 | 推送命令 |
|------|---------|---------|----------|
| **本地** | `gitee` | `https://gitee.com/xinlee0606/football.git` | `git push gitee main` |
| **本地** | `origin` | `https://github.com/xinlee0606/football.git` | `git push origin main` |
| **服务器** | `origin` | `https://gitee.com/xinlee0606/football.git` | `git push origin main` |

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
