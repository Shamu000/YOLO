### 常用命令

| 别名    | 等价命令                                               | 功能描述                 |
| ------- | ------------------------------------------------------ | ------------------------ |
| `gst`   | `git status`                                           | 查看当前状态             |
| `gco`   | `git checkout`                                         | 切换分支或还原文件       |
| `gcb`   | `git checkout -b`                                      | 创建并切换新分支         |
| `gcm`   | `git commit -m`                                        | 提交（带信息）           |
| `gaa`   | `git add -A`                                           | 添加所有变更（包括删除） |
| `ga`    | `git add`                                              | 添加暂存文件             |
| `gca`   | `git commit -a -m`                                     | 提交所有已更改文件       |
| `gcn`   | `git commit --no-edit`                                 | 使用上次提交信息再次提交 |
| `gl`    | `git pull`                                             | 拉取更新                 |
| `gp`    | `git push`                                             | 推送变更                 |
| `gpsup` | `git push --set-upstream origin $(git_current_branch)` | 推送并设定远程分支       |
| `gcl`   | `git clone`                                            | 克隆仓库                 |

### 日志历史查看

| 别名    | 命令                                                         | 说明                 |
| ------- | ------------------------------------------------------------ | -------------------- |
| `glog`  | `git log --oneline --graph --decorate`                       | 图形化日志           |
| `glogf` | `git log --oneline --graph --all --decorate`                 | 全分支图形日志       |
| `gll`   | `git log --pretty=format:'%C(yellow)%h%Creset %s %Cgreen(%cr) %C(bold blue)<%an>' --graph` | 高亮格式日志         |
| `gd`    | `git diff`                                                   | 当前变更差异         |
| `gds`   | `git diff --staged`                                          | 已暂存变更的 diff    |
| `gdt`   | `git difftool`                                               | 用 difftool 比较差异 |

### 分支管理

| 别名   | 命令                    | 描述              |
| ------ | ----------------------- | ----------------- |
| `gb`   | `git branch`            | 查看分支列表      |
| `gba`  | `git branch -a`         | 查看本地+远程分支 |
| `gbd`  | `git branch -d`         | 删除本地分支      |
| `gbr`  | `git branch -r`         | 查看远程分支      |
| `gm`   | `git merge`             | 合并分支          |
| `grb`  | `git rebase`            | 变基操作          |
| `gcp`  | `git cherry-pick`       | 拾取某次提交      |
| `grbc` | `git rebase --continue` | 继续 rebase       |
| `grba` | `git rebase --abort`    | 中止 rebase       |

### 重置清理撤销

| 别名          | 命令                      | 描述                       |
| ------------- | ------------------------- | -------------------------- |
| `gclean`      | `git clean -fd`           | 删除 untracked 文件        |
| `greset`      | `git reset`               | 重置暂存区                 |
| `greset-hard` | `git reset --hard`        | 丢弃所有修改               |
| `gundo`       | `git reset --soft HEAD~1` | 撤销最近一次提交但保留修改 |
| `gunstage`    | `git reset HEAD`          | 取消 add                   |

### 暂存

| 别名      | 命令              | 描述            |
| --------- | ----------------- | --------------- |
| `gstash`  | `git stash`       | 暂存当前修改    |
| `gstashp` | `git stash pop`   | 还原最近 stash  |
| `gsta`    | `git stash apply` | 应用最近 stash  |
| `gsl`     | `git stash list`  | 查看 stash 列表 |
| `gsd`     | `git stash drop`  | 删除一个 stash  |

### 远程操作

| 别名  | 命令                | 描述           |
| ----- | ------------------- | -------------- |
| `gr`  | `git remote`        | 查看远程仓库   |
| `gra` | `git remote add`    | 添加远程仓库   |
| `grv` | `git remote -v`     | 查看远程地址   |
| `gf`  | `git fetch`         | 拉取远程不合并 |
| `gpo` | `git push origin`   | 推送到 origin  |
| `gpu` | `git pull upstream` | 拉取 upstream  |

### 快捷操作

| 别名      | 命令                                         | 说明                    |
| --------- | -------------------------------------------- | ----------------------- |
| `gnew`    | `git checkout -b`                            | 新建分支并切换          |
| `gff`     | `git fetch --all --prune`                    | 拉取 + 删除远程无效分支 |
| `gwip`    | `git add -A && git commit -m "WIP"`          | 快速提交工作中状态      |
| `grebase` | `git fetch origin && git rebase origin/main` | 与主分支对齐            |