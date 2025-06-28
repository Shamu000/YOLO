# DVC使用指南

## 更新/备份数据流程

- 每次修改数据后，使用
```bash
dvc add <data_dir>
```
- 之后会生成对应的.dvc文件，使用git提交该文件即可
```bash
git commit -m "dwadwadwa"
```

## 切换数据版本

- 找到git提交记录中需要切换到的版本a
```bash
git checkout <a>
```

- 然后切换数据版本
```bash
dvc checkout
```