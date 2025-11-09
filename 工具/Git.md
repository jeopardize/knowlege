# Git

## git文件配置

```
git config --global user.name "wang jingxin"
git config --global user.email wangjingxin@163.com
```

## git 初始化

```

git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/jeopardize/knowlege.git
git push origin main

```

## git 分支管理

```
git branch #查看当前分之
# 切换到 xxx 分支
git checkout xxx
git branch checkout -b #复制当前分之

# git rebase
git stash #在当前分之my_modify
git checkout 目标主分之
git pull origin 目标主分之
git chekout my_modify
git rebase 目标主分之
git stash pop
git push origin push #不行 -f

```

## 查看git当前提交记录

```
git log #看是否有额外提交
```



## 意外问题

### 提交上去发现错误

```
git commit --amend --auhor ‘author_name <author@email.com>’ --no-edit
```

### 提交后发现意外需要会退

从git log可以看到你需要回退的步骤

```
git reset @xxx
```

