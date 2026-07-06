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



## 常用操作

### 修改commit信息

还没有push

```
git commit --amend -m "新的提交信息"
git push
```

已经push了

```
git commit --amend -m "正确的提交信息"
git push --force-with-lease
```



### 提交后发现意外需要会退

**还没有push**

```
git reset HEAD~1 
```

已经push

```
git revert HEAD~1
```

