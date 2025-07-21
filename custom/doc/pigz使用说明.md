### 压缩文件夹为tar.gz

```
tar cf - myfolder | pigz > myfolder.tar.gz
```

### 解压pigz压缩的tar.gz

```
pigz -dc 文件名.tar.gz | tar xf -
```

