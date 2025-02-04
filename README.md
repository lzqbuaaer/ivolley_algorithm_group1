# ivolley_algorithm_group1

## Getting started

### environment

`environment.yml`保存了所需的环境配置，环境名称为 volley 。

```bash
conda env create -f environment.yml
```

配置好之后需要同步修改`setting.py`中的`PythonPath`。

### input

`main.py`第56行`ret = solve(os.path.join('videos', 'test.mp4'), 0)`指定了输入的视频。
