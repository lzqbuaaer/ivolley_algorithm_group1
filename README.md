# ivolley_algorithm_group1

## Getting started

### environment

1. `environment.yml`保存了所需的环境配置，环境名称为 volley 。

   ```bash
   conda env create -f environment.yml
   ```

2. 然后根据版本安装pytorch。

3. 安装`mmcv`:

   ```bash
   pip install -U openmim
   mim install "mmcv==2.0.0rc4"
   ```

配置好之后需要同步修改`setting.py`中的`PythonPath`。

### input

`main.py`第56行`ret = solve(os.path.join('videos', 'test.mp4'), 0)`指定了输入的视频。
