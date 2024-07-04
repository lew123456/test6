# 使用官方 Python 3.8 镜像作为基础镜像
FROM python:3.8-slim

# 设置工作目录为 /app
WORKDIR /app

# 将当前目录的内容复制到容器的工作目录
COPY . /app

# 安装所有需要的包
RUN pip install --no-cache-dir -r requirements.txt

# 启动应用
CMD ["python", "api/app.py"]