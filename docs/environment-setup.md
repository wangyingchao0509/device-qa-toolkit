# 测试环境搭建说明书（可执行版）

> 这是我在实际工作中整理的一份**可交付给他人照做**的环境搭建 SOP，作为本仓库的配套文档放在这里。
> 示例以「后台管理系统」为例（端口 8080，库名 `testdb`，路径 `/opt/testenv`），替换成自己的项目即可。
>
> **与本仓库的关系**：本项目为了做到「克隆即可运行」用了内存版 Mock 服务，不需要这套环境；
> 但真实项目的测试环境（数据库、中间件、配置分离、数据基线）都按本文流程搭建。
>
> 阅读顺序：第 0 步需求确认表 → 1~8 步逐步执行 → 第 9 步自检脚本 → 第 10 步故障速查。

---

## 第 0 步：先填这张表（不填完不要动手）

| 项目 | 填写 | 说明 |
|---|---|---|
| 被测版本 | `分支名 / commit` | 必须精确到 commit，不能只写"最新版" |
| 环境用途 | 功能 / 联调 / 压测 / 演示 | 用途不同，资源与数据策略不同 |
| 机器配置 | CPU 核数 / 内存 / 磁盘 | 内存 ≥ 应用堆内存 × 2 |
| 数据库 | 版本 + 库名 | 库名带环境前缀，如 `test_db` |
| 依赖的第三方 | 真实对接 / 挡板 | 支付、短信、外部接口必须明确 |
| 使用人与期限 | 谁用、用到什么时候 | 写进环境台账，避免被长期占用 |

---

## 第 1 步：规划（10 分钟，省后面一天）

### 1.1 目录规范

```bash
sudo mkdir -p /opt/testenv/{app,conf,logs,data,backup,script}
sudo chown -R $USER:$USER /opt/testenv
```

| 目录 | 放什么 |
|---|---|
| `app` | 应用包（jar / dist） |
| `conf` | 配置文件（按环境区分） |
| `logs` | 应用日志、部署日志 |
| `data` | 数据库、中间件的数据卷 |
| `backup` | 数据库备份、配置备份 |
| `script` | 部署、检查、造数脚本 |

### 1.2 端口分配表（填好贴在环境文档里）

| 端口 | 服务 | 说明 |
|---|---|---|
| 22 | SSH | |
| 8080 | 应用 | |
| 3306 | MySQL | 容器映射 |
| 6379 | Redis | 容器映射 |
| 80 / 443 | Nginx | 反向代理 |

### 1.3 命名规范

- 容器名：`{env}-{service}`，如 `test-mysql`
- 库名：`{env}_{业务}`，如 `test_order`
- 日志文件：`{env}-{service}-{yyyyMMdd}.log`

**验证方法**：端口表里每个端口在本机都没被占用：

```bash
ss -lntp | grep -E '8080|3306|6379'
```

无输出 = 干净。

---

## 第 2 步：基础环境

### 2.1 系统与时间

```bash
# 查看系统版本
cat /etc/os-release

# 设置时区（很关键，否则日志时间差 8 小时）
sudo timedatectl set-timezone Asia/Shanghai
timedatectl

# 时间同步
sudo apt install -y chrony        # Ubuntu
sudo systemctl enable --now chrony
chronyc sources
```

### 2.2 常用工具

```bash
# Ubuntu / Debian
sudo apt update
sudo apt install -y curl wget vim unzip net-tools lsof git htop

# CentOS / RHEL
sudo yum install -y curl wget vim unzip net-tools lsof git htop
```

### 2.3 防火墙（按需放行）

```bash
# Ubuntu
sudo ufw status
sudo ufw allow 8080/tcp

# CentOS
sudo firewall-cmd --list-ports
sudo firewall-cmd --add-port=8080/tcp --permanent && sudo firewall-cmd --reload
```

**验证方法**：

```bash
timedatectl | grep "Time zone"     # 应显示 Asia/Shanghai
curl -I https://www.baidu.com      # 外网通（内网环境跳过）
```

---

## 第 3 步：运行时环境

### 3.1 Java（后端常见）

```bash
# 安装 JDK 17
sudo apt install -y openjdk-17-jdk        # Ubuntu
sudo yum install -y java-17-openjdk       # CentOS

java -version
echo $JAVA_HOME   # 为空则手动配置
```

写入环境变量：

```bash
echo 'export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64' >> ~/.bashrc
echo 'export PATH=$JAVA_HOME/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### 3.2 Python（自动化、造数、采集脚本用）

```bash
sudo apt install -y python3 python3-venv python3-pip
python3 -V

# 建独立虚拟环境，不要把包装到系统里
python3 -m venv /opt/testenv/venv
source /opt/testenv/venv/bin/activate
pip install -r /opt/testenv/conf/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

`requirements.txt` 必须锁版本：

```
pytest==8.0.0
requests==2.31.0
pymysql==1.1.0
pandas==2.2.0
```

### 3.3 Node（前端项目）

```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
node -v && npm -v
```

### 3.4 Docker（推荐，环境可重建的关键）

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo systemctl enable --now docker
sudo usermod -aG docker $USER     # 重新登录后生效
docker version
```

**验证方法**：

```bash
java -version
python3 -V
docker run --rm hello-world
```

---

## 第 4 步：数据库与中间件

推荐用 `docker-compose`，一次拉起全部依赖，重装只需一条命令。

### 4.1 编写编排文件

`/opt/testenv/conf/docker-compose.yml`：

```yaml
services:
  mysql:
    image: mysql:8.0
    container_name: test-mysql
    restart: always
    environment:
      MYSQL_ROOT_PASSWORD: Root@123456
      MYSQL_DATABASE: testdb
      TZ: Asia/Shanghai
    command:
      - --character-set-server=utf8mb4
      - --collation-server=utf8mb4_unicode_ci
    ports:
      - "3306:3306"
    volumes:
      - /opt/testenv/data/mysql:/var/lib/mysql
      - /opt/testenv/backup/init:/docker-entrypoint-initdb.d
    healthcheck:
      test: ["CMD", "mysqladmin", "ping", "-h", "127.0.0.1", "-pRoot@123456"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: test-redis
    restart: always
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes --requirepass Redis@123456
    volumes:
      - /opt/testenv/data/redis:/data

  nginx:
    image: nginx:1.25
    container_name: test-nginx
    restart: always
    ports:
      - "80:80"
    volumes:
      - /opt/testenv/conf/nginx:/etc/nginx/conf.d
      - /opt/testenv/logs/nginx:/var/log/nginx
```

### 4.2 启动

```bash
cd /opt/testenv/conf
docker compose up -d
docker compose ps
docker compose logs -f mysql     # 看启动日志，出现 ready for connections 才算好
```

### 4.3 建库与授权（不用 root 跑业务）

```bash
docker exec -it test-mysql mysql -uroot -pRoot@123456
```

```sql
CREATE DATABASE IF NOT EXISTS testdb DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER IF NOT EXISTS 'testapp'@'%' IDENTIFIED BY 'Test@123456';
GRANT ALL PRIVILEGES ON testdb.* TO 'testapp'@'%';
FLUSH PRIVILEGES;
```

**验证方法**：

```bash
docker exec -it test-mysql mysql -utestapp -pTest@123456 -e "SELECT VERSION(); SHOW DATABASES;"
docker exec -it test-redis redis-cli -a Redis@123456 ping     # 应返回 PONG
```

---

## 第 5 步：部署应用

### 5.1 放包

```bash
cd /opt/testenv/app
# 从制品库或 scp 拿包，文件名叫清楚版本
ls -l app-20261116-1.0.0.jar
```

### 5.2 配置文件（与代码分离）

`/opt/testenv/conf/application-test.yml`：

```yaml
server:
  port: 8080
spring:
  datasource:
    url: jdbc:mysql://127.0.0.1:3306/testdb?useUnicode=true&characterEncoding=utf8&serverTimezone=Asia/Shanghai
    username: testapp
    password: Test@123456
  redis:
    host: 127.0.0.1
    port: 6379
    password: Redis@123456
logging:
  file:
    name: /opt/testenv/logs/app.log
third-party:
  pay-url: http://127.0.0.1:9090/mock/pay    # 挡板地址，不要连真实支付
```

### 5.3 用 systemd 托管（不要 nohup，重启机器就没了）

`/etc/systemd/system/testapp.service`：

```ini
[Unit]
Description=Test Environment App
After=network.target

[Service]
Type=simple
User=appuser
WorkingDirectory=/opt/testenv/app
ExecStart=/usr/bin/java -Xms512m -Xmx1024m -jar /opt/testenv/app/app-20261116-1.0.0.jar \
  --spring.config.additional-location=/opt/testenv/conf/application-test.yml
Restart=always
RestartSec=5
StandardOutput=append:/opt/testenv/logs/stdout.log
StandardError=append:/opt/testenv/logs/stderr.log

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now testapp
sudo systemctl status testapp
```

**验证方法**：

```bash
ss -lntp | grep 8080                 # 端口在听
curl -i http://127.0.0.1:8080/health # 返回 200
tail -n 50 /opt/testenv/logs/app.log # 无 ERROR
```

---

## 第 6 步：初始化数据（最容易被忽略的一步）

### 6.1 导入表结构

```bash
docker exec -i test-mysql mysql -utestapp -pTest@123456 testdb < /opt/testenv/backup/schema.sql
```

### 6.2 数据脱敏（涉及客户/患者信息必做）

```sql
-- 姓名打码：真实数据 → 患者+ID
UPDATE patient SET name = CONCAT('患者', id);
-- 手机号打码：只留后四位
UPDATE patient SET phone = CONCAT('138****', RIGHT(phone, 4));
-- 身份证、地址同理，或直接置空
UPDATE patient SET id_card = NULL, address = '测试地址';
```

### 6.3 造测试数据（不要手工点 500 条）

```sql
-- 用数字表批量插入
INSERT INTO patient (name, phone, create_time)
SELECT CONCAT('测试患者', n), CONCAT('1390000', LPAD(n, 4, '0')), NOW()
FROM (
  SELECT @row := @row + 1 AS n
  FROM information_schema.columns, (SELECT @row := 0) t
  LIMIT 1000
) nums;
```

### 6.4 记录基线

```bash
mysqldump -h127.0.0.1 -utestapp -pTest@123456 testdb > /opt/testenv/backup/testdb_baseline_$(date +%Y%m%d).sql
```

**为什么要基线**：测试跑脏了，`source` 一下 5 分钟恢复，不用重装。

**验证方法**：

```sql
SELECT COUNT(*) FROM patient;     -- 数量与预期一致
SELECT * FROM patient LIMIT 3;    -- 数据是脱敏后的
```

---

## 第 7 步：冒烟验证（部署完必须跑）

`/opt/testenv/script/smoke_test.py`：

```python
import sys
import requests

BASE = "http://127.0.0.1:8080"

def check(name, fn):
    try:
        fn()
        print(f"[PASS] {name}")
    except Exception as e:
        print(f"[FAIL] {name} -> {e}")
        sys.exit(1)

check("健康检查", lambda: (
    (lambda r: (_ for _ in ()).throw(AssertionError(r.status_code)) if r.status_code != 200 else None)
    (requests.get(f"{BASE}/health", timeout=5))
))

check("登录接口", lambda: (
    (lambda r: (_ for _ in ()).throw(AssertionError(r.text)) if r.json().get("code") != 0 else None)
    (requests.post(f"{BASE}/api/login", json={"username": "test01", "password": "Test@123456"}, timeout=5))
))

print("冒烟通过，环境可用")
```

```bash
source /opt/testenv/venv/bin/activate
python /opt/testenv/script/smoke_test.py
```

**验证方法**：全部 `[PASS]`，出现任何 `[FAIL]` 就不要交付环境。

---

## 第 8 步：交付文档（一页纸，模板照抄）

```
【测试环境说明】
环境名称：test-功能测试     版本：分支 release/1.0，commit 3f2a9c1
访问地址：http://10.0.0.11:8080
机器配置：4C / 8G / 100G

服务清单
  应用    127.0.0.1:8080     systemctl status testapp
  MySQL   127.0.0.1:3306     docker exec -it test-mysql mysql -utestapp -p
  Redis   127.0.0.1:6379     docker exec -it test-redis redis-cli -a
  Nginx   0.0.0.0:80

账号
  应用测试账号：test01 / Test@123456
  数据库：testapp / Test@123456

数据状态
  基线备份：/opt/testenv/backup/testdb_baseline_20261116.sql
  重置命令：docker exec -i test-mysql mysql -utestapp -pTest@123456 testdb < 备份文件

第三方
  支付、短信走挡板 http://127.0.0.1:9090，不连真实服务

已知限制
  1. 数据量为生产的 1/100，性能结论不可外推
  2. 未接入真实短信通道，验证码固定 123456

重建步骤
  1. docker compose up -d
  2. systemctl start testapp
  3. python script/smoke_test.py

负责人 / 有效期：王颖超 / 至 2026-12-31
```

---

## 第 9 步：环境自检脚本（交付前跑一遍）

`/opt/testenv/script/check_env.sh`：

```bash
#!/usr/bin/env bash
# 测试环境自检：部署完 / 出故障时先跑这个
set -u

echo "===== 1. 系统信息 ====="
timedatectl | grep -E "Time zone|System clock"
echo "磁盘:"; df -h / | tail -1
echo "内存:"; free -h | head -2

echo "===== 2. 端口监听 ====="
for p in 8080 3306 6379 80; do
  if ss -lntp 2>/dev/null | grep -q ":$p "; then
    echo "  [OK]   $p 在监听"
  else
    echo "  [FAIL] $p 未监听"
  fi
done

echo "===== 3. 服务状态 ====="
systemctl is-active testapp
docker compose -f /opt/testenv/conf/docker-compose.yml ps 2>/dev/null

echo "===== 4. 应用健康检查 ====="
curl -s -o /dev/null -w "  HTTP %{http_code}  耗时 %{time_total}s\n" http://127.0.0.1:8080/health

echo "===== 5. 数据库连通性 ====="
docker exec test-mysql mysql -utestapp -pTest@123456 -e "SELECT 1;" testdb 2>/dev/null \
  && echo "  [OK] 数据库可连接" || echo "  [FAIL] 数据库连接失败"

echo "===== 6. 日志最近错误 ====="
grep -iE "error|exception" /opt/testenv/logs/app.log 2>/dev/null | tail -5 || echo "  无 ERROR"
```

```bash
chmod +x /opt/testenv/script/check_env.sh
/opt/testenv/script/check_env.sh
```

---

## 第 10 步：故障速查（症状 → 原因 → 命令）

| 症状 | 最可能原因 | 先敲这个 |
|---|---|---|
| 接口 502 Bad Gateway | 应用没起 / 端口写错 | `ss -lntp \| grep 8080` |
| Connection refused | 服务未监听或防火墙拦 | `ss -lntp`、`ufw status` |
| Access denied for user | 授权 host 不对（`localhost` vs `%`） | `SELECT user,host FROM mysql.user;` |
| 日志时间差 8 小时 | 容器时区没设 | `docker exec test-mysql date`，补 `TZ=Asia/Shanghai` |
| 中文乱码 | 字符集不是 utf8mb4 | `SHOW VARIABLES LIKE 'character%';` |
| 磁盘写满 | 日志没轮转 | `df -h`、`du -sh /opt/testenv/logs/*` |
| 进程莫名消失 | 内存不足被 OOM Kill | `dmesg -T \| grep -i "out of memory"` |
| 端口被占用 | 旧进程没杀干净 | `lsof -i:8080` 或 `ss -lntp \| grep 8080` |
| 改了配置没生效 | 没重启 / 配置没挂载 | `systemctl restart testapp`、`docker compose restart` |
| 只有你这台机器慢 | DNS 或 MTU 问题 | `dig 域名`、`ping -M do -s 1472 目标` |

---

## 附录：三条原则（比步骤更重要）

1. **一切可重建**：所有配置进文件、文件进 Git，`docker compose up -d` 能拉起环境，才算搭完。
2. **一切可追溯**：哪个环境跑哪个 commit、数据基线是哪天、谁在用，写在台账里，不靠记忆。
3. **先查环境再提缺陷**：服务起不来、接口 502 这类问题，先跑 `check_env.sh` 排除环境，再判断是不是产品问题——这一步能省掉大量无效缺陷单，也是主管最看重的判断力。
