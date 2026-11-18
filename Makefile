.PHONY: help install test smoke consistency analysis report perf llm-evals llm-deepeval dashboard clean

help:
	@echo "make install    - 创建虚拟环境并安装依赖"
	@echo "make test       - 运行全部用例"
	@echo "make smoke      - 只跑冒烟用例"
	@echo "make consistency- 只跑并发一致性用例"
	@echo "make analysis   - 生成样例数据并输出分析报告"
	@echo "make report     - 运行用例并生成 HTML 报告"
	@echo "make perf       - 运行负载测试并输出性能报告"
	@echo "make llm-evals  - 跑一轮大模型评测并出报告"
	@echo "make llm-deepeval- 用 DeepEval 跑真指标（需 API Key）"
	@echo "make dashboard  - 生成本地网页看板 reports/dashboard.html"
	@echo "make clean      - 清理缓存与产物"

install:
	python3 -m venv .venv
	.venv/bin/pip install -q --upgrade pip
	.venv/bin/pip install -q -r requirements.txt

test:
	.venv/bin/python -m pytest

smoke:
	.venv/bin/python -m pytest -m smoke

consistency:
	.venv/bin/python -m pytest -m consistency

analysis:
	.venv/bin/python scripts/gen_sample_data.py
	.venv/bin/python scripts/run_analysis.py

report:
	.venv/bin/python -m pytest --html=reports/pytest.html --self-contained-html

perf:
	.venv/bin/python perf/run_load_test.py --users 20 --duration 10 --ramp 5

llm-evals:
	.venv/bin/python llm_evals/run_evals.py

llm-deepeval:
	.venv/bin/python -m llm_evals.deepeval_adapter --limit 5

dashboard:
	.venv/bin/python -m pytest -q --junitxml=reports/raw/junit.xml
	.venv/bin/python scripts/build_dashboard.py

clean:
	rm -rf .pytest_cache reports/pytest.html
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
