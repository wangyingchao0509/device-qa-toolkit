.PHONY: help install test smoke consistency analysis report clean

help:
	@echo "make install    - 创建虚拟环境并安装依赖"
	@echo "make test       - 运行全部用例"
	@echo "make smoke      - 只跑冒烟用例"
	@echo "make consistency- 只跑并发一致性用例"
	@echo "make analysis   - 生成样例数据并输出分析报告"
	@echo "make report     - 运行用例并生成 HTML 报告"
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

clean:
	rm -rf .pytest_cache reports/pytest.html
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
