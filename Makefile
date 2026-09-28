.PHONY: test bench detect

export PYTHONPATH := .

test:
	python -m unittest tests.test_core -v

bench:
	python -m visage.bench

detect:
	python -m visage detect samples/three.jpg -o results/three-cli.jpg --json
