.PHONY: build format lint check examples test clean

build:
	cargo build --release

format:
	uvx ruff format

lint:
	uvx ruff check --fix
	uvx marimo check --fix

test:
	cargo test

clean:
	cargo clean
