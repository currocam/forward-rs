.PHONY: build format lint check examples test clean

build:
	cargo build --release

format:
	uvx ruff format

lint:
	uvx ruff check --fix
	uvx marimo check --fix

examples: build
	cargo run --release --bin two_deme_neutral -- --output notebooks/trees/neutral.trees & \
	cargo run --release --bin one_deme_two_traits -- --output notebooks/trees/one_deme.trees & \
	cargo run --release --bin two_deme_divergent -- --output notebooks/trees/divergent.trees & \
	wait

test:
	cargo test

clean:
	cargo clean
