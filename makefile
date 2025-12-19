# Flow Makefile

.PHONY: all build examples test clean fmt vet

# Go parameters
GOCMD=go
GOBUILD=$(GOCMD) build
GOTEST=$(GOCMD) test
GOFMT=$(GOCMD) fmt
GOVET=$(GOCMD) vet
GOMOD=$(GOCMD) mod

# Directories
SRC_DIR=src
EXAMPLES_DIR=examples
BUILD_DIR=build

# Source files
SRC_FILES=$(wildcard $(SRC_DIR)/*.go)
EXAMPLE_FILES=$(wildcard $(EXAMPLES_DIR)/*.go)

all: build examples

# Initialize go module if not exists
init:
	@if [ ! -f go.mod ]; then \
		$(GOMOD) init flow; \
	fi

# Build the library
build: init
	@echo "Building Flow library..."
	@mkdir -p $(BUILD_DIR)
	$(GOBUILD) -o $(BUILD_DIR)/ ./$(SRC_DIR)/...
	@echo "Build complete."

# Build examples
examples: init
	@echo "Building examples..."
	@mkdir -p $(BUILD_DIR)/examples
	$(GOBUILD) -o $(BUILD_DIR)/examples/xor.exe ./$(EXAMPLES_DIR)/xor.go
	$(GOBUILD) -o $(BUILD_DIR)/examples/mnist.exe ./$(EXAMPLES_DIR)/mnist.go
	$(GOBUILD) -o $(BUILD_DIR)/examples/regression.exe ./$(EXAMPLES_DIR)/regression.go
	@echo "Examples built."

# Run tests
test: init
	@echo "Running tests..."
	$(GOTEST) -v ./$(SRC_DIR)/...

# Format code
fmt:
	@echo "Formatting code..."
	$(GOFMT) ./$(SRC_DIR)/...
	$(GOFMT) ./$(EXAMPLES_DIR)/...

# Vet code
vet: init
	@echo "Vetting code..."
	$(GOVET) ./$(SRC_DIR)/...
	$(GOVET) ./$(EXAMPLES_DIR)/...

# Clean build artifacts
clean:
	@echo "Cleaning..."
	@rm -rf $(BUILD_DIR)
	@rm -f *.json
	@echo "Clean complete."

# Run XOR example
run-xor: examples
	@echo "Running XOR example..."
	./$(BUILD_DIR)/examples/xor.exe

# Run MNIST example
run-mnist: examples
	@echo "Running MNIST example..."
	./$(BUILD_DIR)/examples/mnist.exe

# Run regression example
run-regression: examples
	@echo "Running regression example..."
	./$(BUILD_DIR)/examples/regression.exe

# Run all examples
run-all: run-xor run-mnist run-regression

# Development: format, vet, build, test
dev: fmt vet build test
	@echo "Development checks complete."

# Help
help:
	@echo "Flow Makefile targets:"
	@echo "  make build          - Build the library"
	@echo "  make examples       - Build all examples"
	@echo "  make test           - Run tests"
	@echo "  make fmt            - Format code"
	@echo "  make vet            - Vet code"
	@echo "  make clean          - Clean build artifacts"
	@echo "  make run-xor        - Build and run XOR example"
	@echo "  make run-mnist      - Build and run MNIST example"
	@echo "  make run-regression - Build and run regression example"
	@echo "  make run-all        - Run all examples"
	@echo "  make dev            - Run all dev checks"
	@echo "  make help           - Show this help"
