#!/bin/bash

# Cursor Bible Repository Setup Script
# Run this in your desired directory

echo "🧠 Setting up Cursor Bible repository structure..."

# Create main repository directory
mkdir -p cursor-bible
cd cursor-bible

# Initialize git repository
git init

# Create main directory structure
mkdir -p .github/{ISSUE_TEMPLATE,workflows}
mkdir -p rules/{frameworks,languages,tools,templates}
mkdir -p guides examples/sample-projects

# AI/ML Framework structure (most relevant for AI engineers)
mkdir -p rules/frameworks/ai-ml/{pytorch,tensorflow,transformers,langchain,llamaindex,openai,anthropic,ollama}
mkdir -p rules/frameworks/ai-ml/{jupyter,streamlit,gradio,fastapi-ai,flask-ai}

# Web frameworks (for AI demos/apps)
mkdir -p rules/frameworks/web/{nextjs,react,vue,svelte,django,flask,fastapi}

# Data/Analytics frameworks
mkdir -p rules/frameworks/data/{pandas,numpy,scikit-learn,polars,dask,apache-spark}

# Cloud/MLOps frameworks
mkdir -p rules/frameworks/mlops/{mlflow,wandb,dvc,kubeflow,airflow,prefect}

# Languages (AI-focused)
mkdir -p rules/languages/{python,typescript,rust,julia,r,sql}

# AI/ML Tools
mkdir -p rules/tools/ai/{huggingface,openai-api,anthropic-api,replicate,pinecone,chroma,weaviate}
mkdir -p rules/tools/data/{postgres,mongodb,redis,elasticsearch,sqlite}
mkdir -p rules/tools/dev/{docker,kubernetes,terraform,pytest,black,ruff}
mkdir -p rules/tools/notebooks/{jupyter,colab,kaggle}

# Templates for different AI project types
mkdir -p rules/templates/{llm-apps,computer-vision,nlp,data-science,ml-research,ai-agents}

# Create initial template files
cat > rules/templates/README.md << 'EOF'
# Cursor Rule Templates

This directory contains template `.cursorrules` files for different types of AI/ML projects:

- `llm-apps/` - Rules for LLM application development
- `computer-vision/` - Rules for CV projects
- `nlp/` - Rules for NLP projects  
- `data-science/` - Rules for data analysis and research
- `ml-research/` - Rules for ML research projects
- `ai-agents/` - Rules for AI agent development

Each template includes:
- Framework-specific configurations
- Best practices for the domain
- Common patterns and conventions
- Error handling and debugging tips
EOF

# Create framework README files
cat > rules/frameworks/ai-ml/README.md << 'EOF'
# AI/ML Framework Rules

Cursor rules optimized for AI and Machine Learning development:

## Deep Learning Frameworks
- `pytorch/` - PyTorch-specific rules and patterns
- `tensorflow/` - TensorFlow/Keras configurations
- `transformers/` - Hugging Face Transformers library

## LLM Frameworks  
- `langchain/` - LangChain application development
- `llamaindex/` - LlamaIndex RAG applications
- `openai/` - OpenAI API integration patterns
- `anthropic/` - Anthropic API best practices

## UI Frameworks for AI
- `streamlit/` - Streamlit app development
- `gradio/` - Gradio interface creation
- `jupyter/` - Jupyter notebook optimization

## API Frameworks
- `fastapi-ai/` - FastAPI for AI services
- `flask-ai/` - Flask for ML APIs
EOF

cat > rules/frameworks/web/README.md << 'EOF'
# Web Framework Rules

Cursor rules for web frameworks commonly used in AI applications:

- `nextjs/` - Next.js for AI-powered web apps
- `react/` - React components for AI interfaces
- `vue/` - Vue.js for ML dashboards
- `svelte/` - Svelte for lightweight AI demos
- `django/` - Django for ML web applications
- `flask/` - Flask for simple ML APIs
- `fastapi/` - FastAPI for high-performance ML APIs
EOF

# Create guide files
cat > guides/getting-started.md << 'EOF'
# Getting Started with Cursor Rules for AI Development

This guide helps AI engineers create effective Cursor rules for their projects.

## Quick Setup

1. Choose a template from `/rules/templates/` that matches your project type
2. Copy the relevant framework rules from `/rules/frameworks/`
3. Customize for your specific use case
4. Test with your actual codebase

## For AI Engineers

- Start with the LLM app template if building chatbots/agents
- Use computer vision template for image/video projects  
- Combine data science + web framework rules for ML dashboards
- Always include error handling patterns for API calls

[More detailed instructions...]
EOF

cat > guides/ai-specific-patterns.md << 'EOF'
# AI-Specific Cursor Patterns

## Common Patterns for AI Development

### API Error Handling
How to configure Cursor for robust API error handling with LLM services.

### Data Pipeline Patterns  
Cursor rules for data preprocessing and feature engineering.

### Model Training Workflows
Configurations for training loops, checkpointing, and experiment tracking.

### Deployment Patterns
Rules for containerizing and deploying ML models.

[Detailed examples...]
EOF

# Create GitHub templates
cat > .github/ISSUE_TEMPLATE/cursor-rule-request.md << 'EOF'
---
name: Cursor Rule Request
about: Request a new Cursor rule for a specific framework or use case
title: '[REQUEST] '
labels: 'enhancement, rule-request'
assignees: ''
---

## Framework/Tool
Which framework or tool needs Cursor rules?

## Use Case
What specific AI/ML use case is this for?

## Current Pain Points
What challenges are you facing without proper Cursor rules?

## Expected Behavior
What would ideal Cursor behavior look like for this scenario?

## Additional Context
Any specific patterns, libraries, or workflows to consider?
EOF

cat > .github/PULL_REQUEST_TEMPLATE.md << 'EOF'
## Description
Brief description of the Cursor rules being added/modified.

## Type of Rules
- [ ] AI/ML Framework rules
- [ ] Language-specific rules  
- [ ] Tool integration rules
- [ ] Project template
- [ ] Documentation update

## Framework/Tool
Which framework or tool do these rules target?

## Testing
- [ ] Tested with real AI/ML project
- [ ] Verified rule syntax and format
- [ ] Includes usage examples
- [ ] Documentation updated

## AI Engineering Focus
How do these rules specifically help AI engineers?

## Checklist
- [ ] Rules follow the established format
- [ ] Includes clear documentation
- [ ] No duplicate content
- [ ] Appropriate categorization
EOF

# Create initial .cursorrules template
cat > rules/templates/llm-apps/.cursorrules << 'EOF'
# LLM Application Development Rules

You are an expert in Python, FastAPI, LangChain, and LLM application development.

Key Principles:
- Write clean, readable Python code with type hints
- Use async/await for API calls and I/O operations  
- Implement proper error handling for LLM API calls
- Follow LangChain patterns for chain composition
- Use Pydantic models for data validation
- Implement streaming responses for better UX

Python/FastAPI:
- Use FastAPI for API endpoints with automatic OpenAPI docs
- Implement proper dependency injection
- Use structured logging with context
- Handle rate limiting and retries for external APIs

LangChain/LLMs:
- Use LangChain Expression Language (LCEL) for chains
- Implement proper prompt templates with variables
- Use callback handlers for streaming and monitoring
- Cache embeddings and expensive computations
- Implement fallback strategies for API failures

Error Handling:
- Catch and handle specific LLM API errors
- Implement circuit breakers for external services
- Provide meaningful error messages to users
- Log errors with sufficient context for debugging

Performance:
- Use connection pooling for database and API calls
- Implement caching strategies (Redis, in-memory)
- Batch API calls when possible
- Monitor token usage and costs
EOF

# Create sample project structure
mkdir -p examples/sample-projects/llm-chatbot/{src,tests,config}
cat > examples/sample-projects/llm-chatbot/README.md << 'EOF'
# Sample LLM Chatbot Project

This example shows how to structure an LLM chatbot project with proper Cursor rules.

## Structure
- `src/` - Application code
- `tests/` - Test files  
- `config/` - Configuration files
- `.cursorrules` - Cursor rules for this project

## Usage
Copy the `.cursorrules` file to your project root and customize as needed.
EOF

# Create README files for main directories
cat > rules/README.md << 'EOF'
# Cursor Rules Collection

This directory contains Cursor rules organized by category:

- `frameworks/` - Framework-specific rules (AI/ML, web, data)
- `languages/` - Language-specific configurations  
- `tools/` - Development tool integrations
- `templates/` - Complete rule templates for project types

## For AI Engineers

Most AI projects will benefit from combining:
1. A base template from `templates/`
2. Framework rules from `ai-ml/` 
3. Language rules for Python/TypeScript
4. Tool rules for your development stack
EOF

echo "✅ Repository structure created successfully!"
echo ""
echo "Next steps:"
echo "1. cd cursor-bible"
echo "2. Create initial commit: git add . && git commit -m 'Initial repository structure'"
echo "3. Add remote: git remote add origin <your-repo-url>"
echo "4. Push: git push -u origin main"
echo ""
echo "🚀 Your Cursor Bible repository is ready for AI engineering!"