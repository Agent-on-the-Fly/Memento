# Memento: Fine-tuning LLM Agents **without** Fine-tuning LLMs

> A memory-based, continual-learning framework that helps LLM agents improve from experience **without** updating model weights.

<p align="center">
  <b>Planner–Executor Architecture</b> • <b>Case-Based Reasoning</b> • <b>MCP Tooling</b> • <b>Memory-Augmented Learning</b>
</p>

---

<table>
  <tr>
    <td align="center" width="50%">
      <img src="Figure/f1_val_test.jpg" width="90%"/>
      <br/>
      <sub><b>Memento vs. Baselines on GAIA validation and test sets.</b></sub>
    </td>
    <td align="center" width="50%">
      <img src="Figure/f1_tasks.jpg" width="90%"/>
      <br/>
      <sub><b>Ablation study of Memento across benchmarks.</b></sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="Figure/f1_iteration.jpg" width="90%"/>
      <br/>
      <sub><b>Continual learning curves across memory designs.</b></sub>
    </td>
    <td align="center" width="50%">
      <img src="Figure/f1_ood.jpg" width="90%"/>
      <br/>
      <sub><b>Memento’s accuracy improvement on OOD datasets.</b></sub>
    </td>
  </tr>
</table>

## 📰 News
- [2025.10.05] We’re excited to announce that our parametric Case-Based Reasoning inference code is now officially open-sourced! 🎉
- [2025.09.05] We’ve added support to deploy a local LLM as the executor using vLLM, please see client/agent_local_server.py. 🎉
- [2025.09.03] We’ve set up a WeChat group to make it easier to collaborate and exchange ideas on this project. Welcome to join the Group to share your thoughts, ask questions, or contribute your ideas! 🔥 🔥 🔥 [Join our WeChat Group Now!](Figure/wechat.jpg)
- [2025.08.30] We’re excited to announce that our no-parametric Case-Based Reasoning inference code is now officially open-sourced! 🎉
- [2025.08.28] We’ve created a Discord server to make discussions and collaboration around this project easier. Feel free to join and share your thoughts, ask questions, or contribute ideas! 🔥 🔥 🔥 [Join our Discord!](https://discord.gg/y4FP2EDXyX)
- [2025.08.27] Thanks for your interest in our work! We’ll release our CBR code next week and our Parametric Memory code next month. We’ll keep updating on our further development.
- [2025.08.27] We add a new Crawler MCP in ```server/ai_crawler.py``` for web crawling and query-aware content compression to reduce token cost.
- [2025.08.26] We add the SerpAPI (https://serpapi.com/search-api) MCP tool to help you avoid using the search Docker and speed up development.

## 🔥 Key Features

- **No LLM weight updates.** Memento reframes continual learning as **memory-based online reinforcement learning** over a **memory-augmented MDP**. A neural **case-selection policy** guides actions; experiences are stored and reused via efficient Read/Write operations.
- **Two-stage planner–executor loop.** A CBR-driven **Planner** decomposes tasks and retrieves relevant cases; an **Executor** runs each subtask as an MCP client, orchestrating tools and writing back outcomes.
- **Comprehensive tool ecosystem.** Built-in support for web search, document processing, code execution, image/video analysis, and more through a unified MCP interface.
- **Strong benchmark performance.** Achieves competitive results across GAIA, DeepResearcher, SimpleQA, and HLE benchmarks.

---

## 🧠 Core Concept

**Learn from experiences, not gradients.** Memento logs successful & failed trajectories into a **Case Bank** and **retrieves by value** to steer planning and execution—enabling low-cost, transferable, and online continual learning.

---

## 🏗️ Architecture

### Core Components

- **Meta-Planner**: Breaks down high-level queries into executable subtasks using GPT-4.1
- **Executor**: Executes individual subtasks using o3 or other models via MCP tools
- **Case Memory**: Stores final-step tuples **(s_T, a_T, r_T)** for experience replay
- **MCP Tool Layer**: Unified interface for external tools and services

### Tool Ecosystem

- **Web Research**: Live search and controlled crawling via SearxNG
- **Document Processing**: Multi-format support (PDF, Office, images, audio, video)
- **Code Execution**: Sandboxed Python workspace with security controls
- **Data Analysis**: Excel processing, mathematical computations
- **Media Analysis**: Image captioning, video narration, audio transcription

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key (or compatible API endpoint)
- SearxNG instance for web search
- FFmpeg (system-level binary required for video processing)
- PyTorch 2.0+ with CUDA support (for Parametric Memory)

📖 **For detailed installation instructions, see [INSTALL.md](INSTALL.md)**

### Installation

#### Method 1: Using uv (Recommended - Fast & Modern)

```bash
# Clone repository
git clone https://github.com/Agent-on-the-Fly/Memento
cd Memento

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies and create virtual environment automatically
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

#### Method 2: Using pip with requirements.txt

```bash
# Clone repository
git clone https://github.com/Agent-on-the-Fly/Memento
cd Memento

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### PyTorch Installation

**For GPU support (Recommended for Parametric Memory):**

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

For more PyTorch installation options, visit: https://pytorch.org/get-started/locally/


### System Dependencies Installation

#### FFmpeg Installation (Required)

**FFmpeg is required for video processing functionality.** The `ffmpeg-python` package in our dependencies requires a system-level FFmpeg binary.

**Windows:**
```bash
# Option 1: Using Conda (Recommended for isolated environment)
conda install -c conda-forge ffmpeg

# Option 2: Download from official website
# Visit https://ffmpeg.org/download.html and add to PATH
```

**macOS:**
```bash
# Using Homebrew
brew install ffmpeg
```

**Linux:**
```bash
# Debian/Ubuntu
sudo apt-get update && sudo apt-get install ffmpeg

```

#### Web Scraping & Search Setup

```bash
# Install and setup crawl4ai
crawl4ai-setup
crawl4ai-doctor

# Install playwright browsers
playwright install
```

### Environment Variables Configuration

After creating the `.env` file, you need to configure the following API keys and service endpoints:

```bash
#===========================================
# OpenAI API Configuration
#===========================================
USE_AZURE_OPENAI=False

OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1  # or your custom endpoint

AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_API_VERSION=your_azure_openai_api_version_here
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint_here

#===========================================
# Tools & Services API
#===========================================
# Chunkr API (https://chunkr.ai/)
CHUNKR_API_KEY=your_chunkr_api_key_here

# Jina API
JINA_API_KEY=your_jina_api_key_here

# ASSEMBLYAI API
ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here
```

**Note**: Replace `your_*_api_key_here` with your actual API keys. Some services are optional depending on which tools you plan to use.


### SearxNG Setup

For web search capabilities, set up SearxNG:
You can follow https://github.com/searxng/searxng-docker/ to set the docker and use our setting.

```bash
# In a new terminal
cd ./Memento/searxng-docker
docker compose up -d
```


### Basic Usage

#### Interactive Mode

```bash
python client/agent.py
```

#### Parametric Memory Mode (Advanced - With Memory Retriever)

**Parametric Memory** enables the agent to learn from past experiences using a trained neural retriever model.

**Step 1: Train the Memory Retriever**

First, you need to train the retriever model with initial training data:

```bash
cd memory

# Train the retriever model
python train_memory_retriever.py \
  --train training_data.jsonl \
  --output_dir ./ckpts/retriever \
  --use_plan \
  --val_ratio 0.1 \
  --batch_size 32 \
  --lr 2e-5 \
  --epochs 10 \
  --save_best
```

**Step 2: Configure Environment Variables**

Add the following to your `.env` file:

```bash
# Memory Configuration
MEMORY_JSONL_PATH=../memory/memory.jsonl
TRAINING_DATA_PATH=../memory/training_data.jsonl
RETRIEVER_MODEL_PATH=../memory/ckpts/retriever/best.pt
MEMORY_TOP_K=8
MEMORY_MAX_POS_EXAMPLES=8
MEMORY_MAX_NEG_EXAMPLES=8
```

**Step 3: Run Parametric Memory Agent**

```bash
cd client

python parametric_memory.py
```
---

## 🔧 Configuration

### Model Selection

- **Planner Model**: Defaults to `gpt-4.1` for task decomposition
- **Executor Model**: Defaults to `o3` for task execution
- **Custom Models**: Support for any OpenAI-compatible API

### Tool Configuration

- **Search**: Configure SearxNG instance URL
- **Code Execution**: Customize import whitelist and security settings
- **Document Processing**: Set cache directories and processing limits

---

## 📊 Performance

### Benchmark Results

- **GAIA**: 87.88% (Val, Pass@3 Top-1) and **79.40%** (Test)
- **DeepResearcher**: **66.6% F1 / 80.4% PM**, with **+4.7–9.6** absolute gains on OOD datasets
- **SimpleQA**: **95.0%**
- **HLE**: **24.4% PM** (close to GPT-5 at 25.32%)

### Key Insights

- **Small, high-quality memory works best**: Retrieval **K=4** yields peak F1/PM
- **Planning + CBR consistently improves performance**
- **Concise, structured planning outperforms verbose deliberation**

---

## 🛠️ Development

### Project Structure

```
Memento/
├── client/                      # Main agent implementation
│   ├── agent.py                # Hierarchical client with planner–executor
│   ├── no_parametric_cbr.py    # Non-parametric case-based reasoning
│   ├── parametric_memory.py    # Parametric memory with neural retriever
│   ├── run_parametric.sh       # Convenience script for parametric mode
│   └── PARAMETRIC_MEMORY_GUIDE.md  # Detailed parametric memory guide
├── server/                      # MCP tool servers
│   ├── code_agent.py           # Code execution & workspace management
│   ├── search_tool.py          # Web search via SearxNG
│   ├── serp_search.py          # SERP-based search tool
│   ├── documents_tool.py       # Multi-format document processing
│   ├── image_tool.py           # Image analysis & captioning
│   ├── video_tool.py           # Video processing & narration
│   ├── excel_tool.py           # Spreadsheet processing
│   ├── math_tool.py            # Mathematical computations
│   ├── craw_page.py            # Web page crawling
│   └── ai_crawler.py           # Query-aware compression crawler
├── interpreters/                # Code execution backends
│   ├── docker_interpreter.py
│   ├── e2b_interpreter.py
│   ├── internal_python_interpreter.py
│   └── subprocess_interpreter.py
├── memory/                      # Memory components / data
│   ├── parametric_memory.py        # Case retriever for inference
│   ├── train_memory_retriever.py  # Retriever training script
│   ├── np_memory.py            # Non-parametric memory utilities
│   ├── retrain.sh              # Convenience script for retraining
│   ├── memory.jsonl            # Memory pool (cases with labels)
│   ├── training_data.jsonl     # Training data for retriever
│   └── ckpts/                  # Model checkpoints
│       └── retriever/
│           ├── best.pt         # Best performing model
│           └── last.pt         # Last epoch model
├── data/                        # Sample data / cases
├── searxng-docker/              # SearxNG Docker setup
├── Figure/                      # Figures for README/paper
├── README.md
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

### Adding New Tools

1. Create a new FastMCP server in the `server/` directory
2. Implement your tool functions with proper error handling
3. Register the tool with the MCP protocol
4. Update the client's server list in `agent.py`

### Custom Interpreters

Extend the `interpreters/` module to add new execution backends:

```python
from interpreters.base import BaseInterpreter

class CustomInterpreter(BaseInterpreter):
    async def execute(self, code: str) -> str:
        # Your custom execution logic
        pass
```

---

## 📋 TODO

### Completed Features ✅

- [x] **Add Case Bank Reasoning**: Implemented parametric memory-based case retrieval with neural retriever
- [x] **Continual Learning Pipeline**: Automated training data collection and model retraining

### Upcoming Features & Improvements

- [ ] **Add User Personal Memory Mechanism**: Implement user-preference search
- [ ] **Refine Tools & Add More Tools**: Enhance existing tools and expand the tool ecosystem
- [ ] **Test More New Benchmarks**: Evaluate performance on additional benchmark datasets
- [ ] **Memory Compression**: Implement efficient memory pruning and compression strategies
- [ ] **Multi-modal Memory**: Extend memory to support images, videos, and other modalities

---

### Limitations

- **Long-horizon tasks**: GAIA Level-3 remains challenging due to compounding errors
- **Frontier knowledge**: HLE performance limited by tooling alone
- **Open-source coverage**: Limited executor validation in fully open pipelines

---

## 🙏 Acknowledgement

* Some of the code in the toolkits and interpreters is adapted from [Camel-AI](https://github.com/camel-ai/camel).

---

## 📚 Citation

If Memento helps your work, please cite:

```bibtex
@article{zhou2025mementofinetuningllmagents,
      title={Memento: Fine-tuning LLM Agents without Fine-tuning LLMs},
      author={Huichi Zhou and Yihang Chen and Siyuan Guo and Xue Yan and Kin Hei Lee and Zihan Wang and Ka Yiu Lee and Guchun Zhang and Kun Shao and Linyi Yang and Jun Wang},
      journal={arXiv preprint arXiv: 2508.16153},
      url={https://arxiv.org/abs/2508.16153},
      year={2025}
}

@article{huang2025deep,
  title={Deep Research Agents: A Systematic Examination And Roadmap},
  author={Huang, Yuxuan and Chen, Yihang and Zhang, Haozheng and Li, Kang and Fang, Meng and Yang, Linyi and Li, Xiaoguang and Shang, Lifeng and Xu, Songcen and Hao, Jianye and others},
  journal={arXiv preprint arXiv:2506.18096},
  year={2025}
}
```

For a broader overview, please check out our survey: [Github](https://github.com/ai-agents-2030/awesome-deep-research-agent)

---

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:

- Bug reports and feature requests
- Code contributions and pull requests
- Documentation improvements
- Tool and interpreter extensions

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Agent-on-the-Fly/Memento&type=Date)](https://www.star-history.com/#Agent-on-the-Fly/Memento&Date)
