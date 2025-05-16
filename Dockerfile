from nikolasigmoid/py-agent-infra:latest
copy requirements.txt requirements.txt

run python -m pip install --no-cache-dir -r requirements.txt
run apt-get update && apt-get install -y libgl1 && rm -rf /var/lib/apt/lists/* && apt-get clean && rm -rf ~/.cache

copy app app
copy deepsearch deepsearch
copy system_prompt.txt system_prompt.txt

env ETERNALAI_MCP_PROXY_URL="http://84532-proxy/prompt"
env PROXY_SCOPE="*api.tavily.com*"
env RETRIEVER="tavily"
env PUBMED_EMAIL="daniel@bvm.network"
env TAVILY_API_KEY="tvly-hahaha"
env FORWARD_ALL_MESSAGES=1
