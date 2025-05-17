from nikolasigmoid/py-agent-infra:latest
copy requirements.txt requirements.txt

run python -m pip install --no-cache-dir -r requirements.txt

copy app app
copy deepsearch deepsearch
copy system_prompt.txt system_prompt.txt

env PROXY_SCOPE="*api.tavily.com*"
env PUBMED_EMAIL="daniel@bvm.network"
env FORWARD_ALL_MESSAGES=1
