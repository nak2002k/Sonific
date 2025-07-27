from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

try:
    llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    response = llm.invoke([HumanMessage(content="Hello, are you working?")])
    print("✅ Success:", response.content)
except Exception as e:
    print("❌ Failed:", str(e))
