from vector_memory import VectorMemory

vm = VectorMemory()
uid = "user_001"

# 存入记忆，参数名 content
vm.add_memory(user_id=uid, content="我去年暑假去青岛海边看日出")
vm.add_memory(user_id=uid, content="我很喜欢海边傍晚的风")

# 检索，函数名、参数名和你项目保持一致
result = vm.search_memory(user_id=uid, query="海边旅行", top_k=2)
print("检索结果：", result)