INITIAL_RESPONSE = " 欢迎使用小智 👋"
def create_prompt(transcript):
    return f"""以下是我和你的聊天对话

{transcript}.

请你先理解对话内容，对我说话的内容进行纪要总结，尽量控制字数，做出最简洁的响应."""