# first line: 222
@memory.cache
def get_GPT_response(instruction, system_prompt, chat_model_id, chat_deployment_id, temperature=0):
    res = fetch_GPT_response(instruction, system_prompt, chat_model_id, chat_deployment_id, temperature)
    return res
