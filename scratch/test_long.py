import minichain

with minichain.start_chain("test_long") as backend:
    prompt = minichain.SimplePrompt(backend.OpenAIChat(model = "gpt-3.5-turbo", max_tokens=128))
    try:
        prompt("a " * 8000)
    except:
        print("There was an error in the request. Moving on.")
    import pdb; pdb.set_trace()
