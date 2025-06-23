import gradio as gr
import ollama

with gr.Blocks() as demo:
	chat_context = []
	chatbot = gr.Chatbot(type="messages", height=300)
	msg = gr.Textbox()
	
	def add_message(user_message, chat_history):
		chat_context.append({"role": "user", "content": user_message})
		return "", chat_history + [{"role": "user", "content": user_message}]

	def respond(chat_history):
		stream = ollama.chat(
			model="qwen2.5:0.5b",
			messages=chat_context,
			stream=True
		)
		
		chat_history.append({"role": "assistant", "content": ""})
		for chunk in stream:
			chat_history[-1]["content"] += chunk["message"]["content"]
			yield chat_history
    # what if remove the bot response from context to reduce the storage , will it perform the same way>
		chat_context.append({"role": "assistant", "content": chat_history[-1]["content"]})

	msg.submit(add_message, [msg, chatbot], [msg, chatbot]).then(
		respond, chatbot, chatbot
	)

demo.launch(server_name="0.0.0.0")
