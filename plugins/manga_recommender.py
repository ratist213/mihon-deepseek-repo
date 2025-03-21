ifrom mihon import Plugin
from transformers import AutoTokenizer, AutoModelForCausalLM

class DeepSeekPlugin(Plugin):
    def init(self):
        super().init()
        self.model = None
        self.tokenizer = None

    def on_load(self):
        # Загрузка модели DeepSeek-R1
        self.tokenizer = AutoTokenizer.from_pretrained("ai-forever/DeepSeek-R1")
        self.model = AutoModelForCausalLM.from_pretrained("ai-forever/DeepSeek-R1")
        
    def recommend(self, query):
        inputs = self.tokenizer(query, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=512)
        return self.tokenizer.decode(outputs[0])

def register(app):
    plugin = DeepSeekPlugin()
    app.add_menu_item("DeepSeek", plugin.recommend)
    return plugin