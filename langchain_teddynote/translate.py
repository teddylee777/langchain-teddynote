import deepl


class Translator:
    def __init__(self, api_key, source_lang, target_lang):
        self.translator = deepl.Translator(api_key)
        self.source_lang = source_lang
        self.target_lang = target_lang

    def __call__(self, text):
        result = self.translator.translate_text(
            text, source_lang=self.source_lang, target_lang=self.target_lang
        )
        return result.text
