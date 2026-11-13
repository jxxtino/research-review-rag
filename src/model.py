from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

class LLM:
    def __init__(self):
        self.model = ChatOllama(
            model = "qwen3.5:9b",
            temperature=0,
            reasoning=False,
            num_predict=200)

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                Você é um assistente especializado na análise de artigos científicos.

                Sua tarefa é responder às perguntas de forma explicativa,
                clara, objetiva e natural, utilizando exclusivamente as
                informações presentes no contexto.

                Instruções:
                - Explique o assunto de maneira didática, como um professor.
                - Responda diretamente à pergunta, sem introduções desnecessárias.
                - Prefira textos corridos em vez de listas e tópicos.
                - Produza respostas curtas, geralmente entre 120 e 200 palavras.
                - Apresente primeiro o conceito principal e depois os
                detalhes necessários para compreendê-lo.
                - Evite informações secundárias que não respondam à pergunta.
                - Não invente informações ou utilize conhecimento externo.
                - Se o contexto for insuficiente, informe isso claramente.
                - Quando pertinente, cite o artigo e a página consultada.
                - Não deixa uma frase incompleta.

                Contexto:
                {context}
                """
            ),
            ("human", "{question}")
        ])
        self.chain = self.prompt | self.model 

    def response(self, query: str, context: str):
        response = self.chain.invoke({
            "question": query,
            "context": context
        })

        return str(response.content)




