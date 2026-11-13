# Scientific Paper RAG

Aplicação em Python para consultar artigos científicos em PDF usando RAG (*Retrieval-Augmented Generation*), busca semântica e reranking.

O sistema recupera trechos dos documentos, reordena os resultados por relevância e utiliza um modelo local (*Qwen3.5:9b*) para gerar a resposta. A qualidade da resposta depende dos trechos recuperados. A consulta não garante a análise de todos os artigos da pasta.

## Instalação

Na pasta do projeto, crie e ative um ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instale as dependências:

```bash
python -m pip install -r requirements.txt
```

Baixe o modelo de linguagem desejado em `src/model.py`:

```bash
ollama pull qwen3.5:9b
```

Mantenha o Ollama em execução. Se necessário, inicie o serviço em outro terminal:

```bash
ollama serve
```

## Funcionamento

1. Leitura dos PDFs e divisão do texto em trechos.
2. Geração de embeddings e armazenamento no Chroma.
3. Recuperação de 15 trechos candidatos para a pergunta.
4. Reranking e seleção dos 5 trechos mais relevantes.
5. Geração da resposta com o modelo local pelo Ollama.

## Tecnologias

- **Chroma:** armazenamento e busca vetorial.
- **Sentence Transformers:** embeddings e reranking.
- **LangChain:** integração dos componentes.
- **Ollama:** execução local do modelo de linguagem.
- **PyPDF:** leitura dos PDFs.

## Observações

A pasta `database/`, responsável por armazenar o banco vetorial, é gerada localmente e não foi enviada ao GitHub.