# langchain-omics

This package contains the LangChain integration with Omics

## Installation

```bash
pip install -U langchain-omics
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`ChatOmics` class exposes chat models from Omics.

```python
from langchain_omics import ChatOmics

llm = ChatOmics()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`OmicsEmbeddings` class exposes embeddings from Omics.

```python
from langchain_omics import OmicsEmbeddings

embeddings = OmicsEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`OmicsLLM` class exposes LLMs from Omics.

```python
from langchain_omics import OmicsLLM

llm = OmicsLLM()
llm.invoke("The meaning of life is")
```
