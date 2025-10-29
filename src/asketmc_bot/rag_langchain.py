# rag_langchain.py — интеграция LangGraph/LangChain для RAG Discord-бота
from langgraph.graph import StateGraph, node, State
import langsmith
import config as cfg

# Импортируй свои функции/объекты, которые нужны (замени на твои!)
from main import (
    extract_lemmas,
    retriever,
    rerank,
    get_filtered_nodes,
    build_context,
    query_model,
)

class RagState(State):
    question: str
    lemmas: set = set()
    retrieved_nodes: list = []
    reranked_nodes: list = []
    filtered_nodes: list = []
    context: str = ""
    answer: str = ""

@node
def lemmas_node(state: RagState):
    state.lemmas = extract_lemmas(state.question)
    return state

@node
async def retrieve_node(state: RagState):
    state.retrieved_nodes = await retriever.aretrieve(state.question)
    return state

@node
async def rerank_node(state: RagState):
    state.reranked_nodes = await rerank(state.question, state.retrieved_nodes)
    return state

@node
async def filter_node(state: RagState):
    state.filtered_nodes = await get_filtered_nodes(state.reranked_nodes, state.lemmas)
    return state

@node
def context_node(state: RagState):
    state.context = build_context(state.filtered_nodes, state.lemmas, cfg.CTX_LEN_LOCAL)
    return state

@node
async def llm_node(state: RagState):
    prompt = (
        f"CONTEXT:\n{state.context}\n\nQUESTION: {state.question}\nANSWER:"
    )
    state.answer, _ = await query_model(
        [
            {"role": "system", "content": cfg.PROMPT_STRICT},
            {"role": "user", "content": prompt},
        ],
        sys_prompt=cfg.PROMPT_STRICT,
        ctx_txt=state.context,
        q=state.question
    )
    return state

# Сборка графа
graph = StateGraph(RagState)
graph.add_node("lemmas", lemmas_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("rerank", rerank_node)
graph.add_node("filter", filter_node)
graph.add_node("context", context_node)
graph.add_node("llm", llm_node)

graph.add_edge("lemmas", "retrieve")
graph.add_edge("retrieve", "rerank")
graph.add_edge("rerank", "filter")
graph.add_edge("filter", "context")
graph.add_edge("context", "llm")

graph.set_entry_point("lemmas")
graph.set_exit_point("llm")

pipeline = graph.compile()

# Функция для запуска из main.py/Discord
async def langchain_rag_pipeline(question: str) -> str:
    langsmith.init()  # для трассировки
    output: RagState = await pipeline.invoke(RagState(question=question))
    return output.answer or "No answer."

