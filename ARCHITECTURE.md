# Architecture 
## Overview
Basically, current architecture is just slightly modified ReAct graph that uses
summarization and turn counter to decide when to clear message history

## Graph
![Graph](/images/graph.png "Graph")

1. We keep count of turns in graph state with node `turn_counter`.
2. State gets pushed to `summarize_conversation` node, where we look at current turn count and decide if we go straight through or condense previous messages and delete them.
3. We get into ReAct part of the graph, where agent decides what tools to call and how to craft an answer.

## Weak points
1. Summarization currently uses naive approach with invocation of base model where we ask it to summarize previous messages. It'd better to use dedicated SummarizationNode from prebuilt tools
2. Tool precedence is set using SystemMessage, that approach can look quite flaky  
