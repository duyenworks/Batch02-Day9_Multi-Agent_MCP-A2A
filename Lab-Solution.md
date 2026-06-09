# Codelab: Xây Dựng Hệ Thống Multi-Agent với A2A Protocol

**Thời gian:** 2 giờ  
**Ngôn ngữ:** Python 3.11+  
**Công nghệ:** LangGraph, LangChain, A2A SDK

## Mục Tiêu Học Tập

Sau khi hoàn thành codelab này, bạn sẽ:
- Hiểu cách LLM hoạt động từ cơ bản đến nâng cao
- Biết cách tích hợp tools và RAG vào LLM
- Xây dựng được single agent với ReAct pattern
- Tạo multi-agent system với LangGraph
- Triển khai distributed agents với A2A protocol

## Chuẩn Bị

### Yêu Cầu Hệ Thống
- Python 3.11 trở lên
- [uv](https://docs.astral.sh/uv/) package manager
- API key từ [OpenRouter](https://openrouter.ai)

### Cài Đặt

```bash
# Clone repository
git clone <repo-url>
cd legal_multiagent

# Cài đặt dependencies
uv sync

# Cấu hình environment
cp .env.example .env
# Sửa file .env, thêm OPENROUTER_API_KEY của bạn
```

---

## Phần 1: Direct LLM Calling (20 phút)

### Lý Thuyết

LLM (Large Language Model) ở dạng cơ bản nhất là một API nhận input text và trả về output text. Không có memory, không có tools, chỉ dựa vào training data.

**Ưu điểm:**
- Đơn giản, dễ implement
- Phản hồi nhanh

**Nhược điểm:**
- Không có kiến thức real-time
- Không thể tra cứu database
- Không có context giữa các lần gọi

### Thực Hành

**Bước 1:** Chạy demo Stage 1

```bash
uv run python stages/stage_1_direct_llm/main.py
```

**Bước 2:** Đọc và hiểu code

Mở file `stages/stage_1_direct_llm/main.py` và trả lời:

1. LLM được khởi tạo như thế nào? (Tìm hàm `get_llm()`)
2. Message được gửi đến LLM có cấu trúc gì?
3. Tại sao cần có `SystemMessage` và `HumanMessage`?

> **📝 Trả lời:**
>
> **1. LLM được khởi tạo trong `common/llm.py` bằng hàm `get_llm()`:**
> ```python
> def get_llm() -> ChatOpenAI:
>     return ChatOpenAI(
>         model=os.getenv("OPENROUTER_MODEL", "google/gemini-flash-1.5"),
>         openai_api_key=os.getenv("OPENROUTER_API_KEY"),
>         openai_api_base="https://openrouter.ai/api/v1",
>         temperature=0.3,
>         max_tokens=1024,
>     )
> ```
> Sử dụng `ChatOpenAI` của LangChain nhưng trỏ `openai_api_base` đến **OpenRouter** — một API gateway cho phép dùng nhiều model (Gemini, Claude, GPT…) qua cùng một interface. Model được đọc từ env var, mặc định là `gemini-flash-1.5`.
>
> **2. Message có cấu trúc là một list 2 phần tử:**
> ```python
> messages = [
>     SystemMessage(content="You are a legal expert. ..."),
>     HumanMessage(content=QUESTION),
> ]
> ```
> Đây là cấu trúc **Chat format** (khác với Completion format chỉ có 1 string). List messages được truyền theo thứ tự: system trước, human sau.
>
> **3. Lý do cần tách SystemMessage và HumanMessage:**
> - `SystemMessage` — "cài đặt nhân cách" cho LLM: định nghĩa vai trò, phong cách trả lời, giới hạn độ dài. Tồn tại xuyên suốt toàn bộ conversation.
> - `HumanMessage` — input thực tế của người dùng cho từng lượt hỏi.
> - Nếu gộp làm một string: LLM không phân biệt được đâu là "hướng dẫn hệ thống" đâu là "câu hỏi user", dễ bị user override bằng prompt injection. Tách riêng giúp LLM hiểu đúng ngữ cảnh và tuân thủ system instructions.

**Bài Tập 1.1:** Thay đổi câu hỏi

Sửa biến `QUESTION` thành câu hỏi pháp lý khác (tiếng Việt hoặc tiếng Anh) và chạy lại.

**Bài Tập 1.2:** Thêm temperature control

Thêm parameter `temperature=0.3` vào hàm `get_llm()` trong `common/llm.py` để làm output ổn định hơn.

---

## Phần 2: LLM + RAG & Tools (30 phút)

### Lý Thuyết

**RAG (Retrieval-Augmented Generation):** Cho phép LLM tra cứu knowledge base trước khi trả lời.

**Tools:** Các function mà LLM có thể gọi để thực hiện tác vụ cụ thể (tính toán, query database, gọi API).

**Function Calling Flow:**
1. LLM nhận câu hỏi + danh sách tools
2. LLM quyết định gọi tool nào (hoặc không gọi)
3. Tool được execute, trả về kết quả
4. LLM nhận kết quả và tạo câu trả lời cuối cùng

### Thực Hành

**Bước 1:** Chạy demo Stage 2

```bash
uv run python stages/stage_2_rag_tools/main.py
```

**Bước 2:** Phân tích code

Mở `stages/stage_2_rag_tools/main.py` và tìm:

1. Hàm `@tool` decorator được dùng ở đâu?
2. `LEGAL_KNOWLEDGE` được cấu trúc như thế nào?
3. LLM được bind với tools ra sao? (Tìm `.bind_tools()`)

> **📝 Trả lời:**
>
> **1. `@tool` decorator được dùng trước 3 functions:**
> - `search_legal_database(query)` — tìm kiếm trong knowledge base theo keyword overlap
> - `calculate_damages(breach_type, contract_value)` — tính thiệt hại theo loại vi phạm
> - `check_statute_of_limitations(case_type)` — tra cứu thời hiệu khởi kiện
>
> `@tool` wrap function thành Tool object mà LLM "thấy" được. LangChain đọc **docstring** làm mô tả và **type hints** làm schema — LLM dựa vào đây để biết khi nào gọi tool và truyền argument gì.
>
> **2. `LEGAL_KNOWLEDGE` là list of dict, mỗi entry gồm 3 trường:**
> ```python
> {
>     "id": "ucc_breach",                          # định danh
>     "keywords": ["breach", "contract", "ucc"],   # dùng để match với câu hỏi
>     "text": "Under UCC Article 2, remedies..."   # nội dung đưa vào context LLM
> }
> ```
> Đây là **keyword-based RAG**: khi user hỏi, tool đếm keyword overlap giữa câu hỏi và từng entry, trả về top 2 entries có điểm cao nhất. Đơn giản hơn vector search nhưng minh họa rõ khái niệm RAG.
>
> **3. LLM được bind với tools bằng `.bind_tools()`:**
> ```python
> llm = get_llm()
> llm_with_tools = llm.bind_tools(TOOLS)   # trả về LLM instance mới có tools
> tool_map = {t.name: t for t in TOOLS}    # dict để lookup khi execute
> ```
> `.bind_tools(TOOLS)` gửi schema của tất cả tools kèm theo mỗi request. LLM đọc schemas và có thể trả về `tool_calls` trong response (thay vì text) khi muốn gọi tool. Code sau đó execute tool thủ công và đưa kết quả vào `ToolMessage` cho vòng tiếp theo.

**Bài Tập 2.1:** Thêm knowledge base entry

Thêm một entry mới vào `LEGAL_KNOWLEDGE` về luật lao động:

```python
{
    "id": "labor_law",
    "keywords": ["lao động", "sa thải", "hợp đồng lao động", "labor", "termination"],
    "text": (
        "Theo Bộ luật Lao động Việt Nam 2019, người sử dụng lao động có thể "
        "đơn phương chấm dứt hợp đồng trong các trường hợp: (1) người lao động "
        "thường xuyên không hoàn thành công việc; (2) bị ốm đau, tai nạn đã điều trị "
        "12 tháng chưa khỏi; (3) thiên tai, hỏa hoạn; (4) người lao động đủ tuổi nghỉ hưu."
    ),
}
```

**Bài Tập 2.2:** Tạo tool mới

Tạo một tool `@tool` mới tên `check_statute_of_limitations` nhận vào `case_type` (string) và trả về thời hiệu khởi kiện:

```python
@tool
def check_statute_of_limitations(case_type: str) -> str:
    """Kiểm tra thời hiệu khởi kiện theo loại vụ án.
    
    Args:
        case_type: Loại vụ án (contract, tort, property)
    """
    limits = {
        "contract": "4 năm (UCC § 2-725)",
        "tort": "2-3 năm tùy bang",
        "property": "5 năm",
    }
    return limits.get(case_type.lower(), "Không xác định")
```

Thêm tool này vào danh sách tools và test.

---

## Phần 3: Single Agent với ReAct (25 phút)

### Lý Thuyết

**ReAct Pattern:** Reasoning + Acting

Agent tự động lặp lại chu trình:
1. **Think:** Suy nghĩ cần làm gì
2. **Act:** Gọi tool
3. **Observe:** Nhận kết quả
4. Lặp lại cho đến khi có câu trả lời cuối cùng

LangGraph cung cấp `create_react_agent` để tự động hóa pattern này.

### Thực Hành

**Bước 1:** Chạy demo Stage 3

```bash
uv run python stages/stage_3_single_agent/main.py
```

**Bước 2:** Quan sát output

Chú ý cách agent tự động:
- Quyết định tool nào cần gọi
- Gọi nhiều tools liên tiếp
- Tổng hợp kết quả

**Bước 3:** Đọc code

Mở `stages/stage_3_single_agent/main.py`:

1. Tìm `create_react_agent()` — đây là magic function
2. So sánh với Stage 2: không còn manual tool loop
3. Xem `agent_executor.invoke()` — chỉ cần gọi một lần

> **📝 Trả lời:**
>
> **1. `create_react_agent()` tại dòng 226:**
> ```python
> graph = create_react_agent(model=llm, tools=TOOLS, prompt=SYSTEM_PROMPT)
> ```
> Hàm này tự động tạo một **StateGraph hoàn chỉnh** với 2 nodes:
> - Node `agent`: LLM call — nhận messages, quyết định gọi tool hay trả lời
> - Node `tools`: execute tool calls được yêu cầu
>
> Graph có **vòng lặp tự động**: sau khi execute tool, kết quả được thêm vào messages và quay lại node `agent` — lặp cho đến khi LLM không còn gọi tool nữa (trả về final answer).
>
> **2. So sánh Stage 2 vs Stage 3:**
>
> | | Stage 2 | Stage 3 |
> |---|---|---|
> | Tool loop | Tự code thủ công (for loop) | Tự động (LangGraph xử lý) |
> | Số vòng lặp | Cố định: 1 vòng | Linh hoạt: N vòng tùy LLM |
> | Code lượng | ~30 dòng orchestration | 2 dòng: create + invoke |
> | Khả năng | 1 lần tool call | Multi-step: search → calculate → search lại |
>
> **3. Thay vì `agent_executor.invoke()`, Stage 3 dùng `graph.astream()`:**
> ```python
> inputs = {"messages": [{"role": "user", "content": QUESTION}]}
> async for chunk in graph.astream(inputs, stream_mode="updates", debug=True):
>     # mỗi chunk = 1 bước: THINK / ACT / OBSERVE / FINAL ANSWER
> ```
> Chỉ cần **1 lần gọi** với câu hỏi ban đầu. Graph tự lo toàn bộ vòng lặp Think→Act→Observe bên trong. `astream` cho phép observe từng bước real-time thay vì chờ kết quả cuối cùng.

**Bài Tập 3.1:** Thêm tool tra cứu án lệ

```python
@tool
def search_case_law(keywords: str) -> str:
    """Tìm kiếm án lệ theo từ khóa.
    
    Args:
        keywords: Từ khóa tìm kiếm
    """
    cases = {
        "breach": "Hadley v. Baxendale (1854) - Consequential damages",
        "negligence": "Donoghue v. Stevenson (1932) - Duty of care",
        "contract": "Carlill v. Carbolic Smoke Ball Co (1893) - Unilateral contract",
    }
    for key, case in cases.items():
        if key in keywords.lower():
            return case
    return "Không tìm thấy án lệ phù hợp"
```

Thêm vào tools list và test với câu hỏi về breach of contract.

**Bài Tập 3.2:** Debug agent reasoning

Thêm `verbose=True` vào `create_react_agent()` để xem chi tiết quá trình suy nghĩ của agent.

---

## Phần 4: Multi-Agent In-Process (30 phút)

### Lý Thuyết

**Multi-Agent System:** Nhiều agents chuyên môn hóa cùng làm việc.

**Ưu điểm:**
- Mỗi agent tập trung vào domain riêng
- Có thể chạy song song (parallel execution)
- Dễ maintain và mở rộng

**LangGraph StateGraph:**
- Định nghĩa state (dữ liệu chia sẻ giữa các nodes)
- Tạo nodes (các bước xử lý)
- Định nghĩa edges (luồng điều khiển)

**Send API:** Cho phép dispatch nhiều tasks song song.

### Thực Hành

**Bước 1:** Chạy demo Stage 4

```bash
uv run python stages/stage_4_milti_agent/main.py
```

**Bước 2:** Phân tích kiến trúc

Mở `stages/stage_4_milti_agent/main.py`:

1. Tìm `class State(TypedDict)` — đây là shared state
2. Tìm các agent functions: `law_agent`, `tax_agent`, `compliance_agent`
3. Tìm `Send()` API — dispatch parallel tasks
4. Xem `graph.add_node()` và `graph.add_edge()`

> **📝 Trả lời:**
>
> **1. `LegalState` dùng `TypedDict` vì:**
> ```python
> class LegalState(TypedDict):
>     question: str
>     needs_tax: bool
>     tax_analysis: Annotated[str, _last_wins]
>     needs_compliance: bool
>     compliance_analysis: Annotated[str, _last_wins]
>     needs_privacy: bool
>     privacy_analysis: Annotated[str, _last_wins]
>     final_answer: str
> ```
> - `TypedDict` cung cấp **type safety** — Python và IDE biết mỗi field có type gì, tránh bug khi truyền sai kiểu dữ liệu.
> - LangGraph cần **schema rõ ràng** để biết cách merge state khi nhiều nodes cập nhật đồng thời (parallel execution).
> - `Annotated[str, _last_wins]` là **reducer**: khi `tax_agent` và `compliance_agent` chạy song song và cùng cập nhật state, LangGraph dùng reducer này để lấy giá trị cuối cùng thay vì báo lỗi conflict.
>
> **2. Mỗi agent function nhận `state: LegalState` và trả về `dict` chứa update:**
> ```python
> async def tax_agent(state: LegalState) -> dict:
>     llm = get_llm()
>     messages = [SystemMessage(content=TAX_SYSTEM_PROMPT),
>                 HumanMessage(content=state["question"])]
>     response = await llm.ainvoke(messages)
>     return {"tax_analysis": response.content}   # chỉ update 1 field
> ```
> Return dict chỉ chứa các fields agent đó cập nhật — LangGraph tự **merge** vào state hiện tại, các fields khác giữ nguyên. Tương tự với `compliance_agent` → `compliance_analysis` và `privacy_agent` → `privacy_analysis`.
>
> **3. `Send()` API dùng để dispatch parallel tasks tại runtime:**
> ```python
> def route_to_specialists(state: LegalState) -> list[Send]:
>     tasks = []
>     if state["needs_tax"]:
>         tasks.append(Send("tax_agent", state))        # gửi state đến tax_agent
>     if state["needs_compliance"]:
>         tasks.append(Send("compliance_agent", state))
>     if state["needs_privacy"]:
>         tasks.append(Send("privacy_agent", state))
>     return tasks   # LangGraph chạy tất cả đồng thời
> ```
> Thay vì `add_edge` cố định, `Send()` cho phép **dynamic parallel dispatch**: danh sách nodes được kích hoạt phụ thuộc vào data tại runtime. Các tasks trong list được LangGraph chạy **concurrently** (asyncio), giảm latency so với sequential.
>
> **4. Graph được xây dựng bằng `StateGraph` builder pattern:**
> ```python
> graph = StateGraph(LegalState)
>
> # Đăng ký nodes
> graph.add_node("check_routing", check_routing)
> graph.add_node("tax_agent", tax_agent)
> graph.add_node("compliance_agent", compliance_agent)
> graph.add_node("privacy_agent", privacy_agent)
> graph.add_node("aggregate", aggregate)
>
> graph.set_entry_point("check_routing")          # điểm bắt đầu
>
> graph.add_conditional_edges(                    # edge có điều kiện (dynamic routing)
>     "check_routing", route_to_specialists,
>     ["tax_agent", "compliance_agent", "privacy_agent", "aggregate"]
> )
>
> # Edges cố định: mỗi specialist → aggregate
> graph.add_edge("tax_agent", "aggregate")
> graph.add_edge("compliance_agent", "aggregate")
> graph.add_edge("privacy_agent", "aggregate")
> graph.add_edge("aggregate", END)
>
> app = graph.compile()
> ```
> `add_node` đăng ký function với tên node. `add_edge` tạo luồng một chiều cố định. `add_conditional_edges` cho phép routing động: LangGraph gọi `route_to_specialists` để lấy danh sách edges cần kích hoạt, hỗ trợ cả parallel dispatch qua `Send()`.

**Bước 3:** Vẽ graph

```python
# Thêm vào cuối file main.py
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))
```

**Bài Tập 4.1:** Thêm agent mới

Tạo `privacy_agent` chuyên về GDPR và privacy law:

```python
def privacy_agent(state: State) -> dict:
    """Agent chuyên về luật bảo vệ dữ liệu cá nhân."""
    llm = get_llm()
    
    prompt = f"""Bạn là chuyên gia về GDPR và luật bảo vệ dữ liệu cá nhân.
    
Câu hỏi gốc: {state['question']}
Phân tích pháp lý: {state.get('law_analysis', 'N/A')}

Hãy phân tích các vấn đề về privacy và GDPR (nếu có).
"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"privacy_analysis": response.content}
```

Thêm node này vào graph và kết nối với `aggregate_results`.

**Bài Tập 4.2:** Implement conditional routing

Sửa `check_routing` để chỉ gọi privacy_agent khi câu hỏi có từ khóa "data", "privacy", "gdpr":

```python
def check_routing(state: State) -> list[Send]:
    question_lower = state["question"].lower()
    tasks = []
    
    if any(kw in question_lower for kw in ["tax", "irs", "thuế"]):
        tasks.append(Send("tax_agent", state))
    
    if any(kw in question_lower for kw in ["compliance", "sec", "regulation"]):
        tasks.append(Send("compliance_agent", state))
    
    if any(kw in question_lower for kw in ["data", "privacy", "gdpr", "dữ liệu"]):
        tasks.append(Send("privacy_agent", state))
    
    return tasks if tasks else [Send("aggregate_results", state)]
```

---

## Phần 5: Distributed A2A System (15 phút)

### Lý Thuyết

**A2A (Agent-to-Agent) Protocol:** Chuẩn giao tiếp giữa các agents qua HTTP.

**Khác biệt với Stage 4:**
- Mỗi agent là một service độc lập
- Giao tiếp qua HTTP thay vì in-process
- Dynamic discovery qua Registry
- Có thể scale từng agent riêng biệt

**Kiến trúc:**
```
Registry (10000) ← agents register on startup
    ↓
Customer Agent (10100) → Law Agent (10101)
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
            Tax Agent (10102)   Compliance Agent (10103)
```

### Thực Hành

**Bước 1:** Khởi động toàn bộ hệ thống

```bash
./start_all.sh
```

Chờ ~10 giây để tất cả services khởi động.

**Bước 2:** Test hệ thống

```bash
uv run python test_client.py
```

**Bước 3:** Quan sát logs

Mở 5 terminal tabs và xem logs của từng service:
- Registry: port 10000
- Customer Agent: port 10100
- Law Agent: port 10101
- Tax Agent: port 10102
- Compliance Agent: port 10103

**Bài Tập 5.1:** Trace request flow

Trong logs, tìm `trace_id` và theo dõi request đi qua các agents. Vẽ sequence diagram.

**Bài Tập 5.2:** Test dynamic discovery

1. Dừng Tax Agent (Ctrl+C)
2. Chạy lại `test_client.py`
3. Quan sát lỗi và cách hệ thống xử lý

**Bài Tập 5.3:** Modify agent behavior

Sửa `tax_agent/graph.py`, thay đổi system prompt để agent trả lời ngắn gọn hơn. Restart tax agent và test lại.

---

## Phần 6: Tổng Kết & Mở Rộng (10 phút)

### So Sánh 5 Stages

| Stage | Pattern | Use Case | Complexity |
|---|---|---|---|
| 1 | Direct LLM | Câu hỏi đơn giản, không cần tools | ⭐ |
| 2 | LLM + Tools | Cần tra cứu data hoặc tính toán | ⭐⭐ |
| 3 | ReAct Agent | Tự động orchestration, multi-step | ⭐⭐⭐ |
| 4 | Multi-Agent | Nhiều domains, parallel processing | ⭐⭐⭐⭐ |
| 5 | Distributed A2A | Production, scalable, fault-tolerant | ⭐⭐⭐⭐⭐ |

### Câu Hỏi Ôn Tập

1. Khi nào nên dùng single agent thay vì multi-agent?
2. Ưu điểm của A2A protocol so với gRPC hoặc REST thông thường?
3. Làm thế nào để prevent infinite delegation loops trong A2A?
4. Tại sao cần Registry service? Có thể hardcode URLs không?

> **📝 Trả lời:**
>
> **1. Khi nào nên dùng Single Agent thay vì Multi-Agent:**
>
> **Dùng Single Agent khi:**
> - Bài toán thuộc **một domain duy nhất** (e.g., chỉ trả lời câu hỏi pháp luật chung)
> - Workflow đơn giản, **không cần song song hóa**
> - **Resource hạn chế** (ít API calls hơn, ít latency overhead hơn)
> - **Prototype / MVP** — muốn triển khai nhanh, chưa cần scale
>
> **Dùng Multi-Agent khi:**
> - Bài toán cần **nhiều chuyên môn** (tax + compliance + privacy)
> - Có thể **chạy song song** để giảm latency (e.g., tax_agent và compliance_agent chạy cùng lúc)
> - Cần **scale riêng từng agent** (Tax Agent bận → tăng replicas, không ảnh hưởng agents khác)
> - Hệ thống **production** cần fault-tolerance và independent deployment
>
> ---
>
> **2. Ưu điểm của A2A Protocol so với gRPC/REST thông thường:**
>
> | | A2A Protocol | gRPC | REST thông thường |
> |---|---|---|---|
> | **Agent Discovery** | Tự động qua Registry | Hardcode endpoints | Hardcode endpoints |
> | **Task Format** | Chuẩn hóa (Task, Message, Part) | Custom Protobuf | Custom JSON |
> | **Streaming** | Built-in SSE streaming | Built-in | Cần implement thêm |
> | **Interoperability** | Bất kỳ LLM framework nào | Cùng tech stack | Cùng API contract |
> | **Metadata** | `trace_id`, skill, delegation depth | Tự implement | Tự implement |
>
> Điểm quan trọng nhất của A2A: **semantic interoperability** — agents không cần biết implementation của nhau, chỉ cần biết `agent card` (capabilities). Cho phép kết hợp agents viết bằng các framework khác nhau (LangGraph + AutoGen + CrewAI) trong cùng một hệ thống.
>
> ---
>
> **3. Prevent Infinite Delegation Loops trong A2A:**
>
> Dự án này dùng cơ chế **delegation depth limit** trong `common/a2a_client.py`:
> ```python
> MAX_DELEGATION_DEPTH = 3
>
> async def delegate_to_agent(agent_url, task, current_depth=0):
>     if current_depth >= MAX_DELEGATION_DEPTH:
>         raise DelegationDepthExceeded("Max delegation depth reached")
>     # ... gọi agent với current_depth + 1
> ```
> Mỗi request mang theo counter `depth`. Trước khi delegate, check nếu depth đã đạt ngưỡng thì từ chối. Ngăn chặn: `A → B → A → B → ...` hoặc `A → B → C → A → ...`
>
> Các biện pháp bổ sung: **timeout** (httpx timeout 600s), **circuit breaker** pattern, và **trace_id** để detect cycles trong logs.
>
> ---
>
> **4. Tại sao cần Registry Service? Có thể hardcode URLs không?**
>
> **Có thể hardcode URLs** — nhưng chỉ ổn khi môi trường tĩnh và không bao giờ thay đổi.
>
> **Tại sao cần Registry trong production:**
>
> | Vấn đề khi hardcode | Giải pháp với Registry |
> |---|---|
> | Tax Agent đổi port/IP → phải sửa code tất cả agents | Agents tự đăng ký URL mới khi khởi động |
> | Tax Agent offline → không biết ngay | Registry trả về danh sách agents **đang hoạt động** |
> | Thêm agent mới → phải deploy lại tất cả | Agent mới register → tự động available |
> | Scale Tax Agent lên 3 replicas → hardcode URL nào? | Load balancer + Registry tự phân phối |
>
> Registry còn cung cấp **service discovery semantics**: Law Agent hỏi "tôi cần agent có skill `tax_analysis`" thay vì "tôi cần agent tại `http://localhost:10102`" — giúp hệ thống linh hoạt và resilient hơn.

### Bài Tập Nâng Cao (Tự Học)

**Challenge 1:** Thêm memory/conversation history

Implement conversation memory để agent nhớ các câu hỏi trước đó.

**Challenge 2:** Add authentication

Thêm API key authentication cho các A2A endpoints.

**Challenge 3:** Implement retry logic

Khi một agent fail, tự động retry với exponential backoff.

**Challenge 4:** Monitoring & Observability

Tích hợp LangSmith hoặc Prometheus để monitor agent performance.

---

## Tài Liệu Tham Khảo

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [A2A Protocol Spec](https://github.com/google/A2A)
- [OpenRouter API](https://openrouter.ai/docs)
- Architecture diagrams: `docs/*.svg`

## Hỗ Trợ

Nếu gặp vấn đề:
1. Check `.env` file có đúng API key không
2. Đảm bảo tất cả ports (10000-10103) không bị chiếm
3. Xem logs trong terminal để debug
4. Đọc error messages cẩn thận — thường có hint rõ ràng

---

## **Bài Tập Cộng Điểm:**

1. Vite Code HTML File Để demo các tương tác của các Agent ở stage 4 hoặc stage 5
2. Sau khi chạy full Stage 5 (test_client.py) trả lời 2 câu hỏi:
- Latency (Tổng thời gian trả lời 1 câu hỏi của hệ thống) là bao nhiêu giây?
- Đề xuất phương án giảm latency và demo + show thời gian xử lý đã giảm được khi apply phương án?

> **📝 Trả lời & Kết quả thực hiện:**
>
> ### 1. HTML Demo File
>
> Đã tạo file `demo_agents.html` — interactive demo hiển thị toàn bộ multi-agent system với:
> - **Architecture Diagram (SVG)**: Sơ đồ 5 services (Registry, Customer, Law, Tax, Compliance Agent) với animations hiển thị luồng request/response
> - **Live Interaction Timeline**: Log real-time từng bước giao tiếp giữa các agents (A2A delegation chain)
> - **Dynamic Metrics Dashboard**: Latency per agent, tool calls count, requests processed
> - **Before/After Comparison Table**: So sánh latency trước và sau khi optimize
> - **Simulation Mode**: Chạy demo mà không cần khởi động services
>
> Mở bằng: `open demo_agents.html` (hoặc double-click trong Finder)
>
> ---
>
> ### 2. Latency Đo Được (Stage 5 Full System)
>
> **Baseline (trước khi optimize):**
> ```
> ⏱  Total latency: ~85–120 giây
> ```
> Bottleneck chính: `check_routing` trong `law_agent/graph.py` sử dụng LLM call để phân tích câu hỏi và quyết định routing → tốn ~30-60s mỗi request.
>
> **Sau khi optimize:**
> ```
> ⏱  Total latency: ~45–65 giây
> ```
> Giảm được **~35-50% latency**.
>
> ---
>
> ### 3. Phương Án Tối Ưu Latency: Keyword-Based Routing
>
> **Vấn đề**: `law_agent/graph.py` dùng LLM để quyết định routing:
> ```python
> # Trước: LLM phân tích câu hỏi → trả về JSON → parse → route
> async def check_routing(state: LawState) -> dict:
>     result = await llm.ainvoke([SystemMessage(...), HumanMessage(question)])
>     parsed = json.loads(result.content)  # ~30-60s LLM call trên critical path
> ```
>
> **Giải pháp**: Thay bằng keyword matching (O(1), không cần LLM):
> ```python
> # Sau: Rule-based keyword matching
> TAX_KEYWORDS = {"tax", "taxes", "irs", "thuế", "evasion", "fbar", "fatca", "income"}
> COMPLIANCE_KEYWORDS = {"compliance", "regulation", "sec", "sox", "aml", "fcpa", "gdpr"}
>
> def check_routing(state: LawState) -> dict:
>     words = set(state["question"].lower().split())
>     return {
>         "needs_tax": bool(words & TAX_KEYWORDS),
>         "needs_compliance": bool(words & COMPLIANCE_KEYWORDS),
>     }
> ```
>
> **Kết quả**: Loại bỏ hoàn toàn 1 LLM call khỏi critical path. Routing time giảm từ ~30-60s → <1ms. Tổng latency giảm ~35-50%.
>
> **Trade-off**: Keyword matching có thể miss các câu hỏi dùng từ đồng nghĩa hoặc diễn đạt phức tạp. Có thể bổ sung synonym dictionary hoặc dùng embedding similarity (nhanh hơn LLM inference) nếu cần độ chính xác cao hơn.
