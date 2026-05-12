# CodeAgent-RL: Replication of the RL pipeline described in the Composer 2 report for tool-using code agents

Ever since the [Composer 2 Technical Report](https://arxiv.org/abs/2603.24477) came out, I had been wanting to replicate its core RL pipeline. The main reason was not to train some especially strong model, but to understand the infrastructure behind it properly: how distributed training is wired up, how rollout generation and training run together, how weight syncing fits in, and what RL for tool-using code agents actually looks like in code. This project was much more about reasoning through the system and making the full pipeline work end to end. In the end, I managed to get the whole thing running with 3 H200 GPUs, vLLM, and DeepSpeed ZeRO-3.

This blog is long because I tried to document the implementation in as much detail as I reasonably could. Some parts, especially the training section, are heavier and more paragraph-driven than a polished tutorial would probably be. That was intentional. I wanted this to read more like a careful walkthrough of what is actually happening in the code than a short high-level summary that skips the messy parts.

Before reading, it will help a lot if you have already gone through the Composer report itself. I made a few changes from the original design for practicality and convenience. The overall structure is still quite close, but there are places where I may not call out every deviation immediately. So having the original report in your head makes it much easier to tell what came from Composer and what came from my own implementation choices.

All of the code is available here: [CodeAgent-RL on GitHub](https://github.com/anaumghori/CodeAgent-RL)

<figure class="blog-theme-diagram">
  <img class="theme-image theme-image-light" src="images/CodeAgent-RL/light.png" alt="Full pipeline diagram, light version" />
  <img class="theme-image theme-image-dark" src="images/CodeAgent-RL/dark.png" alt="Full pipeline diagram, dark version" />
</figure>

<p><em>This architecture diagram was generated with Claude. If you want to properly inspect the architecture details, it is worth zooming in and exploring it closely.</em></p>

# Early Decisions

Choosing the base model was probably the easiest decision in this entire setup because I already had my eyes on Hermes-4-14B from NousResearch long before I started wiring the rest of the pipeline together. The bigger issue was not picking a good model in isolation, but picking one that still made sense after skipping several stages from the original Composer tech report. Composer goes through pretraining continuation and lightweight SFT before RL ever begins, which means their RL stage starts from a model that is already heavily prepared for coding-style reasoning and tool usage. Since I was not reproducing those earlier stages, the starting checkpoint mattered much more for me.

I needed something that already understood coding reasonably well out of the box and could follow structured instructions reliably. Hermes fit that almost immediately. It has 14.77 billion parameters that spread across 40 layers and the BF16 checkpoint itself occupies around 27.5 GB on disk, which also matches the memory footprint reported by vLLM during startup. More importantly, it already supports structured tool calls through `<tool_call>{...}</tool_call>` formatting, hybrid `<think>...</think>` reasoning traces, and the kind of instruction-following behavior that this project depends on pretty heavily.

Once the model decision was locked in, the next thing I spent time on was datasets. Since the datasets originally used by the Composer team are private and not publicly available, I had to make my own decisions regarding the training mixture and overall data composition. I eventually settled on two datasets from Hugging Face:
* `nebius/SWE-rebench-V2` which mainly contains repository-level bug fixing and software-engineering-style tasks, covering roughly 32K instances across 20 programming languages. This dataset made up 70% of the final training mixture, and I also reserved a stratified held-out validation subset containing 50 problems.
* `caijanfeng/CodeContests-O` which contains around 11.4K large-scale algorithmic programming problems. This formed the remaining 30% of the training mixture and I used 50 samples from its validation split as part of the fixed evaluation setup.


### Sample anatomy

The anatomy here is just a prompt-construction step. The original dataset samples contain multiple fields or columns, and the implementation selects the ones that matter for the task itself, then reshapes them into a prompt that reads like a concrete assignment. 

For SWE-rebench-V2, that means starting from dataset fields such as `repo`, `base_commit`, `problem_statement`, and `pr_description`. The prompt uses the repository field, the base commit field, and the issue text itself. That issue text comes from the dataset's `problem_statement` field, with `pr_description` used as a fallback when needed. Those selected fields are then transformed into a repository-oriented prompt like this:

:::
Repository: wtforms/wtforms  
Base commit: 848d28d67e45cda7a06c4c8ed2768e6a8cb1c016  
Issue / PR description:  
[the original problem statement]  

Diagnose the problem, locate the relevant files, make a minimal patch that satisfies the failing tests without breaking existing tests, and run the test suite to verify the fix.  
:::

CodeContests-O needed the same treatment, but with a different set of fields because the task itself is different. There is no repository field here and no commit to anchor against. Instead, the important columns are the dataset's `name` field and `description` field, which together define the programming problem. Those are the fields that get lifted into the prompt body, after which the workflow instructions are added around them.

:::
Problem: 584_B. Kolya and Tanya  
[full problem description]  
Workflow requirements:  

1. Use `write_file` to create `solution.py` containing the full runnable Python solution.
2. Use `run_command` to execute `python solution.py` against the provided sample cases before finishing.
3. Only provide a final answer after both steps above are complete.
4. Do not put the solution code only in the final message; the code must be written to `solution.py` via the tool.  

Read input from stdin and write the answer to stdout.  
The rollout is incomplete unless `solution.py` is created and verified through tools.  
:::


### GPU setup

The first place where I actually had to stop and think carefully was the GPU setup itself. I had already decided from the beginning that I would be using Modal because of the free monthly credits, so the real question was figuring out what GPU configuration would make the most sense for this kind of training pipeline. Initially I thought about using two H100 GPUs with 80 GB VRAM each, but the more I mapped the system design out, the less practical that setup became. In the Composer tech report, rollout generation and training run simultaneously on separate GPUs. That separation matters a lot because the system is constantly generating trajectories while training updates are happening in parallel.

With only two H100s, I would effectively end up alternating between rollout generation and training because the training step would require access to both GPUs due to the memory overhead from the 14.77B Hermes parameters alongside optimizer states, gradients, activations, KV cache growth and other runtime allocations. Once training on both GPUs finished, I would then have to separately switch back into rollout generation mode again on either one or both GPUs to produce fresh trajectories before continuing training. That completely breaks the concurrent rollout-and-train structure that Composer relies on. I could technically still make the setup work, but it would stop resembling the original design philosophy in any meaningful way. I also specifically wanted room to experiment with DeepSpeed and parallelism strategies during training instead of running the entire system in a rigid single-purpose configuration. If both GPUs are already occupied handling rollout generation or training, there is barely any room left to properly explore distributed setups in practice.

At that point I started considering H200s instead which provide 141 GB VRAM per GPU. Even then, I moved from two H200s to three because running right at the memory ceiling is usually a terrible idea during RL training. All it takes is one unexpected spike in activation memory to trigger OOM failures halfway through a run.

A reasonable question here is why I did not simply scale the H100 count upward instead. The answer mostly came down to cost efficiency and memory density. To get enough usable headroom with H100s, I would realistically need four of them. At Modal pricing, each H100 costs around $3.95 per hour, which pushes the total to roughly $15.80 every hour. The three H200 setup costs around $4.54 per GPU per hour, which comes out to about $13.62 hourly in total. So despite H200s being individually more expensive, the overall setup actually ended up cheaper by about $2.18 per hour while simultaneously giving me significantly more VRAM headroom and a much more flexible training layout. And honestly, part of me also just wanted an excuse to finally work with H200s directly.

In the most recent execution I did, once vLLM had finished its startup profiling and graph capture on the inference GPU, it reported about `95.18 GiB` of available KV-cache memory, a cache capacity of `623,792` tokens, and an estimated `19.04x` maximum concurrency at the configured `32,768` token context length. It also spent about `0.79 GiB` on the CUDA graph pool.

# Environment Setup and Parallel Runtime

Before getting into rollout generation, training, rewards, vLLM, and the rest of the lower-level implementation details, I first want to build the mental model for how the whole system is moving. That makes the later sections much easier to follow because there are several parts running in parallel and feeding into each other continuously.

The entrypoint for this project is `modal run [src/modal_app.py](https://github.com/anaumghori/CodeAgent-RL/tree/main/src/modal_app.py)`. That launches the whole pipeline inside one Modal job, which is exactly how I ran it throughout the project. That part matters because the SWE side depends on Modal-native sandboxes created from the dataset images, so the training system and the repository execution layer are both living inside the same Modal-managed runtime from the start.

The environments themselves also differ by task type. CodeContests is the simpler case. It mostly creates a temporary local workspace where the model writes `solution.py`, runs commands, and gets evaluated against the provided test cases. SWE is heavier. It needs an actual repository environment where the model can inspect files, edit code, run shell commands, and execute tests in isolation. In this implementation, that isolation layer is a Modal sandbox created directly from the dataset image for the specific SWE instance.

With that environment context in place, we can look at the actual runtime as three parallel pieces working together:

#### **1. GPU 2 handles inference.**

This is the vLLM side of the system. A persistent inference engine sits on GPU 2 and keeps generating rollouts from the current policy. One prompt goes in, and the model samples four separate rollouts for that prompt. These four rollouts stay grouped together all the way through reward computation and GRPO advantage calculation. When the group is finished, it gets pushed into the rollout buffer. 

#### **2. GPU 0 and GPU 1 handle training.**

These two GPUs are paired together for training under DeepSpeed ZeRO-3. At the very beginning, training does not start immediately because it has nothing to train on yet. It first waits for GPU 2 to generate the initial rollout groups and push them into the rollout buffer. Once rollout data starts showing up in that buffer, the training workers begin pulling batches from it and training on them. After that point, the system settles into the steady loop: inference keeps adding rollout data, and training keeps consuming that data in parallel.

#### **3. The CPUs handle prompt and environment preparation in parallel.**

The model cannot generate a useful rollout unless it has both a task prompt and a live environment to interact with. Those environments are not only created when inference asks for them at the last second. The code tries to prewarm them ahead of time. Once upcoming prompts are selected, their environments start getting prepared during startup and then keep getting prepared in the background as new prompts are queued up. If a prewarmed environment is ready, inference picks it up immediately and keeps moving. If it is not ready in time, the system falls back to building that environment inline and continues from there. So the point of this CPU-side work is to hide as much setup latency as possible, keep ready environments waiting for rollout generation, and only pay the full setup cost directly when the prewarm path misses.

You might naturally wonder what inference does if prewarming is still happening when it needs an environment. The answer: it waits, but only for a bounded amount of time. In [src/orchestrator.py](https://github.com/anaumghori/CodeAgent-RL/tree/main/src/orchestrator.py), the inference path waits up to 1 second for a CodeContests environment and up to 10 seconds for a non-CodeContests environment before giving up on the prewarm path. That waiting logic is handled through `acquire_for_prompt()` in [src/environments/pool.py](https://github.com/anaumghori/CodeAgent-RL/tree/main/src/environments/pool.py). If a matching prewarmed environment becomes ready within that window, inference takes it immediately. If not, inference claims that prompt for inline setup and builds the environment itself. A background prewarm worker might still finish a similar environment afterward, but the pool tracks that inline claim and discards the late prewarm result instead of keeping two duplicate ready environments around. So the real behavior is very explicit: wait briefly, use the prewarmed environment if it shows up in time, and otherwise switch to inline setup without letting the system pile up redundant copies for the same prompt.

| Phase (recent run) | Environment setup timing | What it represents |
|---|---|---|
| Initial cold-start phase | 47.03s p50, 47.21s max | Early environment builds before the system and queue fully warm up |
| Post-warmup steady state | 2.56s p50 | Environment setup after prewarming and queue stabilization |

Prewarming hides most of the cold-start setup cost from the rollout critical path, allowing rollout generation and inference to behave much more like a continuously moving pipeline. 

After that initial fill, the interaction looks more like this:

1. Prompts are selected from the dataset stream.
2. CPU workers prewarm the environments for those upcoming prompts.
3. GPU 2 runs inference and generates four rollouts for a prompt.
4. The completed rollout group gets evaluated and pushed into the rollout buffer.
5. GPUs 0 and 1 pull rollout groups from that buffer and run training steps on them.
6. While that training is happening, inference is already generating more rollouts and the CPUs are already preparing later environments.

There is also a small but important parent-side concurrency detail here. The orchestrator is not just one thread doing everything in order. The main thread owns the training loop and checkpoint/eval cadence, but rollout generation itself runs in a background inference thread, and there is also a separate IPC dispatcher thread whose whole job is to listen for asynchronous messages from trainer rank 0. That dispatcher thread is what services the vLLM pause / update / resume requests during weight sync. So even though the whole thing is one parent process, there are really several control paths moving at once inside it. 

# vLLM and rollout generation

Before even getting into the inference engine itself, it is worth looking at what we are actually sending into it. Rollout generation here is driven by two task-specific prompt paths, one for SWE-style repository tasks and one for CodeContests-style programming tasks. The task prompts themselves were already defined earlier in the [Sample anatomy](#sample-anatomy) section, and those same task prompts are what get used here during rollout generation. What changes at this stage is that a system prompt gets placed above them, and the tool definitions also get injected directly into the chat template. A rollout group consists of four separate rollouts for the same prompt, but the code loops through them one by one, where each rollout gets its own fresh environment, its own conversation history, and its own sampled trajectory. So when I say “four rollouts for one prompt,” what that really means in practice is four independent attempts by the same policy to solve the same task under the same prompt.

The tool side stays the same throughout the runtime. The model is given five tools in total: `read_file`, `write_file`, `run_command`, `search_code`, and `list_directory`. These are passed into Hermes through the tokenizer chat template, so the model sees them as part of the prompt itself rather than as some external runtime magic. That matters because the rollout loop expects Hermes to return tool calls in its native `<tool_call>{...}</tool_call>` format, which the code then parses and dispatches back into the environment.

For SWE rollouts, the system prompt is:

:::
Before every visible response, reason privately inside `<think>...</think>` and keep that reasoning out of the visible answer except through the final result.

You are an expert software engineer. Use the provided tools to inspect code, run commands, write fixes, and verify your work with the test suite.
:::

For CodeContests, the thinking instruction stays the same, but the rest of the system prompt becomes stricter because the workflow is narrower and easier to game if the rules are not explicit:

:::
Before every visible response, reason privately inside `<think>...</think>` and keep that reasoning out of the visible answer except through the final result.

You are solving a competitive programming task inside a strict tool-using harness. You must use tools to complete the task. The required order is: (1) use `write_file` to create `solution.py`, (2) use `run_command` to execute and verify `solution.py`, (3) only then provide a brief final response. Do not treat a prose answer or a markdown code block as a valid solution submission. If `solution.py` has not been written and verified through tools, the task is incomplete.
:::

The central object here is vLLM's `LLM` class. It is basically the long-lived inference engine that holds the model in memory and gives the rollout loop a direct way to send chat conversations into the model and get back generated text, token IDs, and logprobs.

```python
# Source: src/inference/vllm_server.py and src/inference/rollout.py
from vllm import LLM, SamplingParams
from vllm.config import WeightTransferConfig

# Long-lived vLLM engine on the inference GPU.
llm = LLM(
    model="NousResearch/Hermes-4-14B",      # Base model used for rollout generation
    tensor_parallel_size=1,                 # One dedicated inference GPU
    dtype="bfloat16",                       # BF16 inference
    gpu_memory_utilization=0.90,            # Fraction of GPU memory vLLM may use
    max_model_len=32768,                    # Maximum context length for inference
    enable_prefix_caching=True,             # Reuse shared prompt prefixes across rollouts
    enable_sleep_mode=True,                 # Allow pause / wake during weight sync
    weight_transfer_config=WeightTransferConfig(backend="nccl"),
)

# Sampling configuration used by the rollout loop.
sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    max_tokens=8192,
    logprobs=1,                             # Return per-token logprobs for RL training
)

# Rollout generation calls chat(...) directly.
outputs = llm.chat(
    messages=messages,                      # System + task + prior tool outputs
    sampling_params=sampling_params,
    tools=TOOL_DEFINITIONS,                 # Tool schema injected into the chat template
)

text = outputs[0].outputs[0].text
token_ids = list(outputs[0].outputs[0].token_ids)
logprobs = outputs[0].outputs[0].logprobs
```

Every rollout starts from the same general scaffold: a system prompt, a task prompt, and the same tool definitions. The conversations then branch based on tool outputs and model decisions, but the front half of the prompt structure is highly repetitive. That is the kind of pattern where prefix caching helps, because the expensive shared prefix does not have to be recomputed from scratch every single time.

One small but important change here was the generation budget for CodeContests. In the beginning, I was using the same maximum generation budget for both SWE and CodeContests. Later, that got split in code so the default rollout path still uses `max_generation_tokens = 8192`, but CodeContests specifically switches to `codecontests_max_generation_tokens = 4096`. This override is applied inside `_sampling_params()` in [src/inference/rollout.py](https://github.com/anaumghori/CodeAgent-RL/tree/main/src/inference/rollout.py).

```python
# Source: src/inference/rollout.py
def _sampling_params(self, prompt: Prompt):
    from vllm import SamplingParams
    max_tokens = self.cfg.sequence.max_generation_tokens
    if prompt.source == SOURCE_CODECONTESTS:
        max_tokens = self.cfg.sequence.codecontests_max_generation_tokens
    return SamplingParams(
        temperature=self.cfg.sampling.temperature,
        top_p=self.cfg.sampling.top_p,
        top_k=self.cfg.sampling.top_k,
        max_tokens=max_tokens,
        logprobs=1,
    )
```

This change came from a pretty obvious failure mode once I started looking at actual rollouts. CodeContests tasks were generally easier than SWE, but the model would still keep generating for far too long and drift away from the task. It would keep talking without producing useful tool calls, without writing `solution.py`, and without reaching a real completion. In a lot of those cases, all four sampled rollouts looked almost identical and equally useless. The fix was not just lowering the token budget. The CodeContests system prompt was also rewritten to make the required tool order much stricter. After that, and with the `4096` token cap in place, the rollouts became much more useful. Instead of four long drifting failures, it became much more common for at least one or two rollouts in the group to actually produce the right tool calls and write `solution.py` the way the pipeline expected.

Now let's understand what a rollout is allowed to do before it gets cut off or compressed. Each rollout can go up to `24` turns, where a turn starts when the current conversation state is rendered and sent into vLLM, and ends only after that generation has been fully handled. So if the model answers directly, that entire response counts as one turn. If the model emits one or more tool calls, those tools are executed immediately and their results are appended back into the same conversation as `tool` messages, while still remaining part of the same turn. The next turn only begins once the model is called again on that updated conversation. Tool calls and tool outputs do not create extra turns by themselves because the code increments `n_turns` once per model generation.

Self-summarization is one of the central ideas discussed in the Composer report, where long coding trajectories are not treated as one giant uninterrupted prompt-response pair. Instead, the rollout gets periodically compressed into structured summaries generated by the model itself, allowing long tool-using trajectories with shell outputs, retries, file reads, and tool results to continue without collapsing under growing context size. The summary is not supposed to be an external note written for the model. It is part of the model's own trajectory.

That is also exactly how our code works. Each rollout is tracked as one or more `RolloutSegment`s. A segment is one contiguous stretch of conversation plus its token-level metadata. As the rollout grows, the generator watches two thresholds. The soft trigger fires when visible tokens reach `10000`. The hard trigger fires when the full segment reaches `12000` tokens. Once either one is hit, the current segment is closed and the model is asked to summarize its own progress before continuing. Because that summary is generated by the policy itself, that summary then becomes part of the rollout itself. The model generates it, its tokens and logprobs are stored just like any other model output, and then a new segment begins. But the rollout does not restart from zero. The next segment is rebuilt from the original system prompt and task prompt, then the summary is inserted as prior work, and then the most recent `2` turns from the previous segment are also kept. So the model gets a compressed memory of everything earlier plus a short local window of the most recent interaction history. That is the practical meaning of self-summarization here.

The summary prompt is:

:::
Produce a STRUCTURED SUMMARY of work so far. Cover: current understanding of the task, files examined and modified, tests run and outcomes, hypotheses explored, and remaining work. Keep it concise but information-dense.
:::

Once you have that framing in mind, the actual rollout flow becomes much easier to follow. A rollout begins by rendering the current conversation with the tokenizer chat template and sending that into vLLM. vLLM returns generated text, token IDs, and per-token logprobs. The code then separates the model output into private thinking inside `<think>...</think>` and the visible assistant content after it. The visible content is scanned for Hermes-style tool calls, and those calls are executed inside the current environment before the next turn begins.

The tool-call parsing itself is very small and direct:

```python
# Source: src/inference/rollout.py
TOOL_CALL_RE = re.compile(r"<tool_call>\\s*(\\{.*?\\})\\s*</tool_call>", re.DOTALL)

def _parse_tool_calls(text: str) -> list[dict]:
    calls = []
    for match in TOOL_CALL_RE.finditer(text):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "name" in payload:
            calls.append(payload)
    return calls
```

So a typical rollout usually looks like this:

1. Before anything is generated, the rollout is represented as a structured chat conversation made of messages like `system`, `user`, `assistant`, and `tool`. The chat template is the formatting layer that turns those structured messages into the exact prompt text and token layout the model expects.
2. The conversation is rendered with the system prompt, task prompt, and tool definitions.
3. vLLM generates one assistant step containing private reasoning and, if needed, one or more `<tool_call>` blocks.
4. The code strips the `<think>` section away from the visible answer.
5. That visible answer is appended back into the running conversation as the assistant message for the current turn.
6. If tool calls were emitted, they are parsed and executed immediately inside the environment.
7. The resulting tool outputs are appended back into the same conversation as `tool` messages, then re-rendered through the same chat template before being stitched back into the rollout state. This means the model does not just receive raw text blobs from the environment, but instead sees properly structured tool responses in the exact conversation format used during generation.
8. Only after that does the next turn begin, with vLLM being called again on the updated conversation state.
9. If the segment grows too large, self-summarization compresses the earlier work and the rollout continues in a fresh segment.
10. The rollout ends when the model returns a final visible answer with no tool calls left to execute, or when it hits the turn limit.

So what does one finished rollout group actually contain by the time it leaves the inference side? It contains four separate rollouts for the same prompt, and each rollout carries:

1. Its own environment interaction history, including the full conversation state built from assistant messages and tool outputs
2. Per-token `logprobs`, `token_ids`, and `loss_mask`, so training can tell exactly what the model generated and where gradient should apply
3. Segment boundaries and segment-level token counts for thinking text, tool-call text, tool-output text, and final visible text
4. Rollout-level counts such as total tool calls and total turns
5. The final visible response that the model ended on
6. Test metadata collected after the environment run


# Rewards 

Once we have four completed rollouts for the same prompt, the next step is assigning rewards. This reward pass sits directly at the boundary between inference and training, because we only push the rollout group into the rollout buffer after each rollout already has its final scalar reward and group-relative advantage attached to it. By the time training sees the trajectories, they are no longer just raw rollout generations but fully scored training examples. Let's move onto defining the rewards that our code contains. 

#### Correctness reward is task-specific

SWE and CodeContests do not use the same correctness function and I think that distinction matters a lot. For SWE, the reward is based on the repository-style `FAIL_TO_PASS` / `PASS_TO_PASS` structure. The model gets full credit only when it fixes the target failing tests without breaking the ones that were already passing. If it introduces regressions, it gets penalized. If it makes partial progress without regressions, it gets partial credit.

```python
# Source: src/training/rewards.py
def compute_swe_correctness(...):
    if pass_to_pass_total > 0 and pass_to_pass_passed < pass_to_pass_total:
        return -0.2
    if fail_to_pass_total == 0:
        return 0.0
    if fail_to_pass_passed == fail_to_pass_total:
        return 1.0
    if fail_to_pass_passed > 0:
        return 0.5 * (fail_to_pass_passed / fail_to_pass_total)
    return 0.0
```

For CodeContests, the correctness side is much simpler: it is just pass rate over the test set. But there is one extra rule that ended up mattering a lot in practice. If the model does not follow the required workflow, correctness is forced to `0.0` even if the answer text looks reasonable. So for CodeContests, correctness is not just about whether some code could have been right in theory. It is tied to whether the agent actually used the environment the way the pipeline expects.

```python
# Source: src/training/rewards.py
def compute_codecontests_correctness(tests_passed: int, tests_total: int) -> float:
    if tests_total == 0:
        return 0.0
    return tests_passed / tests_total


def compute_correctness_reward(test_result: dict, source: str, cfg):
    if source.startswith("swe"):
        raw = compute_swe_correctness(...)
    else:
        if not test_result.get("required_workflow_followed", False):
            return 0.0
        raw = compute_codecontests_correctness(
            test_result.get("tests_passed", 0),
            test_result.get("tests_total", 0),
        )
    return cfg.reward.correctness_weight * raw
```

#### Auxiliary rewards

I did not want the whole training signal to collapse into "did the tests pass or not," especially for coding agents where two failed trajectories can still differ a lot in quality. I also did not want these extra rewards to overpower the main objective. So the code keeps correctness as the center of gravity and uses a fixed set of much smaller auxiliary rewards to gently shape behavior around it. On top of correctness, the code adds:

- Syntax validity for written Python files
- A penalty for leaving `TODO`, `FIXME`, or `XXX` markers behind
- A minimal-diff penalty when the agent touches files that were not relevant to the task
- A tool-hygiene penalty for redundant repeated tool calls
- A verification/workflow signal based on whether the agent actually followed the intended solve-and-test pattern

The weights are intentionally small compared to correctness:

```python
# Source: src/config/config.py
correctness_weight = 1.0
aux_syntax_weight = 0.05
aux_no_todo_weight = 0.05
aux_minimal_diff_weight = 0.05
aux_tool_hygiene_weight = 0.05
aux_test_discipline_weight = 0.05
```

#### Length penalty keeps the model from sprawling

If you only reward correctness, the model has a very real tendency to sprawl. It will keep thinking, keep calling tools, keep printing outputs, and on easy tasks it can waste a lot of motion doing work that simply was not necessary. To counter that, the reward code computes a single scalar effort variable `x`. That number is just a compact way of saying, "how much work did this rollout do?" It combines the main things that make a trajectory expensive or unnecessarily long: thinking tokens, tool-call tokens, tool-output tokens, final visible tokens, total tool calls and total turns. Once those are collapsed into one effort score, the code applies a nonlinear penalty to it and subtracts that from the reward. So the mechanism here is very direct: more effort increases `x`, higher `x` increases the penalty, and a higher penalty pulls the total reward down.

The three config parameters controlling that penalty are `k`, `q`, and `lambda`. `lambda` is the overall strength of the penalty. `k` controls how quickly the penalty starts ramping up as effort increases. `q` controls the curvature, which is what gives the function its concave shape instead of making it grow linearly forever. In practice, that means the model still has room to spend effort on genuinely hard tasks, but it pays a clearer cost for dragging out easy ones. So if two rollouts both solve the task, the cleaner and more efficient one should usually come out ahead. That was the whole point of adding this term in the first place.

```python
# Source: src/training/rewards.py
# Nonlinear length-penalty config.
k = 0.15
q = 1.5
lambda_ = 0.05

x = (
    think_tokens / 512.0
    + tool_call_tokens / 1024.0
    + tool_output_tokens / 4096.0
    + final_tokens / 512.0
    + 0.25 * n_tool_calls
    + 0.5 * n_turns
)
# Turn that effort score into a concave increasing penalty and subtract it.
penalty = lambda_ * (((1.0 + k * x) ** (1.0 - q) - 1.0) / (k * (1.0 - q)))
total_reward = correctness + auxiliary_total - penalty
```

#### Advantages and Dr. GRPO

Once the scalar reward has been computed for each of the four rollouts in the group, the next job is to turn those rewards into advantages. In GRPO, the advantage is basically the "better or worse than expected" signal attached to each rollout. Since the four rollouts in a group all came from the same prompt, GRPO compares them against each other instead of comparing them to some separate value model.

In a standard GRPO, that relative comparison is usually followed by an extra normalization step so the scale of the advantages is standardized within the group. This implementation intentionally does not do that. It keeps the group-mean subtraction, but it does not divide by the group standard deviation. That is the main Dr. GRPO change being used here, with Dr. GRPO acting as the variant of GRPO that drops that extra normalization step. I think the easiest mental model here is just: the pipeline samples four attempts for one prompt, scores all four, computes the group's average reward, and then asks which rollouts landed above that average and which landed below it. Those above-average rollouts get positive advantage, those below it get negative advantage, and that is the signal that gets passed into training.

```python
# Source: src/training/rewards.py
def compute_group_advantages(rewards: list[float]) -> list[float]:
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    # For a group with rewards [r1, r2, r3, r4], each advantage is Ai = ri - mean(group).
    return [r - mean for r in rewards]
```

The cleanest example I was able to capture through the logs came from the `584_B. Kolya and Tanya` sample, where the same prompt and same policy version still produced four very different trajectories inside a single GRPO group. As we can see, two rollouts drifted into weaker workflow behavior and fell below zero once penalties were applied, while the other two followed the tool path correctly and ended up far above the group mean. The Group success rate: was `0.5`.

| Rollout | Reward | Advantage | Outcome |
|---|---|---|---|
| 1 | -0.273 | -0.554 | Poor workflow behavior |
| 2 | -0.374 | -0.654 | Poor workflow behavior |
| 3 | 0.889 | 0.609 | Strong tool usage and test passing |
| 4 | 0.879 | 0.599 | Strong tool usage and test passing |


The SWE side gave a different but equally useful example, which came from the logs for the `wtforms__wtforms-614` sample. This rollout clearly did not fully solve the task, but it still made enough real progress that flattening it down to the same reward as a useless trajectory would have thrown away signal the trainer could still learn from.

| Metric | Value |
|---|---|
| FAIL_TO_PASS tests solved | 242 / 262 |
| PASS_TO_PASS regressions | 0 |
| Raw correctness reward | 0.462 |
| Length penalty | 0.294 |
| Final reward | 0.168 |

#### Rollout buffer and bounded staleness

Once rewards and advantages have been attached, the rollout group is finally ready to leave the inference side. At that point it gets pushed into a single shared `RolloutBuffer` that sits between inference and training. Each entry in that buffer is one fully processed `RolloutGroup`: one prompt, four completed rollouts, and all of the reward-related fields that were added after verification.

The buffer itself is intentionally simple. It is an in-memory FIFO queue, so groups are appended in the order inference produces them and pulled from the front in the order training consumes them. In the code, the buffer was configured to hold at most `50` rollout groups at once. Since each group contains `4` rollouts, that means the system can hold up to `200` completed rollouts in memory before inference has to slow down. That limit is there on purpose. I did not want inference to keep generating indefinitely if training fell behind, because then the system would just accumulate older and older samples that were becoming less relevant to the current policy.

So when the buffer fills up, inference blocks and waits for space to open. In the other direction, when training asks for groups and the buffer is empty, training waits for new ones to arrive. If the wait times out, the trainer can still receive a partial batch or nothing at all, and the code tracks that as an underrun. That behavior is what keeps the pipeline honest. It does not let one side run arbitrarily far ahead of the other just because the architecture is asynchronous.

There is also a second guardrail on top of the size limit: policy staleness. Every rollout group is tagged with the `policy_version` that generated it, and training only consumes groups that are still close enough to the current model version. In the RL implementation, the maximum policy staleness was set to `2`, which means groups that fall more than two policy versions behind are dropped instead of being reused indefinitely. That is a pretty important part of the design, because the whole point of the buffer is to preserve concurrency without letting the system drift too far off-policy while training is still moving forward.

# Training

Once rollout groups have rewards, advantages, and `policy_version` tags attached to them, the system moves into the training stage. This is the point where the pipeline becomes much more about moving tensors around correctly. The high-level idea is simple enough: inference produces finished trajectories, training revisits those exact trajectories under the current weights, and then the updated weights get pushed back into the live vLLM engine so the next rollouts come from a newer policy. But in practice, getting that loop to behave cleanly required being pretty explicit about process boundaries, batching, and synchronization.

### Wrapper versus training architecture

The first thing I want to separate is deployment from internal structure.

- `modal run [src/modal_app.py](https://github.com/anaumghori/CodeAgent-RL/tree/main/src/modal_app.py)`: this launches one remote job inside a single Modal container and gives that container the full runtime bundle the pipeline needs: `3×H200` GPUs, CPU, RAM, checkpoint storage, secrets, and the rest of the surrounding infrastructure.

Once that job is actually running, the runtime effectively splits into two major roles. 

1. The first role is the main Python process (parent process) running the `Orchestrator` class, which stays on the CPU side and acts as the control plane for the entire system. It owns the rollout buffer, prompt queue, environment pool, vLLM wrapper, evaluator, checkpoint and evaluation scheduling, along with the IPC channel to trainer rank 0, making it responsible for coordinating the overall rollout and training flow.

2. The second role is the two trainer subprocesses launched from [src/training/launcher.py](https://github.com/anaumghori/CodeAgent-RL/tree/main/src/training/launcher.py), one for each training GPU. These are the actual DeepSpeed workers handling the distributed training side of the pipeline. The setup uses `torch.distributed`, which is the PyTorch layer that allows multiple Python processes to coordinate together as one distributed training job. Each subprocess gets mapped to its own GPU, and once both subprocesses join the same distributed process group, they become rank 0 and rank 1 across the two training GPUs. Each trainer subprocess loads the model, joins the distributed group, and then participates in the actual training step: forward passes, backward passes, optimizer steps, ZeRO-3 parameter gathers, all-reduces for shared metrics, and later the collectives used for weight transfer back into vLLM.

`CUDA_VISIBLE_DEVICES` is the environment variable that decides which physical GPU a process is allowed to see. The launcher gives each trainer subprocess its own mapping so that each subprocess sees exactly one training GPU instead of all three GPUs in the node. That matters because GPU 2 is already being used by the long-lived vLLM inference engine. If the training side were allowed to treat all three GPUs as part of its own world, the training processes and vLLM would start stepping on each other’s device visibility and ownership assumptions. Narrowing visibility per subprocess avoids that. Rank 0 sees one training GPU, rank 1 sees the other training GPU, and the parent process separately stays attached to the inference GPU.

### What the trainer actually receives each step

At steady state, the orchestrator keeps pulling ready groups out of the rollout buffer and sends them to trainer rank 0 over the Unix-domain socket IPC channel. Rank 0 is the only worker that receives the full Python payload. Rank 1 does not get the raw rollout objects directly, which means rank 0 is where the training step first gets assembled into something the workers can actually consume.

The number of rollout groups sent per optimizer step comes from the effective batch size configuration. Training uses a `micro_batch_size` of `1`, `gradient_accumulation_steps` of `16`, and `2` training GPUs. Since the effective batch size is calculated as `micro_batch_size × gradient_accumulation_steps × num_training_gpus`, the final batch size becomes `32`. Each GRPO group contains `4` rollouts, so the orchestrator targets `8` rollout groups per optimizer step.

The trainer flattens every rollout into one or more `TrainingSequence` objects. If a rollout never had to self-summarize, it may contribute just one sequence. If it got segmented mid-trajectory, resulting training sequences can become larger. Earlier in the pipeline, long trajectories are compressed through self-summarization, and every time that happens the rollout is split into another segment. A segment is just one contiguous chunk of the rollout after that compression boundary. By the time the training side receives a rollout group, one rollout might still be a single uninterrupted sequence, while another might already be split across multiple segments because it got too long and had to summarize itself mid-trajectory. Training therefore converts each segment into its own training sequence with token IDs, the loss mask, the old logprobs from sampling time, and the advantage inherited from the full rollout. That is what solves the sequence-growth problem: long rollouts are still trainable, but they are trainable as several linked sequences rather than one giant monolithic example.

In practical terms, those training sequences are also almost the same thing as the microbatches the trainer actually consumes. Each GPU processes one sequence at a time, so the per-GPU microbatch size is intentionally set to `1`. That is mostly a memory-management choice. Once sequences get long, trying to pack several of them into one forward/backward pass would make the step much harder to fit cleanly. So instead of making each GPU handle a larger local batch, the code runs many one-sequence microbatches in a row and relies on DeepSpeed gradient accumulation to combine them into one optimizer step. That is why the earlier numbers line up the way they do: `16` accumulation steps on each of the `2` training GPUs is what turns these one-sequence microbatches into an effective batch size of `32`.

After the trainer has those sequences, it still has to decide how to divide them across the two training GPUs without making one worker handle mostly cheap short examples while the other gets stuck with all the expensive long ones. Once those training sequences exist, the next job is to distribute them across the two training GPUs without making one worker handle mostly cheap short examples while the other gets stuck with all the expensive long ones. That packing logic is our own implementation in [src/data/sequence_packing.py](https://github.com/anaumghori/CodeAgent-RL/tree/main/src/data/sequence_packing.py). The code first keeps all segments from the same rollout together on the same GPU. After that, it uses a simple greedy packing strategy: it sorts the rollout bundles from most expensive to least expensive, then keeps assigning the next bundle to whichever GPU currently has the lowest estimated load. The estimated load is based on sequence length squared rather than raw token count, because attention cost grows faster than linearly as sequences get longer.

There is also a small training-length curriculum inside this step. The length limit here applies to one individual rollout segment at a time. Training does not start by allowing the longest possible per-sequence context right away. Instead, it begins with `max_training_seq_length = 4096`. We define `seq_length_extension_step = 500` setting which means that after the trainer has completed 500 optimizer steps, it switches the per-sequence cap upward to `8192`. So the thing being extended is the maximum length allowed for each individual training sequence before it is packed onto a GPU. This part is our own training-policy choice rather than a framework feature. The point is to keep early training cheaper and more stable, then allow longer segments later once the run has already settled in. If a sequence is longer than the currently active limit, it is truncated before packing. 

Once rank 0 has flattened the rollout groups and packed the resulting sequences, there is still one more step in the handoff: rank 1 has to end up with the same step structure even though it never received the full rollout payload from the orchestrator. This is where plain distributed PyTorch comes in. Rank 0 becomes the source of truth for the packed step and shares only the processed training representation, not the original bulky rollout objects.

First, rank 0 sends lightweight metadata with `broadcast_object_list`, mainly the per-rank microbatch shapes. Then it broadcasts the actual tensors for `input_ids`, `attention_mask`, `loss_mask`, `old_logprobs`, and `advantages`. Rank 1 reconstructs only the microbatches assigned to itself.

This is one of those implementation choices that looks slightly fussy until you compare it to the alternative. Serializing bulky rollout-group Python objects into every worker process would have been a much messier control path. This approach keeps the heavy rollout payload on rank 0, then moves only the tensorized form through distributed collectives.

The DeepSpeed config behind that overall setup looks like this:

```python
# Source: src/config/deepspeed_config.py
{
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": int(5e7),
        "stage3_prefetch_bucket_size": int(5e7),
        "stage3_param_persistence_threshold": int(1e5),
        "stage3_gather_16bit_weights_on_model_save": True,
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-6,
            "betas": [0.9, 0.999],
            "weight_decay": 0.01,
        },
    },
    "gradient_clipping": 1.0,
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 16,
    "activation_checkpointing": {
        "partition_activations": True,
        "contiguous_memory_optimization": True,
        "cpu_checkpointing": False,
    },
}
```

At that point, both training ranks have their assigned microbatches and the actual optimization step can begin.

### The GRPO loss is replaying old trajectories under new weights

By the time training starts, the rollout side has already done the expensive part: it generated the trajectories, ran the tests, computed the rewards, and converted those rewards into group-relative advantages. So the trainer is receiving finished trajectories together with a signal telling it which rollouts were better and which were worse for the same prompt.

At the token level, training still operates on the exact sampled rollout generated earlier during inference. The model is not producing a new answer during training. Instead, it revisits the original trajectory token by token, recomputes the current logprobs over that same sequence, and compares them against the rollout-time logprobs to form the probability ratio `r_t = pi_theta / pi_old`. Tokens from higher-advantage trajectories are pushed toward higher probability, while lower-advantage or negative-advantage tokens are pushed downward, effectively making stronger rollout behavior more likely over time. To keep the policy from changing too aggressively in a single update, PPO uses the clipped surrogate objective `-min(r_t * A_t, clip(r_t, 1 - ε, 1 + ε) * A_t)`, which stabilizes training by limiting how far the update can move from the original rollout policy.

That gives the PPO-style ratio:

```python
# Source: src/training/loss.py
log_ratio = new_logprobs - old_logprobs
ratio = torch.exp(log_ratio)

pg_loss1 = ratio * adv
pg_loss2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv
pg_loss = -torch.min(pg_loss1, pg_loss2)

kl_penalty = kl_coeff * (old_logprobs - new_logprobs)
loss = ((pg_loss + kl_penalty) * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
```

There is also a KL regularization term implemented with the `k1` estimator `old_logprobs - new_logprobs`. While PPO clipping stabilizes local updates, the KL penalty discourages the current policy from drifting too far away from the rollout-time policy that originally generated the trajectories. Together, they keep the model learning from sampled rollouts without letting the policy change too abruptly between updates. So the PPO clipping and the KL penalty are doing related but distinct jobs: the clip prevents overly aggressive local updates, and the KL term adds a broader pressure against policy drift. 

The `loss_mask` decides where the gradient is actually allowed to land. Tokens with mask value `1` are tokens the model itself generated during rollout, so they are eligible for gradient. Tokens with mask value `0` are prompt or environment context tokens, such as the original prompt text or tool outputs, so they stay inside the sequence for conditioning but do not receive credit or blame during the update. That means the loss only applies to model-generated tokens: thinking text, tool-call text, self-summaries, and final visible answers. The normalization is token-level rather than response-length-normalized, which keeps the optimization aligned with the way these rollout sequences were actually constructed.

The warmup effect became much clearer once I compared the earliest optimizer steps against the later steady-state behavior from the same run. Most of the gap came from the system paying its initial setup and warmup costs early on, while the later steps reflected the actual steady-state training workload much more closely.

| Metric | Step 1 | Step 2 |
|---|---|---|
| Total step time | 6.52s | 2.54s |
| Forward pass | 2.59s | 0.64s |
| Backward pass | 3.49s | 1.62s |
| Optimizer step | 0.31s | 0.23s |

The memory metrics also made the H200 choice feel justified on the training side. This was not catastrophically close to the limit, but it was definitely operating in the regime where small mistakes in sequence length, activation memory, or accumulation settings could have turned into very annoying OOM failures.

| Training memory metric | Observed range |
|---|---|
| Allocated memory per GPU | 110.19 GiB |
| Peak allocated memory | 116–120 GiB |
| Peak reserved memory | 135–136 GiB |

### What happens after optimization

Once both ranks finish their local forward/backward work, the trainer still has to collapse the step back into one shared view of what just happened. Each rank has its own local timings, loss values, and memory usage, so the trainer all-reduces the scalar metrics across both ranks and turns them into one averaged step summary. That includes the GRPO diagnostics, rollout-level statistics, timing numbers, and GPU memory snapshots, so the system is not just tracking the loss curve but also what the two training GPUs were actually doing during the step.

After that, rank 0 sends the consolidated result back to the orchestrator over IPC. That returned payload includes the new `training_step`, the current `policy_version`, and all of the trainer metrics. Those two counters move together, but they are not the same thing: `training_step` counts optimizer steps, while `policy_version` tracks which version of the policy future rollout groups should be associated with once the weight sync has happened.

Back in the parent process, the orchestrator folds that result into the main loop. It updates its copy of `training_step` and `policy_version`, records the trainer metrics, adds rollout-buffer depth, staleness drops, and underruns, and also attaches the inference-side GPU memory snapshot so the logging view reflects both the training GPUs and the inference GPU around the same point in time. Then the loop simply continues: the orchestrator goes back to pulling the next ready rollout groups from the buffer while inference keeps generating and training keeps consuming. That steady-state behavior is the real shape of the system once it has fully warmed up.

Checkpointing and evaluation are layered directly on top of that same loop rather than being treated as some separate offline phase. In the current config, evaluation runs every `100` training steps. When that trigger hits, the orchestrator first asks trainer rank 0 to write a recovery checkpoint through DeepSpeed. The saved client state includes the current `training_step`, `policy_version`, prompt-stream position, and wandb run ID, so a resumed run can continue from roughly the right place instead of starting the prompt stream over from zero.

Only after that does evaluation run. The evaluator uses fixed held-out subsets for SWE and CodeContests, temporarily forces greedy decoding by setting evaluation temperature to `0.0`, and runs fresh rollouts through the same rollout generator and environment types. So even the evaluation path is still using the same basic machinery as the training pipeline. It is just inserted periodically into the live asynchronous run rather than being split off into an entirely different workflow.

There is also a quiet bit of runtime plumbing here that matters more than it looks at first glance: the orchestrator keeps checking whether any background piece has died while it is waiting on IPC completions. So if the inference thread crashes, if the IPC dispatcher crashes, or if one of the trainer subprocesses exits unexpectedly, the parent does not just sit there blocked forever waiting for a message that will never arrive. It notices, raises the failure, and then tears the system down in a coordinated way: stop the inference side, send shutdown to the trainer, unblock the rollout buffer, shut down the environment pool, stop vLLM, close logging, and reap the subprocesses.

# Weight syncing with vLLM

Weight-syncing solves a straightforward problem: GPU 2 is running a long-lived vLLM engine that keeps generating rollouts, while GPUs 0 and 1 are updating the policy weights. If we never synchronized those updated weights back into vLLM, inference would just keep sampling from an older model forever. So the system needs a way to push fresh weights from the DeepSpeed side into the already-running vLLM engine without tearing the engine down and rebuilding it every step.

vLLM now has a native weight-transfer feature for exactly this kind of RL setup. At the `LLM` level, there are two main APIs involved: `init_weight_transfer_engine(...)` and `update_weights(...)`. The first is the setup step. It tells vLLM what transfer mechanism the two sides are going to use so both the trainer side and the inference side can join the same synchronization path. The second is the call the running vLLM engine waits on when a fresh set of weights is about to arrive. In our setup, both sides use `WeightTransferConfig(backend="nccl")`, which means the actual tensor movement during weight sync happens over NCCL.

In our implementation, this feature is wired directly into the same offline `LLM` object that is already being used for rollout generation inside the orchestrator. By “offline `LLM` path,” I just mean the in-process vLLM `LLM(...)` engine we keep alive inside the Python runtime and call directly from Python during rollouts. So to put it simply, weight transfer is being plugged straight into the same long-lived rollout-generation engine that is already part of the orchestrator.

That upstream API looks like this at the `LLM` level:

```python
# Source: src/checkpointing/weight_sync.py
llm.init_weight_transfer_engine(
    WeightTransferInitRequest(init_info={...})
)

llm.update_weights(
    WeightTransferUpdateRequest(
        update_info={
            "names": names,
            "dtype_names": dtype_names,
            "shapes": shapes,
        }
    )
)
```

#### Weight-sync setup

vLLM has to see the exact same parameters, names, shapes, and dtypes in the original full-tensor ordering that will later be sent during the NCCL broadcast. If that ordering is wrong, the whole synchronization path becomes unreliable because the receiver and sender are no longer talking about the same tensors in the same sequence. To avoid that, the trainer side captures the full parameter metadata before DeepSpeed wraps the model with ZeRO-3 and later checks that the live parameter order still matches what was captured earlier. After ZeRO-3 has already partitioned the weights, reconstructing that original full-tensor view becomes much more awkward, which is why this has to happen up front.

The coordination around the weight-transfer setup still stays inside the same local process structure used throughout the rest of the training pipeline. Before any real weight synchronization begins, the different sides first go through a short initialization and rendezvous sequence so that by the time actual weight transfer starts, the parent process, trainer ranks, and vLLM engine have already agreed on the transfer path they are going to use.

- Trainer rank 0 first signals the parent process once DeepSpeed initialization is complete.
- The parent process then tells rank 0 to begin the NCCL setup handshake while it concurrently initializes the inference side of the weight-transfer engine.
- After that, trainer rank 0 initializes the trainer side of the transfer engine.
- Rank 0 sends a final completion signal once that rendezvous finishes.
- The run only continues once the trainer side and the in-process vLLM engine have both joined the same NCCL rendezvous.

In other words, there is an explicit little handshake before steady-state training starts. Rank 0 sends a "DeepSpeed init is done" message to the parent, the parent replies with the signal that kicks off NCCL setup, both sides block while the weight-transfer engine is initialized, and only then does the rest of the pipeline proceed. 

Once training is underway, the actual update sequence is more structured than a simple "send weights now" call. At this point there are really two things the system is trying to protect at once. 

#### 1. First, we do not want vLLM to keep scheduling rollout work while its weights are in the middle of changing. 

When the trainer hits a sync point, it first asks the parent process to pause vLLM and begin a weight update. The parent handles that through the IPC dispatcher by calling `pause_for_weight_sync()` and `begin_weight_update(...)` on the `VLLMServer` wrapper. On the vLLM side, that pause does two important things. It puts scheduling to sleep with `sleep(level=0, mode="keep")`, which stops the engine from taking on new rollout work, and it resets the prefix cache so no reusable KV state survives across the weight change. After that, the parent kicks off `llm.update_weights(...)` in a background thread. That means the receiver side is now sitting there waiting for incoming tensors by the time the trainer starts the NCCL broadcast.

That part is worth showing directly because it explains why the orchestration looks slightly strange:

```python
# Source: src/inference/vllm_server.py
def pause_for_weight_sync(self) -> None:
    self.llm.sleep(level=0, mode="keep")
    self.llm.reset_prefix_cache(reset_running_requests=True)

def begin_weight_update(self, names, dtype_names, shapes) -> None:
    request = WeightTransferUpdateRequest(
        update_info=dict(
            names=names,
            dtype_names=dtype_names,
            shapes=shapes,
            packed=True,
        )
    )
    self._update_thread = threading.Thread(
        target=lambda: self.llm.update_weights(request),
        daemon=True,
    )
    self._update_thread.start()
```

#### 2. Second, we do not want the trainer side to start broadcasting tensors before the receiver side is actually ready to accept them. 

Once the receiver side is ready, the trainer ranks still need to synchronize before any weights can be sent. Under ZeRO-3, parameters are normally sharded across both training ranks for memory efficiency, so rank 0 does not hold a complete copy of the model during regular training. Before weights can move to vLLM, the tensors first have to be reconstructed through `deepspeed.zero.GatheredParameters(...)`, which requires participation from both ranks. That is why the synchronization barriers matter: they ensure rank 0 does not begin the NCCL transfer while rank 1 is still behind in the gather path. After both ranks enter the gather flow together and DeepSpeed materializes the full tensor view, only trainer rank 0 proceeds with the actual NCCL transfer through `NCCLWeightTransferEngine.trainer_send_weights(...)` using `packed=True`, which batches the transfers more efficiently. So even though the communication flow is intentionally asymmetric and only rank 0 talks directly to vLLM, rank 1 still remains essential because the gather cannot complete correctly without both ranks participating.

```python
# Source: src/checkpointing/weight_sync.py
def _gathered_named_parameters(self):
    import deepspeed

    for name, param in self._named_parameters():
        if hasattr(param, "ds_id"):
            with deepspeed.zero.GatheredParameters([param], modifier_rank=None, enabled=True):
                yield name, param.data
        else:
            yield name, param.data


if self.rank == 0:
    trainer_args = NCCLTrainerSendWeightsArgs(group=self.group, packed=True)
    NCCLWeightTransferEngine.trainer_send_weights(
        iterator=self._gathered_named_parameters(),
        trainer_args=trainer_args,
    )
else:
    for _ in self._gathered_named_parameters():
        pass
```

#### Confirm the transfer and resume scheduling

After the broadcast completes, there is another barrier before the system moves on. That second barrier exists for the same reason as the first one: it keeps both trainer ranks in lockstep so rank 0 does not race ahead and tell the parent to resume inference before the transfer side has fully settled. Once both ranks clear that point, rank 0 asks the parent to finish the update and resume inference. The parent joins the background update thread with `end_weight_update()`, re-raises any update error if one happened, and then wakes scheduling back up with `wake_up(tags=["scheduling"])`. Only after that final acknowledgment does the trainer continue.

So the full sequence is very explicit: pause vLLM, prepare the receiver, gather the full tensors under ZeRO-3, broadcast them over NCCL, wait for both trainer ranks to finish cleanly, and only then resume scheduling. That is what makes the synchronization path feel a little more involved than a normal parameter copy, but it is also what keeps the rollout engine and the trainer from drifting into an inconsistent state while the weights are in motion.

One subtle but important detail here is the sync cadence. In the current config, `weight_sync_interval = 1`, so the system synchronizes weights after every optimizer step. That means `policy_version` also advances step by step, and inference is usually never more than a small number of versions behind the trainer. Combined with the rollout-buffer staleness filter, that keeps the asynchronous setup reasonably tight without trying to do anything more complicated like mid-rollout weight swaps.

# Throughput, Bottlenecks, and One Reward-Hacking Failure

At least in the setup I actually ran, the system is inference-bound. That is mostly because only one GPU is currently dedicated to inference while two GPUs are dedicated to training. The training side is perfectly capable of consuming data faster than the rollout side can keep producing it, so the bottleneck here is not that the implementation is somehow doing something obviously wasteful on the training side. It is more that one inference GPU simply is not enough to fully saturate the rest of the pipeline once the trainer is ready to move quickly. If I wanted to improve end-to-end utilization in a serious way, the real answer would be more inference capacity rather than trying to squeeze tiny optimizations out of the GRPO step itself.

One other thing that showed up pretty clearly in the older CodeContests rollouts was a reward-hacking failure around `solution.py`. At one point I had a fallback path where, if the model failed to create `solution.py` through the tool call, the environment would still try to recover the code by extracting it from the model's visible response. In practice that meant the rollout could still get treated like it had produced a solution even if the model never actually followed the intended workflow. After enough exposure to that behavior, the model started leaning on it. Instead of using `write_file` properly, it would often make zero tool calls and dump the whole Python solution directly into the final answer. The older logs show this very explicitly: `tool_calls` would be `0`, `solution.py` would get recovered from the model's final response, and the recorded `solution_origin` would come through as `final_response_fallback`. So the model had effectively learned that it could bypass the actual tool-writing requirement and still get most of what it wanted.

That was obviously not the behavior I wanted to reinforce. The whole point of the CodeContests setup was to train the model to actually use the tool harness correctly, not to learn that a nicely formatted explanation with code pasted into it was good enough. So I removed that fallback and tightened the CodeContests prompting and did some other minor tweaks. After that, the behavior started moving back in the direction I wanted. The model gradually stopped treating the final response as a hidden submission channel and started relearning that if it wanted reward, it actually had to create `solution.py` through the tool path and verify it there.

# Conclusion

In this blog, we walked through the complete CodeAgent-RL system from end to end. More than anything, this project was an attempt to understand the real systems work behind modern RL pipelines for tool-using code agents and to make the whole thing run as one coherent setup. I hope this made the full architecture easier to follow and easier to reason about. Until next time~
