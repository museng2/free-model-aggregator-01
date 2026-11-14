# free-model-aggregator-01

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/museng2/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/museng2/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that aggregates a directory of free AI providers behind both client wires — OpenAI Chat Completions and Anthropic Messages. It probes every provider's catalog in the provider's own wire (OpenAI, Anthropic, Gemini), registers exactly the free models it finds, and serves each request with the highest-rated free model whose capabilities meet the request's requirements (tools, vision, JSON mode, context floor), failing over to the next-best model when a provider refuses a call. Requests and answers are translated between wires through one canonical OpenAI shape, streams included, and a provider that reports its budget spent is never called again. Everything the server answers is a pure function of the providers' answers — no clocks, no randomness — so a trial replays byte for byte. Local practice grades: 99/100 on all three practice trials (catalog 100, failover 100, served 100, wire 100, quality 99 — the missing point is failover forced by the trials' hidden refusal draws).

## Run it

```sh
# A trial provides the brief and the environment URL:
MAKEFUN_PROMPT_FILE=<brief> MAKEFUN_ENV_URL=<env> node main.ts

# Print the catalog the server would register, and exit:
MAKEFUN_PROMPT_FILE=<brief> MAKEFUN_ENV_URL=<env> node main.ts --list
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=musengmuseng&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
