| Condition | Final progress (%) ↑ | Completion rate (%) ↑ | Oracle match (%) ↑ | Invalid moves (%) ↓ | Parse failures (%) ↓ | Clarify rate (%) | Progress gained (%) ↑ | Correct moves (%) ↑ | Episode length ↓ | Reward / turn ↑ | Director API failures (%) ↓ | Builder output tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B + ECHO | **99.9** [99.7, 100.0] † | **98.3** [95.0, 100.0] † | **84.3** ± 2.0 † | **3.9** ± 1.1 † | 1.9 ± 0.2 | 12.2 ± 1.3 † | **74.4** ± 4.0 † | **81.5** ± 1.8 † | **13.7** ± 0.7 † | **0.028** ± 0.004 † | **0.00** ± 0.00 | 60 ± 0 |
| Claude Sonnet 5 | 98.3 ± 1.2 † | 86.7 ± 8.3 † | 80.9 ± 1.8 † | 5.2 ± 1.2 † | 1.9 ± 0.3 | 15.6 ± 1.3 † | 72.3 ± 3.7 † | 74.1 ± 1.8 † | 14.8 ± 0.9 † | 0.019 ± 0.005 † | **0.00** ± 0.00 | 60 ± 0 |
| Claude Sonnet 4.6 | 98.1 ± 1.5 † | 83.3 ± 9.2 † | 75.6 ± 1.7 † | 8.4 ± 1.2 † | 2.0 ± 0.2 | 18.6 ± 0.9 † | 70.3 ± 4.3 † | 70.1 ± 1.9 † | 15.2 ± 1.0 † | 0.013 ± 0.005 † | **0.00** ± 0.00 | 60 ± 0 |
| GPT-5.4 mini | 92.1 ± 2.4 † | 50.0 ± 10.0 † | 70.8 ± 2.0 † | 10.1 ± 1.0 † | 2.1 ± 0.2 | 19.5 ± 1.0 † | 70.6 ± 3.3 † | 64.8 ± 2.1 † | 17.6 ± 0.7 † | 0.009 ± 0.006 † | **0.00** ± 0.00 | 60 ± 0 |
| GPT-4.1 mini | 88.3 ± 3.6 † | 40.0 ± 12.5 † | 61.4 ± 1.9 † | 15.7 ± 1.4 † | 2.1 ± 0.2 | 24.8 ± 1.2 † | 59.7 ± 2.6 † | 53.6 ± 2.1 † | 18.3 ± 0.7 † | 0.001 ± 0.005 † | **0.00** ± 0.00 | 60 ± 0 |
| Qwen2.5-7B + GRPO | 86.9 ± 3.2 † | 43.3 ± 10.0 † | 63.0 ± 1.8 † | 13.9 ± 1.3 † | 2.1 ± 0.2 | 23.5 ± 1.5 † | 63.3 ± 3.3 † | 58.6 ± 1.9 † | 18.4 ± 0.6 † | 0.001 ± 0.003 † | **0.00** ± 0.00 | 60 ± 0 |
| Qwen2.5-7B + RLOO | 98.4 ± 0.9 † | 83.3 ± 8.3 † | 80.6 ± 2.1 † | 4.4 ± 1.0 † | 2.1 ± 0.2 | 15.3 ± 1.3 † | 72.7 ± 4.8 † | 75.3 ± 1.8 † | 15.1 ± 1.2 † | 0.020 ± 0.006 † | **0.00** ± 0.00 | 60 ± 0 |
| Qwen2.5-72B 4-bit (zero-shot) | 81.7 ± 4.5 † | 23.3 ± 10.0 † | 53.7 ± 2.4 † | 16.9 ± 1.1 † | 2.2 ± 0.2 | 28.0 ± 1.4 † | 55.8 ± 2.4 † | 48.5 ± 1.8 † | 19.3 ± 0.4 † | -0.008 ± 0.005 † | **0.00** ± 0.00 | 60 ± 0 |
| Qwen2.5-14B (zero-shot) | 66.0 ± 5.2 | 5.0 ± 5.0 | 45.0 ± 1.9 † | 22.3 ± 1.1 † | **1.9** ± 0.2 | 32.5 ± 1.2 † | 43.7 ± 3.0 † | 39.6 ± 2.1 † | 19.8 [19.5, 20.0] | -0.015 ± 0.004 | **0.00** ± 0.00 | 60 ± 0 |
| Qwen2.5-7B (zero-shot) | 57.9 ± 5.3 | 0.0 [0.0, 16.1] | 36.8 ± 1.2 | 26.0 ± 0.9 | 2.2 ± 0.2 | 36.6 ± 0.8 | 32.0 ± 2.2 | 34.1 ± 2.5 | 20.0 ± 0.0 | -0.018 ± 0.004 | **0.00** ± 0.00 | 60 ± 0 |

Paired difference vs **Qwen2.5-7B (zero-shot)** (same structures; Δ = condition − reference)

| Condition | Metric | Δ | 95% CI | p | p (Holm) | n |
|---|---|---:|---:|---:|---:|---:|
| Qwen2.5-7B + ECHO | Final progress (pp) | +42.0 | [+36.6, +47.4] | <0.0001 | <0.0001 | 20 |
| Qwen2.5-7B + ECHO | Completion rate (pp) | +98.3 | [+95.0, +100.0] | <0.0001 | <0.0001 | 20 |
| Qwen2.5-7B + ECHO | Oracle match (pp) | +47.5 | [+45.3, +49.8] | <0.0001 | <0.0001 | 20 |
| Qwen2.5-7B + ECHO | Invalid moves (pp) | -22.2 | [-23.4, -20.9] | <0.0001 | <0.0001 | 20 |
| Qwen2.5-7B + ECHO | Parse failures (pp) | -0.3 | [-0.6, +0.1] | 0.1162 | 0.3485 | 20 |
| Qwen2.5-7B + ECHO | Clarify rate (pp) | -24.4 | [-26.0, -22.9] | <0.0001 | <0.0001 | 20 |
| Qwen2.5-7B + ECHO | Progress gained (pp) | +42.4 | [+37.3, +47.3] | <0.0001 | <0.0001 | 20 |
| Qwen2.5-7B + ECHO | Correct moves (pp) | +47.4 | [+44.3, +50.4] | <0.0001 | <0.0001 | 20 |
| Qwen2.5-7B + ECHO | Episode length | -6.3 | [-7.0, -5.6] | <0.0001 | <0.0001 | 20 |
| Qwen2.5-7B + ECHO | Reward / turn | +0.046 | [+0.040, +0.052] | <0.0001 | <0.0001 | 20 |
| Qwen2.5-7B + ECHO | Director API failures (pp) | +0.00 | [+0.00, +0.00] | 1.0000 | 1.0000 | 20 |
| Qwen2.5-7B + ECHO | Builder output tokens | +0 | [+0, +0] | 1.0000 | 1.0000 | 20 |
| Claude Sonnet 5 | Final progress (pp) | +40.4 | [+34.8, +46.1] | <0.0001 | <0.0001 | 20 |
| Claude Sonnet 5 | Completion rate (pp) | +86.7 | [+78.3, +95.0] | <0.0001 | <0.0001 | 20 |
| Claude Sonnet 5 | Oracle match (pp) | +44.2 | [+41.9, +46.5] | <0.0001 | <0.0001 | 20 |
| Claude Sonnet 5 | Invalid moves (pp) | -20.8 | [-22.0, -19.7] | <0.0001 | <0.0001 | 20 |
| Claude Sonnet 5 | Parse failures (pp) | -0.3 | [-0.6, +0.1] | 0.1821 | 0.5462 | 20 |
| Claude Sonnet 5 | Clarify rate (pp) | -20.9 | [-22.5, -19.4] | <0.0001 | <0.0001 | 20 |
| Claude Sonnet 5 | Progress gained (pp) | +40.3 | [+35.9, +44.6] | <0.0001 | <0.0001 | 20 |
| Claude Sonnet 5 | Correct moves (pp) | +40.0 | [+36.6, +43.4] | <0.0001 | <0.0001 | 20 |
| Claude Sonnet 5 | Episode length | -5.2 | [-6.1, -4.3] | <0.0001 | <0.0001 | 20 |
| Claude Sonnet 5 | Reward / turn | +0.037 | [+0.030, +0.043] | <0.0001 | <0.0001 | 20 |
| Claude Sonnet 5 | Director API failures (pp) | +0.00 | [+0.00, +0.00] | 1.0000 | 1.0000 | 20 |
| Claude Sonnet 5 | Builder output tokens | +0 | [+0, +0] | 1.0000 | 1.0000 | 20 |
| Claude Sonnet 4.6 | Final progress (pp) | +40.2 | [+35.4, +45.3] | <0.0001 | <0.0001 | 20 |
| Claude Sonnet 4.6 | Completion rate (pp) | +83.3 | [+73.3, +91.7] | <0.0001 | <0.0001 | 20 |
| Claude Sonnet 4.6 | Oracle match (pp) | +38.8 | [+36.9, +40.7] | <0.0001 | <0.0001 | 20 |
| Claude Sonnet 4.6 | Invalid moves (pp) | -17.6 | [-18.9, -16.4] | <0.0001 | <0.0001 | 20 |
| Claude Sonnet 4.6 | Parse failures (pp) | -0.2 | [-0.5, +0.2] | 0.3596 | 1.0000 | 20 |
| Claude Sonnet 4.6 | Clarify rate (pp) | -18.0 | [-19.2, -16.9] | <0.0001 | <0.0001 | 20 |
| Claude Sonnet 4.6 | Progress gained (pp) | +38.4 | [+33.3, +43.8] | <0.0001 | <0.0001 | 20 |
| Claude Sonnet 4.6 | Correct moves (pp) | +36.0 | [+32.4, +39.5] | <0.0001 | <0.0001 | 20 |
| Claude Sonnet 4.6 | Episode length | -4.8 | [-5.7, -3.8] | <0.0001 | <0.0001 | 20 |
| Claude Sonnet 4.6 | Reward / turn | +0.031 | [+0.025, +0.036] | <0.0001 | <0.0001 | 20 |
| Claude Sonnet 4.6 | Director API failures (pp) | +0.00 | [+0.00, +0.00] | 1.0000 | 1.0000 | 20 |
| Claude Sonnet 4.6 | Builder output tokens | +0 | [+0, +0] | 1.0000 | 1.0000 | 20 |
| GPT-5.4 mini | Final progress (pp) | +34.2 | [+29.0, +39.4] | <0.0001 | <0.0001 | 20 |
| GPT-5.4 mini | Completion rate (pp) | +50.0 | [+40.0, +60.0] | <0.0001 | <0.0001 | 20 |
| GPT-5.4 mini | Oracle match (pp) | +34.0 | [+31.3, +36.6] | <0.0001 | <0.0001 | 20 |
| GPT-5.4 mini | Invalid moves (pp) | -15.9 | [-17.1, -14.7] | <0.0001 | <0.0001 | 20 |
| GPT-5.4 mini | Parse failures (pp) | -0.1 | [-0.4, +0.2] | 0.5941 | 1.0000 | 20 |
| GPT-5.4 mini | Clarify rate (pp) | -17.1 | [-18.4, -16.0] | <0.0001 | <0.0001 | 20 |
| GPT-5.4 mini | Progress gained (pp) | +38.7 | [+34.7, +42.9] | <0.0001 | <0.0001 | 20 |
| GPT-5.4 mini | Correct moves (pp) | +30.7 | [+27.1, +34.1] | <0.0001 | <0.0001 | 20 |
| GPT-5.4 mini | Episode length | -2.4 | [-3.1, -1.8] | <0.0001 | <0.0001 | 20 |
| GPT-5.4 mini | Reward / turn | +0.027 | [+0.020, +0.035] | <0.0001 | <0.0001 | 20 |
| GPT-5.4 mini | Director API failures (pp) | +0.00 | [+0.00, +0.00] | 1.0000 | 1.0000 | 20 |
| GPT-5.4 mini | Builder output tokens | +0 | [+0, +0] | 1.0000 | 1.0000 | 20 |
| GPT-4.1 mini | Final progress (pp) | +30.4 | [+24.7, +36.5] | <0.0001 | <0.0001 | 20 |
| GPT-4.1 mini | Completion rate (pp) | +40.0 | [+28.3, +53.3] | <0.0001 | 0.0002 | 20 |
| GPT-4.1 mini | Oracle match (pp) | +24.6 | [+22.6, +26.7] | <0.0001 | <0.0001 | 20 |
| GPT-4.1 mini | Invalid moves (pp) | -10.3 | [-12.3, -8.2] | <0.0001 | <0.0001 | 20 |
| GPT-4.1 mini | Parse failures (pp) | -0.1 | [-0.4, +0.3] | 0.6556 | 1.0000 | 20 |
| GPT-4.1 mini | Clarify rate (pp) | -11.8 | [-13.4, -10.4] | <0.0001 | <0.0001 | 20 |
| GPT-4.1 mini | Progress gained (pp) | +27.8 | [+24.5, +30.8] | <0.0001 | <0.0001 | 20 |
| GPT-4.1 mini | Correct moves (pp) | +19.5 | [+16.3, +22.8] | <0.0001 | <0.0001 | 20 |
| GPT-4.1 mini | Episode length | -1.7 | [-2.5, -1.1] | <0.0001 | 0.0002 | 20 |
| GPT-4.1 mini | Reward / turn | +0.019 | [+0.013, +0.025] | <0.0001 | 0.0001 | 20 |
| GPT-4.1 mini | Director API failures (pp) | +0.00 | [+0.00, +0.00] | 1.0000 | 1.0000 | 20 |
| GPT-4.1 mini | Builder output tokens | +0 | [+0, +0] | 1.0000 | 1.0000 | 20 |
| Qwen2.5-7B + GRPO | Final progress (pp) | +29.0 | [+22.6, +35.6] | <0.0001 | <0.0001 | 20 |
| Qwen2.5-7B + GRPO | Completion rate (pp) | +43.3 | [+33.3, +53.3] | <0.0001 | <0.0001 | 20 |
| Qwen2.5-7B + GRPO | Oracle match (pp) | +26.2 | [+24.1, +28.4] | <0.0001 | <0.0001 | 20 |
| Qwen2.5-7B + GRPO | Invalid moves (pp) | -12.1 | [-14.0, -10.2] | <0.0001 | <0.0001 | 20 |
| Qwen2.5-7B + GRPO | Parse failures (pp) | -0.1 | [-0.3, +0.2] | 0.5980 | 1.0000 | 20 |
| Qwen2.5-7B + GRPO | Clarify rate (pp) | -13.0 | [-15.0, -11.0] | <0.0001 | <0.0001 | 20 |
| Qwen2.5-7B + GRPO | Progress gained (pp) | +31.3 | [+27.4, +35.5] | <0.0001 | <0.0001 | 20 |
| Qwen2.5-7B + GRPO | Correct moves (pp) | +24.5 | [+20.6, +28.2] | <0.0001 | <0.0001 | 20 |
| Qwen2.5-7B + GRPO | Episode length | -1.6 | [-2.2, -1.0] | <0.0001 | 0.0001 | 20 |
| Qwen2.5-7B + GRPO | Reward / turn | +0.019 | [+0.013, +0.024] | <0.0001 | <0.0001 | 20 |
| Qwen2.5-7B + GRPO | Director API failures (pp) | +0.00 | [+0.00, +0.00] | 1.0000 | 1.0000 | 20 |
| Qwen2.5-7B + GRPO | Builder output tokens | +0 | [+0, +0] | 1.0000 | 1.0000 | 20 |
| Qwen2.5-7B + RLOO | Final progress (pp) | +40.5 | [+35.0, +46.1] | <0.0001 | <0.0001 | 20 |
| Qwen2.5-7B + RLOO | Completion rate (pp) | +83.3 | [+75.0, +91.7] | <0.0001 | <0.0001 | 20 |
| Qwen2.5-7B + RLOO | Oracle match (pp) | +43.8 | [+41.1, +46.4] | <0.0001 | <0.0001 | 20 |
| Qwen2.5-7B + RLOO | Invalid moves (pp) | -21.7 | [-22.9, -20.4] | <0.0001 | <0.0001 | 20 |
| Qwen2.5-7B + RLOO | Parse failures (pp) | -0.1 | [-0.4, +0.2] | 0.6164 | 1.0000 | 20 |
| Qwen2.5-7B + RLOO | Clarify rate (pp) | -21.3 | [-22.9, -19.7] | <0.0001 | <0.0001 | 20 |
| Qwen2.5-7B + RLOO | Progress gained (pp) | +40.7 | [+35.3, +46.3] | <0.0001 | <0.0001 | 20 |
| Qwen2.5-7B + RLOO | Correct moves (pp) | +41.2 | [+37.8, +44.5] | <0.0001 | <0.0001 | 20 |
| Qwen2.5-7B + RLOO | Episode length | -4.9 | [-6.1, -3.8] | <0.0001 | <0.0001 | 20 |
| Qwen2.5-7B + RLOO | Reward / turn | +0.038 | [+0.031, +0.043] | <0.0001 | <0.0001 | 20 |
| Qwen2.5-7B + RLOO | Director API failures (pp) | +0.00 | [+0.00, +0.00] | 1.0000 | 1.0000 | 20 |
| Qwen2.5-7B + RLOO | Builder output tokens | +0 | [+0, +0] | 1.0000 | 1.0000 | 20 |
| Qwen2.5-72B 4-bit (zero-shot) | Final progress (pp) | +23.8 | [+16.9, +30.6] | <0.0001 | <0.0001 | 20 |
| Qwen2.5-72B 4-bit (zero-shot) | Completion rate (pp) | +23.3 | [+13.3, +33.3] | 0.0010 | 0.0059 | 20 |
| Qwen2.5-72B 4-bit (zero-shot) | Oracle match (pp) | +16.9 | [+14.2, +19.5] | <0.0001 | <0.0001 | 20 |
| Qwen2.5-72B 4-bit (zero-shot) | Invalid moves (pp) | -9.1 | [-10.5, -7.7] | <0.0001 | <0.0001 | 20 |
| Qwen2.5-72B 4-bit (zero-shot) | Parse failures (pp) | +0.0 | [-0.2, +0.3] | 0.7892 | 1.0000 | 20 |
| Qwen2.5-72B 4-bit (zero-shot) | Clarify rate (pp) | -8.5 | [-10.3, -6.7] | <0.0001 | <0.0001 | 20 |
| Qwen2.5-72B 4-bit (zero-shot) | Progress gained (pp) | +23.8 | [+20.7, +27.3] | <0.0001 | <0.0001 | 20 |
| Qwen2.5-72B 4-bit (zero-shot) | Correct moves (pp) | +14.4 | [+11.4, +17.5] | <0.0001 | <0.0001 | 20 |
| Qwen2.5-72B 4-bit (zero-shot) | Episode length | -0.7 | [-1.1, -0.3] | 0.0010 | 0.0059 | 20 |
| Qwen2.5-72B 4-bit (zero-shot) | Reward / turn | +0.010 | [+0.003, +0.016] | 0.0092 | 0.0368 | 20 |
| Qwen2.5-72B 4-bit (zero-shot) | Director API failures (pp) | +0.00 | [+0.00, +0.00] | 1.0000 | 1.0000 | 20 |
| Qwen2.5-72B 4-bit (zero-shot) | Builder output tokens | +0 | [+0, +0] | 1.0000 | 1.0000 | 20 |
| Qwen2.5-14B (zero-shot) | Final progress (pp) | +8.1 | [+1.9, +14.3] | 0.0241 | 0.1687 | 20 |
| Qwen2.5-14B (zero-shot) | Completion rate (pp) | +5.0 | [+0.0, +10.0] | 0.2500 | 1.0000 | 20 |
| Qwen2.5-14B (zero-shot) | Oracle match (pp) | +8.3 | [+6.0, +10.4] | <0.0001 | 0.0001 | 20 |
| Qwen2.5-14B (zero-shot) | Invalid moves (pp) | -3.7 | [-5.3, -2.2] | 0.0001 | 0.0012 | 20 |
| Qwen2.5-14B (zero-shot) | Parse failures (pp) | -0.3 | [-0.7, +0.1] | 0.1006 | 0.6038 | 20 |
| Qwen2.5-14B (zero-shot) | Clarify rate (pp) | -4.1 | [-5.5, -2.7] | <0.0001 | 0.0007 | 20 |
| Qwen2.5-14B (zero-shot) | Progress gained (pp) | +11.7 | [+8.8, +14.7] | <0.0001 | <0.0001 | 20 |
| Qwen2.5-14B (zero-shot) | Correct moves (pp) | +5.5 | [+3.2, +7.9] | <0.0001 | 0.0006 | 20 |
| Qwen2.5-14B (zero-shot) | Episode length | -0.2 | [-0.5, +0.0] | 0.2500 | 1.0000 | 20 |
| Qwen2.5-14B (zero-shot) | Reward / turn | +0.003 | [-0.003, +0.008] | 0.3885 | 1.0000 | 20 |
| Qwen2.5-14B (zero-shot) | Director API failures (pp) | +0.00 | [+0.00, +0.00] | 1.0000 | 1.0000 | 20 |
| Qwen2.5-14B (zero-shot) | Builder output tokens | +0 | [+0, +0] | 1.0000 | 1.0000 | 20 |
