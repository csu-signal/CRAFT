"""
CRAFTEchoTrainer -- GRPO subclass implementing ECHO-style per-turn credit for
the CRAFT builder, plus the episode_return and rloo_per_turn baselines,
selected via `advantage_mode`. 
"""
import pickle
from pathlib import Path

import numpy as np
import torch
from trl import GRPOConfig, GRPOTrainer
from trl.models import unwrap_model_for_generation

from advantages import compute_multiturn_advantages
from reward import BuilderRewardFunction
from rollout import run_builder_episode
from agents.builder_agent import BUILDER_SYSTEM_PROMPT_ORACLE, BUILDER_SYSTEM_PROMPT_BASE


class CRAFTEchoTrainer(GRPOTrainer):
    BUILDER_LOG_KEYS = [
        "progress_delta", "matched_oracle", "parse_failure", "move_invalid", "completed",
        "off_list_penalty", "invalid_move_penalty", "clarify_penalty",
        "completion_bonus", "efficiency_bonus",
        "action", "turn", "training_reward", "final_progress",
    ]

    def __init__(
        self, *args,
        structures=None,
        advantage_mode="echo",
        oracle_n=5,
        max_turns=15,
        director_model_name="gpt-4.1-mini",
        director_api_key=None,
        director_mode="api",
        shared_director_model=None,
        shared_director_tokenizer=None,
        run_id=0,
        logprob_batch_size=16,
        train_micro_batch_size=4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.structures = structures
        self.advantage_mode = advantage_mode
        self.oracle_n = oracle_n
        self.max_turns = max_turns
        self.director_model_name = director_model_name
        self.director_api_key = director_api_key
        self.director_mode = director_mode
        self.shared_director_model = shared_director_model
        self.shared_director_tokenizer = shared_director_tokenizer
        self.run_id = run_id

        self.logprob_batch_size = logprob_batch_size
        # See training_step() override below -- caps how many turn-sequences
        # get forward+backward'd through the model in one shot.
        self.train_micro_batch_size = train_micro_batch_size

        self._builder_logs = {k: [] for k in self.BUILDER_LOG_KEYS}

        self._builder_reward_func = None
        for func in self.reward_funcs:
            if isinstance(func, BuilderRewardFunction):
                self._builder_reward_func = func
                break
        if self._builder_reward_func is None:
            raise ValueError("CRAFTEchoTrainer requires a BuilderRewardFunction in reward_funcs")

        self._buffered_reward_infos = None
        self._buffered_group_sizes = None
        self._current_group_sizes = None

    # ---- micro-batched forward+backward ------------------------------------
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        chunking the forward+backward pass
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)

        total = inputs["advantages"].shape[0]
        seq_keys = [
            k for k, v in inputs.items()
            if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == total
        ]
        chunk_size = max(1, self.train_micro_batch_size)
        chunk_starts = list(range(0, total, chunk_size))
        n_chunks = len(chunk_starts)

        prev_gas = self.current_gradient_accumulation_steps
        self.current_gradient_accumulation_steps = n_chunks

        total_loss = torch.zeros((), device=self.accelerator.device)
        try:
            for start in chunk_starts:
                end = min(start + chunk_size, total)
                chunk_inputs = {}
                for k, v in inputs.items():
                    if k in seq_keys:
                        chunk_inputs[k] = v[start:end]
                    elif k == "num_items_in_batch":
                        chunk_inputs[k] = inputs["completion_mask"][start:end].sum()
                    else:
                        chunk_inputs[k] = v

                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, chunk_inputs)
                self.accelerator.backward(loss)
                total_loss += loss.detach()
                del chunk_inputs, loss
        finally:
            self.current_gradient_accumulation_steps = prev_gas

        return total_loss.detach()

    # ---- logging helpers (mirrors ECHOTrainer) ----------------------------
    def _clear_builder_logs(self):
        for k in self.BUILDER_LOG_KEYS:
            self._builder_logs[k] = []

    def _safe_mean(self, vals):
        vals = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    def _safe_rate(self, bools):
        vals = [1.0 if bool(v) else 0.0 for v in bools if v is not None]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    def _compute_and_log_builder_metrics(self, reward_infos, group_sizes, mode):
        self._clear_builder_logs()
        for info in reward_infos:
            for k in self.BUILDER_LOG_KEYS:
                if k in info:
                    self._builder_logs[k].append(info[k])

        prefix = "" if mode == "train" else "eval_"
        L, M = self._builder_logs, self._metrics[mode]

        M.setdefault(f"{prefix}builder/progress_delta_mean", []).append(self._safe_mean(L["progress_delta"]))
        M.setdefault(f"{prefix}builder/completed_rate", []).append(self._safe_rate(L["completed"]))
        M.setdefault(f"{prefix}builder/oracle_match_rate", []).append(
            self._safe_rate([v for v in L["matched_oracle"] if v is not None]))
        M.setdefault(f"{prefix}builder/clarify_rate", []).append(
            self._safe_rate([a == "clarify" for a in L["action"]]))
        M.setdefault(f"{prefix}builder/parse_failure_rate", []).append(self._safe_rate(L["parse_failure"]))
        M.setdefault(f"{prefix}builder/invalid_move_rate", []).append(self._safe_rate(L["move_invalid"]))
        M.setdefault(f"{prefix}builder/mean_episode_length", []).append(
            float(np.mean(group_sizes)) if group_sizes else float("nan"))
        # The actual objective (terminal overall_progress, ideally 1.0) --
        # distinct from progress_delta_mean, which is the dense per-turn
        # training signal. This is only set on each episode's last turn (see
        # rollout.run_builder_episode), so L["final_progress"] naturally has
        # one entry per episode in this window, not one per turn.
        M.setdefault(f"{prefix}builder/final_progress_mean", []).append(self._safe_mean(L["final_progress"]))

        rewards = [info.get("training_reward", 0.0) for info in reward_infos]
        M.setdefault("reward", []).append(float(np.mean(rewards)) if rewards else float("nan"))
        M.setdefault("reward_std", []).append(float(np.std(rewards)) if rewards else float("nan"))

    # ---- generation-batch buffering (ported unchanged from ECHOTrainer) ---
    def _prepare_inputs(self, generation_batch):
        mode = "train" if self.model.training else "eval"

        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            is_generation_step = (self._step % generate_every == 0 or self._buffered_inputs is None)

            if is_generation_step:
                generation_batch = self._generate_and_score_completions(generation_batch)
                self._buffered_inputs = self._split_by_episodes(
                    generation_batch, self._current_group_sizes, self.args.steps_per_generation,
                )

            step_in_gen = self._step % self.args.steps_per_generation
            if self._buffered_reward_infos is not None:
                self._compute_and_log_builder_metrics(
                    self._buffered_reward_infos[step_in_gen], self._buffered_group_sizes[step_in_gen], mode,
                )

            inputs = self._buffered_inputs[step_in_gen]
            self._step += 1
        else:
            inputs = self._generate_and_score_completions(generation_batch)
            if self._buffered_reward_infos is not None:
                all_infos = [info for sl in self._buffered_reward_infos for info in sl]
                all_gsizes = [gs for sl in self._buffered_group_sizes for gs in sl]
                self._compute_and_log_builder_metrics(all_infos, all_gsizes, mode)

        return inputs

    def _split_by_episodes(self, output, group_sizes, steps_per_generation):
        ep_per_step = len(group_sizes) // steps_per_generation
        turn_boundaries = [0]
        for gs in group_sizes:
            turn_boundaries.append(turn_boundaries[-1] + gs)

        slices = []
        for i in range(steps_per_generation):
            start_turn = turn_boundaries[i * ep_per_step]
            end_turn = turn_boundaries[(i + 1) * ep_per_step]
            sl = {}
            for k, v in output.items():
                if k == "num_items_in_batch":
                    sl[k] = output["completion_mask"][start_turn:end_turn].sum()
                elif isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == output["advantages"].shape[0]:
                    sl[k] = v[start_turn:end_turn]
                else:
                    sl[k] = v
            slices.append(sl)
        return slices

    def _generate_and_score_completions(self, inputs):
        device = self.accelerator.device
        structure_indices = [int(inp["structure_idx"]) for inp in inputs]

        all_prompt_ids, all_completion_ids, all_rewards, all_reward_infos = [], [], [], []
        group_sizes = []

        total_episodes = len(structure_indices) * self.num_generations
        episode_num = 0
        for structure_idx in structure_indices:
            structure_data = self.structures[structure_idx]
            for _ in range(self.num_generations):
                episode_num += 1
                print(f"[rollout] generation batch: episode {episode_num}/{total_episodes} (structure_idx={structure_idx})")
                episode = self._run_single_episode(structure_data, structure_idx, device)
                all_prompt_ids.extend(episode["prompt_ids"])
                all_completion_ids.extend(episode["completion_ids"])
                all_rewards.extend(episode["rewards"])
                all_reward_infos.extend(episode["reward_infos"])
                group_sizes.append(len(episode["rewards"]))

        if self.args.save_steps and self.state.global_step % self.args.save_steps == 0:
            accum_path = Path(self.args.output_dir) / "episodes_all_steps.pkl"
            accum = []
            if accum_path.exists():
                with open(accum_path, "rb") as f:
                    accum = pickle.load(f)
            accum.append({
                "step": self.state.global_step,
                "structure_indices": structure_indices,
                "group_sizes": group_sizes,
                "reward_infos": all_reward_infos,
            })
            with open(accum_path, "wb") as f:
                pickle.dump(accum, f)

        pad_id = self.processing_class.pad_token_id
        max_p = max(t.shape[0] for t in all_prompt_ids)
        max_c = max(t.shape[0] for t in all_completion_ids)

        prompt_ids_padded = torch.stack([
            torch.cat([torch.full((max_p - t.shape[0],), pad_id), t])
            for t in all_prompt_ids
        ]).to(device)
        completion_ids_padded = torch.stack([
            torch.cat([t, torch.full((max_c - t.shape[0],), pad_id)])
            for t in all_completion_ids
        ]).to(device)

        completion_mask = (completion_ids_padded != pad_id).int()
        rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32, device=device)
        prompt_completion_ids = torch.cat([prompt_ids_padded, completion_ids_padded], dim=1)
        attention_mask_full = torch.cat([
            (prompt_ids_padded != pad_id).int(), completion_mask
        ], dim=1)

        ref_per_token_logps = None
        if self.beta != 0.0:
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model, prompt_completion_ids, attention_mask_full,
                        max_c, batch_size=self.logprob_batch_size,
                        compute_entropy=False,
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model, prompt_completion_ids, attention_mask_full,
                            max_c, batch_size=self.logprob_batch_size,
                            compute_entropy=False,
                        )

        with torch.no_grad():
            old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                self.model, prompt_completion_ids, attention_mask_full,
                max_c, batch_size=self.logprob_batch_size,
                compute_entropy=False,
            )

        B = len(structure_indices)
        advantages = compute_multiturn_advantages(
            rewards=rewards_tensor,
            group_sizes=group_sizes,
            B=B,
            G=self.num_generations,
            mode=self.advantage_mode,
            device=device,
        )

        self._current_group_sizes = group_sizes
        ep_per_step = len(group_sizes) // self.args.steps_per_generation
        turn_boundaries = [0]
        for gs in group_sizes:
            turn_boundaries.append(turn_boundaries[-1] + gs)

        self._buffered_reward_infos = []
        self._buffered_group_sizes = []
        for i in range(self.args.steps_per_generation):
            start_ep = i * ep_per_step
            end_ep = (i + 1) * ep_per_step
            start_turn = turn_boundaries[start_ep]
            end_turn = turn_boundaries[end_ep]
            self._buffered_reward_infos.append(all_reward_infos[start_turn:end_turn])
            self._buffered_group_sizes.append(group_sizes[start_ep:end_ep])

        output = {
            "prompt_ids": prompt_ids_padded,
            "prompt_mask": (prompt_ids_padded != pad_id).int(),
            "completion_ids": completion_ids_padded,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": completion_mask.sum(),
            "old_per_token_logps": old_per_token_logps,
        }
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        return output

    # ---- CRAFT-specific rollout --------------------------------------------
    def _run_single_episode(self, structure_data, structure_idx, device):
        def generate_fn(prompt_text, oracle_moves):
            system_prompt = BUILDER_SYSTEM_PROMPT_ORACLE if oracle_moves else BUILDER_SYSTEM_PROMPT_BASE
            chat_text = self.processing_class.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text},
                ],
                tokenize=False, add_generation_prompt=True,
            )
            input_ids = self.processing_class.encode(
                chat_text, return_tensors="pt",
                truncation=True, max_length=self.args.max_prompt_length,
                add_special_tokens=False,
            ).to(device)

            with torch.no_grad():
                with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped:
                    output = unwrapped.generate(
                        input_ids,
                        max_new_tokens=self.args.max_completion_length,
                        do_sample=True,
                        temperature=self.args.temperature,
                        top_p=0.9,
                        pad_token_id=self.processing_class.eos_token_id,
                    )
            new_tokens = output[0][input_ids.shape[1]:]
            decoded = self.processing_class.decode(new_tokens, skip_special_tokens=True).strip()
            return input_ids[0].cpu(), new_tokens.cpu(), decoded

        return run_builder_episode(
            structure_data=structure_data,
            generate_fn=generate_fn,
            structure_index=structure_idx,
            run_id=self.run_id,
            oracle_n=self.oracle_n,
            max_turns=self.max_turns,
            director_model_name=self.director_model_name,
            director_api_key=self.director_api_key,
            director_mode=self.director_mode,
            shared_director_model=self.shared_director_model,
            shared_director_tokenizer=self.shared_director_tokenizer,
        )

    def log(self, logs, start_time=None):
        mode = "train" if self.model.training else "eval"
        metrics = {k: sum(v) / len(v) for k, v in self._metrics[mode].items() if v}
        if mode == "eval":
            metrics = {f"eval_{k}": v for k, v in metrics.items()}
        logs = {**logs, **metrics}

        step = self.state.global_step
        prefix = "" if mode == "train" else "eval_"

        def _g(key):
            return metrics.get(f"{prefix}builder/{key}", float("nan"))

        print(
            f"\n{'='*60}\n[{mode.upper()}] step={step}\n"
            f"  reward           = {logs.get('reward', float('nan')):.4f}\n"
            f"  final_progress   = {_g('final_progress_mean'):.4f}  <- the actual objective (target: 1.0)\n"
            f"  progress_delta   = {_g('progress_delta_mean'):.4f}  (dense per-turn training signal)\n"
            f"  completed_rate   = {_g('completed_rate'):.3f}\n"
            f"  oracle_match     = {_g('oracle_match_rate'):.3f}\n"
            f"  clarify_rate     = {_g('clarify_rate'):.3f}\n"
            f"  parse_failure    = {_g('parse_failure_rate'):.3f}\n"
            f"  episode_length   = {_g('mean_episode_length'):.2f}\n"
            f"{'='*60}"
        )
        super().log(logs, start_time)


class Fixed_GRPOConfig(GRPOConfig):
    """Ported unchanged from echo-edp/train_echo.py -- fixes TRL's mutual-
    exclusion bug for generation_batch_size / steps_per_generation."""

    def __post_init__(self):
        import transformers
        transformers.TrainingArguments.__post_init__(self)

        num_processes = self.world_size
        if self.generation_batch_size is None and self.steps_per_generation is None:
            self.steps_per_generation = self.gradient_accumulation_steps
            self.generation_batch_size = (
                self.per_device_train_batch_size * num_processes * self.steps_per_generation
            )
        elif self.generation_batch_size is not None and self.steps_per_generation is None:
            global_batch_size = self.per_device_train_batch_size * num_processes
            if self.generation_batch_size % global_batch_size != 0:
                raise ValueError(
                    f"generation_batch_size ({self.generation_batch_size}) must be "
                    f"divisible by global batch size ({global_batch_size})."
                )
            self.steps_per_generation = self.generation_batch_size // global_batch_size
        elif self.generation_batch_size is None and self.steps_per_generation is not None:
            self.generation_batch_size = (
                self.per_device_train_batch_size * num_processes * self.steps_per_generation
            )
        else:
            self.generation_batch_size = (
                self.per_device_train_batch_size * num_processes * self.steps_per_generation
            )

        if self.do_eval and self.eval_strategy != "no":
            if (self.per_device_eval_batch_size * num_processes) % self.num_generations != 0:
                raise ValueError("Global eval batch size must be divisible by num_generations.")

        if self.generation_batch_size % self.num_generations != 0:
            raise ValueError("generation_batch_size must be divisible by num_generations.")

        if self.num_generations < 2:
            raise ValueError("GRPO requires at least 2 generations per prompt.")
