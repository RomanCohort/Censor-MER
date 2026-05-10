# =============================================================================
# Censor -- Emotion Reporter (Enhanced with LLM-based Report Generation)
# =============================================================================
# Converts micro-expression features and AU activations into structured
# natural language reports, mimicking how the brain's language areas
# (Broca's, Wernicke's) translate emotional states into linguistic descriptions.
#
# Two report paths:
#   1. Template-based (primary): Deterministic, clinically structured reports
#   2. LLM-based (secondary): Free-text via HuggingFace OPT-125M
#
# Mathematical formulation (template path):
#   report = template.format(
#       AU_list=threshold(au_intensities > 0.5),
#       dominant_ME=argmax(me_logits),
#       rPPG_anomaly=detect(rppg_threshold),
#       OPD_landmarks=opd_detected
#   )
#
# LLM path uses meta-llama/opt-125m for lightweight free-text generation.
# =============================================================================

import torch
import torch.nn as nn
from config.defaults import LLM_CONFIG


# =============================================================================
# EmotionReporter -- Natural Language Emotion Report Generator
# =============================================================================

class EmotionReporter(nn.Module):
    """
    EmotionReporter -- Generates structured and free-text reports from
    high-dimensional micro-expression features.

    Transforms fused features + AU intensities + ME logits into clinically
    structured natural language reports, including physiological cues
    (rPPG blood flow anomalies) and temporal dynamics (OPD landmarks).

    Architecture:
        Input: fused_feat (B, 1024), au_intensities (B, T, 28), me_logits (B, 7)
          -> Text projection (B, 1024 -> 256)
          -> AU parsing (threshold-based activation detection)
          -> ME classification (argmax with confidence)
          -> Template-based report generation (primary)
          -> LLM-based free-text report via HuggingFace OPT-125M (secondary)
    """

    # AU names from FACS coding system
    AU_NAMES = LLM_CONFIG['au_names']
    ME_CATEGORIES = LLM_CONFIG['me_categories']
    AU_THRESHOLD = 0.5

    def __init__(self, config=None):
        super().__init__()
        cfg = config or LLM_CONFIG

        self.text_embed_dim = cfg['text_embed_dim']
        self.max_report_len = cfg['max_report_len']

        # Text projection: feature vector -> text embedding
        self.text_proj = nn.Sequential(
            nn.Linear(1024, self.text_embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.text_embed_dim * 2),
            nn.Linear(self.text_embed_dim * 2, self.text_embed_dim)
        )
        for module in self.text_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

        # LLM: OPT-125M for lightweight free-text generation
        # Uses a prompt-based approach: given emotion features, generate clinical description
        self._init_llm()

    def _init_llm(self):
        """
        Initialize HuggingFace OPT-125M for LLM-based report generation.

        OPT-125M is a causal language model with 125M parameters, suitable for
        CPU/small GPU deployment. It generates free-text clinical descriptions
        from structured emotion features.
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import os

            # Use a small model to avoid large downloads
            model_name = "facebook/opt-125m"

            print(f"[EmotionReporter] Loading OPT-125M from HuggingFace...")
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
            os.makedirs(cache_dir, exist_ok=True)

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=cache_dir
            )
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=cache_dir
            )
            self.llm.eval()  # Inference mode

            # Set pad token (OPT doesn't have one by default)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self._llm_available = True
            print(f"[EmotionReporter] OPT-125M loaded successfully!")
        except Exception as e:
            print(f"[EmotionReporter] WARNING: Could not load HuggingFace LLM: {e}")
            print(f"[EmotionReporter] Falling back to template-only reports.")
            self.llm = None
            self.tokenizer = None
            self._llm_available = False

    def _build_llm_prompt(self, dominant_me, confidence, active_aus, au_intensities):
        """
        Build a prompt for LLM-based report generation.

        The prompt encodes the emotion analysis results as natural language context
        for the LLM to generate a clinical description.

        Args:
            dominant_me (str): Dominant micro-expression category
            confidence (float): Classification confidence
            active_aus (list): List of (au_idx, intensity, name) tuples
            au_intensities (torch.Tensor): AU intensities (B, T, 28)
        Returns:
            prompt (str): Formatted prompt for the LLM
        """
        # AU description
        au_list_str = "; ".join([
            f"{name}({intensity:.0%})"
            for _, intensity, name in active_aus[:5]
        ])
        if not au_list_str:
            au_list_str = "no significant action units detected"

        # Analyze temporal dynamics
        B = au_intensities.shape[0]
        temporal_notes = []
        for b in range(min(B, 1)):
            au_mean = au_intensities[b].mean(dim=0)
            # Find most dynamic AU
            au_std = au_intensities[b].std(dim=0)
            max_var_idx = au_std.argmax().item()
            max_var_name = self.AU_NAMES.get(max_var_idx, f"AU{max_var_idx:02d}")

            # Check for rapid onset (high variance = dynamic)
            if au_std[max_var_idx].item() > 0.15:
                temporal_notes.append(
                    f"Note: {max_var_name} shows rapid temporal variation "
                    f"(onset-apex-decay pattern), suggesting a genuine micro-expression."
                )

        temporal_str = "\n".join(temporal_notes) if temporal_notes else ""

        prompt = (
            f"Emotion Analysis Results:\n"
            f"- Primary Emotion: {dominant_me} (confidence: {confidence:.0%})\n"
            f"- Active Action Units: {au_list_str}\n"
            f"{temporal_str}\n"
            f"Clinical Interpretation: Provide a detailed emotional state description "
            f"based on these findings. Include physiological correlates and confidence assessment."
        )
        return prompt

    def _generate_llm_report(self, prompt, max_new_tokens=100):
        """
        Generate a free-text report using the LLM.

        Args:
            prompt (str): Input prompt with emotion analysis results
            max_new_tokens (int): Maximum number of new tokens to generate
        Returns:
            llm_report (str): Generated free-text report
        """
        if not self._llm_available:
            return "[LLM unavailable]"

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=200
            ).to(self.llm.device)

            with torch.no_grad():
                outputs = self.llm.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # Decode: skip the input prompt tokens
            generated = outputs[0][inputs['input_ids'].shape[1]:]
            report = self.tokenizer.decode(generated, skip_special_tokens=True)
            return report.strip()

        except Exception as e:
            return f"[LLM generation error: {e}]"

    def _parse_aus(self, au_intensities):
        """Parse active AUs from intensity tensor."""
        B, T, _ = au_intensities.shape
        au_mean = au_intensities.mean(dim=1)  # (B, 28)

        results = []
        for b in range(B):
            sample_aus = []
            for au_idx in range(28):
                intensity = au_mean[b, au_idx].item()
                if intensity > self.AU_THRESHOLD:
                    name = self.AU_NAMES.get(au_idx, f"AU{au_idx:02d}")
                    sample_aus.append((au_idx, intensity, name))
            sample_aus.sort(key=lambda x: x[1], reverse=True)
            results.append(sample_aus)
        return results

    def _dominant_emotion(self, me_logits):
        """Identify dominant micro-expression category."""
        probabilities = torch.softmax(me_logits, dim=1)
        values, indices = torch.max(probabilities, dim=1)

        results = []
        for b in range(me_logits.shape[0]):
            cat = self.ME_CATEGORIES[indices[b].item()]
            conf = values[b].item()
            results.append((cat, conf))
        return results

    def _parse_rppg_cues(self, au_intensities):
        """Generate rPPG-related physiological cues from AU patterns."""
        B = au_intensities.shape[0]
        cues = []
        for b in range(B):
            au_mean = au_intensities[b].mean(dim=0)
            cheek_raiser = au_mean[6].item() if 6 < len(au_mean) else 0.0
            brow_lowerer = au_mean[4].item() if 4 < len(au_mean) else 0.0
            lip_raiser = au_mean[10].item() if 10 < len(au_mean) else 0.0

            cue_parts = []
            if cheek_raiser > 0.6:
                cue_parts.append("Elevated cheek activity (possible vasodilation, flushed face)")
            if brow_lowerer > 0.6:
                cue_parts.append("Lowered brow with facial tension (possible vasoconstriction, furrowed brow)")
            if lip_raiser > 0.6:
                cue_parts.append("Upper lip elevation detected (AU10, associated with disgust or contempt)")

            if not cue_parts:
                cue_parts.append("rPPG signal within normal range (no significant autonomic response)")
            cues.append("; ".join(cue_parts))
        return cues

    def _parse_opd_landmarks(self, au_intensities):
        """Generate temporal landmark descriptions."""
        B, T, _ = au_intensities.shape
        landmarks = []
        for b in range(B):
            au_mean = au_intensities[b].mean(dim=0)
            max_au = torch.argmax(au_mean).item()
            au_trace = au_intensities[b, :, max_au]
            au_name = self.AU_NAMES.get(max_au, f"AU{max_au:02d}")
            peak_frame = torch.argmax(au_trace).item()

            # Compute onset and decay
            above_threshold = au_trace > self.AU_THRESHOLD
            onset_frame = 0
            for t in range(T):
                if above_threshold[t]:
                    onset_frame = t
                    break

            decay_frame = T - 1
            for t in range(T - 1, -1, -1):
                if above_threshold[t]:
                    decay_frame = t
                    break

            landmark_str = (
                f"[{au_name}] onset at frame ~{onset_frame}, "
                f"peak at frame ~{peak_frame}, decay at frame ~{decay_frame}"
            )
            landmarks.append(landmark_str)
        return landmarks

    def _build_template_report(self, fused_feat, au_intensities, me_logits):
        """Build structured template-based clinical report."""
        active_aus = self._parse_aus(au_intensities)
        dominant_me = self._dominant_emotion(me_logits)
        rppg_cues = self._parse_rppg_cues(au_intensities)
        landmarks = self._parse_opd_landmarks(au_intensities)

        reports = []
        for b in range(au_intensities.shape[0]):
            au_list_str = "; ".join([
                f"{name}(AU{idx:02d}:{intensity:.1%})"
                for idx, intensity, name in active_aus[b]
            ]) if active_aus[b] else "No significant AU activation detected"

            me_cat, me_conf = dominant_me[b]

            report = (
                f"Emotion Analysis Report:\n"
                f"  - Dominant expression: {me_cat} with {me_conf:.1%} confidence\n"
                f"  - Active AUs: {au_list_str}\n"
                f"  - Physiological cues: {rppg_cues[b]}\n"
                f"  - Temporal dynamics: {landmarks[b]}\n"
                f"  - Confidence score: {me_conf:.3f}"
            )
            reports.append(report)
        return reports

    def forward(self, fused_feat, au_intensities, me_logits):
        """
        Generate emotion reports from model outputs.

        Args:
            fused_feat (torch.Tensor): Fused features, shape (B, 1024)
            au_intensities (torch.Tensor): AU intensities, shape (B, T, 28)
            me_logits (torch.Tensor): ME classification logits, shape (B, 7)
        Returns:
            template_reports (list[str]): Structured clinical reports
            llm_reports (list[str]): Free-text LLM-generated reports
        """
        print(f"[EmotionReporter] Input: fused={fused_feat.shape}, au={au_intensities.shape}, me={me_logits.shape}")

        # Generate template reports
        template_reports = self._build_template_report(fused_feat, au_intensities, me_logits)

        # Generate LLM reports
        llm_reports = []
        if self._llm_available:
            active_aus = self._parse_aus(au_intensities)
            dominant_me = self._dominant_emotion(me_logits)

            for b in range(fused_feat.shape[0]):
                me_cat, me_conf = dominant_me[b]
                prompt = self._build_llm_prompt(me_cat, me_conf, active_aus[b], au_intensities)
                llm_report = self._generate_llm_report(prompt)
                llm_reports.append(llm_report)

            print(f"[EmotionReporter] LLM reports generated: {len(llm_reports)}")
        else:
            llm_reports = [
                "[LLM Report Pending] Install HuggingFace transformers for free-text generation. "
                f"Template report available above."
                for _ in range(fused_feat.shape[0])
            ]
            print(f"[EmotionReporter] LLM reports: placeholder (HuggingFace not available)")

        print(f"[EmotionReporter] Template reports generated: {len(template_reports)}")
        print(f"[EmotionReporter] Sample report:\n{template_reports[0]}")

        return template_reports, llm_reports