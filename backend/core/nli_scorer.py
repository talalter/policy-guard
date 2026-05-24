"""NLI-based contradiction scorer using a ModernBERT cross-encoder.

Splits context and response into sentences, uses a bi-encoder to select the
top-K most semantically similar premise candidates per hypothesis (filtered by
a similarity threshold and lexical overlap gate), then scores those pairs in
mini-batches via ModernBERT, yielding results as each mini-batch completes.

ModernBERT advantages over DeBERTa-v3:
- Flash Attention 2 for faster inference on CUDA
- 8 192-token context window (vs 512) - handles long LLM outputs without truncation
- Rotary position embeddings (RoPE) that generalise better to out-of-distribution lengths

Default model: dleemiller/ModernCE-base-nli - a cross-encoder fine-tuned on
AllNLI (MNLI + SNLI), achieving 92% on MNLI-mismatched. The "CE" suffix
signals it is purpose-built for pairwise sequence classification, exactly the
pattern used here.

Pairs are sorted by bi-encoder similarity descending before scoring so the
highest-confidence candidates arrive first.
"""

import logging
from collections.abc import Iterator

import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from backend.config import settings
from backend.models import NLIResult, SentencePair
from backend.utils.text import flatten_tool_context, is_tool_context, split_sentences

logger = logging.getLogger(__name__)

_MODEL_NAME = settings.nli_model
_BI_ENCODER_MODEL = settings.bi_encoder_model
_NLI_TOP_K = settings.nli_top_k
_NLI_MIN_SIMILARITY = settings.nli_min_similarity
_NLI_MINI_BATCH_SIZE = settings.nli_mini_batch_size
_NLI_MAX_LENGTH = settings.nli_max_length


# BGE models require an instruction prefix on the query (hypothesis) side only.
# Passage (premise) encodings are left as-is.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _needs_query_prefix(model_name: str) -> bool:
    """Return True for BGE bi-encoders that require asymmetric query prefixing."""
    return "bge" in model_name.lower()


def _log_pair_result(
    log: logging.Logger,
    winning_label: str,
    confidence: float,
    contradiction_score: float,
    entailment_score: float,
    neutral_score: float,
    pair: "SentencePair",
) -> None:
    """Log one scored pair: INFO for confirmed contradictions, DEBUG for everything else."""
    if winning_label == "contradiction":
        log.debug(
            "Contradiction hit conf=%.3f contradiction=%.3f entailment=%.3f neutral=%.3f",
            confidence,
            contradiction_score,
            entailment_score,
            neutral_score,
        )
        log.debug("Full pair | premise=%r | hypothesis=%r", pair.premise, pair.hypothesis)
    else:
        log.debug(
            "Pair scored label=%s conf=%.3f contradiction=%.3f premise=%r hypothesis=%r",
            winning_label,
            confidence,
            contradiction_score,
            pair.premise[:60],
            pair.hypothesis[:60],
        )


class NLIScorer:
    """Scores (premise, hypothesis) pairs for contradiction using ModernBERT NLI.

    Pipeline:
    1. Bi-encoder computes an (M×N) cosine similarity matrix.
    2. For each hypothesis, top-K premises are selected then filtered by a
       minimum similarity threshold and a lexical overlap gate.
    3. Surviving pairs are sorted by similarity descending and scored in
       mini-batches, yielding NLIResult objects as each batch completes.
    """

    def __init__(self) -> None:
        """Load the bi-encoder and ModernBERT tokenizer/model once at construction time."""
        logger.info("Loading bi-encoder: %s", _BI_ENCODER_MODEL)
        self._bi_encoder = SentenceTransformer(_BI_ENCODER_MODEL)

        logger.info("Loading NLI model: %s", _MODEL_NAME)
        self._tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        self._model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
        self._model.eval()
        self._label2idx = {v.lower(): k for k, v in self._model.config.id2label.items()}
        logger.info("NLI model loaded - label map: %s", self._label2idx)

    def _compute_similarity_matrix(
        self, premises: list[str], hypotheses: list[str]
    ) -> torch.Tensor:
        """Encode premises and hypotheses, return (M x N) cosine similarity matrix."""
        premise_embs = self._bi_encoder.encode(premises, convert_to_tensor=True, show_progress_bar=False)
        queries = (
            [_BGE_QUERY_PREFIX + h for h in hypotheses]
            if _needs_query_prefix(_BI_ENCODER_MODEL)
            else hypotheses
        )
        hyp_embs = self._bi_encoder.encode(queries, convert_to_tensor=True, show_progress_bar=False)
        return util.cos_sim(premise_embs, hyp_embs)  # shape: (M, N)

    def _build_pairs(
        self,
        premises: list[str],
        hypotheses: list[str],
        sim_matrix: torch.Tensor,
        top_k: int,
    ) -> tuple[list[SentencePair], list[float]]:
        """Select top-K premise candidates per hypothesis, apply similarity threshold.

        Filters pairs below _NLI_MIN_SIMILARITY. The cross-encoder is the right
        place to reject bad pairs - no lexical overlap gate is applied here, so
        policy rules that use different vocabulary from the agent action are not
        silently dropped before NLI sees them.

        Returns:
            Parallel (pairs, sim_scores) lists.
        """
        pairs: list[SentencePair] = []
        sim_scores: list[float] = []
        k = min(top_k, len(premises))

        for h_idx, hypothesis in enumerate(hypotheses):
            top_indices = sim_matrix[:, h_idx].topk(k).indices.tolist()

            for p_idx in top_indices:
                sim_score = float(sim_matrix[p_idx, h_idx])

                if sim_score < _NLI_MIN_SIMILARITY:
                    continue

                pairs.append(SentencePair(premise=premises[p_idx], hypothesis=hypothesis))
                sim_scores.append(sim_score)

        return pairs, sim_scores

    def _score_batch(self, pairs: list[SentencePair]) -> list[NLIResult]:
        """Run one mini-batch of pairs through ModernBERT and return NLIResult list."""
        premise_texts = [p.premise for p in pairs]
        hypothesis_texts = [p.hypothesis for p in pairs]

        batch_encoding = self._tokenizer(
            premise_texts,
            hypothesis_texts,
            padding=True,
            truncation=True,
            max_length=_NLI_MAX_LENGTH,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = self._model(**batch_encoding).logits  # shape: (B, 3)

        probs = torch.softmax(logits, dim=-1)  # shape: (B, 3)

        batch_results = []
        for pair, pair_probs in zip(pairs, probs):
            contradiction_score = float(pair_probs[self._label2idx["contradiction"]])
            entailment_score = float(pair_probs[self._label2idx["entailment"]])
            neutral_score = float(pair_probs[self._label2idx["neutral"]])

            label_scores = {
                "contradiction": contradiction_score,
                "entailment": entailment_score,
                "neutral": neutral_score,
            }
            winning_label = max(label_scores, key=label_scores.__getitem__)
            confidence = label_scores[winning_label]

            nli_result = NLIResult(
                pair=pair,
                label=winning_label,
                confidence=confidence,
                contradiction_score=contradiction_score,
                entailment_score=entailment_score,
                neutral_score=neutral_score,
            )

            _log_pair_result(
                logger,
                winning_label,
                confidence,
                contradiction_score,
                entailment_score,
                neutral_score,
                pair,
            )

            batch_results.append(nli_result)

        return batch_results

    def _score_pairs_stream(
        self,
        pairs: list[SentencePair],
        sim_scores: list[float],
    ) -> Iterator[NLIResult]:
        """Sort pairs by similarity descending, score in mini-batches, yield as ready."""
        sorted_indices = sorted(range(len(pairs)), key=lambda i: sim_scores[i], reverse=True)
        sorted_pairs = [pairs[i] for i in sorted_indices]

        for i in range(0, len(sorted_pairs), _NLI_MINI_BATCH_SIZE):
            batch = sorted_pairs[i : i + _NLI_MINI_BATCH_SIZE]
            yield from self._score_batch(batch)

    def score(self, context: str, response: str) -> Iterator[NLIResult]:
        """Score sentence pairs between context and response, yielding as results arrive.

        Uses bi-encoder similarity to pre-filter candidates, then streams
        NLIResult objects in mini-batches sorted highest similarity first.

        Args:
            context: The source document the response should be faithful to.
            response: The LLM-generated response to evaluate.

        Yields:
            NLIResult for each scored pair, highest-similarity pairs first.
        """
        tool_ctx = is_tool_context(context)
        if tool_ctx:
            context = flatten_tool_context(context)
            logger.debug("Tool call context detected - applied prose normalisation")

        premises = split_sentences(context)
        hypotheses = split_sentences(response)

        logger.debug(
            "Sentence split: %d premise(s) from context, %d hypothesis(es) from response",
            len(premises),
            len(hypotheses),
        )

        if not premises or not hypotheses:
            logger.warning("No sentence pairs to score - empty context or response")
            return

        full_cross_product = len(premises) * len(hypotheses)
        sim_matrix = self._compute_similarity_matrix(premises, hypotheses)
        pairs, sim_scores = self._build_pairs(
            premises, hypotheses, sim_matrix, _NLI_TOP_K
        )

        logger.debug(
            "Pair selection: %d/%d pairs survive",
            len(pairs),
            full_cross_product,
        )

        yield from self._score_pairs_stream(pairs, sim_scores)
