# Inference Optimizations

`logits_to_keep=1` already avoids materializing logits for every prompt
position. The main remaining cost is model prefill, so optimize prefix reuse,
cached-tail batching, and padding before micro-optimizations.

## Highest-impact opportunities

1. Batch cached-prefix suffixes.

   `score_with_prefix()` currently loops through suffixes, running one forward
   call per suffix. For many suffixes that share a prefix, repeat/expand the
   prefix KV cache and score all tails in one batched forward pass.

2. Bucket uncached batches by token length.

   `score_batch()` left-pads each chunk to its longest prompt. Sorting or
   bucketing by tokenized length reduces padding, attention, and KV work when
   prompts differ substantially in size. Restore the original order before
   returning results.

3. Reduce cached-path lock contention.

   `_letter_logits_cached()` holds `_cache_lock` over tokenization, inference,
   tensor conversion, and cache cropping. Protect LRU lookup/insertion with a
   global lock, but use a per-cache-entry lock while its KV cache is extended
   and cropped. Independent prefixes can then run concurrently.

## Low-risk improvements

4. Gather A-Z logits once.

   The engine calls `batch_letter_logits()` and `batch_letter_mass()`, each of
   which gathers the same 26 token IDs. Gather the `[batch, 26]` tensor once;
   use it both to build the returned letter-logit dictionaries and to calculate
   the letter-side log-sum-exp.

5. Cache letter IDs as a device tensor.

   During model load, create a `torch.long` tensor containing the 26 token IDs
   on the model device. Use `index_select` rather than rebuilding a Python list
   and using advanced indexing on each inference call.

6. Make full-vocabulary letter mass optional.

   `batch_letter_mass()` performs a full-vocabulary `logsumexp` for every
   prompt. It is useful health telemetry, but can be gated behind debug mode,
   tests, or sampled monitoring when maximum throughput is more important.

## Conditional optimization

7. Tokenize only the suffix on prefix-cache hits.

   Cache hits currently tokenize `prefix + suffix` and validate its leading
   IDs against cached prefix IDs, correctly guarding against tokenizer merges at
   the concatenation boundary. If rendering can guarantee a merge-safe boundary,
   tokenize only the suffix to avoid repeatedly tokenizing the long prefix.
   Retain periodic or test-time full-tokenization validation before relying on
   this optimization.

## Suggested order

Implement the shared A-Z gather first, then benchmark cached-tail batching and
length bucketing against representative production prompt shapes. The latter
two should deliver the larger latency and throughput gains.
