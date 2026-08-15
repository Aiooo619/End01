# v003 initial checkpoint comparison

- Base: Illustrious XL v2.0
- Prompt: alpine signal engineer, fixed across all candidates
- Seed: 42
- Resolution: 768x1024
- Checkpoints: epoch 2, epoch 4, epoch 6, final
- LoRA strengths tested: 0.5 and 0.3
- Discord sessions: `cmp-dcc61d01ce6f`, `cmp-c9e25ecef465`

## Findings

- Anatomy and clothing separation are substantially more stable than v002.
- No obvious extra legs or severe sleeve/mechanical fusion appeared.
- All candidates converge on a similar short-haired female, oversized coat,
  animal tail, and salute-like hand pose.
- Lowering strength from 0.5 to 0.3 did not remove these repeated attributes.
- Prompt compliance is partial: navy/orange insulated workwear is present, but
  the requested signal-engineer concept and tools are weak.
- Epoch 4 and epoch 6 are the most structurally stable. Final is not clearly
  superior and should not be promoted without further tests.

## Decision

Keep v003 as a draft. Do not promote yet. The next dataset revision should
rebalance species, gender, pose, coat length, and silhouette; captions should
separate identity/pose attributes from clothing structure before v004.
