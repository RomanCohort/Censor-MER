# Onset-Apex Difference Mode (Lateral Inhibition / Contrast Sensitivity)

## Problem
Censor has ~millions of parameters but only ~150 training samples.
Full video input (B,3,16,224,224) has too much redundant information for
the model to learn meaningful micro-expression features from so few samples.

## Solution: Onset-Apex Difference Input
In biology, retinal ganglion cells use **lateral inhibition** to enhance
edges and suppress uniform regions -- we only perceive *change*, not absolute
intensity. Similarly, micro-expression information is concentrated in the
difference between the onset (neutral) and apex (peak) frames.

## Changes

### 1. dataset_frames.py -- Return onset-apex diff as extra channel
- In `__getitem__`: compute onset-apex difference frame
- Return shape changes from (3,T,H,W) to (4,T,H,W) -- RGB + diff channel
- The diff channel is broadcast across T frames (same spatial diff for all t)
- Actually better: return onset frame, apex frame, and diff as separate items
- Simplest approach: add a `--diff_mode` flag that returns (4,T,H,W) with diff channel

### 2. config/defaults.py -- Update input channels
- FAST_PATHWAY_CONFIG: input_channels stays 2 (flow)
- SLOW_PATHWAY_CONFIG: input_channels 6 -> 7 (RGB + rPPG + diff)
  - Or keep 6 and replace rPPG with diff (rPPG is noisy anyway)
  - Better: add diff as 7th channel, let model learn to use it

### 3. main.py -- Compute diff in preprocessing
- Add onset-apex diff computation in Stage 1
- Concatenate diff channel to rgb_rppg for slow pathway
- This is the "lateral inhibition" step in the biomimetic narrative

### 4. train_frames.py -- Add --diff_mode flag
- Pass diff_mode to dataset so it returns the diff channel

## Minimal Implementation (safest approach)

Instead of changing the model architecture (risky), we can:
1. In dataset `__getitem__`: compute onset-apex diff frame
2. Replace the rPPG channel with the diff channel
   - rPPG is noisy and doesn't help much
   - Diff channel directly encodes the micro-expression change
   - No model architecture change needed (still 6 channels: RGB + diff*3)
3. The diff frame is replicated 3 times to match rPPG's 3 channels

This way:
- No model changes needed
- No config changes needed
- Only dataset_frames.py needs modification
- The diff channel replaces useless rPPG with high-signal onset-apex difference

## Implementation Steps

1. **dataset_frames.py**: In `__getitem__`, after sampling T frames:
   - Find onset and apex frame indices within the sampled sequence
   - Compute diff = apex_frame - onset_frame (per-pixel RGB difference)
   - Replace rPPG with diff (replicated 3x to match shape)
   - Actually, we can't do this in dataset because rPPG is computed in main.py

2. **Better approach**: Do it in main.py's forward pass
   - After saliency detection, compute onset-apex diff from the input
   - Replace rPPG extraction with diff computation
   - This keeps the biomimetic narrative: "lateral inhibition replaces blood-flow estimation"

3. **Even simpler**: Modify dataset to return diff as 4th channel,
   then in main.py concatenate diff*3 instead of rPPG

Let me go with approach 2: modify main.py to compute diff instead of rPPG
when diff_mode is enabled. This is the cleanest and requires minimal changes.

## Final Plan

### File: dataset_frames.py
- Add `diff_mode` parameter to `__getitem__`
- When diff_mode=True: also return onset_idx and apex_idx as metadata
- Actually, the dataset already returns (video, me_label, au_label)
- We need to pass onset/apex info to main.py somehow
- Simplest: compute diff in main.py from the input video itself
  - onset = first frame of the sequence
  - apex = frame with maximum motion (can detect from flow)
  - Or just use first and middle frame as onset/apex proxy

### File: main.py
- Add `diff_mode` parameter to Censor.__init__
- In forward(), when diff_mode=True:
  - Compute diff = x[:,:,T//2,:,:] - x[:,:,0,:,:]  (apex - onset)
  - Expand diff to (B,3,T,H,W) by replicating across time
  - Use diff instead of rPPG heatmap
- This is the "lateral inhibition" mechanism

### File: train_frames.py
- Add `--diff_mode` flag
- Pass to Censor(fast_preprocess=True, verbose=False, diff_mode=True)

### File: config/defaults.py
- No changes needed (still 6 channels for slow pathway)
