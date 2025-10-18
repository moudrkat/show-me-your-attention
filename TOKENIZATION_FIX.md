# Automatic Tokenization Handling

## The Problem

Words can split into multiple tokens:
- `"Scare"` → `["Sc", "are"]`
- `"IMPORTANT"` → `["IM", "PORT", "ANT"]`
- `"cat"` → `["Ġcat"]` (single token)

**Previous approach**: User had to manually specify subword tokens like `"Sc"` instead of `"scare"`.

**Issues**:
1. User needs to understand tokenization
2. Error-prone (which subword to use?)
3. Results depend on which subword you pick
4. Substring matching was naive: `"cat"` would match both `"cat"` AND `"scare"`

## The Solution

### New Method: `_find_word_token_indices(word, prompt)`

Uses the tokenizer's **offset_mapping** to accurately find ALL tokens that belong to a word:

```python
# Example: "Scare the cat"
# Tokens: ['Sc', 'are', 'Ġthe', 'Ġcat']

extractor._find_word_token_indices("scare", "Scare the cat")
# Returns: [0, 1]  ← Both tokens of "scare"

extractor._find_word_token_indices("cat", "Scare the cat")
# Returns: [3]  ← Only "cat", NOT "scare"!
```

### How It Works

1. **Find word boundaries** using regex: `\b{word}\b` (whole word only)
2. **Tokenize with offset mapping** to get character-to-token alignment
3. **Match overlapping tokens** to the word's character span
4. **Return all token indices** that contribute to the word

### Key Features

✅ **Automatic multi-token handling**: "scare" finds both ["Sc", "are"]
✅ **Word boundary detection**: "cat" doesn't match inside "scare"
✅ **Case insensitive**: Works with "IMPORTANT", "important", etc.
✅ **Multiple occurrences**: Finds all instances of a word in the prompt

## Computing Attention Values Correctly

### When a word spans multiple tokens:

**Target word**: "scare" → tokens [0, 1]
**Instruction word**: "cat" → token [3]

The attention from "cat" (token 3) to "scare" is computed as:
```
attention = average(attention[head, 3, 0], attention[head, 3, 1])
```

This is the **average attention to ALL tokens** of the word.

### Why averaging makes sense:

1. The word "scare" is represented by BOTH tokens ["Sc", "are"]
2. To understand how much "cat" attends to "scare", we need to consider attention to both parts
3. Averaging gives the overall binding strength to the complete word

### Implementation in code:

```python
# For each head, for each target token, for each instruction token
for head_idx in range(attn.shape[0]):
    for inst_idx in instruction_indices:  # [0, 1] for "scare"
        for tgt_idx in target_indices:    # [3] for "cat"
            attention_from_target.append(attn[head_idx, tgt_idx, inst_idx])

# Then compute mean across all these attention values
target_to_instruction_mean = np.mean(attention_from_target)
```

## Usage in App

**Before** (manual):
```python
target_word = "cat"
instruction_word = "Sc"  # Had to know tokenization!
```

**Now** (automatic):
```python
target_word = "cat"
instruction_word = "scare"  # Just use the full word!
```

The system automatically:
1. Finds ALL tokens for "scare" (whether it's 1 or 5 tokens)
2. Computes attention across all those tokens
3. Returns the mean binding strength

## Test Results

From `test_tokenization.py`:

```
Prompt: 'Scare the cat'
Tokens: ['Sc', 'are', 'Ġthe', 'Ġcat']
Indices for 'scare': [0, 1]  ✅ Both tokens found
Indices for 'cat': [3]        ✅ Only cat, not scare
```

```
Prompt: 'I beg you, scare the cat'
Tokens: ['I', 'Ġbeg', 'Ġyou', ',', 'Ġscare', 'Ġthe', 'Ġcat']
Indices for 'scare': [4]      ✅ Single token (different context)
Indices for 'cat': [6]
```

**Attention values computed correctly** across all 3 prompts!
