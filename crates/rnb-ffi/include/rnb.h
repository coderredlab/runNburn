/*
 * runNburn C API
 *
 * Licensed under the Apache License, Version 2.0.
 */

#ifndef RNB_H_INCLUDED
#define RNB_H_INCLUDED

#include <stdint.h>

#define RNB_API_VERSION_MAJOR 0
#define RNB_API_VERSION_MINOR 1
#define RNB_API_VERSION_PATCH 0

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RnbContext RnbContext;

typedef struct RnbStats {
    float prefill_ms;
    float decode_ms;
    uint32_t prefill_tokens;
    uint32_t decode_tokens;
} RnbStats;

/* Load a GGUF model. Returns NULL on failure. */
RnbContext *rnb_load(const char *model_path);

/* Load a GGUF model with a host RAM budget. Zero selects the automatic policy. */
RnbContext *rnb_load_with_ram_budget(const char *model_path,
                                     uint64_t ram_budget_bytes);

/* Apply the built-in Qwen-style chat wrapper and prefill the prompt. */
int32_t rnb_submit(RnbContext *ctx, const char *prompt);

/* Prefill an already rendered prompt without adding chat markup. */
int32_t rnb_submit_raw(RnbContext *ctx, const char *rendered_prompt);

/* Return the next UTF-8 fragment, or NULL after EOS or an error. */
const char *rnb_next_token(RnbContext *ctx);

/* Update sampler parameters without reloading the model. */
int32_t rnb_set_sampler(RnbContext *ctx,
                        float temperature,
                        float top_p,
                        uint32_t top_k,
                        float repetition_penalty);

/* Copy generation statistics into out. */
void rnb_get_stats(RnbContext *ctx, RnbStats *out);

/* Reset sequence state for a new conversation. */
void rnb_reset(RnbContext *ctx);

/* Release a context returned by rnb_load or rnb_load_with_ram_budget. */
void rnb_free(RnbContext *ctx);

#ifdef __cplusplus
}
#endif

#endif /* RNB_H_INCLUDED */
