/* Vision-Exp attention of one query token for up to 32 heads, one head per
 * wave.  The keys of a query are a run of the raw ring (raw_bounds: the causal
 * window of a text token, the bidirectional span of an image token) followed
 * by its compressed rows: the visible part of the indexer's selection when
 * topk is set, otherwise every visible row, optionally with an additive mask.
 * A compressed row is visible once its last source token has arrived.
 *
 * Keys are staged eight at a time and folded into an online softmax, as the
 * text kernels do.  There is no score matrix: the memory of a launch depends
 * neither on the chunk nor on the context length, where heads x queries x
 * keys floats were gigabytes of scratch deep in a long context. */
__global__ static void attention_visual_mixed_online_kernel(
        float          *heads,
        const float    *sinks,
        const float    *q,
        const float    *raw_kv,
        const float    *comp_kv,
        const int32_t  *topk,
        const float    *comp_mask,
        const uint32_t *raw_bounds,
        uint32_t        n_tokens,
        uint32_t        pos0,
        uint32_t        first_raw_pos,
        uint32_t        raw_cap,
        uint32_t        raw_start,
        uint32_t        n_comp,
        uint32_t        top_k,
        uint32_t        ratio,
        uint32_t        n_head) {
    enum { STAGE = 8, HEAD_DIM = 512 };
    const uint32_t t = blockIdx.x;
    if (t >= n_tokens) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t head = blockIdx.y * 32u + (threadIdx.x >> 5u);
    const bool valid_head = head < n_head;

    __shared__ uint32_t comp_rows[DS4_ROCM_ATTENTION_INDEXED_TOPK_CAP];
    __shared__ uint32_t comp_count_s;
    __shared__ float4 kv_shared[STAGE * (HEAD_DIM / 4)];
    __shared__ float add_shared[STAGE];

    const uint32_t raw_lo = raw_bounds[2u * t];
    const uint32_t raw_hi = raw_bounds[2u * t + 1u];
    const uint32_t raw_count = raw_hi >= raw_lo ? raw_hi - raw_lo + 1u : 0u;
    const uint32_t raw_first = raw_lo - first_raw_pos;
    uint32_t visible_comp = ratio ? (pos0 + t + 1u) / ratio : 0u;
    if (visible_comp > n_comp) visible_comp = n_comp;
    if (threadIdx.x == 0) {
        uint32_t count = visible_comp;
        if (topk) {
            count = 0;
            for (uint32_t i = 0; i < top_k; i++) {
                const int32_t c = topk[(uint64_t)t * top_k + i];
                if (c >= 0 && (uint32_t)c < visible_comp) comp_rows[count++] = (uint32_t)c;
            }
        }
        comp_count_s = count;
    }
    __syncthreads();
    const uint32_t n_keys = raw_count + comp_count_s;

    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        const float4 *q4 = (const float4 *)(q + ((uint64_t)t * n_head + head) * HEAD_DIM);
        q0 = q4[lane +  0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }
    const float scale = rsqrtf((float)HEAD_DIM);
    /* The sink is a key with a zero value: it starts the running maximum and
     * holds the weight 1 at that maximum. */
    float max_s = valid_head ? sinks[head] : 0.0f;
    float sum_s = 1.0f;
    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;

    for (uint32_t key0 = 0; key0 < n_keys; key0 += STAGE) {
        const uint32_t nk = n_keys - key0 < STAGE ? n_keys - key0 : STAGE;
        for (uint32_t off = threadIdx.x; off < nk * (HEAD_DIM / 4); off += blockDim.x) {
            const uint32_t key = key0 + off / (HEAD_DIM / 4);
            const uint32_t d4 = off % (HEAD_DIM / 4);
            const float *row;
            float add = 0.0f;
            if (key < raw_count) {
                row = raw_kv + (uint64_t)attention_raw_ring_row(
                        raw_start, raw_cap, raw_first + key) * HEAD_DIM;
            } else {
                const uint32_t c = topk ? comp_rows[key - raw_count] : key - raw_count;
                row = comp_kv + (uint64_t)c * HEAD_DIM;
                if (comp_mask && d4 == 0) add = comp_mask[(uint64_t)t * n_comp + c];
            }
            kv_shared[off] = ((const float4 *)row)[d4];
            if (d4 == 0) add_shared[off / (HEAD_DIM / 4)] = add;
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t k = 0; k < nk; k++) {
                const float4 *kv4 = kv_shared + k * (HEAD_DIM / 4);
                const float4 k0 = kv4[lane +  0u];
                const float4 k1 = kv4[lane + 32u];
                const float4 k2 = kv4[lane + 64u];
                const float4 k3 = kv4[lane + 96u];
                float score = dot4_f32(q0, k0) + dot4_f32(q1, k1) +
                              dot4_f32(q2, k2) + dot4_f32(q3, k3);
                score = warp_sum_f32(score) * scale;
                score = __shfl_sync(FULL_WARP_MASK, score, 0);
                const float add = add_shared[k];
                score = add > -1.0e20f ? score + add : -INFINITY;

                const float new_max = fmaxf(max_s, score);
                const float old_scale = expf(max_s - new_max);
                const float key_scale = expf(score - new_max);
                sum_s = sum_s * old_scale + key_scale;
                o0.x = o0.x * old_scale + k0.x * key_scale;
                o0.y = o0.y * old_scale + k0.y * key_scale;
                o0.z = o0.z * old_scale + k0.z * key_scale;
                o0.w = o0.w * old_scale + k0.w * key_scale;
                o1.x = o1.x * old_scale + k1.x * key_scale;
                o1.y = o1.y * old_scale + k1.y * key_scale;
                o1.z = o1.z * old_scale + k1.z * key_scale;
                o1.w = o1.w * old_scale + k1.w * key_scale;
                o2.x = o2.x * old_scale + k2.x * key_scale;
                o2.y = o2.y * old_scale + k2.y * key_scale;
                o2.z = o2.z * old_scale + k2.z * key_scale;
                o2.w = o2.w * old_scale + k2.w * key_scale;
                o3.x = o3.x * old_scale + k3.x * key_scale;
                o3.y = o3.y * old_scale + k3.y * key_scale;
                o3.z = o3.z * old_scale + k3.z * key_scale;
                o3.w = o3.w * old_scale + k3.w * key_scale;
                max_s = new_max;
            }
        }
        __syncthreads();
    }

    if (valid_head) {
        const float inv_s = 1.0f / sum_s;
        o0.x *= inv_s; o0.y *= inv_s; o0.z *= inv_s; o0.w *= inv_s;
        o1.x *= inv_s; o1.y *= inv_s; o1.z *= inv_s; o1.w *= inv_s;
        o2.x *= inv_s; o2.y *= inv_s; o2.z *= inv_s; o2.w *= inv_s;
        o3.x *= inv_s; o3.y *= inv_s; o3.z *= inv_s; o3.w *= inv_s;
        float4 *out4 = (float4 *)(heads + ((uint64_t)t * n_head + head) * HEAD_DIM);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}

__global__ static void router_select_visual_parallel_kernel(
        int32_t       *selected,
        float         *weights,
        float         *probs,
        const float   *bias,
        const float   *visual_bias,
        const int32_t *hash,
        const float   *logits,
        const int32_t *tokens,
        uint32_t       hash_rows,
        uint32_t       vocab_size,
        uint32_t       n_tokens,
        int            has_bias,
        int            hash_mode) {
    const uint32_t t = blockIdx.x;
    const uint32_t i = threadIdx.x;
    if (t >= n_tokens || i >= 256u) return;
    const float *log = logits + (uint64_t)t * 256u;
    float *prob = probs + (uint64_t)t * 256u;
    int32_t *sel = selected + (uint64_t)t * 6u;
    float *w = weights + (uint64_t)t * 6u;
    __shared__ float sprob[256];

    const float p = ds4_precise_sqrtf(softplus_dev(log[i]));
    sprob[i] = p;
    prob[i] = p;
    __syncthreads();

    if (i != 0u) return;
    const int32_t token = tokens[t];
    const bool image = token >= 0 && (uint32_t)token >= vocab_size;
    if (hash_mode && !image) {
        const uint32_t row = token >= 0 && (uint32_t)token < hash_rows
            ? (uint32_t)token : 0u;
        const int32_t *src = hash + (uint64_t)row * 6u;
        for (uint32_t j = 0; j < 6u; j++) sel[j] = src[j];
    } else {
        for (uint32_t j = 0; j < 6u; j++) sel[j] = -1;
        for (uint32_t e = 0; e < 256u; e++) {
            const float score = sprob[e] +
                (image ? visual_bias[e] : (has_bias ? bias[e] : 0.0f));
            for (uint32_t j = 0; j < 6u; j++) {
                const int32_t old = sel[j];
                const float old_score = old < 0 ? -INFINITY :
                    sprob[old] + (image ? visual_bias[old]
                                        : (has_bias ? bias[old] : 0.0f));
                if (old < 0 || score > old_score) {
                    for (uint32_t k = 5u; k > j; k--) sel[k] = sel[k - 1u];
                    sel[j] = (int32_t)e;
                    break;
                }
            }
        }
    }

    float sum = 0.0f;
    for (uint32_t j = 0; j < 6u; j++) {
        const int32_t e = sel[j];
        const float v = e >= 0 && e < 256 ? sprob[e] : 0.0f;
        w[j] = v;
        sum += v;
    }
    sum = fmaxf(sum, 6.103515625e-5f);
    for (uint32_t j = 0; j < 6u; j++) w[j] = w[j] / sum * 1.5f;
}

static int attention_visual_mixed_launch(
        ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size,
        uint64_t sinks_offset, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv, const ds4_gpu_tensor *comp_kv,
        uint32_t comp_kv_f16, const ds4_gpu_tensor *topk, uint32_t top_k,
        const ds4_gpu_tensor *comp_mask, const int32_t *tokens,
        uint32_t vocab_size, uint32_t n_tokens, uint32_t pos0, uint32_t n_raw,
        uint32_t raw_cap, uint32_t raw_start, uint32_t n_comp, uint32_t window,
        uint32_t ratio, uint32_t n_head, uint32_t head_dim) {
    if (!heads || !model_map || !q || !raw_kv || !tokens || vocab_size == 0 ||
        n_tokens == 0 || n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap ||
        (n_comp != 0 && (!comp_kv || ratio == 0)) || comp_kv_f16 ||
        head_dim != 512u ||
        (topk && (n_comp == 0 || top_k == 0 ||
                  top_k > DS4_ROCM_ATTENTION_INDEXED_TOPK_CAP ||
                  !cuda_tensor_has_elems2(topk, n_tokens, top_k,
                                          sizeof(int32_t)))) ||
        !cuda_model_range_fits(model_size, sinks_offset,
                               (uint64_t)n_head * sizeof(float)) ||
        !cuda_tensor_has_elems3(heads, n_tokens, n_head, head_dim,
                                sizeof(float)) ||
        !cuda_tensor_has_elems3(q, n_tokens, n_head, head_dim,
                                sizeof(float)) ||
        !cuda_tensor_has_elems2(raw_kv, raw_cap, head_dim, sizeof(float)) ||
        (n_comp && !cuda_tensor_has_elems2(comp_kv, n_comp, head_dim,
                                           sizeof(float))) ||
        (comp_mask && !cuda_tensor_has_elems2(comp_mask, n_tokens, n_comp,
                                              sizeof(float))) ||
        pos0 > UINT32_MAX - n_tokens || n_raw > pos0 + n_tokens) return 0;

    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float),
            "visual_attn_sinks");
    if (!sinks) return 0;

    std::vector<uint32_t> bounds((size_t)n_tokens * 2u);
    if (!ds4_deepseek4_attention_bounds(
            (const int *)tokens, n_tokens, vocab_size,
            pos0, n_raw, window, bounds.data())) return 0;
    const uint64_t bounds_bytes = (uint64_t)n_tokens * 2u * sizeof(uint32_t);
    uint32_t *device_bounds =
        (uint32_t *)cuda_tmp_alloc(bounds_bytes, "visual attention bounds");
    if (!device_bounds ||
        !cuda_ok(cudaMemcpy(device_bounds, bounds.data(),
                            (size_t)bounds_bytes, cudaMemcpyHostToDevice),
                 "visual attention bounds upload")) return 0;

    const dim3 grid(n_tokens, (n_head + 31u) / 32u, 1u);
    attention_visual_mixed_online_kernel<<<grid, 1024u>>>(
            (float *)heads->ptr, sinks, (const float *)q->ptr,
            (const float *)raw_kv->ptr,
            n_comp ? (const float *)comp_kv->ptr : NULL,
            topk ? (const int32_t *)topk->ptr : NULL,
            comp_mask ? (const float *)comp_mask->ptr : NULL,
            device_bounds, n_tokens, pos0, pos0 + n_tokens - n_raw, raw_cap,
            raw_start, n_comp, top_k, ratio, n_head);
    return cuda_ok(cudaGetLastError(), "visual attention launch");
}

extern "C" int ds4_gpu_attention_visual_mixed_batch_heads_tensor(
        ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size,
        uint64_t sinks_offset, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv, const ds4_gpu_tensor *comp_kv,
        uint32_t comp_kv_f16, const ds4_gpu_tensor *comp_mask,
        uint32_t use_comp_mask, const int32_t *tokens, uint32_t vocab_size,
        uint32_t n_tokens, uint32_t pos0, uint32_t n_raw, uint32_t raw_cap,
        uint32_t raw_start, uint32_t n_comp, uint32_t window, uint32_t ratio,
        uint32_t n_head, uint32_t head_dim) {
    if (use_comp_mask && !comp_mask) return 0;
    return attention_visual_mixed_launch(
            heads, model_map, model_size, sinks_offset, q, raw_kv, comp_kv,
            comp_kv_f16, NULL, 0, use_comp_mask ? comp_mask : NULL, tokens,
            vocab_size, n_tokens, pos0, n_raw, raw_cap, raw_start, n_comp,
            window, ratio, n_head, head_dim);
}

extern "C" int ds4_gpu_attention_visual_indexed_batch_heads_tensor(
        ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size,
        uint64_t sinks_offset, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv, const ds4_gpu_tensor *comp_kv,
        uint32_t comp_kv_f16, const ds4_gpu_tensor *topk, uint32_t top_k,
        const int32_t *tokens, uint32_t vocab_size, uint32_t n_tokens,
        uint32_t pos0, uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start,
        uint32_t n_comp, uint32_t window, uint32_t ratio, uint32_t n_head,
        uint32_t head_dim) {
    if (!topk) return 0;
    return attention_visual_mixed_launch(
            heads, model_map, model_size, sinks_offset, q, raw_kv, comp_kv,
            comp_kv_f16, topk, top_k, NULL, tokens, vocab_size, n_tokens,
            pos0, n_raw, raw_cap, raw_start, n_comp, window, ratio, n_head,
            head_dim);
}

extern "C" int ds4_gpu_router_select_batch_visual_tensor(
        ds4_gpu_tensor *selected, ds4_gpu_tensor *weights,
        ds4_gpu_tensor *probs, const void *model_map, uint64_t model_size,
        uint64_t bias_offset, uint64_t hash_offset, uint32_t hash_rows,
        bool has_bias, bool hash_mode, const void *vision_map,
        uint64_t vision_size, uint64_t visual_bias_offset,
        const ds4_gpu_tensor *logits, const ds4_gpu_tensor *tokens,
        uint32_t vocab_size, uint32_t n_expert, uint32_t n_expert_used,
        float expert_weight_scale, uint32_t n_tokens) {
    if (!selected || !weights || !probs || !logits || !tokens ||
        !model_map || !vision_map || n_tokens == 0 || vocab_size == 0 ||
        n_expert != 256u || n_expert_used != 6u ||
        fabsf(expert_weight_scale - 1.5f) > 1.0e-6f ||
        !cuda_tensor_has_elems2(logits, n_tokens, 256u, sizeof(float)) ||
        !cuda_tensor_has_elems2(probs, n_tokens, 256u, sizeof(float)) ||
        !cuda_tensor_has_elems2(selected, n_tokens, 6u, sizeof(int32_t)) ||
        !cuda_tensor_has_elems2(weights, n_tokens, 6u, sizeof(float)) ||
        !cuda_tensor_has_i32(tokens, n_tokens)) return 0;

    const float *bias = NULL;
    const int32_t *hash = NULL;
    if (has_bias && !hash_mode) {
        if (!cuda_model_range_fits(model_size, bias_offset,
                                   256u * sizeof(float))) return 0;
        bias = (const float *)cuda_model_range_ptr(
                model_map, bias_offset, 256u * sizeof(float), "router_bias");
        if (!bias) return 0;
    }
    if (hash_mode) {
        if (hash_rows == 0u) return 0;
        const uint64_t hash_bytes = (uint64_t)hash_rows * 6u * sizeof(int32_t);
        if (!cuda_model_range_fits(model_size, hash_offset, hash_bytes))
            return 0;
        hash = (const int32_t *)cuda_model_range_ptr(
                model_map, hash_offset, hash_bytes, "router_hash");
        if (!hash) return 0;
    }
    if (!cuda_model_range_fits(vision_size, visual_bias_offset,
                               256u * sizeof(float))) return 0;
    const float *visual_bias = (const float *)cuda_model_range_ptr(
            vision_map, visual_bias_offset, 256u * sizeof(float),
            "visual_router_bias");
    if (!visual_bias) return 0;

    router_select_visual_parallel_kernel<<<n_tokens, 256>>>(
            (int32_t *)selected->ptr, (float *)weights->ptr,
            (float *)probs->ptr, bias, visual_bias, hash,
            (const float *)logits->ptr, (const int32_t *)tokens->ptr,
            hash_rows, vocab_size, n_tokens,
            has_bias && !hash_mode, hash_mode);
    return cuda_ok(cudaGetLastError(), "visual router select launch");
}
