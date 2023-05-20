#include "gpt2.h"
#include "llama.cpp/ggml.h"

#include "utils.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <iostream>
#if defined(_WIN32) && defined(_MSC_VER)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#include <stdio.h>
#else
#include <unistd.h>
#endif
#include <sstream>
#include <thread>
#include <unordered_set>
#include <regex>

// default hparams (GPT-2 117M)
struct gpt2_hparams
{
    int32_t n_vocab = 50257;
    int32_t n_ctx = 1024;
    int32_t n_embd = 768;
    int32_t n_head = 12;
    int32_t n_layer = 12;
    int32_t ftype = 1;
};

struct gpt2_layer
{
    // normalization
    struct ggml_tensor *ln_1_g;
    struct ggml_tensor *ln_1_b;

    struct ggml_tensor *ln_2_g;
    struct ggml_tensor *ln_2_b;

    // attention
    struct ggml_tensor *c_attn_attn_w;
    struct ggml_tensor *c_attn_attn_b;

    struct ggml_tensor *c_attn_proj_w;
    struct ggml_tensor *c_attn_proj_b;

    // mlp
    struct ggml_tensor *c_mlp_fc_w;
    struct ggml_tensor *c_mlp_fc_b;

    struct ggml_tensor *c_mlp_proj_w;
    struct ggml_tensor *c_mlp_proj_b;
};

struct gpt2_model
{
    gpt2_hparams hparams;

    // normalization
    struct ggml_tensor *ln_f_g;
    struct ggml_tensor *ln_f_b;

    struct ggml_tensor *wte;     // position embedding
    struct ggml_tensor *wpe;     //    token embedding
    struct ggml_tensor *lm_head; // language model head

    std::vector<gpt2_layer> layers;

    // key + value memory
    struct ggml_tensor *memory_k;
    struct ggml_tensor *memory_v;

    //
    struct ggml_context *ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
};

// load the model's weights from a file
bool gpt2_model_load(const std::string &fname, gpt2_model &model, gpt_vocab &vocab)
{
    printf("%s: loading model from '%s'\n", __func__, fname.c_str());

    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin)
    {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return false;
    }

    // verify magic
    {
        uint32_t magic;
        fin.read((char *)&magic, sizeof(magic));
        if (magic != 0x67676d32)
        {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
            return false;
        }
    }

    // load hparams
    {
        auto &hparams = model.hparams;

        fin.read((char *)&hparams.n_vocab, sizeof(hparams.n_vocab));
        fin.read((char *)&hparams.n_ctx, sizeof(hparams.n_ctx));
        fin.read((char *)&hparams.n_embd, sizeof(hparams.n_embd));
        fin.read((char *)&hparams.n_head, sizeof(hparams.n_head));
        fin.read((char *)&hparams.n_layer, sizeof(hparams.n_layer));
        fin.read((char *)&hparams.ftype, sizeof(hparams.ftype));

        // const int32_t qntvr = hparams.ftype / GGML_QNT_VERSION_FACTOR;

        printf("%s: n_vocab = %d\n", __func__, hparams.n_vocab);
        printf("%s: n_ctx   = %d\n", __func__, hparams.n_ctx);
        printf("%s: n_embd  = %d\n", __func__, hparams.n_embd);
        printf("%s: n_head  = %d\n", __func__, hparams.n_head);
        printf("%s: n_layer = %d\n", __func__, hparams.n_layer);
        printf("%s: ftype   = %d\n", __func__, hparams.ftype);
        // printf("%s: qntvr   = %d\n", __func__, qntvr);

        hparams.ftype %= GGML_QNT_VERSION_FACTOR;
    }

    // load vocab
    {
        int32_t n_vocab = 0;
        fin.read((char *)&n_vocab, sizeof(n_vocab));

        if (n_vocab != model.hparams.n_vocab)
        {
            fprintf(stderr, "%s: invalid model file '%s' (bad vocab size %d != %d)\n",
                    __func__, fname.c_str(), n_vocab, model.hparams.n_vocab);
            return false;
        }

        std::string word;
        for (int i = 0; i < n_vocab; i++)
        {
            uint32_t len;
            fin.read((char *)&len, sizeof(len));

            word.resize(len);
            fin.read((char *)word.data(), len);

            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;
        }
    }

    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    ggml_type wtype = ggml_ftype_to_ggml_type((ggml_ftype)(model.hparams.ftype));
    if (wtype == GGML_TYPE_COUNT)
    {
        fprintf(stderr, "%s: invalid model file '%s' (bad ftype value %d)\n",
                __func__, fname.c_str(), model.hparams.ftype);
        return false;
    }

    auto &ctx = model.ctx;

    size_t ctx_size = 0;

    {
        const auto &hparams = model.hparams;

        const int n_embd = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx = hparams.n_ctx;
        const int n_vocab = hparams.n_vocab;

        ctx_size += n_embd * ggml_type_sizef(GGML_TYPE_F32); // ln_f_g
        ctx_size += n_embd * ggml_type_sizef(GGML_TYPE_F32); // ln_f_b

        ctx_size += n_vocab * n_embd * ggml_type_sizef(wtype);       // wte
        ctx_size += n_ctx * n_embd * ggml_type_sizef(GGML_TYPE_F32); // wpe
        ctx_size += n_vocab * n_embd * ggml_type_sizef(wtype);       // lm_head

        ctx_size += n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32)); // ln_1_g
        ctx_size += n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32)); // ln_1_b

        ctx_size += n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32)); // ln_2_g
        ctx_size += n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32)); // ln_2_b

        ctx_size += n_layer * (3 * n_embd * n_embd * ggml_type_sizef(wtype)); // c_attn_attn_w
        ctx_size += n_layer * (3 * n_embd * ggml_type_sizef(GGML_TYPE_F32));  // c_attn_attn_b

        ctx_size += n_layer * (n_embd * n_embd * ggml_type_sizef(wtype)); // c_attn_proj_w
        ctx_size += n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32));  // c_attn_proj_b

        ctx_size += n_layer * (4 * n_embd * n_embd * ggml_type_sizef(wtype)); // c_mlp_fc_w
        ctx_size += n_layer * (4 * n_embd * ggml_type_sizef(GGML_TYPE_F32));  // c_mlp_fc_b

        ctx_size += n_layer * (4 * n_embd * n_embd * ggml_type_sizef(wtype)); // c_mlp_proj_w
        ctx_size += n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32));      // c_mlp_proj_b

        ctx_size += n_ctx * n_layer * n_embd * ggml_type_sizef(GGML_TYPE_F32); // memory_k
        ctx_size += n_ctx * n_layer * n_embd * ggml_type_sizef(GGML_TYPE_F32); // memory_v

        ctx_size += (6 + 12 * n_layer) * 512; // object overhead

        printf("%s: ggml tensor size = %d bytes\n", __func__, (int)sizeof(ggml_tensor));
        printf("%s: ggml ctx size = %6.2f MB\n", __func__, ctx_size / (1024.0 * 1024.0));
    }

    // create the ggml context
    {
        struct ggml_init_params params = {
            .mem_size = ctx_size,
            .mem_buffer = NULL,
            .no_alloc = false,
        };

        model.ctx = ggml_init(params);
        if (!model.ctx)
        {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    // prepare memory for the weights
    {
        const auto &hparams = model.hparams;

        const int n_embd = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx = hparams.n_ctx;
        const int n_vocab = hparams.n_vocab;

        model.layers.resize(n_layer);

        model.ln_f_g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        model.ln_f_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

        model.wte = ggml_new_tensor_2d(ctx, wtype, n_embd, n_vocab);
        model.wpe = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_ctx);
        model.lm_head = ggml_new_tensor_2d(ctx, wtype, n_embd, n_vocab);

        // map by name
        model.tensors["model/ln_f/g"] = model.ln_f_g;
        model.tensors["model/ln_f/b"] = model.ln_f_b;

        model.tensors["model/wte"] = model.wte;
        model.tensors["model/wpe"] = model.wpe;
        model.tensors["model/lm_head"] = model.lm_head;

        for (int i = 0; i < n_layer; ++i)
        {
            auto &layer = model.layers[i];

            layer.ln_1_g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            layer.ln_1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            layer.ln_2_g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            layer.ln_2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            layer.c_attn_attn_w = ggml_new_tensor_2d(ctx, wtype, n_embd, 3 * n_embd);
            layer.c_attn_attn_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3 * n_embd);

            layer.c_attn_proj_w = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.c_attn_proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            layer.c_mlp_fc_w = ggml_new_tensor_2d(ctx, wtype, n_embd, 4 * n_embd);
            layer.c_mlp_fc_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4 * n_embd);

            layer.c_mlp_proj_w = ggml_new_tensor_2d(ctx, wtype, 4 * n_embd, n_embd);
            layer.c_mlp_proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            // map by name
            model.tensors["model/h" + std::to_string(i) + "/ln_1/g"] = layer.ln_1_g;
            model.tensors["model/h" + std::to_string(i) + "/ln_1/b"] = layer.ln_1_b;

            model.tensors["model/h" + std::to_string(i) + "/ln_2/g"] = layer.ln_2_g;
            model.tensors["model/h" + std::to_string(i) + "/ln_2/b"] = layer.ln_2_b;

            model.tensors["model/h" + std::to_string(i) + "/attn/c_attn/w"] = layer.c_attn_attn_w;
            model.tensors["model/h" + std::to_string(i) + "/attn/c_attn/b"] = layer.c_attn_attn_b;

            model.tensors["model/h" + std::to_string(i) + "/attn/c_proj/w"] = layer.c_attn_proj_w;
            model.tensors["model/h" + std::to_string(i) + "/attn/c_proj/b"] = layer.c_attn_proj_b;

            model.tensors["model/h" + std::to_string(i) + "/mlp/c_fc/w"] = layer.c_mlp_fc_w;
            model.tensors["model/h" + std::to_string(i) + "/mlp/c_fc/b"] = layer.c_mlp_fc_b;

            model.tensors["model/h" + std::to_string(i) + "/mlp/c_proj/w"] = layer.c_mlp_proj_w;
            model.tensors["model/h" + std::to_string(i) + "/mlp/c_proj/b"] = layer.c_mlp_proj_b;
        }
    }

    // key + value memory
    {
        const auto &hparams = model.hparams;

        const int n_embd = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx = hparams.n_ctx;

        const int n_mem = n_layer * n_ctx;
        const int n_elements = n_embd * n_mem;

        model.memory_k = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
        model.memory_v = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);

        const size_t memory_size = ggml_nbytes(model.memory_k) + ggml_nbytes(model.memory_v);

        printf("%s: memory size = %8.2f MB, n_mem = %d\n", __func__, memory_size / 1024.0 / 1024.0, n_mem);
    }

    // load weights
    {
        size_t total_size = 0;

        bool has_lm_head = false;

        while (true)
        {
            int32_t n_dims;
            int32_t length;
            int32_t ttype;

            fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            fin.read(reinterpret_cast<char *>(&length), sizeof(length));
            fin.read(reinterpret_cast<char *>(&ttype), sizeof(ttype));

            if (fin.eof())
            {
                break;
            }

            int32_t nelements = 1;
            int32_t ne[2] = {1, 1};
            for (int i = 0; i < n_dims; ++i)
            {
                fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
                nelements *= ne[i];
            }

            std::string name(length, 0);
            fin.read(&name[0], length);

            if (model.tensors.find(name.data()) == model.tensors.end())
            {
                fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.data());
                return false;
            }

            auto tensor = model.tensors[name.data()];
            if (ggml_nelements(tensor) != nelements)
            {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                return false;
            }

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1])
            {
                fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]\n",
                        __func__, name.data(), (int)tensor->ne[0], (int)tensor->ne[1], ne[0], ne[1]);
                return false;
            }

            // for debugging
            if (0)
            {
                printf("%24s - [%5d, %5d], type = %6s, %6.2f MB, %9zu bytes\n", name.data(), ne[0], ne[1], ggml_type_name(ggml_type(ttype)), ggml_nbytes(tensor) / 1024.0 / 1024.0, ggml_nbytes(tensor));
            }

            const size_t bpe = ggml_type_size(ggml_type(ttype));

            if ((nelements * bpe) / ggml_blck_size(tensor->type) != ggml_nbytes(tensor))
            {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                        __func__, name.data(), ggml_nbytes(tensor), nelements * bpe);
                return false;
            }

            fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));

            // GPT-2 models share the WTE tensor as the LM head
            if (name == "model/wte" && has_lm_head == false)
            {
                memcpy(model.lm_head->data, tensor->data, ggml_nbytes(tensor));
            }

            if (name == "model/lm_head")
            {
                has_lm_head = true;
            }

            total_size += ggml_nbytes(tensor);
        }

        printf("%s: model size  = %8.2f MB\n", __func__, total_size / 1024.0 / 1024.0);
    }

    fin.close();

    return true;
}

// evaluate the transformer
//
//   - model:     the model
//   - n_threads: number of threads to use
//   - n_past:    the context size so far
//   - embd_inp:  the embeddings of the tokens in the context
//   - embd_w:    the predicted logits for the next token
//
bool gpt2_eval(
    const gpt2_model &model,
    const int n_threads,
    const int n_past,
    const std::vector<gpt_vocab::id> &embd_inp,
    std::vector<float> &embd_w,
    size_t &mem_per_token)
{
    const int N = embd_inp.size();

    const auto &hparams = model.hparams;

    const int n_embd = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx = hparams.n_ctx;
    const int n_head = hparams.n_head;
    const int n_vocab = hparams.n_vocab;

    static size_t buf_size = 256u * 1024 * 1024;
    static void *buf = malloc(buf_size);

    if (mem_per_token > 0 && mem_per_token * N > buf_size)
    {
        const size_t buf_size_new = 1.1 * (mem_per_token * N); // add 10% to account for ggml object overhead
        // printf("\n%s: reallocating buffer from %zu to %zu bytes\n", __func__, buf_size, buf_size_new);

        // reallocate
        buf_size = buf_size_new;
        buf = realloc(buf, buf_size);
        if (buf == nullptr)
        {
            fprintf(stderr, "%s: failed to allocate %zu bytes\n", __func__, buf_size);
            return false;
        }
    }

    struct ggml_init_params params = {
        .mem_size = buf_size,
        .mem_buffer = buf,
        .no_alloc = false,
    };

    struct ggml_context *ctx0 = ggml_init(params);
    struct ggml_cgraph gf = {};
    gf.n_threads = n_threads;

    struct ggml_tensor *embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    memcpy(embd->data, embd_inp.data(), N * ggml_element_size(embd));

    struct ggml_tensor *position = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    for (int i = 0; i < N; ++i)
    {
        ((int32_t *)position->data)[i] = n_past + i;
    }

    // wte + wpe
    struct ggml_tensor *inpL =
        ggml_add(ctx0,
                 ggml_get_rows(ctx0, model.wte, embd),
                 ggml_get_rows(ctx0, model.wpe, position));

    for (int il = 0; il < n_layer; ++il)
    {
        struct ggml_tensor *cur;

        // norm
        {
            // [ 768, N]
            cur = ggml_norm(ctx0, inpL);

            // cur = ln_1_g*cur + ln_1_b
            // [ 768, N]
            cur = ggml_add(ctx0,
                           ggml_mul(ctx0,
                                    ggml_repeat(ctx0, model.layers[il].ln_1_g, cur),
                                    cur),
                           ggml_repeat(ctx0, model.layers[il].ln_1_b, cur));
        }

        // attn
        // [2304, 768] - model.layers[il].c_attn_attn_w
        // [2304,   1] - model.layers[il].c_attn_attn_b
        // [ 768,   N] - cur (in)
        // [2304,   N] - cur (out)
        //
        // cur = attn_w*cur + attn_b
        // [2304, N]
        {
            cur = ggml_mul_mat(ctx0,
                               model.layers[il].c_attn_attn_w,
                               cur);

            cur = ggml_add(ctx0,
                           ggml_repeat(ctx0, model.layers[il].c_attn_attn_b, cur),
                           cur);
        }

        // self-attention
        {
            struct ggml_tensor *Qcur = ggml_view_2d(ctx0, cur, n_embd, N, cur->nb[1], 0 * sizeof(float) * n_embd);
            struct ggml_tensor *Kcur = ggml_view_2d(ctx0, cur, n_embd, N, cur->nb[1], 1 * sizeof(float) * n_embd);
            struct ggml_tensor *Vcur = ggml_view_2d(ctx0, cur, n_embd, N, cur->nb[1], 2 * sizeof(float) * n_embd);

            // store key and value to memory
            if (N >= 1)
            {
                struct ggml_tensor *k = ggml_view_1d(ctx0, model.memory_k, N * n_embd, (ggml_element_size(model.memory_k) * n_embd) * (il * n_ctx + n_past));
                struct ggml_tensor *v = ggml_view_1d(ctx0, model.memory_v, N * n_embd, (ggml_element_size(model.memory_v) * n_embd) * (il * n_ctx + n_past));

                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Kcur, k));
                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Vcur, v));
            }

            // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
            // [64, N, 12]
            struct ggml_tensor *Q =
                ggml_permute(ctx0,
                             ggml_cpy(ctx0,
                                      Qcur,
                                      ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd / n_head, n_head, N)),
                             0, 2, 1, 3);

            // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
            // [64, n_past + N, 12]
            struct ggml_tensor *K =
                ggml_permute(ctx0,
                             ggml_reshape_3d(ctx0,
                                             ggml_view_1d(ctx0, model.memory_k, (n_past + N) * n_embd, il * n_ctx * ggml_element_size(model.memory_k) * n_embd),
                                             n_embd / n_head, n_head, n_past + N),
                             0, 2, 1, 3);

            // GG: flash attention
            // struct ggml_tensor * V =
            //    ggml_cpy(ctx0,
            //            ggml_permute(ctx0,
            //                ggml_reshape_3d(ctx0,
            //                    ggml_view_1d(ctx0, model.memory_v, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(model.memory_v)*n_embd),
            //                    n_embd/n_head, n_head, n_past + N),
            //                1, 2, 0, 3),
            //            ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_past + N, n_embd/n_head, n_head));

            // struct ggml_tensor * KQV = ggml_flash_attn(ctx0, Q, K, V, true);

            // K * Q
            // [n_past + N, N, 12]
            struct ggml_tensor *KQ = ggml_mul_mat(ctx0, K, Q);

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            // [n_past + N, N, 12]
            struct ggml_tensor *KQ_scaled =
                ggml_scale_inplace(ctx0,
                                   KQ,
                                   ggml_new_f32(ctx0, 1.0f / sqrt(float(n_embd) / n_head)));

            // KQ_masked = mask_past(KQ_scaled)
            // [n_past + N, N, 12]
            struct ggml_tensor *KQ_masked = ggml_diag_mask_inf_inplace(ctx0, KQ_scaled, n_past);

            // KQ = soft_max(KQ_masked)
            // [n_past + N, N, 12]
            struct ggml_tensor *KQ_soft_max = ggml_soft_max_inplace(ctx0, KQ_masked);

            // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
            // [n_past + N, 64, 12]
            struct ggml_tensor *V_trans =
                ggml_cpy(ctx0,
                         ggml_permute(ctx0,
                                      ggml_reshape_3d(ctx0,
                                                      ggml_view_1d(ctx0, model.memory_v, (n_past + N) * n_embd, il * n_ctx * ggml_element_size(model.memory_v) * n_embd),
                                                      n_embd / n_head, n_head, n_past + N),
                                      1, 2, 0, 3),
                         ggml_new_tensor_3d(ctx0, model.memory_v->type, n_past + N, n_embd / n_head, n_head));

            // KQV = transpose(V) * KQ_soft_max
            // [64, N, 12]
            struct ggml_tensor *KQV = ggml_mul_mat(ctx0, V_trans, KQ_soft_max);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            // [64, 12, N]
            struct ggml_tensor *KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

            // cur = KQV_merged.contiguous().view(n_embd, N)
            // [768, N]
            cur = ggml_cpy(ctx0,
                           KQV_merged,
                           ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));
        }

        // projection
        // [ 768, 768] - model.layers[il].c_attn_proj_w
        // [ 768,   1] - model.layers[il].c_attn_proj_b
        // [ 768,   N] - cur (in)
        // [ 768,   N] - cur (out)
        //
        // cur = proj_w*cur + proj_b
        // [768, N]
        {
            cur = ggml_mul_mat(ctx0,
                               model.layers[il].c_attn_proj_w,
                               cur);

            cur = ggml_add(ctx0,
                           ggml_repeat(ctx0, model.layers[il].c_attn_proj_b, cur),
                           cur);
        }

        // add the input
        cur = ggml_add(ctx0, cur, inpL);

        struct ggml_tensor *inpFF = cur;

        // feed-forward network
        {
            // norm
            {
                cur = ggml_norm(ctx0, inpFF);

                // cur = ln_2_g*cur + ln_2_b
                // [ 768, N]
                cur = ggml_add(ctx0,
                               ggml_mul(ctx0,
                                        ggml_repeat(ctx0, model.layers[il].ln_2_g, cur),
                                        cur),
                               ggml_repeat(ctx0, model.layers[il].ln_2_b, cur));
            }

            // fully connected
            // [3072, 768] - model.layers[il].c_mlp_fc_w
            // [3072,   1] - model.layers[il].c_mlp_fc_b
            // [ 768,   N] - cur (in)
            // [3072,   N] - cur (out)
            //
            // cur = fc_w*cur + fc_b
            // [3072, N]
            cur = ggml_mul_mat(ctx0,
                               model.layers[il].c_mlp_fc_w,
                               cur);

            cur = ggml_add(ctx0,
                           ggml_repeat(ctx0, model.layers[il].c_mlp_fc_b, cur),
                           cur);

            // GELU activation
            // [3072, N]
            cur = ggml_gelu(ctx0, cur);

            // projection
            // [ 768, 3072] - model.layers[il].c_mlp_proj_w
            // [ 768,    1] - model.layers[il].c_mlp_proj_b
            // [3072,    N] - cur (in)
            // [ 768,    N] - cur (out)
            //
            // cur = proj_w*cur + proj_b
            // [768, N]
            cur = ggml_mul_mat(ctx0,
                               model.layers[il].c_mlp_proj_w,
                               cur);

            cur = ggml_add(ctx0,
                           ggml_repeat(ctx0, model.layers[il].c_mlp_proj_b, cur),
                           cur);
        }

        // input for next layer
        inpL = ggml_add(ctx0, cur, inpFF);
    }

    // norm
    {
        // [ 768, N]
        inpL = ggml_norm(ctx0, inpL);

        // inpL = ln_f_g*inpL + ln_f_b
        // [ 768, N]
        inpL = ggml_add(ctx0,
                        ggml_mul(ctx0,
                                 ggml_repeat(ctx0, model.ln_f_g, inpL),
                                 inpL),
                        ggml_repeat(ctx0, model.ln_f_b, inpL));
    }

    // inpL = WTE * inpL
    // [ 768, 50257] - model.lm_head
    // [ 768, N]     - inpL
    inpL = ggml_mul_mat(ctx0, model.lm_head, inpL);

    // logits -> probs
    // inpL = ggml_soft_max_inplace(ctx0, inpL);

    // run the computation
    ggml_build_forward_expand(&gf, inpL);
    ggml_graph_compute(ctx0, &gf);

    // if (n_past%100 == 0) {
    //     ggml_graph_print   (&gf);
    //     ggml_graph_dump_dot(&gf, NULL, "gpt-2.dot");
    // }

    // embd_w.resize(n_vocab*N);
    // memcpy(embd_w.data(), ggml_get_data(inpL), sizeof(float)*n_vocab*N);

    // return result just for the last token
    embd_w.resize(n_vocab);
    memcpy(embd_w.data(), (float *)ggml_get_data(inpL) + (n_vocab * (N - 1)), sizeof(float) * n_vocab);

    if (mem_per_token == 0)
    {
        mem_per_token = ggml_used_mem(ctx0) / N;
    }
    // printf("used_mem = %zu\n", ggml_used_mem(ctx0));

    ggml_free(ctx0);

    return true;
}

#define GPT2_MAX_RNG_STATE 64 * 1024

size_t gpt2_get_state_size(const gpt2_model &model)
{
    // we don't know size of rng until we actually serialize it. so reserve more than enough memory for its serialized state.
    // for reference, std::mt19937(1337) serializes to 6701 bytes.
    const size_t s_rng_size = sizeof(size_t);
    const size_t s_rng = GPT2_MAX_RNG_STATE;
    const size_t s_kv_size = sizeof(size_t);
    const size_t s_kv_ntok = sizeof(int);
    const size_t s_total = (+s_rng_size + s_rng + s_kv_size + s_kv_ntok);
    fflush(stdout);
    return s_total;
}

size_t gpt2_copy_state_data(const gpt2_model &model, const std::mt19937 &rng, uint8_t *dest)
{
    uint8_t *out = dest;
    fflush(stdout);
    // copy rng
    {
        std::stringstream rng_ss;
        rng_ss << rng;

        const size_t rng_size = rng_ss.str().size();
        char rng_buf[GPT2_MAX_RNG_STATE];

        memset(&rng_buf[0], 0, GPT2_MAX_RNG_STATE);
        memcpy(&rng_buf[0], rng_ss.str().data(), rng_ss.str().size());

        memcpy(out, &rng_size, sizeof(rng_size));
        out += sizeof(rng_size);
        memcpy(out, &rng_buf[0], GPT2_MAX_RNG_STATE);
        out += GPT2_MAX_RNG_STATE;
    }

    const size_t written = out - dest;
    const size_t expected = gpt2_get_state_size(model);
    assert(written == expected);
    fflush(stdout);
    return written;
}

size_t gpt2_set_state_data(gpt2_model *model, std::mt19937 *rng, const uint8_t *src)
{
    const uint8_t *in = src;

    // set rng
    {
        size_t rng_size;
        char rng_buf[GPT2_MAX_RNG_STATE];

        memcpy(&rng_size, in, sizeof(rng_size));
        in += sizeof(rng_size);
        memcpy(&rng_buf[0], in, GPT2_MAX_RNG_STATE);
        in += GPT2_MAX_RNG_STATE;

        std::stringstream rng_ss;
        rng_ss.str(std::string(&rng_buf[0], rng_size));
        rng_ss >> *rng;

        assert(rng_ss.fail() == false);
    }

    const size_t nread = in - src;
    const size_t expected = gpt2_get_state_size(*model);
    assert(nread == expected);
    fflush(stdout);
    return nread;
}

//
struct GPT2Private
{
    const std::string modelPath;
    bool modelLoaded;
    gpt_vocab vocab;
    gpt2_model *model = nullptr;
    int64_t n_threads = 0;
    size_t mem_per_token = 0;
    std::mt19937 rng;
};

GPT2::GPT2()
    : d_ptr(new GPT2Private)
{
    d_ptr->model = new gpt2_model;
    d_ptr->modelLoaded = false;
}

bool GPT2::loadModel(const std::string &modelPath)
{
    std::mt19937 rng(time(NULL));
    d_ptr->rng = rng;

    auto fin = std::ifstream(modelPath, std::ios::binary);

    // load the model
    if (!gpt2_model_load(modelPath, *d_ptr->model, d_ptr->vocab))
    {
        std::cerr << "GPT-J ERROR: failed to load model from " << modelPath;
        return false;
    }

    d_ptr->n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());
    d_ptr->modelLoaded = true;
    fflush(stdout);
    return true;
}

// set the number of threads in GPT2Private
void GPT2::setThreadCount(int32_t n_threads)
{
    d_ptr->n_threads = n_threads;
}

// get the number of threads to use from d_ptr which is GPT2Private
int32_t GPT2::threadCount()
{
    return d_ptr->n_threads;
}

// delete the model when the object is destroyed
GPT2::~GPT2()
{
    delete d_ptr->model;
}

// just return if the model is loaded or not
bool GPT2::isModelLoaded() const
{
    return d_ptr->modelLoaded;
}

// TODO
size_t GPT2::stateSize() const
{
    return gpt2_get_state_size(*d_ptr->model);
}

size_t GPT2::saveState(uint8_t *dest) const
{
    return gpt2_copy_state_data(*d_ptr->model, d_ptr->rng, dest);
}

size_t GPT2::restoreState(const uint8_t *src)
{
    return gpt2_set_state_data(d_ptr->model, &d_ptr->rng, src);
}

void GPT2::prompt(const std::string &prompt,
                  std::function<bool(int32_t)> promptCallback,
                  std::function<bool(int32_t, const std::string &)> responseCallback,
                  std::function<bool(bool)> recalculateCallback,
                  PromptContext &promptCtx)
{

    if (!isModelLoaded())
    {
        std::cerr << "GPT2 ERROR: prompt won't work with an unloaded model!\n";
        return;
    }

    const int64_t t_main_start_us = ggml_time_us();

    int64_t t_sample_us = 0;
    int64_t t_predict_us = 0;
    int64_t t_prompt_us = 0;

    // tokenize the prompt
    std::vector<gpt_vocab::id> embd_inp = ::gpt_tokenize(d_ptr->vocab, prompt);

    // save the context size
    promptCtx.n_ctx = d_ptr->model->hparams.n_ctx;

    if ((int)embd_inp.size() > promptCtx.n_ctx - 4)
    {
        responseCallback(-1, "ERROR: The prompt size exceeds the context window size and cannot be processed.");
        std::cerr << "GPT2 ERROR: The prompt is" << embd_inp.size() << "tokens and the context window is" << promptCtx.n_ctx << "!\n";
        return;
    }

    promptCtx.n_predict = std::min(promptCtx.n_predict, promptCtx.n_ctx - (int)embd_inp.size());
    promptCtx.n_past = std::min(promptCtx.n_past, promptCtx.n_ctx);

    // determine the required inference memory per token:
    static bool initialized = false;
    static std::vector<gpt_vocab::id> p_instruct;
    static std::vector<gpt_vocab::id> r_instruct;
    if (!initialized)
    {
        gpt2_eval(*d_ptr->model, d_ptr->n_threads, 0, {0, 1, 2, 3}, promptCtx.logits,
                  d_ptr->mem_per_token);
        initialized = true;
    }

    // process the prompt in batches
    size_t i = 0;
    const int64_t t_start_prompt_us = ggml_time_us();
    while (i < embd_inp.size())
    {
        size_t batch_end = std::min(i + promptCtx.n_batch, embd_inp.size());
        std::vector<gpt_vocab::id> batch(embd_inp.begin() + i, embd_inp.begin() + batch_end);

        // Check if the context has run out...
        if (promptCtx.n_past + batch.size() > promptCtx.n_ctx)
        {
            const int32_t erasePoint = promptCtx.n_ctx * promptCtx.contextErase;
            // Erase the first percentage of context from the tokens...
            std::cerr << "GPT2: reached the end of the context window so resizing\n";
            promptCtx.tokens.erase(promptCtx.tokens.begin(), promptCtx.tokens.begin() + erasePoint);
            promptCtx.n_past = promptCtx.tokens.size();
            recalculateContext(promptCtx, recalculateCallback);
            assert(promptCtx.n_past + batch.size() <= promptCtx.n_ctx);
        }

        if (!gpt2_eval(*d_ptr->model, d_ptr->n_threads, promptCtx.n_past, batch, promptCtx.logits,
                       d_ptr->mem_per_token))
        {
            std::cerr << "GPT2 ERROR: Failed to process prompt\n";
            return;
        }

        size_t tokens = batch_end - i;
        for (size_t t = 0; t < tokens; ++t)
        {
            if (promptCtx.tokens.size() == promptCtx.n_ctx)
                promptCtx.tokens.erase(promptCtx.tokens.begin());
            promptCtx.tokens.push_back(batch.at(t));
            if (!promptCallback(batch.at(t)))
                return;
        }
        promptCtx.n_past += batch.size();
        i = batch_end;
    }
    t_prompt_us += ggml_time_us() - t_start_prompt_us;

    int p_instructFound = 0;
    int r_instructFound = 0;

    std::string cachedResponse;
    std::vector<gpt_vocab::id> cachedTokens;
    std::unordered_set<std::string> reversePrompts = {"### Instruction", "### Prompt", "### Response", "### Human", "### Assistant"};

    // predict next tokens
    int32_t totalPredictions = 0;
    for (int i = 0; i < promptCtx.n_predict; i++)
    {

        // sample next token
        const int n_vocab = d_ptr->model->hparams.n_vocab;
        gpt_vocab::id id = 0;
        {
            const int64_t t_start_sample_us = ggml_time_us();
            const size_t n_prev_toks = std::min((size_t)promptCtx.repeat_last_n, promptCtx.tokens.size());
            id = gpt_sample_top_k_top_p(d_ptr->vocab, n_vocab,
                                        promptCtx.tokens.data() + promptCtx.tokens.size() - n_prev_toks,
                                        n_prev_toks,
                                        promptCtx.logits,
                                        promptCtx.top_k, promptCtx.top_p, promptCtx.temp,
                                        promptCtx.repeat_penalty,
                                        d_ptr->rng);

            t_sample_us += ggml_time_us() - t_start_sample_us;
        }

        // Check if the context has run out...
        if (promptCtx.n_past + 1 > promptCtx.n_ctx)
        {
            const int32_t erasePoint = promptCtx.n_ctx * promptCtx.contextErase;
            // Erase the first percentage of context from the tokens...
            std::cerr << "GPT2: reached the end of the context window so resizing\n";
            promptCtx.tokens.erase(promptCtx.tokens.begin(), promptCtx.tokens.begin() + erasePoint);
            promptCtx.n_past = promptCtx.tokens.size();
            recalculateContext(promptCtx, recalculateCallback);
            assert(promptCtx.n_past + 1 <= promptCtx.n_ctx);
        }

        const int64_t t_start_predict_us = ggml_time_us();
        if (!gpt2_eval(*d_ptr->model, d_ptr->n_threads, promptCtx.n_past, {id}, promptCtx.logits,
                       d_ptr->mem_per_token))
        {
            std::cerr << "GPT2 ERROR: Failed to predict next token\n";
            return;
        }
        t_predict_us += ggml_time_us() - t_start_predict_us;

        promptCtx.n_past += 1;
        // display text
        ++totalPredictions;

        if (id == 50256 /*end of text*/)
            goto stop_generating;

        const std::string str = d_ptr->vocab.id_to_token[id];

        // Check if the provided str is part of our reverse prompts
        bool foundPartialReversePrompt = false;
        const std::string completed = cachedResponse + str;
        if (reversePrompts.find(completed) != reversePrompts.end())
        {
            goto stop_generating;
        }

        // Check if it partially matches our reverse prompts and if so, cache
        for (auto s : reversePrompts)
        {
            if (s.compare(0, completed.size(), completed) == 0)
            {
                foundPartialReversePrompt = true;
                cachedResponse = completed;
                break;
            }
        }

        // Regardless the token gets added to our cache
        cachedTokens.push_back(id);

        // Continue if we have found a partial match
        if (foundPartialReversePrompt)
            continue;

        // Empty the cache
        for (auto t : cachedTokens)
        {
            if (promptCtx.tokens.size() == promptCtx.n_ctx)
                promptCtx.tokens.erase(promptCtx.tokens.begin());
            promptCtx.tokens.push_back(t);
            if (!responseCallback(t, d_ptr->vocab.id_to_token[t]))
                goto stop_generating;
        }
        cachedTokens.clear();
    }

stop_generating:

#if 0
    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        std::cout << "GPT2 INFO: mem per token = " << mem_per_token << " bytes\n";
        std::cout << "GPT2 INFO:   sample time = " << t_sample_us/1000.0f << " ms\n";
        std::cout << "GPT2 INFO:   prompt time = " << t_prompt_us/1000.0f << " ms\n";
        std::cout << "GPT2 INFO:  predict time = " << t_predict_us/1000.0f << " ms / " << t_predict_us/1000.0f/totalPredictions << " ms per token\n";
        std::cout << "GPT2 INFO:    total time = " << (t_main_end_us - t_main_start_us)/1000.0f << " ms\n";
        fflush(stdout);
    }
#endif

    return;
}

void GPT2::recalculateContext(PromptContext &promptCtx, std::function<bool(bool)> recalculate)
{
    size_t i = 0;
    promptCtx.n_past = 0;
    while (i < promptCtx.tokens.size())
    {
        size_t batch_end = std::min(i + promptCtx.n_batch, promptCtx.tokens.size());
        std::vector<gpt_vocab::id> batch(promptCtx.tokens.begin() + i, promptCtx.tokens.begin() + batch_end);

        assert(promptCtx.n_past + batch.size() <= promptCtx.n_ctx);

        if (!gpt2_eval(*d_ptr->model, d_ptr->n_threads, promptCtx.n_past, batch, promptCtx.logits,
                       d_ptr->mem_per_token))
        {
            std::cerr << "GPT2 ERROR: Failed to process prompt\n";
            goto stop_generating;
        }
        promptCtx.n_past += batch.size();
        if (!recalculate(true))
            goto stop_generating;
        i = batch_end;
    }
    assert(promptCtx.n_past == promptCtx.tokens.size());

stop_generating:
    recalculate(false);
}
