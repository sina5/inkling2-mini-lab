// inkling-probe: run one prompt (optionally with an image) through Inkling 2 Mini and dump
// expert routing, hidden states, and loop gate activity to a binary file.
//
// Usage: INKLING_PROBE_OUT=out.bin llama-inkling-probe -m model.gguf --mmproj mmproj.gguf [--image img.png] -p "<raw prompt>"
// The raw prompt is not passed through a chat template; put <__media__> where the image goes.
//
// Record format (little endian), repeated until end of file:
//   u32 name_len, name bytes, u32 kind (0 = f32, 1 = i32), i64 ne0, i64 ne1, data[ne0*ne1]
// Gate tensors are reduced to two rows per token: mean(x) and mean(|x|) over channels.
// The final record "tokens" (i32, ne0 = n_tokens, ne1 = 1) holds the token ids, -1 for image positions.

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct probe_writer {
    FILE * f = nullptr;
    std::vector<uint8_t> buf;

    void write_record(const std::string & name, uint32_t kind, int64_t ne0, int64_t ne1, const void * data) {
        const uint32_t n = (uint32_t) name.size();
        fwrite(&n, 4, 1, f);
        fwrite(name.data(), 1, n, f);
        fwrite(&kind, 4, 1, f);
        fwrite(&ne0, 8, 1, f);
        fwrite(&ne1, 8, 1, f);
        fwrite(data, 4, (size_t) (ne0 * ne1), f);
    }
};

static bool starts_with(const char * s, const char * prefix) {
    return strncmp(s, prefix, strlen(prefix)) == 0;
}

static bool wanted(const char * name) {
    return starts_with(name, "ffn_moe_topk-") || starts_with(name, "l_out-") ||
           starts_with(name, "loop_gain_") || starts_with(name, "loop_forget_");
}

static float read_f32(const uint8_t * base, const ggml_tensor * t, int64_t i0, int64_t i1) {
    return *(const float *) (base + i0 * t->nb[0] + i1 * t->nb[1]);
}

static bool probe_cb(ggml_tensor * t, bool ask, void * user_data) {
    if (ask) {
        return wanted(t->name);
    }
    if (!wanted(t->name) || t->ne[2] != 1 || t->ne[3] != 1) {
        return true;
    }
    auto * w = (probe_writer *) user_data;
    w->buf.resize(ggml_nbytes(t));
    ggml_backend_tensor_get(t, w->buf.data(), 0, ggml_nbytes(t));
    const uint8_t * base = w->buf.data();
    const int64_t ne0 = t->ne[0];
    const int64_t ne1 = t->ne[1];

    if (t->type == GGML_TYPE_I32) {
        std::vector<int32_t> out(ne0 * ne1);
        for (int64_t i1 = 0; i1 < ne1; ++i1) {
            for (int64_t i0 = 0; i0 < ne0; ++i0) {
                out[i1 * ne0 + i0] = *(const int32_t *) (base + i0 * t->nb[0] + i1 * t->nb[1]);
            }
        }
        w->write_record(t->name, 1, ne0, ne1, out.data());
        return true;
    }
    if (t->type != GGML_TYPE_F32) {
        return true;
    }
    if (starts_with(t->name, "loop_")) {
        std::vector<float> out(2 * ne1);
        for (int64_t i1 = 0; i1 < ne1; ++i1) {
            double sum = 0.0, sum_abs = 0.0;
            for (int64_t i0 = 0; i0 < ne0; ++i0) {
                const float v = read_f32(base, t, i0, i1);
                sum += v;
                sum_abs += std::fabs(v);
            }
            out[i1] = (float) (sum / ne0);
            out[ne1 + i1] = (float) (sum_abs / ne0);
        }
        w->write_record(t->name, 0, ne1, 2, out.data());
        return true;
    }
    std::vector<float> out(ne0 * ne1);
    for (int64_t i1 = 0; i1 < ne1; ++i1) {
        for (int64_t i0 = 0; i0 < ne0; ++i0) {
            out[i1 * ne0 + i0] = read_f32(base, t, i0, i1);
        }
    }
    w->write_record(t->name, 0, ne0, ne1, out.data());
    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MTMD)) {
        return 1;
    }
    const char * out_path = std::getenv("INKLING_PROBE_OUT");
    if (out_path == nullptr || params.prompt.empty()) {
        LOG_ERR("need -p <raw prompt> and the INKLING_PROBE_OUT environment variable\n");
        return 1;
    }

    probe_writer writer;
    writer.f = fopen(out_path, "wb");
    if (!writer.f) {
        LOG_ERR("cannot open %s\n", out_path);
        return 1;
    }

    llama_backend_init();
    params.cb_eval = probe_cb;
    params.cb_eval_user_data = &writer;
    params.warmup = false;

    auto llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    llama_context * lctx = llama_init->context();
    if (!model || !lctx) {
        return 1;
    }

    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu = params.mmproj_use_gpu;
    mparams.n_threads = params.cpuparams.n_threads;
    mtmd::context_ptr vision(mtmd_init_from_file(params.mmproj.path.c_str(), model, mparams));
    if (!vision) {
        LOG_ERR("cannot load mmproj\n");
        return 1;
    }

    mtmd::bitmaps bitmaps;
    for (const auto & path : params.image) {
        auto res = mtmd_helper_bitmap_init_from_file(vision.get(), path.c_str(), false, mtmd_helper_init_opt_default());
        if (!res.bitmap) {
            LOG_ERR("cannot load image %s\n", path.c_str());
            return 1;
        }
        bitmaps.entries.emplace_back(res.bitmap);
    }

    mtmd_input_text text = {params.prompt.c_str(), params.prompt.size(), /* add_special */ false, /* parse_special */ true};
    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bitmap_ptrs = bitmaps.c_ptr();
    if (mtmd_tokenize(vision.get(), chunks.ptr.get(), &text, bitmap_ptrs.data(), bitmap_ptrs.size()) != 0) {
        LOG_ERR("tokenize failed\n");
        return 1;
    }

    std::vector<int32_t> token_ids;
    for (size_t i = 0; i < chunks.size(); ++i) {
        const mtmd_input_chunk * chunk = chunks[i];
        if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n = 0;
            const llama_token * toks = mtmd_input_chunk_get_tokens_text(chunk, &n);
            token_ids.insert(token_ids.end(), toks, toks + n);
        } else {
            token_ids.insert(token_ids.end(), mtmd_input_chunk_get_n_tokens(chunk), -1);
        }
    }

    llama_pos n_past = 0;
    if (mtmd_helper_eval_chunks(vision.get(), lctx, chunks.ptr.get(), 0, 0, params.n_batch, true, &n_past) != 0) {
        LOG_ERR("eval failed\n");
        return 1;
    }

    writer.write_record("tokens", 1, (int64_t) token_ids.size(), 1, token_ids.data());
    fclose(writer.f);
    LOG_INF("wrote %zu tokens of probe data to %s\n", token_ids.size(), out_path);
    return 0;
}
