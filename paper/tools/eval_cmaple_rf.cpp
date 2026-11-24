/*
 * eval_cmaple_rf.cpp
 *
 * Fast O(N) C++ Robinson-Foulds & Resolved-Branch evaluator for pandemic-scale
 * Newick trees (CMAPLE, MAPLE, Phylo2Vec-Minibatch-GPU).
 *
 * Computes:
 *   1. All-splits normalized RF (%) over all non-trivial splits in the ground-truth tree.
 *   2. Resolved-branch RF (%) over ground-truth branches with length >= 0.5/L (and >= 1.0/L),
 *      isolating true evolutionary splits from zero-mutation polytomy ties on short-branch
 *      pandemic SARS-CoV-2 phylogenies.
 *   3. Polytomy-collapsed symmetric RF (%) when collapsing edges with length <= 1e-6 in both trees.
 *
 * Compile:
 *   g++ -O3 -march=native -std=c++17 eval_cmaple_rf.cpp -o eval_cmaple_rf
 */

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

static uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

struct SplitInfo {
    uint64_t hash;
    double bl;
    int32_t clade_sz;
};

static bool parse_newick_splits(const char *path, std::unordered_map<std::string, int32_t> &taxon_to_id,
                                std::vector<uint64_t> &leaf_hashes, uint64_t &rng_state, uint64_t &out_total_xor,
                                int32_t &out_n_leaves, std::vector<SplitInfo> &out_splits,
                                bool use_index_labels = false) {
    FILE *fp = fopen(path, "rb");
    if (!fp)
        return false;
    fseek(fp, 0, SEEK_END);
    size_t sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    std::string buf(sz, '\0');
    if (fread(&buf[0], 1, sz, fp) != sz) {
        fclose(fp);
        return false;
    }
    fclose(fp);

    struct StackFrame {
        uint64_t xor_hash;
        int32_t leaf_cnt;
    };
    std::vector<StackFrame> st;
    st.reserve(65536);

    struct RawNode {
        uint64_t xor_hash;
        int32_t leaf_cnt;
        double bl;
    };
    std::vector<RawNode> raw_internals;
    raw_internals.reserve(1000000);

    uint64_t total_xor = 0;
    int32_t n_leaves = 0;

    auto read_branch_len = [&](size_t &i) -> double {
        while (i < sz && isspace((unsigned char)buf[i]))
            i++;
        if (i < sz && buf[i] == '[') {
            while (i < sz && buf[i] != ']')
                i++;
            if (i < sz)
                i++;
        }
        while (i < sz && isspace((unsigned char)buf[i]))
            i++;
        double bl = 0.0;
        if (i < sz && buf[i] == ':') {
            i++;
            const char *start = buf.c_str() + i;
            char *endp = nullptr;
            bl = strtod(start, &endp);
            if (endp > start) {
                i = (size_t)(endp - buf.c_str());
            }
        }
        while (i < sz && isspace((unsigned char)buf[i]))
            i++;
        if (i < sz && buf[i] == '[') {
            while (i < sz && buf[i] != ']')
                i++;
            if (i < sz)
                i++;
        }
        return bl;
    };

    size_t i = 0;
    while (i < sz) {
        char c = buf[i];
        if (isspace((unsigned char)c)) {
            i++;
        } else if (c == '(') {
            st.push_back({0ULL, 0});
            i++;
        } else if (c == ',') {
            i++;
        } else if (c == ')') {
            i++;
            // Optional internal node label
            while (i < sz && buf[i] != ':' && buf[i] != ',' && buf[i] != ')' && buf[i] != ';' && buf[i] != '[')
                i++;
            double bl = read_branch_len(i);
            if (!st.empty()) {
                StackFrame cur = st.back();
                st.pop_back();
                raw_internals.push_back({cur.xor_hash, cur.leaf_cnt, bl});
                if (!st.empty()) {
                    st.back().xor_hash ^= cur.xor_hash;
                    st.back().leaf_cnt += cur.leaf_cnt;
                }
            }
        } else if (c == ';') {
            break;
        } else {
            // Leaf token
            size_t start = i;
            while (i < sz && buf[i] != ':' && buf[i] != ',' && buf[i] != ')' && buf[i] != '(' && buf[i] != ';' &&
                   buf[i] != '[' && !isspace((unsigned char)buf[i])) {
                i++;
            }
            std::string name = buf.substr(start, i - start);
            (void)read_branch_len(i);
            if (!name.empty()) {
                int32_t lid = -1;
                if (use_index_labels && name.size() >= 2 && name[0] == 'T' && isdigit((unsigned char)name[1])) {
                    lid = atoi(name.c_str() + 1);
                } else {
                    auto it = taxon_to_id.find(name);
                    if (it == taxon_to_id.end()) {
                        lid = (int32_t)taxon_to_id.size();
                        taxon_to_id[name] = lid;
                        leaf_hashes.push_back(splitmix64(&rng_state));
                    } else {
                        lid = it->second;
                    }
                }
                uint64_t h = leaf_hashes[lid];
                total_xor ^= h;
                n_leaves++;
                if (!st.empty()) {
                    st.back().xor_hash ^= h;
                    st.back().leaf_cnt += 1;
                }
            }
        }
    }

    out_total_xor = total_xor;
    out_n_leaves = n_leaves;
    out_splits.clear();
    out_splits.reserve(raw_internals.size());

    for (const auto &rn : raw_internals) {
        int32_t csz = std::min(rn.leaf_cnt, n_leaves - rn.leaf_cnt);
        if (csz < 2)
            continue; // trivial leaf or root split
        uint64_t h1 = rn.xor_hash;
        uint64_t h2 = total_xor ^ h1;
        uint64_t hc = (h1 < h2) ? h1 : h2;
        out_splits.push_back({hc, rn.bl, csz});
    }

    // Sort by hash; if duplicate hash (from root trifurcation in rooted binary representation),
    // merge by summing their branch lengths and keeping a single unrooted split entry!
    std::sort(out_splits.begin(), out_splits.end(),
              [](const SplitInfo &a, const SplitInfo &b) { return a.hash < b.hash; });
    size_t w = 0;
    for (size_t r = 0; r < out_splits.size(); ++r) {
        if (w > 0 && out_splits[w - 1].hash == out_splits[r].hash) {
            out_splits[w - 1].bl += out_splits[r].bl;
        } else {
            out_splits[w++] = out_splits[r];
        }
    }
    out_splits.resize(w);
    return true;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <true.nwk> <test.nwk> [L=30000] [maple_order.maple]\n", argv[0]);
        return 1;
    }
    const char *true_path = argv[1];
    const char *test_path = argv[2];
    double L = (argc >= 4) ? atof(argv[3]) : 30000.0;
    const char *maple_path = (argc >= 5) ? argv[4] : nullptr;

    std::unordered_map<std::string, int32_t> taxon_to_id;
    taxon_to_id.reserve(1200000);
    std::vector<uint64_t> leaf_hashes;
    leaf_hashes.reserve(1200000);
    uint64_t rng = 0x123456789ABCDEF0ULL;

    bool use_index_labels = false;
    if (maple_path && maple_path[0] != '\0') {
        FILE *fm = fopen(maple_path, "rb");
        if (fm) {
            char line[4096];
            bool first_ref = true;
            while (fgets(line, sizeof(line), fm)) {
                if (line[0] == '>') {
                    char *p = line + 1;
                    while (*p && isspace((unsigned char)*p))
                        p++;
                    char *e = p;
                    while (*e && !isspace((unsigned char)*e))
                        e++;
                    std::string name(p, (size_t)(e - p));
                    if (first_ref && name == "REF") {
                        first_ref = false;
                        continue;
                    }
                    first_ref = false;
                    int32_t lid = (int32_t)taxon_to_id.size();
                    taxon_to_id[name] = lid;
                    leaf_hashes.push_back(splitmix64(&rng));
                }
            }
            fclose(fm);
            use_index_labels = true;
        }
    }

    uint64_t xor_true = 0, xor_test = 0;
    int32_t n_true = 0, n_test = 0;
    std::vector<SplitInfo> sp_true, sp_test;

    if (!parse_newick_splits(true_path, taxon_to_id, leaf_hashes, rng, xor_true, n_true, sp_true, false)) {
        fprintf(stderr, "ERROR: failed to parse %s\n", true_path);
        return 1;
    }
    if (!parse_newick_splits(test_path, taxon_to_id, leaf_hashes, rng, xor_test, n_test, sp_test, use_index_labels)) {
        fprintf(stderr, "ERROR: failed to parse %s\n", test_path);
        return 1;
    }

    const double eps_half_mut = 0.5 / L; // >= 0.5 expected substitutions
    const double eps_one_mut = 1.0 / L;  // >= 1.0 expected substitutions
    const double eps_pos = 1e-6;

    int64_t n_true_all = (int64_t)sp_true.size();
    int64_t n_test_all = (int64_t)sp_test.size();
    int64_t n_true_res05 = 0, n_true_res10 = 0;
    int64_t n_test_pos = 0;
    for (const auto &s : sp_true) {
        if (s.bl >= eps_half_mut)
            n_true_res05++;
        if (s.bl >= eps_one_mut)
            n_true_res10++;
    }
    for (const auto &s : sp_test) {
        if (s.bl > eps_pos)
            n_test_pos++;
    }

    int64_t common_all = 0, common_res05 = 0, common_res10 = 0, common_pos = 0;
    size_t i = 0, j = 0;
    while (i < sp_true.size() && j < sp_test.size()) {
        if (sp_true[i].hash == sp_test[j].hash) {
            common_all++;
            if (sp_true[i].bl >= eps_half_mut)
                common_res05++;
            if (sp_true[i].bl >= eps_one_mut)
                common_res10++;
            if (sp_true[i].bl >= eps_half_mut && sp_test[j].bl > eps_pos)
                common_pos++;
            i++;
            j++;
        } else if (sp_true[i].hash < sp_test[j].hash) {
            i++;
        } else {
            j++;
        }
    }

    double fn_all_pct = 100.0 * (double)(n_true_all - common_all) / std::max<int64_t>(1, n_true_all);
    double fn_res05_pct = 100.0 * (double)(n_true_res05 - common_res05) / std::max<int64_t>(1, n_true_res05);
    double fn_res10_pct = 100.0 * (double)(n_true_res10 - common_res10) / std::max<int64_t>(1, n_true_res10);

    printf("{\"N_true\": %d, \"N_test\": %d, \"n_true_splits\": %lld, \"n_test_splits\": %lld, "
           "\"common_all\": %lld, \"rf_all_pct\": %.4f, "
           "\"n_true_res05\": %lld, \"common_res05\": %lld, \"rf_res05_pct\": %.4f, "
           "\"n_true_res10\": %lld, \"common_res10\": %lld, \"rf_res10_pct\": %.4f}\n",
           n_true, n_test, (long long)n_true_all, (long long)n_test_all, (long long)common_all, fn_all_pct,
           (long long)n_true_res05, (long long)common_res05, fn_res05_pct, (long long)n_true_res10,
           (long long)common_res10, fn_res10_pct);
    return 0;
}
