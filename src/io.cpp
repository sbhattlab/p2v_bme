/*
 * io.cpp
 *
 * FASTA input (packed as in the .bin bundles), Newick start-tree input and unrooted Newick output.
 * Tree traversals are iterative so that trees with millions of leaves do not overflow the stack.
 */
#include <algorithm>
#include <charconv>
#include <ctype.h>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "io.hpp"

namespace p2vbme {

namespace {

/* Packs one aligned sequence; mirrors bme/bundles.py pack_sequences_bits. */
void pack_sequence(const std::string &seq, int32_t n_words, uint64_t *out) {
    for (size_t i = 0; i < seq.size(); ++i) {
        const int c = toupper((unsigned char)seq[i]);
        int plane;
        if (c == 'A')
            plane = 0;
        else if (c == 'C')
            plane = 1;
        else if (c == 'G')
            plane = 2;
        else if (c == 'T' || c == 'U')
            plane = 3;
        else
            continue;
        const uint64_t bit = 1ULL << (i % 64);
        out[(size_t)plane * n_words + i / 64] |= bit;
        out[(size_t)4 * n_words + i / 64] |= bit;
    }
}

bool needs_quotes(const std::string &name) { return name.find_first_of(" \t()[]':;,") != std::string::npos; }

void write_name(FILE *f, const std::string &name) {
    if (!needs_quotes(name)) {
        fputs(name.c_str(), f);
        return;
    }
    fputc('\'', f);
    for (char ch : name) {
        if (ch == '\'')
            fputc('\'', f);
        fputc(ch, f);
    }
    fputc('\'', f);
}

/* The internal nodes of a rooted tree (root 2n-2) in post-order, found from the root: the search reuses internal
 * node ids, so ids alone do not give children before parents. */
std::vector<int32_t> internal_postorder(const Tree &tree, int32_t n) {
    std::vector<int32_t> order, stack = {2 * n - 2};
    order.reserve(n - 1);
    while (!stack.empty()) { /* parents before children, then reversed */
        const int32_t u = stack.back();
        stack.pop_back();
        order.push_back(u);
        if (tree.c1[u] >= n)
            stack.push_back(tree.c1[u]);
        if (tree.c2[u] >= n)
            stack.push_back(tree.c2[u]);
    }
    if ((int32_t)order.size() != n - 1)
        throw std::invalid_argument("the tree does not have " + std::to_string(n - 1) +
                                    " internal nodes below its root");
    std::reverse(order.begin(), order.end());
    return order;
}

} // namespace

Alignment read_fasta(const std::string &path) {
    FILE *f = fopen(path.c_str(), "r");
    if (!f)
        throw std::runtime_error("cannot open " + path);
    Alignment aln;
    std::unordered_map<std::string, int32_t> seen;
    std::string name, seq;
    bool have_record = false;
    auto finish_record = [&]() {
        if (!have_record)
            return;
        if (aln.n_taxa == 0) {
            if (seq.empty())
                throw std::runtime_error(path + ": first sequence '" + name + "' is empty");
            aln.n_sites = (int32_t)seq.size();
            aln.n_words = (aln.n_sites + 63) / 64;
        } else if ((int32_t)seq.size() != aln.n_sites) {
            throw std::runtime_error(path + ": sequence '" + name + "' has " + std::to_string(seq.size()) +
                                     " sites, the first has " + std::to_string(aln.n_sites) +
                                     " (the alignment must be aligned)");
        }
        if (!seen.emplace(name, aln.n_taxa).second)
            throw std::runtime_error(path + ": duplicate name '" + name + "'");
        const size_t row = aln.packed.size();
        aln.packed.resize(row + (size_t)5 * aln.n_words, 0);
        pack_sequence(seq, aln.n_words, aln.packed.data() + row);
        aln.names.push_back(name);
        ++aln.n_taxa;
        seq.clear();
    };

    char *line = NULL;
    size_t cap = 0;
    ssize_t len;
    long line_no = 0;
    try {
        while ((len = getline(&line, &cap, f)) != -1) {
            ++line_no;
            while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
                line[--len] = '\0';
            if (len > 0 && line[0] == '>') {
                finish_record();
                const char *p = line + 1;
                while (*p == ' ' || *p == '\t')
                    ++p;
                const char *q = p;
                while (*q && *q != ' ' && *q != '\t')
                    ++q;
                name.assign(p, q - p);
                if (name.empty())
                    throw std::runtime_error(path + ":" + std::to_string(line_no) + ": empty sequence name");
                have_record = true;
                continue;
            }
            for (ssize_t i = 0; i < len; ++i) {
                const char ch = line[i];
                if (ch == ' ' || ch == '\t')
                    continue;
                if (!have_record)
                    throw std::runtime_error(path + ":" + std::to_string(line_no) +
                                             ": sequence data before the first '>' header");
                seq.push_back(ch);
            }
        }
    } catch (...) { /* release the line buffer and the file before passing the error on */
        free(line);
        fclose(f);
        throw;
    }
    free(line);
    fclose(f);
    finish_record();
    if (aln.n_taxa == 0)
        throw std::runtime_error(path + ": no sequences");
    return aln;
}

/* ---- .maple ------------------------------------------------------------------------------------------------- */

Alignment read_maple(const std::string &path) {
    FILE *f = fopen(path.c_str(), "r");
    if (!f)
        throw std::runtime_error("cannot open " + path);
    std::vector<std::string> lines;
    {
        char *line = NULL;
        size_t cap = 0;
        ssize_t len;
        while ((len = getline(&line, &cap, f)) != -1) {
            while (len > 0 &&
                   (line[len - 1] == '\n' || line[len - 1] == '\r' || line[len - 1] == ' ' || line[len - 1] == '\t'))
                --len;
            lines.emplace_back(line, (size_t)len);
        }
        free(line);
        fclose(f);
    }
    auto header_name = [&](size_t i) {
        const std::string &h = lines[i];
        size_t p = 1;
        while (p < h.size() && (h[p] == ' ' || h[p] == '\t'))
            ++p;
        size_t q = p;
        while (q < h.size() && h[q] != ' ' && h[q] != '\t')
            ++q;
        if (q == p)
            throw std::runtime_error(path + ":" + std::to_string(i + 1) + ": empty sequence name");
        return h.substr(p, q - p);
    };
    size_t i = 0;
    while (i < lines.size() && lines[i].empty())
        ++i;
    if (i == lines.size() || lines[i][0] != '>')
        throw std::runtime_error(path + ": a .maple file starts with the reference sequence ('>' header)");
    /* the reference: every line up to the next header */
    std::string ref;
    for (++i; i < lines.size() && (lines[i].empty() || lines[i][0] != '>'); ++i)
        for (char ch : lines[i])
            if (ch != ' ' && ch != '\t')
                ref.push_back(ch);
    if (ref.empty())
        throw std::runtime_error(path + ": the reference sequence is empty");

    Alignment aln;
    aln.n_sites = (int32_t)ref.size();
    aln.n_words = (aln.n_sites + 63) / 64;
    const int32_t W = aln.n_words, L = aln.n_sites;
    std::vector<uint64_t> ref_row((size_t)5 * W, 0);
    pack_sequence(ref, W, ref_row.data());
    std::unordered_map<std::string, int32_t> seen;
    while (i < lines.size()) {
        const std::string name = header_name(i);
        if (!seen.emplace(name, aln.n_taxa).second)
            throw std::runtime_error(path + ": duplicate name '" + name + "'");
        const size_t row = aln.packed.size();
        aln.packed.insert(aln.packed.end(), ref_row.begin(), ref_row.end());
        uint64_t *r = aln.packed.data() + row;
        /* differences: "<char> <position> [<length>]", positions from 1; A, C, G, T (or U) set the base, anything
         * else (n, -, ambiguity codes) marks the sites missing (as pipeline/rnasim/pack_cmaple.cpp) */
        for (++i; i < lines.size() && (lines[i].empty() || lines[i][0] != '>'); ++i) {
            const std::string &ln = lines[i];
            if (ln.empty())
                continue;
            const std::string where = path + ":" + std::to_string(i + 1) + ": ";
            const char *p = ln.c_str() + 1;
            char *endp = NULL;
            const long pos = strtol(p, &endp, 10);
            if (endp == p || (ln.size() > 1 && ln[1] != ' ' && ln[1] != '\t'))
                throw std::runtime_error(where + "expected '<character> <position> [<length>]', got '" + ln + "'");
            long span = 1;
            const char *q = endp;
            while (*q == ' ' || *q == '\t')
                ++q;
            if (*q) {
                span = strtol(q, &endp, 10);
                if (endp == q || *endp)
                    throw std::runtime_error(where + "expected '<character> <position> [<length>]', got '" + ln + "'");
            }
            if (pos < 1 || pos > L || span < 1 || span > L - pos + 1) /* (no overflow for huge lengths) */
                throw std::runtime_error(where + "position " + std::to_string(pos) + ", length " +
                                         std::to_string(span) + ": outside the reference (sites 1.." +
                                         std::to_string(L) + ")");
            const int c = toupper((unsigned char)ln[0]);
            const int plane = c == 'A' ? 0 : c == 'C' ? 1 : c == 'G' ? 2 : (c == 'T' || c == 'U') ? 3 : -1;
            for (long site = pos - 1; site < pos - 1 + span; ++site) {
                const uint64_t bit = 1ULL << (site % 64);
                const size_t w = (size_t)(site / 64);
                for (int b = 0; b < 5; ++b)
                    r[(size_t)b * W + w] &= ~bit;
                if (plane >= 0) {
                    r[(size_t)plane * W + w] |= bit;
                    r[(size_t)4 * W + w] |= bit;
                }
            }
        }
        aln.names.push_back(name);
        ++aln.n_taxa;
    }
    if (aln.n_taxa == 0)
        throw std::runtime_error(path + ": no sequences after the reference");
    return aln;
}

Alignment read_alignment(const std::string &path) {
    std::string ext = path.size() >= 6 ? path.substr(path.size() - 6) : "";
    for (char &c : ext)
        c = (char)tolower((unsigned char)c);
    return ext == ".maple" ? read_maple(path) : read_fasta(path);
}

/* ---- identical sequences ---------------------------------------------------------------------------------- */

void deduplicate(Alignment &aln, Deduplication &dd) {
    const size_t row_words = (size_t)5 * aln.n_words;
    dd.kept.clear();
    dd.copy_of.assign(aln.n_taxa, -1);
    /* rows grouped by a hash of their words, then compared word by word */
    std::unordered_map<uint64_t, std::vector<int32_t>> by_hash; /* hash -> kept positions of the first copies */
    std::vector<int32_t> group_size;                            /* per kept position: copies of its row seen */
    for (int32_t x = 0; x < aln.n_taxa; ++x) {
        const uint64_t *row = &aln.packed[(size_t)x * row_words];
        uint64_t h = 1469598103934665603ULL;
        for (size_t w = 0; w < row_words; ++w)
            h = (h ^ row[w]) * 1099511628211ULL;
        int32_t first = -1;
        for (int32_t k : by_hash[h])
            if (memcmp(row, &aln.packed[(size_t)dd.kept[k] * row_words], row_words * sizeof(uint64_t)) == 0) {
                first = k;
                break;
            }
        if (first >= 0 && group_size[first] >= 2) { /* a third or later copy: removed */
            dd.copy_of[x] = first;
            ++group_size[first];
            continue;
        }
        if (first >= 0)
            ++group_size[first];
        dd.kept.push_back(x);
        group_size.push_back(first >= 0 ? 0 : 1); /* only the first copy of a group counts the copies */
        if (first < 0)
            by_hash[h].push_back((int32_t)dd.kept.size() - 1);
    }
    /* compact the kept rows in place (no second copy of a large alignment) */
    if (dd.kept.size() == (size_t)aln.n_taxa)
        return;
    for (size_t k = 0; k < dd.kept.size(); ++k) {
        const int32_t x = dd.kept[k];
        if ((size_t)x != k) {
            std::copy(aln.packed.begin() + (size_t)x * row_words, aln.packed.begin() + (size_t)(x + 1) * row_words,
                      aln.packed.begin() + k * row_words);
            aln.names[k] = aln.names[x];
        }
    }
    aln.n_taxa = (int32_t)dd.kept.size();
    aln.packed.resize((size_t)aln.n_taxa * row_words);
    aln.names.resize(aln.n_taxa);
}

Tree reattach(const Tree &tree, const Deduplication &dd) {
    const int32_t M = (int32_t)dd.kept.size(), N = (int32_t)dd.copy_of.size();
    if ((int32_t)tree.c1.size() != 2 * M - 1)
        throw std::invalid_argument("reattach: the tree is not on the kept sequences");
    const bool with_bl = !tree.branch_lengths.empty();
    if (with_bl && (int32_t)tree.branch_lengths.size() < 2 * M - 2)
        throw std::invalid_argument("reattach: fewer branch lengths than edges");
    std::vector<std::vector<int32_t>> extra(M); /* removed copies, by the kept position of their first copy */
    for (int32_t x = 0; x < N; ++x)
        if (dd.copy_of[x] >= 0)
            extra[dd.copy_of[x]].push_back(x);
    Tree full;
    full.double_precision_lengths = tree.double_precision_lengths;
    full.c1.assign(2 * N - 1, -1);
    full.c2.assign(2 * N - 1, -1);
    if (with_bl) /* the 2N - 2 edges above nodes 0..2N-3, as compute_bme_branch_lengths */
        full.branch_lengths.assign(2 * N - 2, 0.0);
    std::vector<int32_t> top(2 * M - 1, -1); /* node of the full tree standing for each node of the reduced tree */
    int32_t next = N;
    for (int32_t k = 0; k < M; ++k) {
        /* the kept sequence, then each removed copy joined to it by zero-length branches */
        int32_t node = dd.kept[k];
        for (int32_t x : extra[k]) {
            full.c1[next] = node;
            full.c2[next] = x;
            node = next++;
        }
        top[k] = node;
        if (with_bl)
            full.branch_lengths[node] = tree.branch_lengths[k];
    }
    for (int32_t u : internal_postorder(tree, M)) { /* children before parents */
        full.c1[next] = top[tree.c1[u]];
        full.c2[next] = top[tree.c2[u]];
        top[u] = next;
        if (with_bl && u != 2 * M - 2) /* the root has no edge above it */
            full.branch_lengths[next] = tree.branch_lengths[u];
        ++next;
    }
    return full;
}

Tree prune(const Tree &tree, const Deduplication &dd) {
    const int32_t N = (int32_t)dd.copy_of.size(), M = (int32_t)dd.kept.size();
    if ((int32_t)tree.c1.size() != 2 * N - 1)
        throw std::invalid_argument("prune: the tree is not on all sequences");
    std::vector<int32_t> new_id(2 * N - 1, -1); /* reduced id of a kept leaf or of a kept internal node; -1 = gone */
    for (int32_t k = 0; k < M; ++k)
        new_id[dd.kept[k]] = k;
    Tree out;
    out.c1.assign(2 * M - 1, -1);
    out.c2.assign(2 * M - 1, -1);
    int32_t next = M;
    for (int32_t u : internal_postorder(tree, N)) { /* children before parents */
        const int32_t a = new_id[tree.c1[u]], b = new_id[tree.c2[u]];
        if (a >= 0 && b >= 0) {
            out.c1[next] = a;
            out.c2[next] = b;
            new_id[u] = next++;
        } else {
            new_id[u] = a >= 0 ? a : b; /* a node with a removed child is replaced by the other child */
        }
    }
    if (next != 2 * M - 1)
        throw std::logic_error("prune: wrong number of nodes");
    return out;
}

Tree read_newick(const std::string &path, const std::vector<std::string> &names) {
    FILE *f = fopen(path.c_str(), "r");
    if (!f)
        throw std::runtime_error("cannot open " + path);
    std::string s;
    char buf[1 << 16];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0)
        s.append(buf, n);
    fclose(f);

    const int32_t N = (int32_t)names.size();
    std::unordered_map<std::string, int32_t> leaf_id;
    for (int32_t i = 0; i < N; ++i)
        leaf_id.emplace(names[i], i);

    /* Parse into nodes: leaves carry their alignment index, internal nodes their children. */
    struct Node {
        int32_t leaf = -1;
        std::vector<int32_t> kids;
    };
    std::vector<Node> nodes;
    std::vector<int32_t> open; /* stack of internal nodes being filled */
    std::vector<uint8_t> used(N, 0);
    int32_t root = -1;
    size_t i = 0;
    auto skip_ws = [&]() {
        while (i < s.size() && isspace((unsigned char)s[i]))
            ++i;
    };
    auto skip_length = [&]() {
        skip_ws();
        if (i < s.size() && s[i] == ':') {
            ++i;
            while (i < s.size() && s[i] != ',' && s[i] != ')' && s[i] != ';')
                ++i;
        }
    };
    auto read_label = [&]() {
        std::string label;
        skip_ws();
        if (i < s.size() && s[i] == '\'') {
            ++i;
            while (i < s.size()) {
                if (s[i] == '\'') {
                    if (i + 1 < s.size() && s[i + 1] == '\'') {
                        label.push_back('\'');
                        i += 2;
                        continue;
                    }
                    ++i;
                    break;
                }
                label.push_back(s[i++]);
            }
        } else {
            while (i < s.size() && !strchr("(),:;", s[i]) && !isspace((unsigned char)s[i]))
                label.push_back(s[i++]);
        }
        return label;
    };
    while (true) {
        skip_ws();
        if (i >= s.size())
            throw std::runtime_error(path + ": Newick ends before ';'");
        const char c = s[i];
        if (c == '(') {
            nodes.push_back(Node());
            const int32_t id = (int32_t)nodes.size() - 1;
            if (!open.empty())
                nodes[open.back()].kids.push_back(id);
            open.push_back(id);
            ++i;
        } else if (c == ',') {
            ++i;
        } else if (c == ')') {
            if (open.empty())
                throw std::runtime_error(path + ": unbalanced ')'");
            const int32_t id = open.back();
            open.pop_back();
            ++i;
            read_label(); /* internal labels (e.g. support values) are ignored */
            skip_length();
            if (open.empty())
                root = id;
        } else if (c == ';') {
            break;
        } else {
            const std::string label = read_label();
            if (label.empty())
                throw std::runtime_error(path + ": unexpected character '" + std::string(1, c) + "'");
            auto it = leaf_id.find(label);
            if (it == leaf_id.end())
                throw std::runtime_error(path + ": leaf '" + label + "' is not in the alignment");
            if (used[it->second])
                throw std::runtime_error(path + ": leaf '" + label + "' appears twice");
            used[it->second] = 1;
            nodes.push_back(Node());
            nodes.back().leaf = it->second;
            if (open.empty())
                throw std::runtime_error(path + ": leaf outside parentheses");
            nodes[open.back()].kids.push_back((int32_t)nodes.size() - 1);
            skip_length();
        }
    }
    if (root < 0 || !open.empty())
        throw std::runtime_error(path + ": unbalanced parentheses");
    for (int32_t x = 0; x < N; ++x)
        if (!used[x])
            throw std::runtime_error(path + ": alignment sequence '" + names[x] + "' is not in the tree");

    /* An unrooted tree (three children at the root) is rooted by joining the first two. */
    if (nodes[root].kids.size() == 3) {
        Node joined;
        joined.kids = {nodes[root].kids[0], nodes[root].kids[1]};
        nodes.push_back(joined);
        nodes[root].kids = {(int32_t)nodes.size() - 1, nodes[root].kids[2]};
    }
    for (const Node &nd : nodes)
        if (nd.leaf < 0 && nd.kids.size() != 2)
            throw std::runtime_error(path + ": the start tree must be binary (a node has " +
                                     std::to_string(nd.kids.size()) + " children)");

    /* Number internal nodes in post-order: N, N+1, ..., root = 2N - 2. */
    Tree t;
    t.c1.assign(2 * N - 1, -1);
    t.c2.assign(2 * N - 1, -1);
    std::vector<int32_t> id_of(nodes.size(), -1);
    int32_t next = N;
    std::vector<std::pair<int32_t, int32_t>> st;
    st.push_back(std::make_pair(root, 0));
    while (!st.empty()) {
        const int32_t u = st.back().first;
        if (nodes[u].leaf >= 0) {
            id_of[u] = nodes[u].leaf;
            st.pop_back();
            continue;
        }
        const int32_t state = st.back().second;
        if (state < 2) {
            st.back().second = state + 1;
            st.push_back(std::make_pair(nodes[u].kids[state], 0));
            continue;
        }
        id_of[u] = next++;
        t.c1[id_of[u]] = id_of[nodes[u].kids[0]];
        t.c2[id_of[u]] = id_of[nodes[u].kids[1]];
        st.pop_back();
    }
    if (next != 2 * N - 1)
        throw std::runtime_error(path + ": the tree does not have " + std::to_string(N) + " leaves");
    return t;
}

void write_newick_unrooted(const std::string &path, const std::vector<std::string> &names, const Tree &tree,
                           bool neg_as_zero) {
    const int32_t N = (int32_t)names.size();
    if (N < 3)
        throw std::runtime_error("an unrooted tree needs at least 3 leaves");
    FILE *f = fopen(path.c_str(), "w");
    if (!f)
        throw std::runtime_error("cannot write " + path);
    setvbuf(f, NULL, _IOFBF, 1 << 24);
    const bool with_bl = !tree.branch_lengths.empty();
    const int32_t root = 2 * N - 2;
    const int32_t a = tree.c1[root], b = tree.c2[root];
    /* The root's internal child becomes the trifurcation; its sibling hangs from it by the merged edge. */
    const int32_t centre = a >= N ? a : b;
    const int32_t sibling = a >= N ? b : a;
    /* the length of the edge above u, in the lengths' precision (float: the sum of the root edge in float) */
    auto edge = [&](int32_t u) -> double {
        if (tree.double_precision_lengths) {
            const double len = u == sibling ? tree.branch_lengths[a] + tree.branch_lengths[b] : tree.branch_lengths[u];
            return (neg_as_zero && !(len > 0.0)) ? 0.0 : len;
        }
        const float len = u == sibling ? (float)tree.branch_lengths[a] + (float)tree.branch_lengths[b]
                                       : (float)tree.branch_lengths[u];
        return (neg_as_zero && !(len > 0.0f)) ? 0.0f : len;
    };
    /* ":length", the shortest decimal that reads back to the same value of the lengths' type, float or double
     */
    auto put_length = [&](double len) {
        char buf[40];
        buf[0] = ':';
        const char *end = tree.double_precision_lengths ? std::to_chars(buf + 1, buf + sizeof(buf), len).ptr
                                                        : std::to_chars(buf + 1, buf + sizeof(buf), (float)len).ptr;
        fwrite(buf, 1, (size_t)(end - buf), f);
    };
    auto write_subtree = [&](int32_t top) {
        std::vector<std::pair<int32_t, int32_t>> st;
        st.push_back(std::make_pair(top, 0));
        while (!st.empty()) {
            const int32_t u = st.back().first;
            if (u < N) {
                write_name(f, names[u]);
                if (with_bl)
                    put_length(edge(u));
                st.pop_back();
                continue;
            }
            const int32_t state = st.back().second;
            if (state == 0) {
                fputc('(', f);
                st.back().second = 1;
                st.push_back(std::make_pair(tree.c1[u], 0));
            } else if (state == 1) {
                fputc(',', f);
                st.back().second = 2;
                st.push_back(std::make_pair(tree.c2[u], 0));
            } else {
                fputc(')', f);
                if (with_bl)
                    put_length(edge(u));
                st.pop_back();
            }
        }
    };
    fputc('(', f);
    write_subtree(tree.c1[centre]);
    fputc(',', f);
    write_subtree(tree.c2[centre]);
    fputc(',', f);
    write_subtree(sibling);
    fputs(");\n", f);
    if (fclose(f) != 0)
        throw std::runtime_error("cannot write " + path);
}

} // namespace p2vbme
