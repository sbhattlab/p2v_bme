/*
 * io.cpp
 *
 * Alignment input by file type, Newick start-tree input and unrooted Newick output.
 * Tree traversals are iterative so that trees with millions of leaves do not overflow the stack.
 */
#include <charconv>
#include <ctype.h>
#include <stdexcept>
#include <stdio.h>
#include <string.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "fasta.hpp"
#include "io.hpp"
#include "maple.hpp"

namespace p2vbme {

namespace {

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

} // namespace

Alignment read_alignment(const std::string &path) {
    std::string ext = path.size() >= 6 ? path.substr(path.size() - 6) : "";
    for (char &c : ext)
        c = (char)tolower((unsigned char)c);
    return ext == ".maple" ? read_maple(path) : read_fasta(path);
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
