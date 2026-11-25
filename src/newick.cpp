/*
 * newick.cpp
 *
 * Newick start-tree input and unrooted Newick output (one place for the quoting rules).
 * Tree traversals are iterative so that trees with millions of leaves do not overflow the stack.
 */
#include <charconv>
#include <ctype.h>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "newick.hpp"

namespace vecme {

namespace {

bool needs_quotes(const std::string &name) { return name.find_first_of(" \t()[]':;,") != std::string::npos; }

void write_name(std::ostream &out, const std::string &name) {
    if (!needs_quotes(name)) {
        out << name;
        return;
    }
    out.put('\'');
    for (char ch : name) {
        if (ch == '\'')
            out.put('\'');
        out.put(ch);
    }
    out.put('\'');
}

} // namespace

namespace {

// Parses the Newick tree that starts at s[i], up to its ';' (i then points after it).
// Leaves: by name in names; if learn, names starts empty and receives the leaf names in the order they appear.
// lengths: keep the branch lengths (as double), if the tree has them.
// Errors name `source` (a file, or a file and a tree number).
Tree parse_tree(const std::string &s, size_t &i, const std::string &source, std::vector<std::string> &names, bool learn,
                bool lengths) {
    std::unordered_map<std::string, int32_t> leaf_id;
    for (int32_t k = 0; k < (int32_t)names.size(); ++k)
        leaf_id.emplace(names[k], k);

    // Parse into nodes: leaves carry their alignment index, internal nodes their children.
    struct Node {
        int32_t leaf = -1;
        std::vector<int32_t> kids;
        double length = 0.0;
    };
    std::vector<Node> nodes;
    std::vector<int32_t> open; // stack of internal nodes being filled
    std::vector<uint8_t> used(names.size(), 0);
    bool has_lengths = false;
    int32_t root = -1;
    auto skip_ws = [&]() {
        while (i < s.size() && isspace((unsigned char)s[i]))
            ++i;
    };
    // ":length" after a node: kept as the node's length if lengths, else skipped
    auto read_length = [&](int32_t node) {
        skip_ws();
        if (i < s.size() && s[i] == ':') {
            ++i;
            const size_t start = i;
            while (i < s.size() && s[i] != ',' && s[i] != ')' && s[i] != ';')
                ++i;
            if (lengths) {
                const std::string_view text(s.data() + start, i - start);
                size_t a = 0, b = text.size();
                while (a < b && isspace((unsigned char)text[a]))
                    ++a;
                while (b > a && isspace((unsigned char)text[b - 1]))
                    --b;
                double len = 0.0;
                const auto [end, ec] = std::from_chars(text.data() + a, text.data() + b, len);
                if (ec != std::errc() || end != text.data() + b)
                    throw std::runtime_error(source + ": bad branch length '" + std::string(text) + "'");
                nodes[node].length = len;
                has_lengths = true;
            }
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
            while (i < s.size() && std::string_view("(),:;").find(s[i]) == std::string_view::npos &&
                   !isspace((unsigned char)s[i]))
                label.push_back(s[i++]);
        }
        return label;
    };
    while (true) {
        skip_ws();
        if (i >= s.size())
            throw std::runtime_error(source + ": Newick ends before ';'");
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
                throw std::runtime_error(source + ": unbalanced ')'");
            const int32_t id = open.back();
            open.pop_back();
            ++i;
            read_label(); // internal labels (e.g. support values) are ignored
            read_length(id);
            if (open.empty())
                root = id;
        } else if (c == ';') {
            ++i;
            break;
        } else {
            const std::string label = read_label();
            if (label.empty())
                throw std::runtime_error(source + ": unexpected character '" + std::string(1, c) + "'");
            auto it = leaf_id.find(label);
            if (it == leaf_id.end()) {
                if (!learn)
                    throw std::runtime_error(source + ": leaf '" + label + "' is not in the alignment");
                it = leaf_id.emplace(label, (int32_t)names.size()).first;
                names.push_back(label);
                used.push_back(0);
            }
            if (used[it->second])
                throw std::runtime_error(source + ": leaf '" + label + "' appears twice");
            used[it->second] = 1;
            nodes.push_back(Node());
            nodes.back().leaf = it->second;
            if (open.empty())
                throw std::runtime_error(source + ": leaf outside parentheses");
            nodes[open.back()].kids.push_back((int32_t)nodes.size() - 1);
            read_length((int32_t)nodes.size() - 1);
        }
    }
    if (root < 0 || !open.empty())
        throw std::runtime_error(source + ": unbalanced parentheses");
    const int32_t N = (int32_t)names.size();
    for (int32_t x = 0; x < N; ++x)
        if (!used[x])
            throw std::runtime_error(source + ": alignment sequence '" + names[x] + "' is not in the tree");

    // An unrooted tree (three children at the root) is rooted by joining the first two, with a zero-length edge.
    if (nodes[root].kids.size() == 3) {
        Node joined;
        joined.kids = {nodes[root].kids[0], nodes[root].kids[1]};
        nodes.push_back(joined);
        nodes[root].kids = {(int32_t)nodes.size() - 1, nodes[root].kids[2]};
    }
    for (const Node &nd : nodes)
        if (nd.leaf < 0 && nd.kids.size() != 2)
            throw std::runtime_error(source + ": the tree must be binary (a node has " +
                                     std::to_string(nd.kids.size()) + " children)");

    // Number internal nodes in post-order: N, N+1, ..., root = 2N - 2.
    Tree t;
    t.c1.assign(2 * N - 1, -1);
    t.c2.assign(2 * N - 1, -1);
    if (has_lengths) {
        t.branch_lengths.assign(2 * N - 2, 0.0);
        t.double_precision_lengths = true;
    }
    std::vector<int32_t> id_of(nodes.size(), -1);
    int32_t next = N;
    std::vector<std::pair<int32_t, int32_t>> st;
    st.push_back(std::make_pair(root, 0));
    while (!st.empty()) {
        const int32_t u = st.back().first;
        if (nodes[u].leaf >= 0) {
            id_of[u] = nodes[u].leaf;
            st.pop_back();
        } else {
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
        if (has_lengths && u != root)
            t.branch_lengths[id_of[u]] = nodes[u].length;
    }
    if (next != 2 * N - 1)
        throw std::runtime_error(source + ": the tree does not have " + std::to_string(N) + " leaves");
    return t;
}

// The whole file in one string; the file closes itself, also when an error is thrown.
std::string read_text(const std::string &path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("cannot open " + path);
    std::ostringstream text;
    text << in.rdbuf();
    return text.str();
}

} // namespace

Tree read_newick(const std::string &path, const std::vector<std::string> &names) {
    const std::string s = read_text(path);
    std::vector<std::string> leaves = names;
    size_t i = 0;
    return parse_tree(s, i, path, leaves, false, false);
}

Tree read_newick_with_names(const std::string &path, std::vector<std::string> &names) {
    const std::string s = read_text(path);
    names.clear();
    size_t i = 0;
    return parse_tree(s, i, path, names, true, true);
}

int32_t read_newick_trees(const std::string &path, std::vector<std::string> &names,
                          const std::function<void(const Tree &)> &fn) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("cannot open " + path);
    names.clear();
    // one tree at a time: the lines up to its ';'
    std::string text, line;
    int32_t n_trees = 0;
    while (std::getline(in, line)) {
        text += line;
        text.push_back('\n');
        size_t i = 0;
        while (text.find(';', i) != std::string::npos) {
            const std::string source = path + " (tree " + std::to_string(n_trees + 1) + ")";
            fn(parse_tree(text, i, source, names, n_trees == 0, false));
            ++n_trees;
        }
        text.erase(0, i);
    }
    if (in.bad())
        throw std::runtime_error("cannot read " + path);
    for (const char c : text)
        if (!isspace((unsigned char)c))
            throw std::runtime_error(path + ": Newick ends before ';'");
    return n_trees;
}

namespace {

// Writes the tree unrooted to a stream, as write_newick_unrooted describes, ending with ";\n".
void put_newick_unrooted(std::ostream &out, const std::vector<std::string> &names, const Tree &tree, bool neg_as_zero,
                         const std::vector<std::string> &labels) {
    const int32_t N = (int32_t)names.size();
    if (N < 3)
        throw std::runtime_error("an unrooted tree needs at least 3 leaves");
    const bool with_bl = !tree.branch_lengths.empty();
    const int32_t root = 2 * N - 2;
    if (!labels.empty() && labels.size() != tree.c1.size())
        throw std::runtime_error("one label per node of the tree is needed");
    const int32_t a = tree.c1[root], b = tree.c2[root];
    // The root's internal child becomes the trifurcation; its sibling hangs from it by the merged edge.
    const int32_t centre = a >= N ? a : b;
    const int32_t sibling = a >= N ? b : a;
    // the length of the edge above u, in the lengths' precision (float: the sum of the root edge in float)
    auto edge = [&](int32_t u) -> double {
        if (tree.double_precision_lengths) {
            const double len = u == sibling ? tree.branch_lengths[a] + tree.branch_lengths[b] : tree.branch_lengths[u];
            return (neg_as_zero && !(len > 0.0)) ? 0.0 : len;
        }
        const float len = u == sibling ? (float)tree.branch_lengths[a] + (float)tree.branch_lengths[b]
                                       : (float)tree.branch_lengths[u];
        return (neg_as_zero && !(len > 0.0f)) ? 0.0f : len;
    };
    // ":length", the shortest decimal that reads back to the same value of the lengths' type, float or double
    auto put_length = [&](double len) {
        char buf[40];
        buf[0] = ':';
        const char *end = tree.double_precision_lengths ? std::to_chars(buf + 1, buf + sizeof(buf), len).ptr
                                                        : std::to_chars(buf + 1, buf + sizeof(buf), (float)len).ptr;
        out.write(buf, end - buf);
    };
    auto write_subtree = [&](int32_t top) {
        std::vector<std::pair<int32_t, int32_t>> st;
        st.push_back(std::make_pair(top, 0));
        while (!st.empty()) {
            const int32_t u = st.back().first;
            if (u < N) {
                write_name(out, names[u]);
                if (with_bl)
                    put_length(edge(u));
                st.pop_back();
                continue;
            }
            const int32_t state = st.back().second;
            if (state == 0) {
                out.put('(');
                st.back().second = 1;
                st.push_back(std::make_pair(tree.c1[u], 0));
            } else if (state == 1) {
                out.put(',');
                st.back().second = 2;
                st.push_back(std::make_pair(tree.c2[u], 0));
            } else {
                out.put(')');
                if (!labels.empty())
                    out << labels[u];
                if (with_bl)
                    put_length(edge(u));
                st.pop_back();
            }
        }
    };
    out.put('(');
    write_subtree(tree.c1[centre]);
    out.put(',');
    write_subtree(tree.c2[centre]);
    out.put(',');
    write_subtree(sibling);
    out << ");\n";
}

// Writes the tree to path, appending or not: (path, mode, tree...) -> the file.
void put_newick_file(const std::string &path, std::ios::openmode mode, const std::vector<std::string> &names,
                     const Tree &tree, bool neg_as_zero, const std::vector<std::string> &labels) {
    std::ofstream out(path, mode);
    if (!out)
        throw std::runtime_error("cannot write " + path);
    put_newick_unrooted(out, names, tree, neg_as_zero, labels);
    // close() flushes; a failed write or flush leaves the stream failed.
    out.close();
    if (!out)
        throw std::runtime_error("cannot write " + path);
}

} // namespace

void write_newick_unrooted(const std::string &path, const std::vector<std::string> &names, const Tree &tree,
                           bool neg_as_zero, const std::vector<std::string> &labels) {
    put_newick_file(path, std::ios::out, names, tree, neg_as_zero, labels);
}

void append_newick_unrooted(const std::string &path, const std::vector<std::string> &names, const Tree &tree) {
    put_newick_file(path, std::ios::app, names, tree, true, {});
}

} // namespace vecme
