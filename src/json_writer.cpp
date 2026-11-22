/*
 * json_writer.cpp
 *
 * The JSON writer of json_writer.hpp.
 *
 * State: out_ (the text so far), stack_ (the objects and arrays still open, innermost last; each knows its bracket
 * and whether it is still empty) and pending_key_ (a key was written, its value not yet). Every key, and every value
 * in an array, first gets its prefix from separator(): a comma unless it is the first element, then a line break
 * and the indentation. A value in an object needs no prefix: it follows its "key": on the same line.
 * The comments above the functions below give their steps, in order, separated by "->".
 */
#include "json_writer.hpp"

#include <charconv>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace p2vbme {

namespace {

// The digits of a \uXXXX escape, lower case
constexpr char HEX_DIGITS[] = "0123456789abcdef";
// Characters in the longest integers, -9223372036854775808 (INT64_MIN) and 18446744073709551615 (UINT64_MAX)
constexpr size_t MAX_INTEGER_CHARS = 20;
// Characters in the longest double std::to_chars writes: sign, 17 digits, point and exponent, e.g.
// -2.2250738585072014e-308.
// Floats are shorter.
constexpr size_t MAX_REAL_CHARS = 24;

// Length of the valid UTF-8 sequence starting at s[i] (2-4), or 0 if it is not valid UTF-8.
// length from the first byte -> must fit in s -> the next bytes must be continuation bytes (10xxxxxx) -> second-byte
// limits that rule out overlong forms, surrogates and code points beyond U+10FFFF
size_t utf8_length(std::string_view s, size_t i) {
    const unsigned char c = (unsigned char)s[i];
    size_t n = c >= 0xF0 && c <= 0xF4 ? 4 : c >= 0xE0 && c <= 0xEF ? 3 : c >= 0xC2 && c <= 0xDF ? 2 : 0;
    if (n == 0 || i + n > s.size())
        return 0;
    for (size_t k = 1; k < n; ++k)
        if (((unsigned char)s[i + k] & 0xC0) != 0x80)
            return 0;
    const unsigned char d = (unsigned char)s[i + 1];
    if ((c == 0xE0 && d < 0xA0) || (c == 0xED && d > 0x9F) || (c == 0xF0 && d < 0x90) || (c == 0xF4 && d > 0x8F))
        return 0; // overlong, surrogate or beyond U+10FFFF
    return n;
}

} // namespace

// open() and close() do the work: checks, bracket, stack
void JsonWriter::begin_object() { open('{'); }
void JsonWriter::end_object() { close('}'); }
void JsonWriter::begin_array() { open('['); }
void JsonWriter::end_array() { close(']'); }

// check a key may go here -> prefix (separator) -> "key": -> pending_key_ = true (a value must follow)
void JsonWriter::key(std::string_view k) {
    // Throws at top level, directly inside an array, or right after another key (which still needs its value)
    if (stack_.empty() || stack_.back().kind != '{' || pending_key_)
        throw std::logic_error("JsonWriter: key outside an object or twice");
    separator();
    write_string(k);
    out_ += ": ";
    pending_key_ = true;
}

// Every value: before_value() (checks, and the prefix in an array) -> the value itself

// before_value -> quoted, escaped string (write_string)
void JsonWriter::value(std::string_view s) {
    before_value();
    write_string(s);
}
// -> the string_view version
void JsonWriter::value(const char *s) { value(std::string_view(s)); }
void JsonWriter::value(const std::string &s) { value(std::string_view(s)); }
// before_value -> true or false
void JsonWriter::value(bool b) {
    before_value();
    out_ += b ? "true" : "false";
}
// -> integer()
void JsonWriter::value(int32_t x) { integer((int64_t)x); }
void JsonWriter::value(int64_t x) { integer(x); }
// before_value -> digits (own function: values above INT64_MAX do not fit integer()'s int64_t)
void JsonWriter::value(uint64_t x) {
    before_value();
    char buf[MAX_INTEGER_CHARS];
    out_.append(buf, std::to_chars(buf, buf + sizeof(buf), x).ptr);
}
// -> real()
void JsonWriter::value(double x) { real(x); }
void JsonWriter::value(float x) { real(x); }
// before_value -> null
void JsonWriter::null() {
    before_value();
    out_ += "null";
}

// check nothing is still open and no key waits for its value -> final line break
void JsonWriter::finish() {
    if (!stack_.empty() || pending_key_)
        throw std::logic_error("JsonWriter: unclosed object or array");
    out_ += '\n';
}

// line break -> two spaces per open object or array
void JsonWriter::indent() {
    out_ += '\n';
    out_.append(2 * stack_.size(), ' ');
}

// comma unless this is the container's first element -> mark the container non-empty -> indent()
void JsonWriter::separator() {
    if (!stack_.back().empty)
        out_ += ',';
    stack_.back().empty = false;
    indent();
}

// top level: only one value, no prefix | in an object: key() must come first, pending_key_ = false, no prefix |
// in an array: separator()
void JsonWriter::before_value() {
    if (stack_.empty()) {
        if (!out_.empty())
            throw std::logic_error("JsonWriter: more than one top-level value");
        return;
    }
    if (stack_.back().kind == '{') {
        if (!pending_key_)
            throw std::logic_error("JsonWriter: value without a key in an object");
        pending_key_ = false;
    } else {
        separator();
    }
}

// before_value (an object or array is itself a value) -> { or [ -> push a new, empty level
void JsonWriter::open(char kind) {
    before_value();
    out_ += kind;
    stack_.push_back({kind, true});
}

// check it closes the innermost level and no key waits -> pop the level -> indent() unless it was empty -> } or ]
void JsonWriter::close(char kind) {
    const char open_kind = kind == '}' ? '{' : '[';
    if (stack_.empty() || stack_.back().kind != open_kind || pending_key_)
        throw std::logic_error("JsonWriter: mismatched close");
    const bool empty = stack_.back().empty;
    stack_.pop_back();
    if (!empty)
        indent();
    out_ += kind;
}

// before_value -> digits (std::to_chars)
void JsonWriter::integer(int64_t x) {
    before_value();
    char buf[MAX_INTEGER_CHARS];
    out_.append(buf, std::to_chars(buf, buf + sizeof(buf), x).ptr);
}

// before_value -> null if NaN or infinite (JSON has neither) -> else the shortest decimal that reads back the same
// (std::to_chars)
template <class F> void JsonWriter::real(F x) {
    before_value();
    if (!std::isfinite(x)) {
        out_ += "null";
        return;
    }
    char buf[MAX_REAL_CHARS];
    out_.append(buf, std::to_chars(buf, buf + sizeof(buf), x).ptr);
}

// opening quote -> each byte: valid UTF-8 copied as is, invalid UTF-8 -> U+FFFD, quote and backslash escaped,
// control characters as \n, \t, ... or \u00XX -> closing quote
void JsonWriter::write_string(std::string_view s) {
    out_ += '"';
    for (size_t i = 0; i < s.size(); ++i) {
        const unsigned char c = (unsigned char)s[i];
        if (c >= 0x80) { // valid UTF-8 passes through; anything else becomes U+FFFD, so the output stays JSON
            const size_t n = utf8_length(s, i);
            if (n == 0) {
                out_ += "\\ufffd";
            } else {
                out_.append(s.substr(i, n));
                i += n - 1;
            }
            continue;
        }
        switch (c) {
        case '"':
            out_ += "\\\"";
            break;
        case '\\':
            out_ += "\\\\";
            break;
        case '\b':
            out_ += "\\b";
            break;
        case '\f':
            out_ += "\\f";
            break;
        case '\n':
            out_ += "\\n";
            break;
        case '\r':
            out_ += "\\r";
            break;
        case '\t':
            out_ += "\\t";
            break;
        default:
            if (c < 0x20) { // \u00XX: c fits in the last two hex digits
                out_ += "\\u00";
                out_ += HEX_DIGITS[c >> 4];
                out_ += HEX_DIGITS[c & 0xF];
            } else {
                out_ += (char)c;
            }
        }
    }
    out_ += '"';
}

} // namespace p2vbme
