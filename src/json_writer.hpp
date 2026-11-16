/*
 * json_writer.hpp
 *
 * A small JSON writer for the run report: objects and arrays with keys in
 * written order, escaped strings, numbers as the shortest decimal that reads back to the same value
 * (std::to_chars, independent of the locale), NaN and infinities as null, 64-bit integers exactly.
 * Indented by two spaces per level.
 */
#pragma once

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include <charconv>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace p2vbme {

class JsonWriter {
   public:
    explicit JsonWriter(std::string &out) : out_(out) {}

    // Open or close an object or array
    void begin_object() { open('{'); }
    void end_object() { close('}'); }
    void begin_array() { open('['); }
    void end_array() { close(']'); }

    /* The key of the next value; only inside an object. */
    void key(std::string_view k) {
        if (stack_.empty() || stack_.back().kind != '{' || pending_key_)
            throw std::logic_error("JsonWriter: key outside an object or twice");
        separator();
        write_string(k);
        out_ += ": ";
        pending_key_ = true;
    }

    // Next element of an array, or value after key() in an object.
    void value(std::string_view s) {
        before_value();
        write_string(s);
    }
    void value(const char *s) { value(std::string_view(s)); }
    void value(const std::string &s) { value(std::string_view(s)); }
    void value(bool b) {
        before_value();
        out_ += b ? "true" : "false";
    }
    void value(int32_t x) { integer((int64_t)x); }
    void value(int64_t x) { integer(x); }
    void value(uint64_t x) {
        before_value();
        char buf[24];
        out_.append(buf, std::to_chars(buf, buf + sizeof(buf), x).ptr);
    }
    void value(double x) { real(x); }
    void value(float x) { real(x); }
    void null() {
        before_value();
        out_ += "null";
    }

    /* key + value */
    template <class T>
    void field(std::string_view k, const T &v) {
        key(k);
        value(v);
    }
    template <class T>
    void array_field(std::string_view k, const std::vector<T> &v) {
        key(k);
        begin_array();
        for (const T &x : v) value(x);
        end_array();
    }

    /* Throws unless every object and array has been closed. */
    void finish() {
        if (!stack_.empty() || pending_key_)
            throw std::logic_error("JsonWriter: unclosed object or array");
        out_ += '\n';
    }

   private:
    struct Level {
        char kind;
        bool empty;
    };

    void indent() {
        out_ += '\n';
        out_.append(2 * stack_.size(), ' ');
    }
    /* comma and line break before an element of the current object or array */
    void separator() {
        if (!stack_.back().empty) out_ += ',';
        stack_.back().empty = false;
        indent();
    }
    void before_value() {
        if (stack_.empty()) {
            if (!out_.empty()) throw std::logic_error("JsonWriter: more than one top-level value");
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
    void open(char kind) {
        before_value();
        out_ += kind;
        stack_.push_back({kind, true});
    }
    void close(char kind) {
        const char open_kind = kind == '}' ? '{' : '[';
        if (stack_.empty() || stack_.back().kind != open_kind || pending_key_)
            throw std::logic_error("JsonWriter: mismatched close");
        const bool empty = stack_.back().empty;
        stack_.pop_back();
        if (!empty) indent();
        out_ += kind;
    }
    void integer(int64_t x) {
        before_value();
        char buf[24];
        out_.append(buf, std::to_chars(buf, buf + sizeof(buf), x).ptr);
    }
    template <class F>
    void real(F x) {
        before_value();
        if (!isfinite(x)) {
            out_ += "null";
            return;
        }
        char buf[40];
        out_.append(buf, std::to_chars(buf, buf + sizeof(buf), x).ptr);
    }
    /* Length of the valid UTF-8 sequence starting at s[i] (2-4), or 0 if it is not valid UTF-8. */
    static size_t utf8_length(std::string_view s, size_t i) {
        const unsigned char c = (unsigned char)s[i];
        size_t n = c >= 0xF0 && c <= 0xF4 ? 4 : c >= 0xE0 ? 3 : c >= 0xC2 && c <= 0xDF ? 2 : 0;
        if (n == 0 || i + n > s.size()) return 0;
        for (size_t k = 1; k < n; ++k)
            if (((unsigned char)s[i + k] & 0xC0) != 0x80) return 0;
        const unsigned char d = (unsigned char)s[i + 1];
        if ((c == 0xE0 && d < 0xA0) || (c == 0xED && d > 0x9F) || (c == 0xF0 && d < 0x90) ||
            (c == 0xF4 && d > 0x8F))
            return 0; /* overlong, surrogate or beyond U+10FFFF */
        return n;
    }

    void write_string(std::string_view s) {
        out_ += '"';
        for (size_t i = 0; i < s.size(); ++i) {
            const unsigned char c = (unsigned char)s[i];
            if (c >= 0x80) { /* valid UTF-8 passes through; anything else becomes U+FFFD, so the
                                output stays JSON */
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
                    if (c < 0x20) {
                        char buf[8];
                        snprintf(buf, sizeof(buf), "\\u%04x", c);
                        out_ += buf;
                    } else {
                        out_ += (char)c;
                    }
            }
        }
        out_ += '"';
    }

    std::string &out_;
    std::vector<Level> stack_;
    bool pending_key_ = false;
};

}  // namespace p2vbme
