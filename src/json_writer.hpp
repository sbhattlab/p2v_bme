/*
 * json_writer.hpp
 *
 * A small JSON writer for the run report: objects and arrays with keys in
 * written order, escaped strings, numbers as the shortest decimal that reads back to the same value
 * (std::to_chars, independent of the locale), NaN and infinities as null, 64-bit integers exactly.
 * Indented by two spaces per level.
 */
#ifndef P2VBME_JSON_WRITER_HPP
#define P2VBME_JSON_WRITER_HPP

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace p2vbme {

class JsonWriter {
  public:
    // Writes into out, which must start empty
    explicit JsonWriter(std::string &out) : out_(out) {}

    // Open or close an object or array
    void begin_object();
    void end_object();
    void begin_array();
    void end_array();

    // The key of the next value; only inside an object.
    void key(std::string_view k);

    // Next element of an array, or value after key() in an object.
    void value(std::string_view s);
    void value(const char *s);
    void value(const std::string &s);
    void value(bool b);
    void value(int32_t x);
    void value(int64_t x);
    void value(uint64_t x);
    void value(double x);
    void value(float x);
    void null();

    // key + value
    template <class T> void field(std::string_view k, const T &v) {
        key(k);
        value(v);
    }
    // key + the elements of v as an array
    template <class T> void array_field(std::string_view k, const std::vector<T> &v) {
        key(k);
        begin_array();
        for (const T &x : v)
            value(x);
        end_array();
    }

    // Throws unless every object and array has been closed.
    void finish();

  private:
    struct Level {
        // { or [
        char kind;
        // true if the object or array is empty so far
        bool empty;
    };

    // Line break, then two spaces per open level
    void indent();
    // comma and line break before an element of the current object or array
    void separator();
    // Checks a value may go here (after a key in an object, once at top level); in an array, writes the separator
    void before_value();
    // Writes { or [ as the next value and opens a level
    void open(char kind);
    // Writes } or ]; throws unless it matches the innermost open level
    void close(char kind);
    // An integer value, exactly
    void integer(int64_t x);
    // A float or double as the shortest decimal that reads back the same; NaN and infinities as null
    template <class F> void real(F x);
    // s as a quoted JSON string: escapes ", \ and control characters; invalid UTF-8 becomes U+FFFD
    void write_string(std::string_view s);

    // The output, appended to
    std::string &out_;
    // List of objects/arrays that are currently open, innermost last
    std::vector<Level> stack_;
    // true between key() and its value
    bool pending_key_ = false;
};

} // namespace p2vbme

#endif // P2VBME_JSON_WRITER_HPP
