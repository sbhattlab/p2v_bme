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

#include <stdint.h>

#include <string>
#include <string_view>
#include <vector>

namespace p2vbme {

class JsonWriter {
  public:
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
        char kind;
        bool empty;
    };

    void indent();
    // comma and line break before an element of the current object or array
    void separator();
    void before_value();
    void open(char kind);
    void close(char kind);
    void integer(int64_t x);
    template <class F> void real(F x);
    void write_string(std::string_view s);

    std::string &out_;
    std::vector<Level> stack_;
    bool pending_key_ = false;
};

} // namespace p2vbme

#endif // P2VBME_JSON_WRITER_HPP
